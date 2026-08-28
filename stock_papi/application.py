# app.py
# v5.8 穩定版：新增首頁健康檢查端點防休眠，並梳理重複路由確保 Flask 正常啟動
# --------------------------------------------------

import os
import queue
import threading
import time
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import requests
import twstock
import json
import hmac
import re

from market_insights import build_industries, build_supply_chains
from reporting.config import MAX_CANONICAL_REPORT_BYTES

# Compatibility export: legacy tests still patch app.render_template.
from flask import render_template, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, PostbackEvent, TextMessage, TextSendMessage,
)
from line_state import (
    FirestoreStore, StateError, StoreError, add_alert, add_watch,
    consume_pending, evaluate_alert, remove_watch, start_pending, top_signals,
)
from stock_papi.settings import (
    ALERT_TASK_TOKEN,
    BROADCAST_TOKEN,
    FINMIND_PASSWORD,
    FINMIND_USER,
    GCP_PROJECT_ID,
    GEMINI_API_KEY,
    LINE_CHANNEL_ACCESS_TOKEN,
    LINE_CHANNEL_SECRET,
    LINE_STATE_READ_BUDGET_SECONDS,
    LINE_STATE_READ_MAX_WORKERS,
    LOCAL_HOST,
    MARKETAUX_API_TOKEN,
    OPENALICE_API_TOKEN,
    OPENALICE_API_URL,
    QUANT_SNAPSHOT_BUCKET,
    REPORT_INDEX_MAX_BYTES,
    REPORT_PDF_MAX_BYTES,
    SENTIMENT_WINDOW_DAYS,
    SUPABASE_KEY,
    SUPABASE_URL,
)
from stock_papi.runtime import (
    _LazyGeminiModel,
    _LazyModule,
    get_gcp_access_token as _runtime_get_gcp_access_token,
)
from stock_papi.config.capabilities import PredictionCapabilityState
from stock_papi.shared.formatting import clamp as _clamp
from stock_papi.shared.formatting import format_sentiment_summary as _format_sentiment_summary
from stock_papi.shared.formatting import safe_float as _safe_float
from stock_papi.shared.validation import is_crypto_query as _is_crypto_query
from stock_papi.shared.validation import is_us_ticker
from stock_papi.shared.logging import (
    RedactingFormatter,
    install_redacting_formatters,
    redact_secrets,
    safe_exception_text,
)
from stock_papi.services.observation_view import build_stock_observation
from stock_papi.integrations.line.flex import (
    _alert_condition_text,
    _alert_management_card,
    _signal_card,
    _watchlist_card,
    build_alert_menu_flex,
    build_alert_push_flex,
    build_alerts_flex,
    build_calculator_help_flex,
    build_calculator_menu_flex,
    build_line_navigation_flex,
    build_line_summary_card,
    build_observation_watchlist_flex,
    build_stock_observation_flex,
    build_stock_flex_message,
    build_strong_signals_flex,
    build_tutorial_flex,
    build_watchlist_flex,
    build_welcome_flex,
)
from stock_papi.integrations.line.notifications import run_alert_checks
from stock_papi.integrations.line.webhook import register_line_routes
from stock_papi.integrations.line.handlers import (
    find_matching_alert as _line_find_matching_alert,
    handle_message_impl as _line_handle_message_impl,
    handle_postback_impl as _line_handle_postback_impl,
    require_same_pending as _line_require_same_pending,
    resolve_postback_stock as _line_resolve_postback_stock,
)
from stock_papi.integrations.line.presentation import (
    _build_sector_signal_row as _line_build_sector_signal_row,
    _build_stock_row as _line_build_stock_row,
    build_category_quick_reply as _line_build_category_quick_reply,
    build_industry_carousel as _line_build_industry_carousel,
    build_projection_flex as _line_build_projection_flex,
    build_sector_signal_carousel as _line_build_sector_signal_carousel,
)
from stock_papi.integrations.line.state import (
    get_line_state as _line_get_state,
    get_line_state_bounded as _line_get_state_bounded,
    load_sector_signal_snapshot as _line_load_sector_signal_snapshot,
    refresh_sector_signals as _line_refresh_sector_signals,
    save_sector_signal_snapshot as _line_save_sector_signal_snapshot,
    store_error_text as _line_store_error_text,
    update_line_state as _line_update_state,
    _system_document_url as _line_system_document_url,
)
from stock_papi.integrations.market_data.tw_exchange import fetch_market_activity
from stock_papi.integrations.market_data.provider import (
    FinMindFetchError,
    fetch_finmind_dataset as _provider_fetch_finmind_dataset,
    fetch_option_context_history as _provider_fetch_option_context_history,
    fetch_yfinance_price_history as _provider_fetch_yfinance_price_history,
    finmind_login as _provider_finmind_login,
    get_stock_name as _provider_get_stock_name,
    search_stock_code as _provider_search_stock_code,
)
from stock_papi.integrations.market_data.tw_security_master import (
    TaiwanSecurityMasterResolver,
    fetch_taiwan_security_master,
    is_taiwan_symbol,
)
from stock_papi.integrations.news.provider import (
    fetch_marketaux_news as _fetch_marketaux_news,
    fetch_news_rss,
    fetch_stocktwits_sentiment as _fetch_stocktwits_sentiment,
    normalize_and_dedupe,
    parse_marketaux_items,
    parse_news_items,
    parse_stocktwits_sentiment as _parse_stocktwits_sentiment,
)
from stock_papi.repositories.gcs import get_allowed_object
from stock_papi.repositories.dashboard_snapshots import (
    DASHBOARD_CACHE as _DASHBOARD_CACHE,
    load_dashboard_snapshot,
    load_preview_dashboard_snapshot,
)
from stock_papi.repositories.market_insights import (
    MARKET_INSIGHTS_CACHE as _MARKET_INSIGHTS_CACHE,
    load_market_insights,
)
from stock_papi.repositories.quant_snapshots import (
    MAX_QUANT_ARTIFACT_COMPRESSED_BYTES,
    MAX_QUANT_ARTIFACT_UNCOMPRESSED_BYTES,
    QUANT_MANIFEST_CACHE as _QUANT_MANIFEST_CACHE,
    QUANT_MANIFEST_CACHE_SECONDS,
    fetch_quant_snapshot,
    published_quant_manifest,
)
from stock_papi.repositories.prediction_snapshots import (
    PREDICTION_CACHE as _PREDICTION_CACHE,
    load_prediction_snapshot,
)
from stock_papi.repositories.report_store import (
    load_report_index,
    load_report_metadata,
    load_report_metadata_by_sha,
    load_report_pdf,
)
from stock_papi.repositories.auth_store import FirestoreAuthStore
from stock_papi.quant.projection import (
    _annualized_percent,
    calculate_investment_projection,
)
from stock_papi.quant.constants import (
    DATA_QUALITY_FEATURES,
    ENTRY_THRESHOLD,
    MARKET_FEATURES,
    MODEL_FEATURES,
    OPTION_FEATURES,
    PREDICTION_HORIZON,
    PRICE_DIFF_WARNING_THRESHOLD,
    ROUND_TRIP_COST,
)
from stock_papi.quant.data import (
    add_market_context_features as _add_market_context_features,
    add_option_context_features as _add_option_context_features,
    add_price_quality_features as _add_price_quality_features,
    clean_df as _clean_quant_df,
    foreign_flow_mask as _quant_foreign_flow_mask,
    get_data as _get_quant_data,
    market_feature_frame as _quant_market_feature_frame,
    merge_chip_data as _merge_chip_data,
    neutral_market_features as _neutral_quant_market_features,
    option_close_frame as _quant_option_close_frame,
    summarize_foreign_flow as _summarize_foreign_flow,
)
from stock_papi.quant.features import (
    add_prediction_target as _add_prediction_target,
    calc_all as _calc_all,
)
from stock_papi.quant.backtest import (
    build_time_splits as _build_time_splits,
    score_oos_predictions as _score_oos_predictions,
)
from stock_papi.quant.model import (
    run_ai_engine as _run_ai_engine,
    run_latest_inference as _run_latest_inference,
)
from stock_papi.services.model_evidence import sanitize_analysis
from stock_papi.services.sentiment import (
    NEWS_MAJOR_EVENTS,
    NEWS_NEGATIONS,
    NEWS_OPINION_TERMS,
    NEWS_SENTIMENT_RULES,
    aggregate_news_sentiment,
    analyze_sentiment,
    analyze_sentiment_detail,
    score_news_item,
)
from stock_papi.services.dashboard import (
    build_market_heatmap,
    cached_opportunities as _dashboard_cached_opportunities,
    dashboard_sector_cards as _dashboard_sector_cards,
    dashboard_top_picks,
)
from stock_papi.services.auth import LineLoginConfig, utc_now, verify_opaque_token
from stock_papi.services.market import (
    build_market_map as _build_market_map,
    build_sector_signal_snapshot as _build_sector_signal_snapshot,
    find_industry_peers as _find_industry_peers,
    sector_candidates,
    sector_signal_item as _sector_signal_item,
    sector_signal_score,
)
from stock_papi.services.stock_analysis import (
    analyze_cached as _analyze_cached,
    analyze_uncached as _analyze_uncached,
    snapshot_dataframe as _build_snapshot_dataframe,
)
from stock_papi.services.papi import (
    AbsorbResearchService,
    get_ai_insight_for_broadcast as _get_ai_insight_for_broadcast,
)
from stock_papi.services.news import get_news as _get_news
from stock_papi.services.market_insights import (
    market_insights_payload as _market_insights_payload,
)
from stock_papi.web.legacy_html import render_web
from absorb.conversation.context import MemoryContextStore
from absorb.conversation.errors import InputRejected
from absorb.conversation.orchestrator import ConversationOrchestrator
from absorb.conversation.policies import looks_like_prompt_injection, validate_question
from absorb.conversation.provider import GeminiConversationProvider
from absorb.conversation.renderers import render_line
from absorb.conversation.schemas import ConversationAnswer
from absorb.conversation.tools import build_registry, resolve_entities
from absorb.conversation.command_bridge import is_fixed_command
from absorb.conversation.metrics import record_metric








pd = _LazyModule("pandas")
np = _LazyModule("numpy")

# ==================================================
# 1. 基本設定與系統快取
# ==================================================
finmind_token = None
_line_state_read_slots = threading.BoundedSemaphore(LINE_STATE_READ_MAX_WORKERS)

APPLICATION_ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLE_REPORT_FILENAME = "absorb-tw-industry-daily-SAMPLE.pdf"
_SAMPLE_DIRECTORY = os.path.join(APPLICATION_ROOT, "static", "samples")
_ABSORB_SAMPLE_PATH = os.path.join(_SAMPLE_DIRECTORY, SAMPLE_REPORT_FILENAME)
_LEGACY_SAMPLE_PATH = os.path.join(_SAMPLE_DIRECTORY, "stock-papi-tw-industry-daily-SAMPLE.pdf")
SAMPLE_REPORT_PATH = _ABSORB_SAMPLE_PATH if os.path.isfile(_ABSORB_SAMPLE_PATH) else _LEGACY_SAMPLE_PATH
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

if supabase_client:
    from line_state import SupabaseStore
    line_store = SupabaseStore(supabase_client)
else:
    line_store = FirestoreStore(GCP_PROJECT_ID) if GCP_PROJECT_ID else None

line_login_config = LineLoginConfig.from_env()
line_auth_store = FirestoreAuthStore(GCP_PROJECT_ID) if GCP_PROJECT_ID else None

gemini_model = _LazyGeminiModel(GEMINI_API_KEY) if GEMINI_API_KEY else None
conversation_context_store = MemoryContextStore(ttl_seconds=1800)
_conversation_provider_cache = {"model": None, "provider": None}
prediction_capability = PredictionCapabilityState.from_environment()
PREVIEW_CANDIDATE_PREFIX = prediction_capability.preview_candidate_prefix or ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("absorb")


def _load_taiwan_security_master():
    return fetch_taiwan_security_master(session=requests.Session())


taiwan_security_master = TaiwanSecurityMasterResolver(
    _load_taiwan_security_master,
    fallback_registry=lambda: twstock.codes,
)


def runtime_logging_secrets():
    return (
        LINE_CHANNEL_ACCESS_TOKEN,
        LINE_CHANNEL_SECRET,
        FINMIND_PASSWORD,
        GEMINI_API_KEY,
        BROADCAST_TOKEN,
        ALERT_TASK_TOKEN,
        OPENALICE_API_TOKEN,
        MARKETAUX_API_TOKEN,
        SUPABASE_KEY,
        finmind_token,
    )


install_redacting_formatters(runtime_logging_secrets)

_FINMIND_BLOCKED_UNTIL = 0
CATEGORY_PAGE_SIZE = 12
SECTOR_SCAN_LIMIT = 20
SECTOR_DISPLAY_LIMIT = 10
SECTOR_SNAPSHOT_DOC = "sector_signals"
ABSORB_THEME_SECTORS = {
    "AI伺服器": {"鴻海", "廣達", "緯創", "緯穎", "英業達", "仁寶", "和碩", "神達", "勤誠"},
    "PC／筆電": {"華碩", "宏碁", "微星", "技嘉", "神基", "藍天"},
    "散熱機構": {"雙鴻", "奇鋐", "建準", "勤誠", "營邦", "迎廣"},
    "工業電腦": {"研華", "樺漢", "凌華", "友通", "艾訊"},
    "網通設備": {"智邦", "啟碁", "中磊", "正文", "台揚", "明泰"},
    "半導體製造": {"台積電", "聯電", "世界", "力積電", "南亞科", "華邦電"},
    "IC設計ASIC": {"聯發科", "瑞昱", "創意", "世芯-KY", "力旺", "M31"},
    "封測設備": {"日月光投控", "矽格", "京元電子", "辛耘", "弘塑", "家登"},
}
# Compatibility alias for callers that still patch the pre-migration identifier.
PAPI_THEME_SECTORS = ABSORB_THEME_SECTORS
_SYSTEM_CACHE = {}
CACHE_EXPIRY_SECONDS = 3600
_YFINANCE_CACHE = {}
YFINANCE_CACHE_SECONDS = 3600

# ==================================================
# 2. 資料抓取與清洗模組
# ==================================================
def finmind_login():
    global finmind_token
    finmind_token = _provider_finmind_login(
        finmind_token,
        FINMIND_USER,
        FINMIND_PASSWORD,
        requests,
        logger=logger,
    )

def fetch_finmind_dataset(dataset, code, start_date, end_date):
    global _FINMIND_BLOCKED_UNTIL
    try:
        frame, _FINMIND_BLOCKED_UNTIL = _provider_fetch_finmind_dataset(
            dataset,
            code,
            start_date,
            end_date,
            blocked_until=_FINMIND_BLOCKED_UNTIL,
            now=time.time,
            login=finmind_login,
            token=lambda: finmind_token,
            requests_module=requests,
            pd=pd,
            logger=logger,
            sleep_fn=time.sleep,
            retry_attempts=2,
        )
    except FinMindFetchError as exc:
        if exc.blocked_until is not None:
            _FINMIND_BLOCKED_UNTIL = exc.blocked_until
        if exc.category == "empty_dataset":
            return pd.DataFrame()
        raise
    return frame


def fetch_yfinance_price_history(tickers, start_date, end_date=None):
    return _provider_fetch_yfinance_price_history(
        tickers,
        start_date,
        end_date,
        cache=_YFINANCE_CACHE,
        cache_seconds=YFINANCE_CACHE_SECONDS,
        now=time.time,
        pd=pd,
        logger=logger,
    )


def fetch_option_context_history(start_date, end_date=None):
    return _provider_fetch_option_context_history(
        start_date,
        end_date,
        fetch_yfinance_price_history,
        ThreadPoolExecutor,
        pd,
        logger,
    )


def get_stock_name(code, target_date=None, require_authoritative=False):
    if is_taiwan_symbol(code):
        return taiwan_security_master.resolve_name(
            code,
            target_date,
            require_authoritative=require_authoritative,
        )
    return _provider_get_stock_name(
        code, twstock.codes, is_us_ticker, taiwan_security_master,
        target_date, require_authoritative,
    )


def get_stock_name_for_date(code, target_date=None, require_authoritative=False):
    return get_stock_name(code, target_date, require_authoritative)

def search_stock_code(keyword):
    return _provider_search_stock_code(
        keyword, twstock.codes, is_us_ticker, get_stock_name,
        taiwan_security_master,
    )


def taiwan_security_codes():
    master = taiwan_security_master.get_master()
    return master.entries if master is not None else twstock.codes


def get_gcp_access_token():
    return _runtime_get_gcp_access_token(line_store, requests)


def _gcs_get_allowed_object(object_name, max_bytes, allowed_prefix):
    return get_allowed_object(
        object_name,
        max_bytes,
        allowed_prefix,
        bucket=QUANT_SNAPSHOT_BUCKET,
        enabled=line_store is not None,
        token_provider=get_gcp_access_token,
        http_get=requests.get,
    )


def _gcs_get_object(object_name, max_bytes):
    """只允許讀取既有 quant/v1 私有物件。"""
    return _gcs_get_allowed_object(object_name, max_bytes, "quant/v1/")


def _gcs_get_report_object(object_name, max_bytes):
    """只允許讀取 reports/v1 私有物件。"""
    return _gcs_get_allowed_object(object_name, max_bytes, "reports/v1/")


def _gcs_get_report_v2_object(object_name, max_bytes):
    """只允許讀取 reports/v2 私有物件。"""
    return _gcs_get_allowed_object(object_name, max_bytes, "reports/v2/")


_CANONICAL_OBJECT_PATH_RE = re.compile(r"^objects/canonical/[0-9a-f]{64}\.json$")
_REGRESSION_OBJECT_PATH_RE = re.compile(r"^objects/regression/[0-9a-f]{64}\.json$")


def load_canonical_object(object_path, max_bytes=MAX_CANONICAL_REPORT_BYTES):
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or not (1 <= max_bytes <= MAX_CANONICAL_REPORT_BYTES)
    ):
        return None
    if not isinstance(object_path, str) or not _CANONICAL_OBJECT_PATH_RE.match(object_path):
        return None
    full_object_name = f"reports/v2/{object_path}"
    data = _gcs_get_report_v2_object(full_object_name, max_bytes)
    if not isinstance(data, bytes) or len(data) == 0 or len(data) > max_bytes:
        return None
    return data


def load_regression_artifact(object_path, max_bytes=2_000_000):
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or not (1 <= max_bytes <= 2_000_000)
    ):
        return None
    if not isinstance(object_path, str) or not _REGRESSION_OBJECT_PATH_RE.fullmatch(object_path):
        return None
    full_object_name = f"reports/v2/{object_path}"
    data = _gcs_get_report_v2_object(full_object_name, max_bytes)
    if not isinstance(data, bytes) or len(data) == 0 or len(data) > max_bytes:
        return None
    return data


def _published_us_securities_observation():
    """Load the latest US securities section through the full report binding."""
    import hashlib

    from reporting.exceptions import ReportWebError
    from reporting.professional_binding import validate_professional_report_binding
    from reporting.professional_schema import ProfessionalPostCloseReport

    reports = _published_report_index_v2(market="US")
    if not isinstance(reports, list):
        raise ReportWebError("美股報告索引暫時無法使用")
    item = next(
        (value for value in reports if value.get("report_type") == "post_close"),
        None,
    )
    if item is None:
        raise ReportWebError("美股盤後報告暫時無法使用")
    metadata = load_report_metadata(
        item,
        load_object=_gcs_get_report_v2_object,
        version="v2",
        expected_market="US",
    )
    pointer = metadata.get("professional_report") if isinstance(metadata, dict) else None
    if not isinstance(pointer, dict):
        raise ReportWebError("美股個股觀察來源暫時無法使用")
    raw = load_canonical_object(pointer.get("object"))
    if (
        not isinstance(raw, bytes)
        or hashlib.sha256(raw).hexdigest() != pointer.get("sha256")
    ):
        raise ReportWebError("美股個股觀察來源驗證失敗")
    try:
        report = ProfessionalPostCloseReport.from_document(json.loads(raw))
        validate_professional_report_binding(
            route_source_date=item.get("source_market_date"),
            metadata=metadata,
            pointer=pointer,
            report=report,
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ReportWebError("美股個股觀察來源驗證失敗") from exc
    if report.identity.market != "US" or report.securities.status != "available":
        raise ReportWebError("美股個股觀察市場或狀態不符")
    data = report.securities.data
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("stock_events"), list)
        or not isinstance(data.get("etf_observations"), list)
    ):
        raise ReportWebError("美股個股觀察格式不合法")
    return {
        "stock_events": list(data.get("stock_events") or []),
        "etf_observations": list(data.get("etf_observations") or []),
    }



def _gcs_get_dashboard_object(object_name, max_bytes):
    return _gcs_get_allowed_object(object_name, max_bytes, "dashboard/v1/")


def _gcs_get_preview_object(object_name, max_bytes):
    return _gcs_get_allowed_object(object_name, max_bytes, "previews/")


def _gcs_get_prediction_object(object_name, max_bytes):
    return _gcs_get_allowed_object(object_name, max_bytes, "predictions/v1/")


def _published_dashboard_snapshot(today=None):
    if PREVIEW_CANDIDATE_PREFIX:
        return load_preview_dashboard_snapshot(
            PREVIEW_CANDIDATE_PREFIX,
            load_object=_gcs_get_preview_object,
            cache=_DASHBOARD_CACHE,
        )
    return load_dashboard_snapshot(
        today=today,
        load_object=_gcs_get_dashboard_object,
        cache=_DASHBOARD_CACHE,
    )


def _published_prediction_snapshot(market, today=None):
    return load_prediction_snapshot(
        market,
        today=today,
        load_object=_gcs_get_prediction_object,
        cache=_PREDICTION_CACHE,
    )


def _published_report_index():
    return load_report_index(
        load_object=_gcs_get_report_object,
        max_bytes=REPORT_INDEX_MAX_BYTES,
    )


def _published_report_index_v2(market="TW"):
    return load_report_index(
        load_object=_gcs_get_report_v2_object,
        max_bytes=REPORT_INDEX_MAX_BYTES,
        version="v2",
        market=market,
    )


def _published_quant_manifest(market, today=None):
    return published_quant_manifest(
        market,
        today=today,
        load_object=_gcs_get_object,
        cache=_QUANT_MANIFEST_CACHE,
    )


def fetch_published_quant_snapshot(code, today=None):
    return fetch_quant_snapshot(
        code,
        today=today,
        is_us_ticker_fn=is_us_ticker,
        load_manifest=_published_quant_manifest,
        load_object=_gcs_get_object,
    )


def fetch_market_insights(today=None):
    return load_market_insights(
        today=today,
        load_object=_gcs_get_object,
        cache=_MARKET_INSIGHTS_CACHE,
    )







def _foreign_flow_mask(frame):
    return _quant_foreign_flow_mask(frame, pd=pd)


def merge_chip_data(price, institutional=None, margin=None):
    return _merge_chip_data(price, institutional, margin, pd=pd)


def _neutral_market_features(frame):
    return _neutral_quant_market_features(frame)


def _market_feature_frame(market, prefix):
    return _quant_market_feature_frame(market, prefix, pd=pd)


def add_market_context_features(price, market=None, etf50=None):
    return _add_market_context_features(price, market, etf50, pd=pd, np=np)


def _option_close_frame(frame, column):
    return _quant_option_close_frame(frame, column, pd=pd)


def add_option_context_features(price, vix=None, vix9d=None, vix3m=None):
    return _add_option_context_features(price, vix, vix9d, vix3m, pd=pd, np=np)


def add_price_quality_features(price, yf_price=None):
    return _add_price_quality_features(price, yf_price, pd=pd, np=np)


def _clean_df(df):
    return _clean_quant_df(df, pd=pd, np=np)






def summarize_foreign_flow(df):
    return _summarize_foreign_flow(df, pd=pd)

def get_data(code, days=730, as_of=None):
    return _get_quant_data(
        code,
        days,
        as_of=as_of,
        datetime=datetime,
        pd=pd,
        is_us_ticker=is_us_ticker,
        twstock_codes=taiwan_security_codes(),
        taiwan_security_master=taiwan_security_master,
        fetch_yfinance=fetch_yfinance_price_history,
        fetch_finmind=fetch_finmind_dataset,
        fetch_option_context=fetch_option_context_history,
        add_price_quality=add_price_quality_features,
        add_market_context=add_market_context_features,
        add_option_context=add_option_context_features,
        merge_chip=merge_chip_data,
        clean=_clean_df,
    )

# ==================================================
# 3. 核心運算模組 (LGBM)
# ==================================================
def fetch_marketaux_news(name):
    return _fetch_marketaux_news(
        name,
        api_token=MARKETAUX_API_TOKEN,
        parse_items=parse_marketaux_items,
    )


def parse_stocktwits_sentiment(payload, code, now=None):
    return _parse_stocktwits_sentiment(
        payload, code, now=now, window_days=SENTIMENT_WINDOW_DAYS
    )


def fetch_stocktwits_sentiment(code):
    return _fetch_stocktwits_sentiment(
        code,
        window_days=SENTIMENT_WINDOW_DAYS,
        parse_items=parse_stocktwits_sentiment,
    )


def get_news(name, code=None):
    return _get_news(
        name,
        code,
        ThreadPoolExecutor,
        fetch_news_rss,
        parse_news_items,
        fetch_marketaux_news,
        fetch_stocktwits_sentiment,
        normalize_and_dedupe,
        SENTIMENT_WINDOW_DAYS,
    )

def calc_all(df):
    return _calc_all(df, pd=pd, np=np)

def add_prediction_target(df):
    return _add_prediction_target(df, np=np)

def build_time_splits(n_samples):
    return _build_time_splits(n_samples, np=np)

def score_oos_predictions(future_returns, probabilities):
    return _score_oos_predictions(future_returns, probabilities, pd=pd, np=np)

def run_ai_engine(df, *, include_oos=False):
    return _run_ai_engine(
        df,
        add_prediction_target=add_prediction_target,
        build_time_splits=build_time_splits,
        score_oos_predictions=score_oos_predictions,
        pd=pd,
        np=np,
        logger=logger,
        include_oos=include_oos,
    )


def _published_report_index_v2(market="TW"):
    return load_report_index(
        load_object=_gcs_get_report_v2_object,
        max_bytes=REPORT_INDEX_MAX_BYTES,
        version="v2",
        market=market,
    )


def run_latest_inference(df):
    return _run_latest_inference(
        df,
        add_prediction_target=add_prediction_target,
        pd=pd,
        np=np,
        logger=logger,
    )

def get_ai_insight_for_broadcast(name, data, bt, news):
    return _get_ai_insight_for_broadcast(
        name, data, bt, news, gemini_model
    )

# ==================================================
# 4. 分析總控
# ==================================================









def _snapshot_dataframe(snapshot):
    return _build_snapshot_dataframe(snapshot, pd=pd)


def _do_analyze(code):
    return _analyze_uncached(
        code,
        fetch_snapshot=fetch_published_quant_snapshot,
        build_snapshot_frame=_snapshot_dataframe,
        get_data=get_data,
        calc_all=calc_all,
        run_ai_engine=run_ai_engine,
        get_stock_name=get_stock_name,
        get_news=get_news,
        analyze_sentiment_detail=analyze_sentiment_detail,
        summarize_foreign_flow=summarize_foreign_flow,
        calculate_projection=calculate_investment_projection,
        pd=pd,
        json=json,
        datetime=datetime,
    )

def analyze(code):
    value = _analyze_cached(
        code,
        cache=_SYSTEM_CACHE,
        expiry_seconds=CACHE_EXPIRY_SECONDS,
        now=time.time,
        analyze_fn=_do_analyze,
    )
    dashboard = _published_dashboard_snapshot()
    return sanitize_analysis(value, dashboard) if dashboard is not None else value

def cached_opportunities(limit=5):
    return _dashboard_cached_opportunities(
        _SYSTEM_CACHE, time.time, CACHE_EXPIRY_SECONDS, limit=limit
    )

def dashboard_sector_cards(limit=6):
    dashboard = _published_dashboard_snapshot()
    if PREVIEW_CANDIDATE_PREFIX and isinstance(dashboard, dict):
        sector_snapshot = dashboard.get("sector_snapshot")
        if isinstance(sector_snapshot, dict):
            preview_snapshot = dict(sector_snapshot)
            preview_snapshot["baseline_status"] = dashboard.get("baseline_status")
            preview_snapshot["presentation"] = dashboard.get("presentation") or {}
            return _dashboard_sector_cards(
                lambda _store: preview_snapshot,
                None,
                cached_opportunities,
                _safe_float,
                limit=limit,
            )
    return _dashboard_sector_cards(
        load_sector_signal_snapshot,
        line_store,
        cached_opportunities,
        _safe_float,
        limit=limit,
    )




def market_forecast(): return analyze("TAIEX")

# ==================================================
# 5. UI 渲染
# ==================================================



# ==================================================
# 6. 動態產業分類與選單生成
# ==================================================
def call_openalice(prompt):
    return _papi_service().call_openalice(prompt)


def _extract_stock_from_papi_prompt(prompt):
    return _papi_service().extract_stock(prompt)


def _match_sector_from_prompt(prompt):
    return _papi_service().match_sector(prompt)


def _build_single_stock_context(data):
    return _papi_service().build_single_context(data)


def _gather_sector_data(codes, max_fresh=2, max_total=5):
    return _papi_service().gather_sector_data(
        codes, max_fresh=max_fresh, max_total=max_total
    )


def _build_papi_sector_examples(limit=3):
    return _papi_service().build_sector_examples(limit=limit)


def _build_papi_prompt(prompt):
    return _papi_service().build_prompt(prompt)


def call_papi_gemini_fallback(prompt):
    return _papi_service().call_gemini(prompt)


def _conversation_provider():
    cached = _conversation_provider_cache
    if cached["provider"] is None or cached["model"] is not gemini_model:
        cached["model"] = gemini_model
        cached["provider"] = GeminiConversationProvider(gemini_model)
    return cached["provider"]


def _conversation_sector_ranking():
    rows = []
    for card in dashboard_sector_cards()[:10]:
        leader = card.get("leader") if isinstance(card, dict) else None
        if not isinstance(leader, dict):
            continue
        recommendation = leader.get("recommendation") if isinstance(leader.get("recommendation"), dict) else {}
        probability = leader.get("prob")
        probability = probability / 100 if isinstance(probability, (int, float)) else None
        rows.append({
            "industry": card.get("name"),
            "symbol": leader.get("code"),
            "name": leader.get("name"),
            "five_day_probability": (
                probability
                if leader.get("model_output_label") == "五日上漲機率"
                else None
            ),
            "model_direction_score": (
                probability
                if leader.get("model_output_label") == "模型方向分數"
                else None
            ),
            "model_output_label": leader.get("model_output_label"),
            "calibration_notice": leader.get("calibration_notice"),
            "trend": leader.get("trend"),
            "action_label": recommendation.get("action"),
            "data_as_of": leader.get("as_of"),
        })
    return rows


def _conversation_report_lookup(report_type, *, market="TW"):
    try:
        item = next(
            (row for row in _published_report_index_v2(market=market) if row.get("report_type") == report_type),
            None,
        )
    except Exception:
        item = None
    if not isinstance(item, dict):
        return {
            "market": market, "report_type": report_type,
            "data_quality": "unavailable", "limitations": ["report unavailable"],
        }
    return {
        "market": item.get("market", "TW"),
        "report_type": report_type,
        "title": item.get("title"),
        "summary": list(item.get("summary") or [])[:5],
        "source_market_date": item.get("source_market_date"),
        "applicable_trading_date": item.get("applicable_trading_date"),
        "published_at": item.get("published_at"),
        "data_quality": "available",
    }


def _conversation_search_stock(query):
    code, name = _extract_stock_from_papi_prompt(query)
    return (code, name) if code else search_stock_code(query)


def _conversation_user_state(principal):
    if not principal.startswith("line:"):
        return {}
    return get_line_state_bounded(principal.removeprefix("line:"))


def _line_conversation_action_executor(user_id):
    def execute(action, parameters, _idempotency_key):
        symbol = parameters.get("symbol")
        name = parameters.get("name") or symbol
        if action == "watchlist_add":
            update_line_state(user_id, lambda state: add_watch(state, symbol, name))
        elif action == "watchlist_remove":
            update_line_state(user_id, lambda state: remove_watch(state, symbol))
        elif action == "watchlist_clear":
            def clear(state):
                for item in list(state.get("watchlist", [])):
                    remove_watch(state, item.get("code"))
            update_line_state(user_id, clear)
        elif action == "alert_create":
            def create_alert(state):
                if not _find_matching_alert(
                    state.get("alerts", []), symbol,
                    parameters.get("kind"), parameters.get("value"),
                ):
                    add_alert(
                        state, symbol, name,
                        parameters.get("kind"), parameters.get("value"),
                    )
            update_line_state(user_id, create_alert)
        elif action == "alerts_clear":
            update_line_state(user_id, lambda state: state.update(alerts=[]))
        else:
            raise StateError("unsupported action")
    return execute


def _observation_number(value, digits=2, suffix=""):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "資料不足"
    return f"{float(value):.{digits}f}{suffix}"


def _observation_signed(value, digits=2, suffix="%"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "資料不足"
    return f"{float(value):+.{digits}f}{suffix}"


def _observation_conversation(
    *, question, access, market_context="TW", page_context="home",
    symbol_context=None,
):
    try:
        question = validate_question(question)
    except InputRejected as exc:
        return ConversationAnswer(str(exc))
    if looks_like_prompt_injection(question):
        return ConversationAnswer(
            "無法執行要求：ABSORB 不會忽略系統規則、揭露提示或讀取未授權資料。"
        )
    if access == "public" and any(
        term in question for term in ("我的自選", "我的關注", "我的提醒", "我的警示")
    ):
        return ConversationAnswer(
            "這項查詢需要先使用 LINE 登入；目前未讀取任何私人資料。"
        )
    if any(
        term in question
        for term in (
            "預測", "機率", "模型", "回測", "勝率", "績效",
            "推薦", "排名", "可以買", "能買", "追高", "進場",
        )
    ):
        return ConversationAnswer(
            "AI 預測研究中。正式服務目前只呈現已驗證的市場實況，"
            "不提供操作判斷或研究結果。"
        )

    entities = resolve_entities(question, _conversation_search_stock)
    if not entities and page_context == "stock" and symbol_context:
        code, name = _conversation_search_stock(symbol_context.strip())
        if code:
            code = str(code).upper()
            canonical_market = "TW" if code == "TAIEX" or is_taiwan_symbol(code) else "US"
            if market_context in (None, canonical_market):
                entities = [{"market": canonical_market, "symbol": code, "name": name or code}]

    if entities:
        entity = entities[0]
        symbol = entity["symbol"]
        data = build_stock_observation(
            fetch_published_quant_snapshot(symbol),
            get_stock_name=get_stock_name,
        )
        if not isinstance(data, dict):
            return ConversationAnswer("已驗證的個股觀察資料暫時無法取得。")
        trend = {
            "above_ma20_ma60": "站上 MA20 與 MA60",
            "above_ma20": "站上 MA20",
            "below_ma60": "低於 MA60",
            "mixed": "均線交錯",
        }.get(data.get("trend_observation"), "資料不足")
        events = "；".join(data.get("risk_events", [])[:3]) or "未觸發額外事件"
        text = (
            f"{data['name']}（{data['code']}）市場觀察："
            f"最新收盤 {_observation_number(data.get('price'))}，"
            f"均線狀態為{trend}，"
            f"RSI {_observation_number(data.get('rsi'), 1)}，"
            f"量比 {_observation_number(data.get('volume_ratio'))}。"
            f"已觸發事件：{events}。\n\n"
            f"資料截至：{data.get('as_of') or '未提供'}｜"
            "內容只描述已發生資料。"
        )
        return ConversationAnswer(
            text,
            data_as_of=data.get("as_of"),
            data_quality="available",
            tools_used=("verified_observation_snapshot",),
        )

    if market_context == "US":
        report = _conversation_report_lookup("post_close", market="US")
        if report.get("data_quality") != "available":
            return ConversationAnswer("已驗證的美股市場觀察資料暫時無法取得。")
        summary = "；".join(report.get("summary") or []) or report.get("title")
        return ConversationAnswer(
            f"美股市場實況：{summary}\n\n資料截至：{report.get('source_market_date') or '未提供'}｜內容只描述已發布資料。",
            data_as_of=report.get("source_market_date"),
            data_quality="available",
            tools_used=("verified_us_report_index",),
        )

    snapshot = _published_dashboard_snapshot()
    if not isinstance(snapshot, dict) or snapshot.get("product_mode") != "observation":
        return ConversationAnswer("已驗證的市場觀察資料暫時無法取得。")
    industries = snapshot.get("industry_observations", [])
    industry = next(
        (
            item for item in industries
            if isinstance(item, dict)
            and str(item.get("name") or "") in question
        ),
        None,
    )
    if industry is not None:
        text = (
            f"{industry['name']}產業觀察："
            f"單日實際報酬 {_observation_signed(industry.get('return_1d_pct'))}，"
            f"近 5 日相對大盤報酬 "
            f"{_observation_signed(industry.get('relative_return_5d_pct'))}，"
            f"上漲家數比例 "
            f"{_observation_number(industry.get('advancing_ratio_pct'), 1, '%')}，"
            f"站上 MA20 比例 "
            f"{_observation_number(industry.get('ma20_breadth_pct'), 1, '%')}。\n\n"
            f"資料截至：{snapshot.get('observation_as_of')}｜"
            "內容只描述已發生資料。"
        )
        return ConversationAnswer(
            text,
            data_as_of=snapshot.get("observation_as_of"),
            data_quality="available",
            tools_used=("verified_observation_dashboard",),
        )
    if any(term in question for term in ("台股", "大盤", "市場", "盤勢", "今天")):
        market = snapshot.get("market_observation", {})
        risk = {
            "normal": "一般",
            "cautious": "謹慎",
            "elevated": "升高",
        }.get(market.get("risk_state"), "資料不足")
        text = (
            f"市場實況：單日中位報酬 "
            f"{_observation_signed(market.get('return_1d_pct'))}，"
            f"上漲 {market.get('advancing_count', '—')} 檔、"
            f"下跌 {market.get('declining_count', '—')} 檔，"
            f"站上 MA20 比例 "
            f"{_observation_number(market.get('ma20_breadth_pct'), 1, '%')}，"
            f"風險狀態為{risk}。\n\n"
            f"資料截至：{snapshot.get('observation_as_of')}｜"
            "內容只描述已發生資料。"
        )
        return ConversationAnswer(
            text,
            data_as_of=snapshot.get("observation_as_of"),
            data_quality="available",
            tools_used=("verified_observation_dashboard",),
        )
    return ConversationAnswer(
        "AI 預測研究中。你可以詢問市場實況、產業實際強弱、"
        "個股價格、均線、技術指標、籌碼或已觸發事件。"
    )


def run_absorb_conversation(
    *, principal, question, access="public", action_executor=None,
    market_context="TW", page_context="home", symbol_context=None,
):
    if prediction_capability.mode == "research":
        return _observation_conversation(
            question=question, access=access,
            market_context=market_context, page_context=page_context,
            symbol_context=symbol_context,
        )
    state_lookup = (lambda: _conversation_user_state(principal)) if access == "authenticated" else None
    orchestrator = ConversationOrchestrator(
        context_store=conversation_context_store,
        tool_registry=build_registry(
            analyze=lambda symbol: analyze(symbol),
            sector_ranking=_conversation_sector_ranking,
            report_lookup=lambda report_type: _conversation_report_lookup(
                report_type, market=market_context
            ),
            watchlist_lookup=state_lookup,
            alerts_lookup=state_lookup,
        ),
        search_stock=_conversation_search_stock,
        provider=_conversation_provider(),
        action_executor=action_executor,
    )
    return orchestrator.handle(
        principal=principal, question=question, access=access,
        market_context=market_context, page_context=page_context,
        symbol_context=symbol_context,
    )


def run_absorb_web_conversation(
    *, principal, question, access="public", market_context="TW", page_context="home",
    symbol_context=None,
):
    action_executor = None
    if access == "authenticated" and principal.startswith("line:"):
        action_executor = _line_conversation_action_executor(principal.removeprefix("line:"))
    return run_absorb_conversation(
        principal=principal,
        question=question,
        access=access,
        action_executor=action_executor,
        market_context=market_context,
        page_context=page_context,
        symbol_context=symbol_context,
    )


def _web_conversation_identity(http_request):
    if not line_login_config.configured or line_auth_store is None:
        return None
    session_id = verify_opaque_token(
        http_request.cookies.get(line_login_config.session_cookie_name),
        line_login_config.session_secret,
    )
    if not session_id:
        return None
    try:
        session = line_auth_store.load_session(session_id, utc_now())
    except Exception:
        return None
    user_id = str((session or {}).get("line_user_id") or "")
    csrf_token = (session or {}).get("csrf_token")
    supplied = http_request.headers.get("X-CSRF-Token")
    if (
        re.fullmatch(r"U[0-9a-f]{32}", user_id) is None
        or not isinstance(csrf_token, str)
        or not isinstance(supplied, str)
        or not hmac.compare_digest(supplied, csrf_token)
    ):
        return None
    return f"line:{user_id}", "authenticated"


def sector_signal_item(code, data):
    return _sector_signal_item(code, data, get_stock_name=get_stock_name)


def build_sector_signal_snapshot(market_map, analyze_fn, now=None, activity=None):
    return _build_sector_signal_snapshot(
        market_map,
        analyze_fn,
        now=now,
        activity=activity,
        scan_limit=SECTOR_SCAN_LIMIT,
        display_limit=SECTOR_DISPLAY_LIMIT,
        get_stock_name=get_stock_name,
    )


def build_market_map():
    return _build_market_map(twstock.codes, ABSORB_THEME_SECTORS)

industry_map = build_market_map()


def find_industry_peers(code, market_map=None, limit=5):
    return _find_industry_peers(code, market_map or industry_map, limit=limit)

def build_category_quick_reply(page=1):
    return _line_build_category_quick_reply(
        industry_map.keys(), CATEGORY_PAGE_SIZE, page=page
    )

# ==================================================
# 7. 自動化發報引擎
# ==================================================
# ==================================================
# 8. 路由與 LINE 基礎指令 (💡 確保名稱不重複版)
# ==================================================
def get_line_state(user_id):
    return _line_get_state(line_store, user_id)


def get_line_state_bounded(user_id, timeout=LINE_STATE_READ_BUDGET_SECONDS):
    return _line_get_state_bounded(
        line_store,
        user_id,
        timeout,
        _line_state_read_slots,
        logger,
        LINE_STATE_READ_MAX_WORKERS,
        queue,
        threading,
    )


def update_line_state(user_id, mutate):
    return _line_update_state(line_store, user_id, mutate)


def _store_error_text():
    return _line_store_error_text(line_store)




















def build_projection_flex(code, name, data, amount, base_url):
    return _line_build_projection_flex(code, name, data, amount, base_url)










def _system_document_url(store, document_id):
    return _line_system_document_url(store, document_id)


def save_sector_signal_snapshot(store, snapshot):
    return _line_save_sector_signal_snapshot(store, snapshot)


def load_sector_signal_snapshot(store):
    dashboard = _published_dashboard_snapshot()
    if dashboard is not None:
        return {
            **dashboard["sector_snapshot"],
            "baseline_status": dashboard.get("baseline_status"),
            "presentation": dashboard.get("presentation") or {},
        }
    return _line_load_sector_signal_snapshot(store)


def refresh_sector_signals(store):
    return _line_refresh_sector_signals(
        store,
        fetch_market_activity,
        build_sector_signal_snapshot,
        save_sector_signal_snapshot,
        industry_map,
        analyze,
    )







def _build_stock_row(code):
    return _line_build_stock_row(code, get_stock_name)

def build_industry_carousel(cat, arr):
    return _line_build_industry_carousel(cat, arr, get_stock_name)


def _build_sector_signal_row(item):
    return _line_build_sector_signal_row(item)


def build_sector_signal_carousel(category, items):
    return _line_build_sector_signal_carousel(category, items, SECTOR_DISPLAY_LIMIT)


def market_insights_payload():
    return _market_insights_payload(
        fetch_market_insights,
        get_stock_name,
        industry_map,
        datetime.date.today,
        build_industries,
        build_supply_chains,
    )


def _reply_text(event, text):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))


def _current_web_root():
    return request.host_url.replace("http://", "https://").rstrip("/")


def _require_same_pending(state, expected_pending):
    return _line_require_same_pending(state, expected_pending)


def _find_matching_alert(alerts, code, kind, value):
    return _line_find_matching_alert(alerts, code, kind, value)


def _resolve_postback_stock(code):
    return _line_resolve_postback_stock(code, search_stock_code)


@handler.add(PostbackEvent)
def handle_postback(event):
    try:
        _handle_postback_impl(event)
    except Exception:
        logger.error("Unexpected error in handle_postback")
        try:
            _reply_text(event, "系統暫時忙碌中，請稍後再試 🙏")
        except Exception:
            logger.error("Failed to send fallback reply in postback")

def _handle_postback_impl(event):
    return _line_handle_postback_impl(event, {
        "reply_text": _reply_text,
        "update_line_state": update_line_state,
        "store_error_text": _store_error_text,
        "resolve_stock": _resolve_postback_stock,
        "line_bot_api": line_bot_api,
        "analyze": analyze,
        "build_projection_flex": build_projection_flex,
        "current_web_root": _current_web_root,
        "find_matching_alert": _find_matching_alert,
        "observation_mode": prediction_capability.mode == "research",
    })


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        _handle_message_impl(event)
    except Exception:
        logger.error("Unexpected error in handle_message")
        try:
            _reply_text(event, "系統暫時忙碌中，請稍後再試 🙏")
        except Exception:
            logger.error("Failed to send fallback reply in message")

def _handle_message_impl(event):
    return _line_handle_message_impl(event, {
        "reply_text": _reply_text,
        "line_store": line_store,
        "line_bot_api": line_bot_api,
        "get_line_state_bounded": get_line_state_bounded,
        "update_line_state": update_line_state,
        "require_same_pending": _require_same_pending,
        "find_matching_alert": _find_matching_alert,
        "store_error_text": _store_error_text,
        "now": time.time,
        "is_crypto_query": _is_crypto_query,
        "openalice_url": OPENALICE_API_URL,
        "openalice_token": OPENALICE_API_TOKEN,
        "call_openalice": call_openalice,
        "call_papi_fallback": call_papi_gemini_fallback,
        "logger": logger,
        "search_stock_code": search_stock_code,
        "analyze": analyze,
        "observe": lambda code: build_stock_observation(
            fetch_published_quant_snapshot(code),
            get_stock_name=get_stock_name,
        ),
        "dashboard_snapshot": _published_dashboard_snapshot,
        "observation_mode": prediction_capability.mode == "research",
        "build_projection_flex": build_projection_flex,
        "build_category_quick_reply": build_category_quick_reply,
        "industry_map": industry_map,
        "load_sector_signal_snapshot": load_sector_signal_snapshot,
        "build_sector_signal_carousel": build_sector_signal_carousel,
        "web_root": request.host_url.replace("http://", "https://").rstrip("/"),
        "request_host_url": request.host_url,
        "conversation": lambda prompt, user_id: render_line(
            run_absorb_conversation(
                principal=f"line:{user_id}",
                question=prompt,
                access="authenticated",
                action_executor=_line_conversation_action_executor(user_id),
            )
        ),
        "is_fixed_command": is_fixed_command,
        "record_metric": record_metric,
    })


def route_dependencies():
    return {
        "search_stock": lambda query: search_stock_code(query),
        "load_report_index": lambda: _published_report_index(),
        "load_report_index_v2": lambda market="TW": _published_report_index_v2(market=market),
        "load_report_pdf": lambda item: load_report_pdf(
            item, load_object=_gcs_get_report_object
        ),
        "load_report_metadata": lambda item: load_report_metadata(
            item, load_object=_gcs_get_report_object
        ),
        "load_report_metadata_v2": lambda item, expected_market: load_report_metadata(
            item, load_object=_gcs_get_report_v2_object, version="v2",
            expected_market=expected_market,
        ),
        "load_report_metadata_v2_by_sha": lambda metadata_sha256, expected_market: load_report_metadata_by_sha(
            metadata_sha256, load_object=_gcs_get_report_v2_object,
            expected_market=expected_market,
        ),
        "load_canonical_object": lambda object_path, max_bytes=5_000_000: load_canonical_object(
            object_path, max_bytes=max_bytes
        ),
        "load_regression_artifact": lambda object_path, max_bytes=2_000_000: load_regression_artifact(
            object_path, max_bytes=max_bytes
        ),
        "sample_report_path": SAMPLE_REPORT_PATH,
        "sample_report_filename": SAMPLE_REPORT_FILENAME,
        "max_pdf_bytes": REPORT_PDF_MAX_BYTES,
        "line_login_config": line_login_config,
        "get_auth_store": lambda: line_auth_store,
        "auth_http_post": requests.post,
        "auth_now": utc_now,
        "converse": lambda **kwargs: run_absorb_web_conversation(**kwargs),
        "resolve_conversation_identity": lambda http_request: _web_conversation_identity(http_request),
        "analyze": lambda code: analyze(code),
        "stock_observation": lambda code: build_stock_observation(
            fetch_published_quant_snapshot(code),
            get_stock_name=get_stock_name,
        ),
        "dashboard_sector_cards": lambda: dashboard_sector_cards(),
        "dashboard_snapshot": lambda: _published_dashboard_snapshot(),
        "prediction_snapshot": lambda market: _published_prediction_snapshot(market),
        "us_securities_observation": lambda: _published_us_securities_observation(),
        "prediction_capability": prediction_capability,
        "cached_opportunities": lambda: cached_opportunities(),
        "build_market_heatmap": build_market_heatmap,
        "dashboard_top_picks": dashboard_top_picks,
        "industry_map": lambda: industry_map,
        "market_insights_payload": lambda: market_insights_payload(),
        "twstock_codes": taiwan_security_codes,
        "is_us_ticker": is_us_ticker,
        "find_industry_peers": lambda code: find_industry_peers(code),
        "get_stock_name": lambda code: get_stock_name(code),
        "handler": handler,
        "get_line_bot_api": lambda: line_bot_api,
        "get_line_store": lambda: line_store,
        "get_broadcast_token": lambda: BROADCAST_TOKEN,
        "get_alert_task_token": lambda: ALERT_TASK_TOKEN,
        "get_broadcast_insight": lambda name, data, bt, news: get_ai_insight_for_broadcast(
            name, data, bt, news
        ),
        "refresh_sector_signals": lambda store: refresh_sector_signals(store),
        "run_alert_checks": (
            lambda store, analyze_fn, push, today, root, *, prediction_allowed=True:
            run_alert_checks(
                store,
                analyze_fn,
                push,
                today,
                root,
                prediction_allowed=prediction_allowed,
            )
        ),
    }
def _papi_service():
    return AbsorbResearchService(
        requests_module=requests,
        openalice_url=OPENALICE_API_URL,
        openalice_token=OPENALICE_API_TOKEN,
        search_stock=search_stock_code,
        get_stock_name=get_stock_name,
        twstock_codes=taiwan_security_codes(),
        industry_map=industry_map,
        analyze=analyze,
        system_cache=_SYSTEM_CACHE,
        cache_expiry_seconds=CACHE_EXPIRY_SECONDS,
        line_store=line_store,
        load_sector_snapshot=load_sector_signal_snapshot,
        safe_float=_safe_float,
        gemini_model=gemini_model,
        now=time.time,
        sleep=time.sleep,
        logger=logger,
        build_prompt_fn=_build_papi_prompt,
        extract_stock_fn=_extract_stock_from_papi_prompt,
        match_sector_fn=_match_sector_from_prompt,
        gather_sector_data_fn=_gather_sector_data,
        build_single_context_fn=_build_single_stock_context,
        build_sector_examples_fn=_build_papi_sector_examples,
        taiwan_security_master=taiwan_security_master,
    )
