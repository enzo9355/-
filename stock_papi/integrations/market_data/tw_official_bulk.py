"""Fail-closed TWSE/TPEx bulk snapshot integration for TW post-close runs."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_source,
    store_cached_source,
)

SOURCE_SCHEMA_VERSION = "tw-official-bulk-v1"
PARSER_VERSION = "tw-official-parser-v1"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_ATTEMPTS = 2
MAX_FINMIND_FALLBACK_REQUESTS = 20


class OfficialSourceFailure(RuntimeError):
    """Safe structured failure for a public official market source."""

    def __init__(
        self,
        source_id: str,
        category: str,
        *,
        http_status: int | None = None,
        retryable: bool = False,
        safe_message: str | None = None,
    ):
        self.source_id = str(source_id)
        self.category = str(category)
        self.http_status = int(http_status) if http_status is not None else None
        self.retryable = bool(retryable)
        self.safe_message = str(safe_message or category)
        super().__init__(f"{self.source_id}: {self.safe_message}")


@dataclass(frozen=True)
class OfficialSourceDefinition:
    source_id: str
    market: str
    dataset: str
    url: str
    response_kind: str
    max_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True)
class OfficialSourceResult:
    source_id: str
    market: str
    dataset: str
    target_date: _datetime.date
    rows: tuple[dict[str, Any], ...]
    symbol_count: int
    content_sha256: str
    response_size_bytes: int
    cache_hit: bool
    date_verification: str = "explicit"


@dataclass(frozen=True)
class OfficialRequestBudget:
    planned_minimum_requests: int
    planned_worst_case_requests: int
    official_requests: int
    finmind_requests: int
    capacity_proven: bool
    reason: str


@dataclass(frozen=True)
class OfficialDailySnapshot:
    target_date: _datetime.date
    price_by_symbol: Mapping[str, Mapping[str, Any]]
    institutional_by_symbol: Mapping[str, tuple[Mapping[str, Any], ...]]
    margin_by_symbol: Mapping[str, Mapping[str, Any]]
    source_results: Mapping[str, OfficialSourceResult]
    manifest_sha256: str
    request_count: int
    request_budget: OfficialRequestBudget
    source_mode: str = "tw_official_bulk_v1"
    source_schema_version: str = SOURCE_SCHEMA_VERSION


SOURCE_DEFINITIONS: dict[str, OfficialSourceDefinition] = {
    "twse_price": OfficialSourceDefinition(
        "twse_price", "TWSE", "price",
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", "list",
    ),
    "twse_institutional": OfficialSourceDefinition(
        "twse_institutional", "TWSE", "institutional",
        "https://www.twse.com.tw/rwd/zh/fund/T86", "twse_report",
    ),
    "twse_margin": OfficialSourceDefinition(
        "twse_margin", "TWSE", "margin",
        "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN", "list",
    ),
    "tpex_price": OfficialSourceDefinition(
        "tpex_price", "TPEx", "price",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", "list",
    ),
    "tpex_institutional": OfficialSourceDefinition(
        "tpex_institutional", "TPEx", "institutional",
        "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php", "tpex_report",
        15 * 1024 * 1024,
    ),
    "tpex_margin": OfficialSourceDefinition(
        "tpex_margin", "TPEx", "margin",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_balance", "list",
    ),
}

_EMPTY_MARKERS = {"", "--", "---", "----", "-", "N/A", "NA", "null", "None", "除權息", "暫停交易"}
_SYMBOL_RE = re.compile(r"\d{4,6}")


def normalize_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not _SYMBOL_RE.fullmatch(text):
        raise ValueError("official source symbol is invalid")
    return text


def normalize_market_date(value: Any) -> _datetime.date:
    if isinstance(value, _datetime.datetime):
        return value.date()
    if isinstance(value, _datetime.date):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return _datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    if re.fullmatch(r"\d{7}", text):
        return _datetime.date(int(text[:3]) + 1911, int(text[3:5]), int(text[5:7]))
    match = re.fullmatch(r"(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if match:
        year, month, day = map(int, match.groups())
        if year < 1911:
            year += 1911
        return _datetime.date(year, month, day)
    raise ValueError("official source date is invalid")


def roc_date_text(value: _datetime.date) -> str:
    return f"{value.year - 1911:03d}/{value.month:02d}/{value.day:02d}"


def parse_number(value: Any, *, allow_empty: bool = False) -> float | None:
    if value is None:
        if allow_empty:
            return None
        raise ValueError("official source number is missing")
    if isinstance(value, bool):
        raise ValueError("official source number is invalid")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "").replace("＋", "+").replace("－", "-")
        if text in _EMPTY_MARKERS:
            if allow_empty:
                return None
            raise ValueError("official source number is missing")
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()
        number = float(text)
        if negative:
            number = -number
    if not math.isfinite(number):
        raise ValueError("official source number is not finite")
    return number


def _known(row: Mapping[str, Any], aliases: Sequence[str], *, required: bool = True) -> Any:
    hits = [name for name in aliases if name in row]
    if not hits:
        if required:
            raise ValueError(f"official source field is missing: {aliases[0]}")
        return None
    values = [row[name] for name in hits]
    if len(hits) > 1 and any(value != values[0] for value in values[1:]):
        raise ValueError(f"official source field aliases conflict: {aliases[0]}")
    return values[0]


def _validate_target_date(actual: _datetime.date, target: _datetime.date) -> None:
    if actual != target:
        raise ValueError("official source target date mismatch")


def _validate_price(row: dict[str, Any]) -> None:
    open_value = float(row["open"])
    high = float(row["max"])
    low = float(row["min"])
    close = float(row["close"])
    volume = float(row["Trading_Volume"])
    if min(open_value, high, low, close) <= 0 or volume < 0:
        raise ValueError("official source price values are invalid")
    if high < max(open_value, close, low) or low > min(open_value, close, high):
        raise ValueError("official source OHLC relationship is invalid")


def _dedupe(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]]) -> tuple[dict[str, Any], ...]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = key(row)
        previous = result.get(identity)
        if previous is not None and previous != row:
            raise ValueError("official source contains conflicting duplicate rows")
        result[identity] = row
    return tuple(result[item] for item in sorted(result))


def parse_twse_price(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        raise ValueError("TWSE price schema is invalid")
    rows = []
    for source in payload:
        if not isinstance(source, dict):
            continue
        try:
            symbol = normalize_symbol(_known(source, ("Code", "證券代號")))
        except ValueError:
            continue
        _validate_target_date(normalize_market_date(_known(source, ("Date", "日期"))), target_date)
        values = {
            "date": target_date.isoformat(),
            "stock_id": symbol,
            "open": parse_number(_known(source, ("OpeningPrice", "開盤價")), allow_empty=True),
            "max": parse_number(_known(source, ("HighestPrice", "最高價")), allow_empty=True),
            "min": parse_number(_known(source, ("LowestPrice", "最低價")), allow_empty=True),
            "close": parse_number(_known(source, ("ClosingPrice", "收盤價")), allow_empty=True),
            "Trading_Volume": parse_number(_known(source, ("TradeVolume", "成交股數")), allow_empty=True),
        }
        if any(values[name] is None for name in ("open", "max", "min", "close", "Trading_Volume")):
            continue
        _validate_price(values)
        rows.append(values)
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def parse_tpex_price(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        raise ValueError("TPEx price schema is invalid")
    rows = []
    for source in payload:
        if not isinstance(source, dict):
            continue
        try:
            symbol = normalize_symbol(_known(source, ("SecuritiesCompanyCode", "代號")))
        except ValueError:
            continue
        _validate_target_date(normalize_market_date(_known(source, ("Date", "日期"))), target_date)
        values = {
            "date": target_date.isoformat(),
            "stock_id": symbol,
            "open": parse_number(_known(source, ("Open", "開盤")), allow_empty=True),
            "max": parse_number(_known(source, ("High", "最高")), allow_empty=True),
            "min": parse_number(_known(source, ("Low", "最低")), allow_empty=True),
            "close": parse_number(_known(source, ("Close", "收盤")), allow_empty=True),
            "Trading_Volume": parse_number(_known(source, ("TradingShares", "成交股數")), allow_empty=True),
        }
        if any(values[name] is None for name in ("open", "max", "min", "close", "Trading_Volume")):
            continue
        _validate_price(values)
        rows.append(values)
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def _institutional_rows(target_date: _datetime.date, symbol: str, foreign: tuple[float, float], trust: tuple[float, float], dealer: tuple[float, float]) -> list[dict[str, Any]]:
    result = []
    for name, pair in (("Foreign", foreign), ("InvestmentTrust", trust), ("Dealer", dealer)):
        buy, sell = pair
        if buy < 0 or sell < 0:
            raise ValueError("official institutional buy/sell is negative")
        result.append({"date": target_date.isoformat(), "stock_id": symbol, "name": name, "buy": buy, "sell": sell})
    return result


def _table_rows(payload: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("official tabular report schema is invalid")
    return [str(item) for item in fields], data


def parse_twse_institutional(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise ValueError("TWSE institutional status is invalid")
    _validate_target_date(normalize_market_date(payload.get("date")), target_date)
    fields, data = _table_rows(payload)
    position = {name: index for index, name in enumerate(fields)}
    required = (
        "證券代號", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
        "外資自營商買進股數", "外資自營商賣出股數", "投信買進股數", "投信賣出股數",
        "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
        "自營商買進股數(避險)", "自營商賣出股數(避險)",
    )
    if any(name not in position for name in required):
        raise ValueError("TWSE institutional fields are incomplete")
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) < len(fields):
            continue
        try:
            symbol = normalize_symbol(source[position["證券代號"]])
        except ValueError:
            continue
        num = lambda name: float(parse_number(source[position[name]]))
        rows.extend(_institutional_rows(
            target_date,
            symbol,
            (num("外陸資買進股數(不含外資自營商)") + num("外資自營商買進股數"), num("外陸資賣出股數(不含外資自營商)") + num("外資自營商賣出股數")),
            (num("投信買進股數"), num("投信賣出股數")),
            (num("自營商買進股數(自行買賣)") + num("自營商買進股數(避險)"), num("自營商賣出股數(自行買賣)") + num("自營商賣出股數(避險)")),
        ))
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"], row["name"]))


def _find_tpex_table_rows(payload: Mapping[str, Any]) -> tuple[list[Any], _datetime.date]:
    actual_date = normalize_market_date(payload.get("date"))
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("TPEx institutional tables are missing")
    candidates: list[Any] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        for key in ("data", "aaData", "rows"):
            value = table.get(key)
            if isinstance(value, list) and value:
                candidates.extend(value)
    if not candidates:
        raise ValueError("TPEx institutional rows are missing")
    return candidates, actual_date


def _strip_html(value: Any) -> str:
    text = re.sub(r"<[^>]*>", "", str(value or ""))
    return text.replace("&nbsp;", " ").strip()


def parse_tpex_institutional(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, dict) or str(payload.get("stat") or "OK").upper() not in {"OK", "0"}:
        raise ValueError("TPEx institutional status is invalid")
    data, actual_date = _find_tpex_table_rows(payload)
    _validate_target_date(actual_date, target_date)
    rows = []
    for source in data:
        if isinstance(source, dict):
            try:
                symbol = normalize_symbol(_known(source, ("SecuritiesCompanyCode", "代號", "證券代號")))
            except ValueError:
                continue
            foreign = (float(parse_number(_known(source, ("ForeignInvestorsBuy", "外資及陸資買進股數")))), float(parse_number(_known(source, ("ForeignInvestorsSell", "外資及陸資賣出股數")))))
            trust = (float(parse_number(_known(source, ("InvestmentTrustBuy", "投信買進股數")))), float(parse_number(_known(source, ("InvestmentTrustSell", "投信賣出股數")))))
            dealer = (float(parse_number(_known(source, ("DealerBuy", "自營商買進股數")))), float(parse_number(_known(source, ("DealerSell", "自營商賣出股數")))))
        elif isinstance(source, list) and len(source) >= 24:
            try:
                symbol = normalize_symbol(_strip_html(source[0]))
            except ValueError:
                continue
            foreign = (float(parse_number(_strip_html(source[8]))), float(parse_number(_strip_html(source[9]))))
            trust = (float(parse_number(_strip_html(source[11]))), float(parse_number(_strip_html(source[12]))))
            dealer = (float(parse_number(_strip_html(source[20]))), float(parse_number(_strip_html(source[21]))))
        else:
            continue
        rows.extend(_institutional_rows(target_date, symbol, foreign, trust, dealer))
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"], row["name"]))


def parse_twse_margin(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        raise ValueError("TWSE margin schema is invalid")
    rows = []
    for source in payload:
        if not isinstance(source, dict):
            continue
        try:
            symbol = normalize_symbol(_known(source, ("股票代號", "Code")))
        except ValueError:
            continue
        rows.append({
            "date": target_date.isoformat(), "stock_id": symbol,
            "MarginPurchaseTodayBalance": float(parse_number(_known(source, ("融資今日餘額", "MarginPurchaseBalance")))),
            "ShortSaleTodayBalance": float(parse_number(_known(source, ("融券今日餘額", "ShortSaleBalance")))),
        })
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def parse_tpex_margin(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        raise ValueError("TPEx margin schema is invalid")
    rows = []
    for source in payload:
        if not isinstance(source, dict):
            continue
        try:
            symbol = normalize_symbol(_known(source, ("SecuritiesCompanyCode", "代號")))
        except ValueError:
            continue
        _validate_target_date(normalize_market_date(_known(source, ("Date", "日期"))), target_date)
        rows.append({
            "date": target_date.isoformat(), "stock_id": symbol,
            "MarginPurchaseTodayBalance": float(parse_number(_known(source, ("MarginPurchaseBalance", "融資餘額")))),
            "ShortSaleTodayBalance": float(parse_number(_known(source, ("ShortSaleBalance", "融券餘額")))),
        })
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


PARSERS: Mapping[str, Callable[[Any, _datetime.date], tuple[dict[str, Any], ...]]] = MappingProxyType({
    "twse_price": parse_twse_price,
    "twse_institutional": parse_twse_institutional,
    "twse_margin": parse_twse_margin,
    "tpex_price": parse_tpex_price,
    "tpex_institutional": parse_tpex_institutional,
    "tpex_margin": parse_tpex_margin,
})


def plan_official_request_budget(
    *, cold_source_count: int, retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    fallback_symbols: int = 0, fallback_requests_per_symbol: int = 3,
    fallback_enabled: bool = False,
    max_finmind_fallback_requests: int = MAX_FINMIND_FALLBACK_REQUESTS,
) -> OfficialRequestBudget:
    values = (cold_source_count, retry_attempts, fallback_symbols, fallback_requests_per_symbol, max_finmind_fallback_requests)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("request budget values must be non-negative integers")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be positive")
    finmind_requests = fallback_symbols * fallback_requests_per_symbol if fallback_enabled else 0
    if fallback_symbols and not fallback_enabled:
        return OfficialRequestBudget(cold_source_count, cold_source_count * retry_attempts, cold_source_count, 0, False, "fallback_disabled")
    if finmind_requests > max_finmind_fallback_requests:
        return OfficialRequestBudget(cold_source_count + finmind_requests, cold_source_count * retry_attempts + finmind_requests, cold_source_count, finmind_requests, False, "fallback_budget_exceeded")
    return OfficialRequestBudget(cold_source_count + finmind_requests, cold_source_count * retry_attempts + finmind_requests, cold_source_count, finmind_requests, True, "capacity_proven")


def assert_fallback_capacity(budget: OfficialRequestBudget) -> None:
    if not budget.capacity_proven:
        raise OfficialSourceFailure("finmind_fallback", "capacity_not_proven", safe_message=budget.reason)


def _source_params(definition: OfficialSourceDefinition, target_date: _datetime.date) -> dict[str, str] | None:
    if definition.source_id == "twse_institutional":
        return {"date": target_date.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"}
    if definition.source_id == "tpex_institutional":
        return {"l": "zh-tw", "o": "json", "se": "EW", "t": "D", "d": roc_date_text(target_date), "s": "0,asc"}
    return None


def _request_payload(definition, target_date, *, session, timeout, retry_attempts, sleep_fn):
    attempts = 0
    for attempt in range(retry_attempts):
        attempts += 1
        try:
            response = session.get(definition.url, params=_source_params(definition, target_date), headers={"User-Agent": "ABSORB/1.0"}, timeout=timeout)
        except Exception as exc:
            if attempt + 1 < retry_attempts:
                sleep_fn(0.5 * (2**attempt))
                continue
            raise OfficialSourceFailure(definition.source_id, "transport_error", retryable=True, safe_message=type(exc).__name__) from None
        status = int(getattr(response, "status_code", 0))
        if status in {500, 502, 503, 504} and attempt + 1 < retry_attempts:
            sleep_fn(0.5 * (2**attempt))
            continue
        if status != 200:
            raise OfficialSourceFailure(definition.source_id, "http_error", http_status=status, retryable=status >= 500, safe_message=f"HTTP {status}")
        content = bytes(getattr(response, "content", b""))
        length_header = getattr(response, "headers", {}).get("Content-Length") if hasattr(response, "headers") else None
        if length_header:
            try:
                if int(length_header) > definition.max_bytes:
                    raise OfficialSourceFailure(definition.source_id, "response_too_large", safe_message="content length exceeds limit")
            except ValueError:
                raise OfficialSourceFailure(definition.source_id, "invalid_headers", safe_message="invalid content length") from None
        if not content or len(content) > definition.max_bytes:
            raise OfficialSourceFailure(definition.source_id, "response_too_large", safe_message="response size is invalid")
        try:
            payload = json.loads(content.decode("utf-8-sig"))
        except (UnicodeError, ValueError):
            raise OfficialSourceFailure(definition.source_id, "invalid_json", safe_message="response JSON is invalid") from None
        return payload, len(content), attempts
    raise AssertionError("unreachable")


def _coverage(dataset: str, rows: Sequence[Mapping[str, Any]], minimum: int) -> int:
    symbols = {str(row.get("stock_id")) for row in rows if row.get("stock_id")}
    if len(symbols) < minimum:
        raise ValueError(f"official {dataset} coverage is below minimum")
    return len(symbols)


def build_official_daily_snapshot(
    root: Any, target_date: _datetime.date, *, session: Any = None,
    now: _datetime.datetime | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
    minimum_price_symbols: Mapping[str, int] | None = None,
    minimum_chip_symbols: int = 1,
) -> OfficialDailySnapshot:
    if not isinstance(target_date, _datetime.date):
        raise TypeError("target_date must be a date")
    if session is None:
        import requests
        session = requests.Session()
    checked_at = now or _datetime.datetime.now(_datetime.timezone.utc)
    minimum_price_symbols = dict(minimum_price_symbols or {"TWSE": 500, "TPEx": 500})
    results: dict[str, OfficialSourceResult] = {}
    request_count = 0
    cold_sources = 0
    for source_id, definition in SOURCE_DEFINITIONS.items():
        try:
            cached = load_cached_source(root, source_id=source_id, target_date=target_date, parser_version=PARSER_VERSION)
        except OfficialCacheError as exc:
            raise OfficialSourceFailure(source_id, "cache_invalid", safe_message=str(exc)) from None
        if cached is not None:
            rows = cached.rows
            response_size = cached.compressed_size
            content_sha = cached.content_sha256
            cache_hit = True
            date_verification = "twse_price_proxy" if source_id == "twse_margin" else "explicit"
        else:
            cold_sources += 1
            payload, response_size, attempts = _request_payload(definition, target_date, session=session, timeout=timeout, retry_attempts=retry_attempts, sleep_fn=sleep_fn)
            request_count += attempts
            try:
                rows = PARSERS[source_id](payload, target_date)
                minimum = minimum_price_symbols.get(definition.market, 1) if definition.dataset == "price" else minimum_chip_symbols
                symbol_count = _coverage(definition.dataset, rows, int(minimum))
            except (KeyError, TypeError, ValueError) as exc:
                raise OfficialSourceFailure(source_id, "schema_validation", safe_message=str(exc)) from None
            date_verification = "twse_price_proxy" if source_id == "twse_margin" else "explicit"
            cached = store_cached_source(
                root, source_id=source_id, target_date=target_date, rows=rows,
                symbol_count=symbol_count, parser_version=PARSER_VERSION,
                source_url=definition.url, fetched_at=checked_at,
                date_verification=date_verification,
            )
            content_sha = cached.content_sha256
            cache_hit = False
        symbol_count = _coverage(definition.dataset, rows, int(minimum_price_symbols.get(definition.market, 1) if definition.dataset == "price" else minimum_chip_symbols))
        results[source_id] = OfficialSourceResult(
            source_id=source_id, market=definition.market, dataset=definition.dataset,
            target_date=target_date, rows=tuple(dict(row) for row in rows),
            symbol_count=symbol_count, content_sha256=content_sha,
            response_size_bytes=response_size, cache_hit=cache_hit,
            date_verification=date_verification,
        )

    if results["twse_price"].target_date != target_date or results["twse_margin"].date_verification != "twse_price_proxy":
        raise OfficialSourceFailure("twse_margin", "date_not_verified", safe_message="TWSE margin date proxy is unavailable")

    price: dict[str, dict[str, Any]] = {}
    institutional: dict[str, list[dict[str, Any]]] = {}
    margin: dict[str, dict[str, Any]] = {}
    for result in results.values():
        for row in result.rows:
            symbol = row["stock_id"]
            if result.dataset == "price":
                if symbol in price and price[symbol] != row:
                    raise OfficialSourceFailure(result.source_id, "cross_source_duplicate", safe_message=f"duplicate price symbol {symbol}")
                price[symbol] = dict(row)
            elif result.dataset == "institutional":
                institutional.setdefault(symbol, []).append(dict(row))
            elif result.dataset == "margin":
                if symbol in margin and margin[symbol] != row:
                    raise OfficialSourceFailure(result.source_id, "cross_source_duplicate", safe_message=f"duplicate margin symbol {symbol}")
                margin[symbol] = dict(row)

    frozen_institutional = {symbol: tuple(MappingProxyType(dict(row)) for row in sorted(rows, key=lambda item: item["name"])) for symbol, rows in institutional.items()}
    manifest_document = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "target_date": target_date.isoformat(),
        "sources": {source_id: {"content_sha256": result.content_sha256, "symbol_count": result.symbol_count, "date_verification": result.date_verification} for source_id, result in sorted(results.items())},
    }
    manifest_sha = hashlib.sha256(json.dumps(manifest_document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    budget = plan_official_request_budget(cold_source_count=cold_sources, retry_attempts=retry_attempts)
    return OfficialDailySnapshot(
        target_date=target_date,
        price_by_symbol=MappingProxyType({symbol: MappingProxyType(row) for symbol, row in price.items()}),
        institutional_by_symbol=MappingProxyType(frozen_institutional),
        margin_by_symbol=MappingProxyType({symbol: MappingProxyType(row) for symbol, row in margin.items()}),
        source_results=MappingProxyType(dict(results)),
        manifest_sha256=manifest_sha,
        request_count=request_count,
        request_budget=budget,
    )
