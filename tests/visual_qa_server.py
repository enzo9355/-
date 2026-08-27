"""Local-only fixture server for desktop and mobile visual QA.

The fixture renders production templates with synthetic data and never connects to
LINE, Firestore, market providers, or the formal report publishing pipeline.
"""

import datetime
import json
import math
import os
import tempfile
from pathlib import Path


os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "visual-test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "visual-test")
os.environ.setdefault("LINE_LOGIN_CHANNEL_ID", "1234567890")
os.environ.setdefault("LINE_LOGIN_CHANNEL_SECRET", "visual-test-secret")
os.environ.setdefault(
    "LINE_LOGIN_REDIRECT_URI", "http://localhost:5099/auth/line/callback"
)
os.environ.setdefault("SESSION_SECRET", "visual-test-session-secret-32-bytes")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

import app as stock_app
from flask import redirect

from reporting.observation_v2 import build_post_close_observation_metadata
from reporting.publisher import publish_report_v2
from stock_papi.services.auth import sign_opaque_token
from tests.test_observation_public_surfaces import (
    observation_dashboard,
    quant_snapshot,
)


NOW = datetime.datetime(2026, 7, 13, 4, 0, tzinfo=datetime.timezone.utc)
USER_ID = "U" + "a" * 32
SESSION_ID = "visual-qa-session"


class VisualAuthStore:
    def __init__(self):
        self.sessions = {
            SESSION_ID: {
                "line_user_id": USER_ID,
                "csrf_token": "visual-qa-csrf-token-that-is-long-enough",
                "expires_at": NOW + datetime.timedelta(days=1),
            }
        }
        self.users = {
            USER_ID: {
                "display_name": "測試投資人",
                "picture_url": None,
                "plan": "free",
            }
        }

    def load_session(self, session_id, now):
        value = self.sessions.get(session_id)
        return dict(value) if value and value["expires_at"] > now else None

    def get_user(self, user_id):
        value = self.users.get(user_id)
        return dict(value) if value else None

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)


class VisualLineStore:
    def __init__(self):
        self.state = {
            "watchlist": [
                {"code": "2330", "name": "台積電"},
                {"code": "2454", "name": "聯發科"},
            ],
            "alerts": [{"id": "visual", "code": "2330", "name": "台積電"}],
            "pending": {},
            "signals": {"as_of": None, "items": []},
        }

    def load(self, _user_id):
        return self.state, None

    def update(self, _user_id, mutate):
        mutate(self.state)
        return self.state


def analysis_data():
    candles = [
        {"time": "2026-07-09", "open": 995, "high": 1015, "low": 990, "close": 1010},
        {"time": "2026-07-10", "open": 1010, "high": 1025, "low": 1000, "close": 1020},
        {"time": "2026-07-11", "open": 1020, "high": 1035, "low": 1010, "close": 1030},
    ]
    return {
        "name": "台積電", "code": "2330", "price": 1030.0, "prob": 63,
        "as_of": "2026-07-11", "quant_source": "本地回測快照",
        "trend": "多頭", "rsi": 58.0, "ma20": 998.0, "macd_osc": 0.3,
        "k": 62.0, "d": 54.0, "s_score": 55.0, "s_status": "中性",
        "candles": json.dumps(candles),
        "ma20_line": json.dumps([
            {"time": item["time"], "value": 998 + index * 4}
            for index, item in enumerate(candles)
        ]),
        "prob_h": [],
        "pred": json.dumps([
            {"time": "2026-07-14", "value": 1038},
            {"time": "2026-07-15", "value": 1046},
        ]),
        "news": [{
            "title": "先進製程需求維持穩健", "normalized_title": "先進製程需求維持穩健",
            "link": "https://example.com/news", "source": "測試新聞",
            "published_at": "2026-07-11T09:00:00+08:00", "direction": "positive",
        }],
        "projection": {
            "ok": True, "amount": 100000, "shares": 97, "deployed_amount": 99910,
            "strategy_profit": 7993, "buy_hold_profit": 4996,
            "strategy_annualized": 8.0, "buy_hold_annualized": 5.0,
        },
        "foreign_flow": {
            "available": True, "net_5": 1500, "net_20": 3200,
            "status": "外資偏多", "source": "外資",
        },
        "bt": {
            "days": 100, "accuracy": 54.0, "brier": 0.23, "strat_cum": 8.0,
            "bh_cum": 5.0, "win_rate": 57.0, "trades": 24, "mdd": -6.0,
            "sharpe": 1.1, "conclusion": "風險調整後表現尚可",
            "top_features": ["成交量", "RSI", "法人"],
        },
        "recommendation": {
            "action": "分批布局", "level": "cautious_bullish",
            "headline": "模型與趨勢偏多，但短線不宜追高", "confidence": "可信度中等",
            "supporting_reasons": ["五日上漲機率 63%", "股價站上 MA20"],
            "risk_reasons": ["短線漲幅擴大，追價風險上升"],
            "suggested_action": "等待拉回後分二至三次建立部位。",
            "invalidation_conditions": ["股價跌破 MA20"],
            "unheld_guidance": "等待拉回後分批建立部位",
            "held_guidance": "可續抱但不宜明顯加碼",
            "data_as_of": "2026-07-11", "source_metrics": {"sample_count": 24},
        },
        "backtest_interpretation": {
            "advantage": "過去相同規則的結果優於單純買進持有，但不代表未來仍會維持。",
            "cumulative_return": "投入 10 萬元，歷史結果約變成 10.8 萬元。",
            "maximum_drawdown": "最差階段，10 萬元可能一度剩下約 9.4 萬元。",
            "win_rate": "每 100 次進場約有 57 次獲利；勝率不代表每次盈虧相同。",
            "cash_ratio": "策略約 35% 的時間沒有持有部位。",
            "sharpe": "報酬效率（Sharpe Ratio）為 1.10。",
            "brier": "機率可信度（Brier Score）為 0.230。",
        },
    }


def visual_dashboard():
    snapshot = observation_dashboard()
    dates = []
    current = datetime.date(2026, 7, 15)
    while len(dates) < 65:
        if current.weekday() < 5:
            dates.append(current)
        current -= datetime.timedelta(days=1)
    dates.reverse()
    candles = []
    ma20 = []
    closes = []
    previous_close = 21840.0
    for index, day in enumerate(dates):
        close = 21840 + index * 18.5 + math.sin(index / 4.2) * 155
        open_value = previous_close + math.sin(index * 1.7) * 34
        high = max(open_value, close) + 46 + (index % 4) * 7
        low = min(open_value, close) - 42 - (index % 3) * 8
        candles.append({
            "time": day.isoformat(),
            "open": round(open_value, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
        })
        closes.append(close)
        if len(closes) >= 20:
            ma20.append({
                "time": day.isoformat(),
                "value": round(sum(closes[-20:]) / 20, 2),
            })
        previous_close = close
    latest = candles[-1]
    prior_close = candles[-2]["close"]
    change = latest["close"] - prior_close
    snapshot["market_index"] = {
        "symbol": "TAIEX",
        "name": "加權指數",
        "as_of": latest["time"],
        "price": latest["close"],
        "change": round(change, 2),
        "change_pct": round(change / prior_close * 100, 2),
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "candles": candles,
        "ma20": ma20,
        "prediction_display": {
            "status": "published",
            "as_of": latest["time"],
            "horizon_sessions": 5,
            "direction": "up",
            "probability_up_pct": 68.4,
            "target_price": round(latest["close"] * 1.018, 2),
            "expected_return_pct": 1.8,
            "model_version": "lgbm-ohlc-5d-v1",
            "validation": {
                "oos_samples": 145,
                "direction_accuracy_pct": 56.6,
                "brier": 0.244,
                "price_mae_pct": 2.83,
            },
            "points": [
                {"time": latest["time"], "value": latest["close"]},
                {"time": "2026-07-22", "value": round(latest["close"] * 1.018, 2)},
            ],
        },
    }
    return snapshot


def visual_quant_snapshot(code="2330", market="TW"):
    snapshot = quant_snapshot(symbol=code, market=market)
    if code == "2330":
        as_of = snapshot["as_of"]
        close = float(snapshot["daily"][-1]["Close"])
        snapshot["prediction_display"] = {
            "status": "published",
            "as_of": as_of,
            "horizon_sessions": 5,
            "direction": "up",
            "probability_up_pct": 64.7,
            "target_price": close + 4.6,
            "expected_return_pct": round(4.6 / close * 100, 2),
            "model_version": "lgbm-ohlc-5d-v1",
            "validation": {
                "oos_samples": 105,
                "direction_accuracy_pct": 55.2,
                "brier": 0.251,
                "price_mae_pct": 3.14,
            },
            "points": [
                {"time": as_of, "value": close},
                {"time": "2026-07-17", "value": close + 1.4},
                {"time": "2026-07-20", "value": close + 2.3},
                {"time": "2026-07-21", "value": close + 3.1},
                {"time": "2026-07-22", "value": close + 3.8},
                {"time": "2026-07-23", "value": close + 4.6},
            ],
        }
    return snapshot


REPORT_ITEM = {
    "report_date": "2026-07-11", "data_as_of": "2026-07-11", "coverage": 0.94,
    "market_action": "控制追價", "headline": "市場偏多，但高檔波動仍需控制部位",
    "key_industries": ["半導體", "AI 伺服器"], "model_versions": {"industry": "v1"},
    "page_count": 8, "generated_at": "2026-07-12T06:30:00+08:00",
    "metadata_object": "reports/v1/2026-07-11/report.json", "metadata_sha256": "0" * 64,
}

REPORT_METADATA = {
    "summary": ["市場偏多", "半導體維持領先"], "warnings": ["高檔波動擴大"],
    "public_report": {
        "market_recommendation": {
            "action": "控制追價", "level": "neutral",
            "headline": "市場偏多，但高檔波動仍需控制部位",
            "supporting_reasons": ["多數產業站上中期均線", "龍頭股量能維持"],
            "risk_reasons": ["高檔類股評價已偏緊"],
            "suggested_action": "優先等待拉回，避免一次投入全部資金。",
            "confidence": "可信度中等", "invalidation_conditions": ["市場跌破月線"],
        },
        "key_points": ["半導體仍是主要領漲產業", "追價風險升高", "保留現金等待拉回"],
        "industries": [{
            "name": "半導體", "action": "分批布局", "headline": "趨勢偏多但避免追高",
            "probability": 64.0, "rotation": "領先區", "risk": "評價偏高", "confidence": "可信度中等",
        }],
        "stocks": [{
            "symbol": "2330", "name": "台積電", "action": "分批布局",
            "headline": "趨勢偏多但短線不追價", "probability": 63.0,
            "risks": ["短線漲幅擴大"], "confidence": "可信度中等",
        }],
        "backtest": {
            "industry": "半導體", "periods": 24, "sample_quality": "可信度中等",
            "interpretation": {
                "advantage": "歷史結果略優於買進持有，仍需考慮成本與樣本限制。",
                "cumulative_return": "投入 10 萬元，歷史結果約變成 10.8 萬元。",
                "maximum_drawdown": "最差階段約回落 6%。", "win_rate": "每 100 次約 57 次獲利。",
                "cash_ratio": "策略約 35% 的時間保持空手。",
            },
        },
        "model_quality": {"samples": 24, "direction_accuracy": "54.0%", "brier_score": "0.230"},
    },
}


stock_app._published_dashboard_snapshot = visual_dashboard
stock_app.fetch_published_quant_snapshot = visual_quant_snapshot
stock_app.find_industry_peers = lambda _code: {"category": "半導體", "codes": ["2454"]}
stock_app.get_stock_name = lambda code: "聯發科" if code == "2454" else "台積電"


class VisualCalendar:
    def next_session(self, _value):
        return datetime.date(2026, 7, 16)


_REPORT_ROOT = tempfile.TemporaryDirectory()
_REPORT_PUBLISH = Path(_REPORT_ROOT.name)
publish_report_v2(
    _REPORT_PUBLISH,
    build_post_close_observation_metadata(
        observation_dashboard(),
        VisualCalendar(),
    ),
)
_REPORT_V2_ROOT = _REPORT_PUBLISH / "publish" / "reports" / "v2"
_REPORT_OBJECTS = {
    f"reports/v2/{path.relative_to(_REPORT_V2_ROOT).as_posix()}": path.read_bytes()
    for path in _REPORT_V2_ROOT.rglob("*")
    if path.is_file()
}
stock_app._gcs_get_report_v2_object = (
    lambda path, _size: _REPORT_OBJECTS.get(path)
)
stock_app.line_auth_store = VisualAuthStore()
stock_app.line_store = VisualLineStore()
stock_app.utc_now = lambda: NOW


@stock_app.app.get("/__visual_login")
def visual_login():
    response = redirect("/account")
    response.set_cookie(
        stock_app.line_login_config.session_cookie_name,
        sign_opaque_token(SESSION_ID, stock_app.line_login_config.session_secret),
        httponly=True,
        samesite="Lax",
        path="/",
    )
    return response


if __name__ == "__main__":
    stock_app.app.run(host="127.0.0.1", port=5099, debug=False, use_reloader=False)
