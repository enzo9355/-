import datetime
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")

import app as stock_app
from stock_papi.services.report_view import build_observation_report_view
from tests.report_fixtures import status_stock_document


def observation_dashboard():
    return {
        "schema_version": 2,
        "kind": "absorb-observation-dashboard",
        "product_mode": "observation",
        "market": "TW",
        "observation_as_of": "2026-07-15",
        "generated_at": "2026-07-16T06:35:08Z",
        "source_manifest": (
            "quant/v1/manifests/TW-20260716T063508Z-aaaaaaaaaaaa.json"
        ),
        "source_manifest_sha256": "a" * 64,
        "prediction_capability": {
            "mode": "research",
            "observation_enabled": True,
            "probability_allowed": False,
            "ranking_allowed": False,
            "strong_action_allowed": False,
            "performance_endorsement_allowed": False,
        },
        "market_observation": {
            "return_1d_pct": 0.82,
            "return_5d_pct": 1.35,
            "return_20d_pct": -0.4,
            "return_60d_pct": 3.1,
            "advancing_count": 1200,
            "declining_count": 700,
            "unchanged_count": 35,
            "ma20_breadth_pct": 61.2,
            "ma60_breadth_pct": 55.3,
            "new_high_20d_count": 87,
            "new_low_20d_count": 22,
            "median_volume_ratio": 1.08,
            "median_institution_net_ratio_pct": 0.12,
            "realized_volatility_20d_pct": 17.5,
            "risk_state": "normal",
        },
        "industry_observations": [
            {
                "name": "半導體",
                "component_count": 40,
                "available_count": 39,
                "coverage": 0.975,
                "return_1d_pct": 1.8,
                "return_5d_pct": 3.2,
                "return_20d_pct": 4.1,
                "relative_return_5d_pct": 1.85,
                "relative_return_20d_pct": 4.5,
                "advancing_ratio_pct": 72.5,
                "ma20_breadth_pct": 67.5,
                "median_volume_ratio": 1.2,
                "median_institution_net_ratio_pct": 0.25,
                "phase": "strong",
                "display_order": 1,
            }
        ],
        "heatmap": [
            {
                "name": "半導體",
                "metric_name": "relative_return_5d_pct",
                "metric_value_pct": 1.85,
                "available_count": 39,
                "coverage": 0.975,
                "tone": "steady",
            }
        ],
        "daily_focus": [
            "市場風險狀態：normal",
            "半導體 5 日相對大盤 +1.85%",
        ],
        "stock_events": [
            {
                "symbol": "2330",
                "name": "台積電",
                "event_type": "volume_surge",
                "severity": "medium",
                "metric_value": 2.2,
                "unit": "ratio",
                "observation": "量能異常放大",
                "as_of": "2026-07-15",
            }
        ],
        "etf_observations": [
            {
                "symbol": "0050",
                "name": "元大台灣50",
                "price": 58.2,
                "return_1d_pct": 0.7,
                "return_5d_pct": 1.1,
                "return_20d_pct": 2.3,
                "volume_ratio": 1.05,
                "trend_observation": "above_ma20_ma60",
                "as_of": "2026-07-15",
            }
        ],
        "data_quality": {
            "coverage": 0.997,
            "symbol_count": 2070,
            "failure_count": 6,
        },
        "gates": {"prediction_separation": "PASS"},
    }


def quant_snapshot(symbol="2330", market="TW"):
    start = datetime.date(2026, 5, 13)
    rows = []
    for index in range(65):
        day = start + datetime.timedelta(days=index)
        close = 100 + index
        rows.append(
            {
                "Date": f"{day.isoformat()}T00:00:00.000",
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "MA20": close - 1,
                "MA60": close - 2,
                "RSI": 68,
                "MACD_OSC": 0.3,
                "K": 62,
                "D": 54,
                "VOL_RATIO": 2.2,
                "INST_NET_RATIO": 0.03,
                "ForeignNet": 1000,
                "DATA_PRICE_WARNING": 0,
                "AI_P": 99,
            }
        )
    return {
        "schema_version": 1,
        "market": market,
        "symbol": symbol,
        "name": "台積電" if symbol == "2330" else symbol,
        "as_of": rows[-1]["Date"][:10],
        "model_version": "must-not-leak",
        "daily": rows,
        "backtest": {"accuracy": 100},
    }


def status_observation():
    snapshot = status_stock_document()
    evidence = snapshot["trading_status_evidence"]
    return {
        "symbol": snapshot["symbol"],
        "name": snapshot["name"],
        "status": snapshot["observation_kind"],
        "label": "當日無正常交易",
        "observation_as_of": snapshot["observation_as_of"],
        "latest_regular_price_date": snapshot["latest_regular_price_date"],
        "evidence_sha256": evidence["evidence_sha256"],
        "last_regular_close": 100.0,
    }


class ObservationPublicSurfaceTests(unittest.TestCase):
    forbidden = (
        "五日上漲機率",
        "產業預測",
        "精選標的",
        "支持這項建議",
        "投資金額試算",
        "查看模型與回測詳細數據",
    )

    def test_dashboard_is_observation_only(self):
        response = stock_app.app.test_client().get("/dashboard")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "市場實況",
            "今日焦點",
            "產業觀察",
            "個股與 ETF",
            "ASK ABSORB",
            "AI 五日情境",
        ):
            self.assertIn(label, html)
        for text in self.forbidden:
            self.assertNotIn(text, html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_industry_and_stock_pages_use_actual_observations_only(self, load_snapshot):
        load_snapshot.return_value = observation_dashboard()

        client = stock_app.app.test_client()
        industries = client.get("/industries")
        stocks = client.get("/stocks")
        html = industries.get_data(as_text=True) + stocks.get_data(as_text=True)

        self.assertEqual(industries.status_code, 200)
        self.assertEqual(stocks.status_code, 200)
        for label in (
            "產業強弱與關注清單",
            "半導體",
            "+1.85%",
            "個股異常事件",
            "量能異常放大",
            "ETF 觀察",
        ):
            self.assertIn(label, html)
        for text in self.forbidden:
            self.assertNotIn(text, html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_uses_actual_snapshot_and_hides_model_fields(self, fetch):
        fetch.return_value = quant_snapshot()

        response = stock_app.app.test_client().get("/stock/2330")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        fetch.assert_called_once_with("2330")
        for label in (
            "個股觀察摘要",
            "價格與均線",
            "籌碼觀察",
            "技術指標",
            "風險事件",
            "AI 五日預測尚未發布",
        ):
            self.assertIn(label, html)
        for text in self.forbidden:
            self.assertNotIn(text, html)
        self.assertNotIn("must-not-leak", html)
        chart = html.split('id="stock-chart-data"', 1)[1]
        self.assertNotIn("prediction", chart)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_status_stock_page_and_line_card_never_present_stale_price_as_current(self, fetch):
        fetch.return_value = status_stock_document()

        data = stock_app.build_stock_observation(fetch.return_value)
        response = stock_app.app.test_client().get("/stock/2303")
        html = response.get_data(as_text=True)
        flex = stock_app.build_stock_observation_flex(
            "2303", "測試股票 2303", data, "https://example.com/stock/2303"
        )
        encoded = json.dumps(flex, ensure_ascii=False)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["observation_kind"], "official_no_regular_trade")
        self.assertEqual(data["last_regular_close"], 100.0)
        for key in (
            "price",
            "trend_observation",
            "volume_ratio",
            "rsi",
            "macd_osc",
            "k",
            "d",
            "risk_events",
            "candles",
        ):
            self.assertNotIn(key, data)
        for output in (html, encoded):
            self.assertIn("當日無正常交易", output)
            self.assertIn("2026-07-16", output)
            for forbidden in ("最新收盤", "今日漲跌", "量比", "RSI", "MACD", "K 值", "D 值"):
                self.assertNotIn(forbidden, output)

    def test_report_view_accepts_only_exact_status_observation_shape(self):
        dashboard = observation_dashboard()
        dashboard["trading_status_observations"] = [status_observation()]
        metadata = {
            "schema_version": 2,
            "product_mode": "observation",
            "report_type": "post_close",
            "title": "盤後市場觀察",
            "source_market_date": "2026-07-29",
            "applicable_trading_date": "2026-07-30",
            "published_at": "2026-07-29T10:00:00Z",
            "summary": [],
            "warnings": [],
            "content": {
                key: dashboard[key]
                for key in (
                    "market_observation",
                    "industry_observations",
                    "heatmap",
                    "stock_events",
                    "trading_status_observations",
                    "etf_observations",
                    "daily_focus",
                    "data_quality",
                )
            },
        }

        report = build_observation_report_view(metadata)
        self.assertEqual(
            report.core["trading_status_observations"],
            [status_observation()],
        )

        metadata["content"]["trading_status_observations"][0]["price"] = 100.0
        with self.assertRaisesRegex(Exception, "schema"):
            build_observation_report_view(metadata)

    def test_browser_renderer_has_timeout_and_no_prediction_rendering(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("AbortController", script)
        self.assertIn("finally", script)
        self.assertNotIn(".innerHTML", script)
        self.assertNotIn('title: "五日預測"', script)
        dashboard_renderer = script.split(
            "function renderDashboard(data)", 1
        )[1].split("function loginLocation()", 1)[0]
        for key in (
            "market_observation",
            "industry_observations",
            "stock_events",
            "etf_observations",
        ):
            self.assertIn(key, dashboard_renderer)
        for key in ("probability", "top_picks", "recommendation"):
            self.assertNotIn(key, dashboard_renderer)

    def test_observation_payload_does_not_serialize_model_data(self):
        document = stock_app.build_stock_observation(quant_snapshot())
        encoded = json.dumps(document, ensure_ascii=False)

        for text in (
            "must-not-leak",
            '"AI_P"',
            '"prob"',
            '"backtest"',
            '"recommendation"',
        ):
            self.assertNotIn(text, encoded)

    def test_line_observation_cards_do_not_expose_research_outputs(self):
        data = stock_app.build_stock_observation(quant_snapshot())
        payloads = (
            stock_app.build_stock_observation_flex(
                "2330",
                "台積電",
                data,
                "https://example.com/stock/2330",
            ),
            stock_app.build_line_navigation_flex("https://example.com"),
            stock_app.build_alert_menu_flex(
                "2330", "台積電", prediction_allowed=False
            ),
        )

        encoded = json.dumps(payloads, ensure_ascii=False)
        for text in (
            "五日上漲機率",
            "模型輸出",
            "產業預測",
            "推薦",
            "回測",
            "績效",
            "投資試算",
        ):
            self.assertNotIn(text, encoded)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    @patch.object(stock_app, "search_stock_code")
    def test_formal_conversation_is_deterministic_observation_only(
        self, search, fetch
    ):
        search.return_value = ("2330", "台積電")
        fetch.return_value = quant_snapshot()

        answer = stock_app.run_absorb_conversation(
            principal="web:test",
            question="台積電目前的價格與均線如何？",
        )

        self.assertIn("最新收盤", answer.text)
        self.assertIn("均線狀態", answer.text)
        self.assertEqual(
            answer.tools_used, ("verified_observation_snapshot",)
        )
        for text in (
            "五日上漲機率",
            "推薦",
            "排名",
            "回測",
            "勝率",
            "績效",
        ):
            self.assertNotIn(text, answer.text)


if __name__ == "__main__":
    unittest.main()
