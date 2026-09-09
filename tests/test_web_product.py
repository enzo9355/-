import datetime as dt
import inspect
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask, render_template


os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")

import app as stock_app
from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.web import routes as web_routes

from tests.test_batch_calendar import calendar_document
from tests.test_observation_public_surfaces import (
    observation_dashboard,
    quant_snapshot,
)


def prediction_product(market="TW", symbol="2330", as_of="2026-07-15"):
    return {
        "schema_version": 1,
        "kind": "absorb-five-session-predictions",
        "market": market,
        "as_of": as_of,
        "model_version": "lgbm-5d-v1",
        "backtest_sha256": "b" * 64,
        "entities": {
            symbol: {
                "symbol": symbol,
                "entity_type": "market_index" if symbol.startswith("^") or symbol == "TAIEX" else "security",
                "as_of": as_of,
                "target_session": "2026-07-22",
                "current_price": 23150.25 if symbol == "TAIEX" else 164.0,
                "up_probability": 0.68,
                "predicted_return_5d": 0.0427,
                "predicted_price": 24138.765675 if symbol == "TAIEX" else 171.0028,
                "predicted_change_pct": 4.27,
            }
        },
    }


class WebProductTests(unittest.TestCase):
    @patch.object(stock_app, "_published_prediction_snapshot")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_dashboard_plainly_marks_uncalibrated_research_prediction(
        self, load_snapshot, load_prediction
    ):
        snapshot = observation_dashboard()
        snapshot["market_index"] = {
            "symbol": "TAIEX", "name": "加權指數", "as_of": "2026-07-15",
            "price": 23150.25, "change": 188.4, "change_pct": 0.82,
            "open": 22982.1, "high": 23210.8, "low": 22940.6,
            "candles": [], "ma20": [], "returns": {},
            "source": {"provider": "TWSE", "kind": "official_index_daily"},
        }
        estimate = prediction_product(symbol="TAIEX")
        estimate["schema_version"] = 2
        estimate["validation_mode"] = "research"
        estimate.pop("backtest_sha256")
        load_snapshot.return_value = snapshot
        load_prediction.return_value = estimate

        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)

        self.assertIn("模型推估上漲機率（未校準）", html)
        self.assertIn("未經回測校準", html)
        self.assertNotIn("目前正式預測", html)

    @patch.object(stock_app, "_published_prediction_snapshot")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_dashboard_renders_verified_market_index_level_and_candles(
        self, load_snapshot, load_prediction
    ):
        snapshot = observation_dashboard()
        snapshot["market_index"] = {
            "symbol": "TAIEX",
            "name": "加權指數",
            "as_of": "2026-07-15",
            "price": 23150.25,
            "change": 188.4,
            "change_pct": 0.82,
            "open": 22982.1,
            "high": 23210.8,
            "low": 22940.6,
            "candles": [
                {
                    "time": "2026-07-14",
                    "open": 22800.0,
                    "high": 23010.0,
                    "low": 22760.0,
                    "close": 22961.85,
                },
                {
                    "time": "2026-07-15",
                    "open": 22982.1,
                    "high": 23210.8,
                    "low": 22940.6,
                    "close": 23150.25,
                },
            ],
            "ma20": [
                {"time": "2026-07-14", "value": 22790.0},
                {"time": "2026-07-15", "value": 22820.0},
            ],
            "returns": {},
            "source": {"provider": "TWSE", "kind": "official_index_daily"},
        }
        load_snapshot.return_value = snapshot
        load_prediction.return_value = prediction_product(symbol="TAIEX")

        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)

        for marker in (
            "加權指數",
            "23,150.25",
            "+188.40",
            "+0.82%",
            'id="market-index-chart"',
            'role="img"',
            "加權指數最近五個交易日 OHLC",
            'id="market-index-chart-data"',
            '"time": "2026-07-15"',
            "五日上漲機率",
            "68.0%",
            "24,138.77",
            "+4.27%",
            "2026-07-22",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_chart_rejects_unverified_embedded_prediction_display(self, fetch):
        snapshot = quant_snapshot()
        snapshot["prediction_display"] = {
            "status": "published",
            "as_of": snapshot["as_of"],
            "horizon_sessions": 5,
            "direction": "up",
            "probability_up_pct": 68.0,
            "target_price": 171.0,
            "expected_return_pct": 4.27,
            "model_version": "lgbm-ohlc-5d-v1",
            "validation": {
                "oos_samples": 75,
                "direction_accuracy_pct": 57.3,
                "brier": 0.241,
                "price_mae_pct": 3.2,
            },
            "points": [
                {"time": snapshot["as_of"], "value": 164.0},
                {"time": "2026-07-21", "value": 171.0},
            ],
        }
        fetch.return_value = snapshot

        response = stock_app.app.test_client().get("/stock/2330")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI 五日預測尚未發布", html)
        self.assertIn("個股最近五個交易日 OHLC", html)
        self.assertNotIn("五日上漲機率", html)
        self.assertNotIn("171.00", html)
        self.assertNotIn('"prediction"', html)
        self.assertNotIn('"AI_P"', html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_stock_events_are_split_into_supported_categories_with_severity(self, load):
        snapshot = observation_dashboard()
        snapshot["stock_events"] = [
            {"symbol": "6955", "name": "邦睿生技-創", "event_type": "price_move", "severity": "high", "observation": "單日跌幅異常", "metric_value": -10.83, "unit": "pct", "as_of": "2026-08-26"},
            {"symbol": "3313", "name": "斐成", "event_type": "price_move", "severity": "high", "observation": "單日漲幅異常", "metric_value": 10.0, "unit": "pct", "as_of": "2026-08-26"},
            {"symbol": "2603", "name": "長榮", "event_type": "volume_surge", "severity": "medium", "observation": "量能異常放大", "metric_value": 2.8, "unit": "ratio", "as_of": "2026-08-26"},
            {"symbol": "2330", "name": "台積電", "event_type": "institution_flow", "severity": "medium", "observation": "機構淨流入偏高", "metric_value": 3.1, "unit": "pct", "as_of": "2026-08-26"},
            {"symbol": "2454", "name": "聯發科", "event_type": "rsi_overbought", "severity": "medium", "observation": "RSI 進入過熱區", "metric_value": 74.0, "unit": "index", "as_of": "2026-08-26"},
            {"symbol": "0050", "name": "元大台灣50", "event_type": "data_warning", "severity": "high", "observation": "資料來源價差警示", "metric_value": 1, "unit": "flag", "as_of": "2026-08-26"},
        ]
        snapshot["trading_status_observations"] = [{
            "symbol": "1589", "name": "永冠-KY", "label": "停止買賣",
            "observation_as_of": "2026-08-26",
        }]
        load.return_value = snapshot

        html = stock_app.app.test_client().get("/stocks").get_data(as_text=True)

        self.assertIn("異常上漲", html)
        self.assertIn("異常下跌", html)
        for label in ("量能異常", "法人動向", "技術面", "官方事件", "資料警示"):
            self.assertIn(label, html)
        self.assertIn('data-event-group="up"', html)
        self.assertIn('data-event-group="down"', html)
        self.assertIn('data-event-group="volume"', html)
        self.assertIn('data-event-group="official"', html)
        self.assertIn("極端", html)
        self.assertIn("顯著", html)
        self.assertLess(html.index("斐成"), html.index("邦睿生技-創"))

    def test_shell_uses_collapsible_dashboard_sidebar_with_accessible_toggle(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        script = Path(stock_app.app.static_folder, "app.js").read_text(encoding="utf-8")

        for marker in (
            'class="dashboard-sidebar"',
            'data-sidebar-toggle',
            'aria-controls="dashboard-sidebar"',
            'aria-expanded="true"',
            'id="dashboard-sidebar"',
        ):
            self.assertIn(marker, html)
        self.assertIn("initSidebar", script)
        self.assertIn('event.key === "Escape"', script)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_does_not_train_or_infer_a_forecast_during_request(self, fetch):
        fetch.return_value = quant_snapshot()

        html = stock_app.app.test_client().get("/stock/2330").get_data(as_text=True)

        self.assertIn("AI 五日預測尚未發布", html)
        self.assertNotIn("五日上漲機率", html)

    def test_chart_renderer_draws_market_candles_and_prediction_marker(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        for marker in (
            "initMarketIndexChart",
            'bySelector("#market-index-chart")',
            'bySelector("#market-index-chart-data")',
            "predictionSeries.setMarkers",
            'text: "AI 5日"',
            "LineStyle.Dashed",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, script)

    def test_data_freshness_classifier_preserves_verified_dates_and_states(self):
        classifier = getattr(web_routes.system, "classify_data_freshness", None)

        self.assertIsNotNone(classifier)
        self.assertIn("next_session", inspect.signature(classifier).parameters)
        calendars = TradingCalendarSet.from_documents([
            calendar_document(2026, closed=("2026-07-20",))
        ])
        cases = (
            ("source date", "2026-07-17", "current"),
            ("weekend", "2026-07-18", "current"),
            ("holiday", "2026-07-20", "current"),
            ("applicable session", "2026-07-21", "current"),
            ("next session", "2026-07-22", "updating"),
            ("later missed session", "2026-07-23", "stale"),
        )
        for label, reference_date, expected in cases:
            with self.subTest(label=label):
                actual = classifier(
                    source_market_date="2026-07-17",
                    applicable_trading_date="2026-07-21",
                    reference_date=reference_date,
                    next_session=calendars.next_session,
                )

                self.assertEqual(actual["status"], expected)
                self.assertEqual(actual["source_market_date"], "2026-07-17")
                self.assertEqual(
                    actual["applicable_trading_date"], "2026-07-21"
                )

        unavailable = classifier(
            source_market_date=None,
            applicable_trading_date=None,
            reference_date="2026-07-21",
            next_session=calendars.next_session,
        )
        self.assertEqual(unavailable["status"], "unavailable")
        missing_calendar = classifier(
            source_market_date="2026-07-17",
            applicable_trading_date="2026-07-17",
            reference_date="2026-07-21",
            next_session=None,
        )
        self.assertEqual(missing_calendar["status"], "unavailable")
        self.assertNotEqual(missing_calendar["status"], "current")

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_tw_health_freshness_uses_production_calendar_for_holidays_and_staleness(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-09-24",
        }
        load_index.side_effect = lambda market="TW": [{
            "market": market,
            "report_type": "post_close",
            "source_market_date": "2026-09-24" if market == "TW" else "2026-09-24",
            "applicable_trading_date": "2026-09-24",
        }]
        cases = (
            ("session", dt.datetime(2026, 9, 24, 3, tzinfo=dt.timezone.utc), "current"),
            ("holiday", dt.datetime(2026, 9, 25, 3, tzinfo=dt.timezone.utc), "current"),
            ("holiday eve", dt.datetime(2026, 9, 28, 3, tzinfo=dt.timezone.utc), "current"),
            ("next session", dt.datetime(2026, 9, 29, 3, tzinfo=dt.timezone.utc), "updating"),
            ("missed session", dt.datetime(2026, 9, 30, 3, tzinfo=dt.timezone.utc), "stale"),
        )
        for label, instant, expected in cases:
            with self.subTest(label=label), patch.object(
                web_routes.system, "_utc_now", return_value=instant
            ):
                response = stock_app.app.test_client().get("/health/data")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["markets"]["TW"]["status"], expected)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_tw_health_fails_unavailable_outside_calendar_evidence(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2027-01-04",
        }
        load_index.side_effect = lambda market="TW": [{
            "market": market,
            "report_type": "post_close",
            "source_market_date": "2027-01-04",
            "applicable_trading_date": "2027-01-04",
        }]
        with patch.object(
            web_routes.system,
            "_utc_now",
            return_value=dt.datetime(2027, 1, 5, 3, tzinfo=dt.timezone.utc),
        ):
            response = stock_app.app.test_client().get("/health/data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["markets"]["TW"]["status"], "unavailable"
        )

    def test_market_local_date_uses_aware_tw_and_us_timezones(self):
        market_local_date = getattr(web_routes.system, "market_local_date", None)

        self.assertIsNotNone(market_local_date)
        early_utc = dt.datetime(2026, 8, 24, 3, 30, tzinfo=dt.timezone.utc)
        late_utc = dt.datetime(2026, 8, 24, 16, 30, tzinfo=dt.timezone.utc)

        self.assertEqual(
            market_local_date("TW", now=early_utc), dt.date(2026, 8, 24)
        )
        self.assertEqual(
            market_local_date("US", now=early_utc), dt.date(2026, 8, 23)
        )
        self.assertEqual(
            market_local_date("TW", now=late_utc), dt.date(2026, 8, 25)
        )
        self.assertEqual(
            market_local_date("US", now=late_utc), dt.date(2026, 8, 24)
        )
        with self.assertRaises(ValueError):
            market_local_date("TW", now=dt.datetime(2026, 8, 24, 3, 30))

    def test_invalid_freshness_dates_normalize_to_json_safe_none(self):
        classifier = web_routes.system.classify_data_freshness
        self.assertIn("next_session", inspect.signature(classifier).parameters)

        actual = classifier(
            source_market_date={"not-json-safe"},
            applicable_trading_date="2026-8-24",
            reference_date=dt.date(2026, 8, 24),
            next_session=lambda value: value + dt.timedelta(days=1),
        )

        self.assertEqual(actual["status"], "unavailable")
        self.assertIsNone(actual["source_market_date"])
        self.assertIsNone(actual["applicable_trading_date"])
        json.dumps(actual)

        with stock_app.app.test_request_context("/dashboard"):
            html = render_template(
                "dashboard.html",
                observation={},
                daily_cards={},
                data_freshness={"TW": actual},
            )
        self.assertIn("尚無已驗證日期", html)
        self.assertNotIn("not-json-safe", html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_internal_us_type_error_is_isolated_and_dashboards_remain_renderable(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-08-24",
        }
        calls = []

        def reports_for(*, market="TW"):
            calls.append(market)
            if market == "US":
                raise TypeError("reader implementation failed")
            return [{
                "market": "TW",
                "report_type": "post_close",
                "source_market_date": "2026-08-24",
                "applicable_trading_date": "2026-08-24",
            }]

        load_index.side_effect = reports_for
        fixed_now = dt.datetime(2026, 8, 24, 12, tzinfo=dt.timezone.utc)
        with patch.object(
            web_routes.system, "_utc_now", return_value=fixed_now, create=True
        ):
            health = stock_app.app.test_client().get("/health/data")

        self.assertEqual(health.status_code, 200)
        payload = health.get_json()
        self.assertEqual(payload["markets"]["TW"]["status"], "current")
        self.assertEqual(payload["markets"]["US"]["status"], "unavailable")
        self.assertIsNone(payload["markets"]["US"]["source_market_date"])
        self.assertEqual(calls, ["TW", "US"])

        calls.clear()
        dashboard = stock_app.app.test_client().get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(calls, ["TW"])

        calls.clear()
        us_dashboard = stock_app.app.test_client().get("/us")
        self.assertEqual(us_dashboard.status_code, 503)
        self.assertEqual(calls, ["US"])

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_invalid_index_dates_are_json_safe_and_do_not_fall_back(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-08-24",
        }

        def reports_for(*, market="TW"):
            if market == "TW":
                return [{
                    "market": "TW",
                    "report_type": "post_close",
                    "source_market_date": "2026-08-24",
                    "applicable_trading_date": "2026-08-24",
                }]
            return [{
                "market": "US",
                "report_type": "post_close",
                "source_market_date": {"not-json-safe"},
                "applicable_trading_date": "2026-8-24",
            }]

        load_index.side_effect = reports_for
        fixed_now = dt.datetime(2026, 8, 24, 12, tzinfo=dt.timezone.utc)
        with patch.object(
            web_routes.system, "_utc_now", return_value=fixed_now
        ):
            response = stock_app.app.test_client().get("/health/data")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["markets"]["TW"]["status"], "current")
        self.assertEqual(payload["markets"]["US"]["status"], "unavailable")
        self.assertIsNone(payload["markets"]["US"]["source_market_date"])
        self.assertIsNone(
            payload["markets"]["US"]["applicable_trading_date"]
        )
        json.dumps(payload)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_report_item_market_mismatch_fails_closed_for_us_only(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-08-24",
        }
        load_index.side_effect = lambda market="TW": [{
            "market": "TW",
            "report_type": "post_close",
            "source_market_date": "2026-08-24",
            "applicable_trading_date": "2026-08-24",
        }]

        calendars = TradingCalendarSet.from_documents([
            calendar_document(2026)
        ])
        fixed_now = dt.datetime(2026, 8, 24, 12, tzinfo=dt.timezone.utc)
        with patch.object(
            web_routes.system,
            "_next_session_for_market",
            side_effect=lambda _market, value: calendars.next_session(value),
        ), patch.object(
            web_routes.system, "_utc_now", return_value=fixed_now
        ):
            response = stock_app.app.test_client().get("/health/data")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["markets"]["TW"]["status"], "current")
        self.assertEqual(payload["markets"]["US"]["status"], "unavailable")
        self.assertIsNone(payload["markets"]["US"]["source_market_date"])

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_non_dashboard_page_does_not_load_freshness_data(
        self, load_index, load_dashboard
    ):
        response = stock_app.app.test_client().get("/learn")

        self.assertEqual(response.status_code, 200)
        load_index.assert_not_called()
        load_dashboard.assert_not_called()

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_tw_dashboard_omits_explicitly_mismatched_daily_card(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = observation_dashboard()
        load_index.return_value = [{
            "market": "US",
            "report_type": "post_close",
            "source_market_date": "2026-08-19",
            "applicable_trading_date": "2026-08-20",
            "title": "MISMATCHED US DAILY CARD",
            "summary": ["MISMATCHED US SUMMARY"],
        }]

        response = stock_app.app.test_client().get("/dashboard")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("MISMATCHED US DAILY CARD", html)
        self.assertNotIn("MISMATCHED US SUMMARY", html)
        self.assertNotIn("/reports/2026-08-19/post-close", html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_data_health_uses_one_utc_instant_for_both_markets(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-08-24",
        }

        def reports_for(market="TW"):
            date = "2026-08-24" if market == "TW" else "2026-08-23"
            return [{
                "market": market,
                "report_type": "post_close",
                "source_market_date": date,
                "applicable_trading_date": date,
            }]

        load_index.side_effect = reports_for
        instants = (
            dt.datetime(2026, 8, 24, 3, 30, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 24, 16, 30, tzinfo=dt.timezone.utc),
        )
        with patch.object(
            web_routes.system, "_utc_now", side_effect=instants
        ) as clock:
            response = stock_app.app.test_client().get("/health/data")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["markets"]["TW"]["status"], "current")
        self.assertEqual(payload["markets"]["US"]["status"], "current")
        self.assertEqual(clock.call_count, 1)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_data_health_keeps_service_ok_separate_from_stale_market_data(
        self, load_index, load_dashboard
    ):
        load_dashboard.return_value = {
            "market": "TW",
            "observation_as_of": "2026-08-20",
        }
        load_index.side_effect = lambda market="TW": [
            {
                "market": market,
                "report_type": "post_close",
                "source_market_date": "2026-08-20",
                "applicable_trading_date": "2026-08-20",
            }
        ]

        calendars = TradingCalendarSet.from_documents([
            calendar_document(2026)
        ])
        fixed_now = dt.datetime(2026, 8, 24, 12, tzinfo=dt.timezone.utc)
        with patch.object(
            web_routes.system,
            "_next_session_for_market",
            side_effect=lambda _market, value: calendars.next_session(value),
        ), patch.object(
            web_routes.system, "_utc_now", return_value=fixed_now
        ):
            response = stock_app.app.test_client().get("/health/data")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["service"]["status"], "ok")
        self.assertEqual(payload["markets"]["TW"]["status"], "stale")
        self.assertNotEqual(payload["markets"]["TW"]["status"], "current")
        self.assertEqual(
            payload["markets"]["TW"]["source_market_date"], "2026-08-20"
        )
        self.assertEqual(
            payload["markets"]["TW"]["applicable_trading_date"], "2026-08-20"
        )

    @patch.object(stock_app, "_published_dashboard_snapshot")
    @patch.object(stock_app, "_published_report_index_v2")
    def test_dashboards_render_verified_freshness_with_exact_dates(
        self, load_index, load_dashboard
    ):
        snapshot = observation_dashboard()
        snapshot["observation_as_of"] = "2026-08-20"
        load_dashboard.return_value = snapshot
        load_index.side_effect = lambda market="TW": [
            {
                "market": market,
                "report_type": "post_close",
                "source_market_date": "2026-08-20",
                "applicable_trading_date": "2026-08-20",
            }
        ]

        calendars = TradingCalendarSet.from_documents([
            calendar_document(2026)
        ])
        fixed_now = dt.datetime(2026, 8, 24, 12, tzinfo=dt.timezone.utc)
        with patch.object(
            web_routes.system,
            "_next_session_for_market",
            side_effect=lambda _market, value: calendars.next_session(value),
        ), patch.object(
            web_routes.system, "_utc_now", return_value=fixed_now
        ):
            dashboard = stock_app.app.test_client().get("/dashboard")
        with stock_app.app.test_request_context("/us"):
            us_html = render_template(
                "us_dashboard.html",
                market="US",
                summary={
                    "source_market_date": "2026-08-20",
                    "applicable_trading_date": "2026-08-20",
                    "executive_summary": {
                        "one_line_conclusion": "已驗證摘要",
                        "largest_risk": "測試風險",
                    },
                    "key_events": [],
                    "validation": {"status": "unavailable", "reason": "測試"},
                },
                data_freshness={
                    "US": {
                        "status": "stale",
                        "source_market_date": "2026-08-20",
                        "applicable_trading_date": "2026-08-20",
                    }
                },
            )

        self.assertEqual(dashboard.status_code, 200)
        for html in (dashboard.get_data(as_text=True), us_html):
            self.assertIn('data-freshness-status="stale"', html)
            self.assertIn("資料狀態：資料過期", html)
            self.assertIn("來源交易日", html)
            self.assertIn("適用交易日", html)
            self.assertIn("2026-08-20", html)

    def test_base_shell_renders_with_dashboard_endpoint_without_us_endpoints(self):
        partial_app = Flask("partial-absorb", root_path=stock_app.app.root_path)
        partial_app.jinja_env.globals["STATIC_ASSET_VERSION"] = "test"

        @partial_app.get("/")
        def dashboard_page():
            return render_template("base.html")

        response = partial_app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('href="/us"', html)
        self.assertIn('href="/market"', html)
        self.assertIn('href="/reports"', html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_information_architecture_has_distinct_server_rendered_pages(self, load):
        load.return_value = observation_dashboard()
        client = stock_app.app.test_client()
        expectations = {
            "/": "台股市場研究摘要",
            "/market": "市場實況",
            "/industries": "產業觀察",
            "/stocks": "個股與 ETF",
            "/ask": "ASK ABSORB",
            "/learn": "市場觀察小辭典",
        }

        for path, heading in expectations.items():
            with self.subTest(path=path):
                response = client.get(path)
                html = response.get_data(as_text=True)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(html.count("<h1"), 1)
                self.assertIn(heading, html)

        home = client.get("/").get_data(as_text=True)
        self.assertNotIn('id="industry-observations"', home)
        self.assertNotIn('id="stock-events"', home)
        self.assertIn('href="/industries"', home)
        self.assertIn('href="/stocks"', home)
        self.assertIn('href="/ask"', home)
        self.assertIn('href="/learn"', home)

    def test_legacy_market_map_redirects_to_industries(self):
        response = stock_app.app.test_client().get("/market-map")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/industries"))

    def test_dashboard_starts_with_today_market_preparation_cards(self):
        response = stock_app.app.test_client().get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("今日市場準備", html)
        self.assertIn("盤後觀察", html)
        self.assertIn("盤前風險更新", html)

    def test_every_papi_theme_has_at_least_five_companies(self):
        self.assertTrue(
            all(
                len(names) >= 5
                for names in stock_app.PAPI_THEME_SECTORS.values()
            )
        )

    def test_build_market_heatmap_orders_strongest_first_for_preview(self):
        cards = [
            {
                "name": "弱勢",
                "count": 1,
                "score": 42,
                "leader": {"code": "1101", "prob": 42},
            },
            {
                "name": "強勢",
                "count": 2,
                "score": 68,
                "leader": {"code": "2330", "prob": 68},
            },
        ]

        result = stock_app.build_market_heatmap(cards)

        self.assertEqual([item["name"] for item in result], ["強勢", "弱勢"])
        self.assertEqual(result[0]["tone"], "hot")
        self.assertEqual(result[1]["tone"], "cold")

    def test_find_industry_peers_excludes_current_stock(self):
        market_map = {
            "全市場": ["2330", "2454", "2303"],
            "半導體": ["2330", "2454", "2303"],
        }

        peers = stock_app.find_industry_peers("2330", market_map, limit=2)

        self.assertEqual(
            peers, {"category": "半導體", "codes": ["2454", "2303"]}
        )

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_industries_page_renders_verified_actual_observations(self, load):
        load.return_value = observation_dashboard()

        response = stock_app.app.test_client().get("/industries")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "產業強弱與關注清單",
            "近 5 日相對大盤報酬",
            "關注公司",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "推薦",
            "回測",
            "勝率",
        ):
            self.assertNotIn(forbidden, html)

    def test_root_renders_dashboard_and_search_redirects_known_stock(self):
        client = stock_app.app.test_client()

        root = client.get("/")
        with patch.object(
            stock_app,
            "search_stock_code",
            side_effect=[("2330", "台積電"), (None, None)],
        ):
            found = client.get("/search?q=台積電")
            missing = client.get(
                "/search?q=不存在股票", follow_redirects=True
            )

        self.assertEqual(root.status_code, 200)
        self.assertIn("ABSORB", root.get_data(as_text=True))
        self.assertEqual(found.status_code, 302)
        self.assertTrue(found.headers["Location"].endswith("/stock/2330"))
        self.assertIn("找不到", missing.get_data(as_text=True))

    def test_empty_search_stays_on_dashboard_with_clear_error(self):
        response = stock_app.app.test_client().get(
            "/search?q=", follow_redirects=True
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("找不到", response.get_data(as_text=True))

    def test_base_shell_uses_absorb_brand_and_light_theme(self):
        response = stock_app.app.test_client().get("/dashboard")
        html = response.get_data(as_text=True)
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("ABSORB", html)
        self.assertIn('class="brand-wordmark"', html)
        self.assertIn('data-brand-wordmark', html)
        self.assertIn('aria-label="回到 ABSORB 主畫面"', html)
        self.assertNotIn('class="brand-mark"', html)
        self.assertIn("今天市場", html)
        self.assertIn("使用 LINE 登入", html)
        self.assertIn("已驗證市場觀察", html)
        self.assertIn('data-market-switch', html)
        self.assertIn('href="/us"', html)
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertIn("--absorb-navy:#122643", css)
        self.assertIn("--absorb-canvas:#f7f9fc", css)
        self.assertIn('"Avenir Next",Avenir,"Noto Sans TC"', css)
        self.assertIn(".research-command", css)
        self.assertIn(".market-switch a{display:grid;min-height:44px", css)
        self.assertIn(".quick-ask-backdrop[hidden]{display:none}", css)
        self.assertIn(".quick-ask-header button{display:grid;width:44px;height:44px", css)
        self.assertIn("--command-content-max:3200px", css)
        self.assertIn("--command-muted:#536575", css)
        self.assertIn(".evidence-canvas{", css)
        self.assertIn("background:var(--command-accent-surface)", css)
        self.assertIn(".brand-wordmark:hover{", css)
        self.assertIn("rotate(-1.5deg)", css)
        self.assertIn("@view-transition{navigation:auto}", css)
        self.assertIn("button,input,select,textarea{font:inherit}", css)
        self.assertIn(".button{display:inline-flex", css)
        self.assertIn(".command-metrics{grid-column:1/-1;grid-template-columns:repeat(4,minmax(0,1fr))", css)
        self.assertNotIn("border-left:4px", css.replace(" ", ""))
        self.assertNotIn("border-top:3px", css.replace(" ", ""))
        version = re.search(r'/static/app\.css\?v=([0-9a-f]{12})', html)
        self.assertIsNotNone(version)
        self.assertIn(f'/static/app.js?v={version.group(1)}', html)
        asset = stock_app.app.test_client().get(
            f'/static/app.js?v={version.group(1)}'
        )
        self.assertIn("immutable", asset.headers["Cache-Control"])
        asset.close()

    def test_us_product_navigation_and_stock_detail_keep_us_context(self):
        client = stock_app.app.test_client()
        with patch.object(
            stock_app, "_published_us_securities_observation",
            return_value={"stock_events": [], "etf_observations": []},
        ):
            us_stocks = client.get("/us/stocks")

        self.assertEqual(us_stocks.status_code, 200)
        html = us_stocks.get_data(as_text=True)
        self.assertIn('data-market="US"', html)
        for href in (
            'href="/us"',
            'href="/us/market"',
            'href="/us/industries"',
            'href="/us/stocks"',
            'href="/reports/us"',
        ):
            with self.subTest(href=href):
                self.assertIn(href, html)

        with patch.object(stock_app, "build_stock_observation", return_value={
            "code": "AAPL", "name": "Apple Inc.", "market": "US",
            "observation_kind": "status", "status_label": "資料不足",
            "observation_as_of": "2026-07-15", "evidence_sha256": "a" * 64,
        }):
            detail = client.get("/stock/AAPL")

        self.assertEqual(detail.status_code, 200)
        detail_html = detail.get_data(as_text=True)
        self.assertIn('data-market="US"', detail_html)
        self.assertIn('class="back-link" href="/us"', detail_html)

    def test_us_dashboard_renders_three_index_forecasts_and_actual_charts(self):
        summary = {
            "source_market_date": "2026-08-26",
            "applicable_trading_date": "2026-08-27",
            "executive_summary": {
                "one_line_conclusion": "美股市場摘要",
                "largest_risk": "波動仍需觀察",
            },
            "key_events": [],
            "validation": {"status": "unavailable", "reason": ""},
            "market": {"status": "available"},
            "industries": {"status": "available"},
        }
        forecasts = []
        for symbol, name, price in (
            ("^GSPC", "S&P 500", 6500.0),
            ("^IXIC", "Nasdaq Composite", 22000.0),
            ("^DJI", "道瓊工業指數", 46000.0),
        ):
            forecasts.append({
                "symbol": symbol, "name": name, "status": "current",
                "as_of": "2026-08-26", "target_session": "2026-09-02",
                "current_price": price, "probability_pct": 61.0,
                "probability_label": "五日上漲機率",
                "predicted_price": price * 1.02, "predicted_change_pct": 2.0,
                "line": [
                    {"time": "2026-08-26", "value": price},
                    {"time": "2026-09-02", "value": price * 1.02},
                ],
                "candles": [
                    {"time": "2026-08-25", "open": price - 20, "high": price + 10, "low": price - 30, "close": price - 10},
                    {"time": "2026-08-26", "open": price - 10, "high": price + 20, "low": price - 15, "close": price},
                ],
            })

        with stock_app.app.test_request_context("/us"):
            html = render_template(
                "us_dashboard.html", summary=summary, market="US",
                index_predictions=forecasts, data_freshness={},
            )

        for marker in (
            "S&amp;P 500", "Nasdaq Composite", "道瓊工業指數",
            "五日上漲機率", "61.0%", "2026-09-02",
            'id="us-index-chart"', 'id="us-index-chart-data"',
        ):
            self.assertIn(marker, html)

    def test_us_index_switch_does_not_depend_on_chart_library(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )
        start = script.index("function initUsIndexChart")
        end = script.index("\n}", start)
        body = script[start:end]

        # The three verified index forecasts are server-rendered with `hidden`
        # on all but the first panel, so only this initializer can reveal them.
        # Gating it on the third-party chart library would make the Nasdaq and
        # Dow forecasts unreachable whenever that CDN script fails to load.
        guard = body[: body.index("const select")]
        self.assertNotIn("window.LightweightCharts", guard)

        select_body = body[body.index("const select") :]
        pressed = select_body.index('setAttribute("aria-pressed"')
        panel = select_body.index("dataset.usIndexPanel")
        chart = select_body.index("window.LightweightCharts")
        self.assertLess(pressed, chart)
        self.assertLess(panel, chart)

    def test_us_stocks_disables_tw_dashboard_hydration(self):
        client = stock_app.app.test_client()
        with patch.object(
            stock_app, "_published_us_securities_observation",
            return_value={"stock_events": [], "etf_observations": []},
        ):
            response = client.get("/us/stocks")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-market="US"', html)
        self.assertNotIn('data-dashboard-endpoint', html)
        self.assertNotIn("TAIEX", html)
        self.assertIn("目前沒有異常上漲事件。", html)
        self.assertIn("目前沒有異常下跌事件。", html)
        for label in ("量能異常", "法人動向", "技術面", "官方", "資料警示"):
            self.assertIn(f"目前沒有{label}事件。", html)
        self.assertIn('action="/search"', html)
        self.assertIn('name="market" value="US"', html)

        with patch.object(
            stock_app, "_published_us_securities_observation",
            side_effect=ValueError("unavailable"),
        ):
            unavailable = client.get("/us/stocks")
        self.assertEqual(unavailable.status_code, 503)
        self.assertIn("暫時無法取得", unavailable.get_data(as_text=True))

        with patch.object(
            stock_app, "_published_us_securities_observation", return_value={}
        ):
            malformed = client.get("/us/stocks")
        self.assertEqual(malformed.status_code, 503)
        self.assertNotIn(
            "目前沒有通過條件的異常事件。",
            malformed.get_data(as_text=True),
        )

        with patch.object(
            stock_app, "search_stock_code", return_value=("AAPL", "Apple Inc.")
        ):
            search = client.get("/search?q=AAPL&market=US")

        self.assertEqual(search.status_code, 302)
        self.assertTrue(search.headers["Location"].endswith("/stock/AAPL"))

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_market_page_accepts_production_data_quality_fields(self, load):
        snapshot = observation_dashboard()
        snapshot["data_quality"] = {
            "coverage": 0.998,
            "available_count": 2072,
            "failure_count": 4,
            "universe_count": 2076,
        }
        load.return_value = snapshot

        response = stock_app.app.test_client().get("/market")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("有效標的</dt><dd>2072</dd>", html)
        self.assertNotIn("有效標的</dt><dd>資料不足</dd>", html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_web_security_headers_and_pinned_chart_supply_chain(
        self, fetch
    ):
        response = stock_app.app.test_client().get("/dashboard")
        csp = response.headers["Content-Security-Policy"]
        fetch.return_value = quant_snapshot()
        stock_html = stock_app.app.test_client().get(
            "/stock/2330"
        ).get_data(as_text=True)

        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("form-action 'self'", csp)
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("lightweight-charts@4.2.2", stock_html)
        self.assertIn('integrity="sha384-', stock_html)
        self.assertNotIn("style=", stock_html)

    def test_dashboard_page_is_the_observation_dashboard(self):
        with patch.object(stock_app, "analyze") as analyze, patch.object(
            stock_app,
            "_published_dashboard_snapshot",
            return_value=observation_dashboard(),
        ):
            response = stock_app.app.test_client().get("/dashboard")

        self.assertEqual(response.status_code, 200)
        analyze.assert_not_called()
        html = response.get_data(as_text=True)
        for label in (
            "台股市場研究摘要",
            "市場指揮台",
            "市場廣度",
            "期間報酬",
            "波動與風險",
            "產業相對強度",
            "資料覆蓋",
            "資料基準日 2026-07-15",
            "今日焦點",
            "產業觀察",
            "市場實況",
            "個股與 ETF",
            "ASK ABSORB",
            "AI 五日情境",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "精選標的",
            "產業預測",
            "data-top-picks",
        ):
            self.assertNotIn(forbidden, html)

    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=observation_dashboard()
    )
    def test_dashboard_research_command_uses_verified_data_without_inline_styles(
        self, _load
    ):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)

        for marker in (
            'data-research-summary',
            'data-market-command',
            'data-breadth-visual',
            'data-sector-flow',
            '<progress',
            'value="61.2"',
            '1200',
            '700',
            '17.5%',
            '半導體',
            '+1.85%',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)
        self.assertNotIn("23,742.18", html)
        self.assertNotIn("style=", html)

    def test_quick_ask_reuses_conversation_contract_and_has_dialog_controls(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        for marker in (
            'data-quick-ask-open',
            'data-quick-ask-dialog',
            'role="dialog"',
            'aria-modal="true"',
            'data-conversation-endpoint="/api/conversation"',
            'data-market-context="TW"',
            'data-page-context="home"',
            'maxlength="1200"',
            'aria-live="polite"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)
        for marker in (
            "dataQuickAskOpen",
            'event.key === "Escape"',
            'event.key === "Tab"',
            'event.key.toLowerCase() === "k"',
            "querySelectorAll(\"[data-conversation-form]\")",
            "market: panel.dataset.marketContext",
            "page: panel.dataset.pageContext",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, script)
        self.assertNotIn(".innerHTML", script)

    def test_public_pages_only_request_private_account_state_when_session_cookie_exists(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('data-account-session="anonymous"', html)
        self.assertIn('document.body.dataset.accountSession !== "present"', script)

    def test_report_filters_and_retry_control_have_progressive_enhancement(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )
        unavailable = Path("templates/report_unavailable.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("[data-report-filter]", script)
        self.assertIn("[data-report-type]", script)
        self.assertIn("data-report-retry", unavailable)
        self.assertIn("window.location.reload()", script)

    def test_dashboard_has_route_based_section_navigation(self):
        html = stock_app.app.test_client().get(
            "/dashboard"
        ).get_data(as_text=True)

        for marker in (
            'href="/market"',
            'href="/industries"',
            'href="/stocks"',
            'href="/ask"',
            'href="/learn"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)

    @patch.object(stock_app, "_published_prediction_snapshot")
    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_chart_renders_only_verified_prediction_product(
        self, fetch, load_prediction
    ):
        fetch.return_value = quant_snapshot()
        load_prediction.return_value = prediction_product()

        response = stock_app.app.test_client().get("/stock/2330")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for marker in (
            "五日上漲機率", "68.0%", "171.00", "+4.27%", "2026-07-22",
            '"time": "2026-07-22"', '"value": 171.0028',
        ):
            self.assertIn(marker, html)
        self.assertNotIn("買進", html)
        self.assertNotIn("賣出", html)

    def test_dashboard_destinations_live_in_sidebar_navigation(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        primary_nav = html.split('<nav class="sidebar-nav"', 1)[1].split("</nav>", 1)[0]

        self.assertIn('href="/ask"', primary_nav)
        self.assertIn('<span class="nav-label">ASK ABSORB</span>', primary_nav)
        self.assertIn('href="/learn"', primary_nav)
        self.assertIn('<span class="nav-label">學習</span>', primary_nav)
        self.assertNotIn('class="dashboard-destinations"', html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_industries_merge_strength_and_attention_companies(self, load_snapshot):
        snapshot = observation_dashboard()
        snapshot["industry_observations"][0].update(
            {
                "relative_return_5d_pct": 2.85,
                "ranking_basis": "actual_momentum",
                "attention_companies": [
                    {
                        "symbol": "2330",
                        "name": "台積電",
                        "price": 1245.0,
                        "return_5d_pct": 8.2,
                        "above_ma20": True,
                        "volume_ratio": 1.8,
                        "as_of": "2026-07-15",
                    }
                ],
            }
        )
        load_snapshot.return_value = snapshot

        html = stock_app.app.test_client().get("/industries").get_data(as_text=True)

        self.assertNotIn("產業實際強弱", html)
        self.assertEqual(html.count('data-industry-disclosure'), 1)
        self.assertIn('class="industry-disclosure hot"', html)
        self.assertIn("實際動能排序", html)
        self.assertIn('href="/stock/2330"', html)
        self.assertIn("台積電 · 2330", html)
        self.assertIn("5 日 +8.20%", html)

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_industries_render_verified_ai_fields_when_published(self, load_snapshot):
        snapshot = observation_dashboard()
        snapshot["industry_observations"][0].update(
            {
                "ranking_basis": "verified_ai_forecast",
                "attention_companies": [
                    {
                        "symbol": "2330",
                        "name": "台積電",
                        "price": 1245.0,
                        "return_5d_pct": 8.2,
                        "above_ma20": True,
                        "volume_ratio": 1.8,
                        "as_of": "2026-07-15",
                        "probability_up_pct": 68.4,
                        "target_price": 1272.5,
                    }
                ],
            }
        )
        load_snapshot.return_value = snapshot

        html = stock_app.app.test_client().get("/industries").get_data(as_text=True)

        self.assertIn("AI 五日模型排序", html)
        self.assertIn("68.4%", html)
        self.assertIn("第 5 日 1,272.50", html)

    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=observation_dashboard()
    )
    def test_navigation_has_route_active_state_and_no_primary_hash_links(self, _load):
        client = stock_app.app.test_client()
        for path, label in (
            ("/dashboard", "今天市場"),
            ("/market", "市場實況"),
            ("/industries", "產業觀察"),
            ("/stocks", "個股與 ETF"),
            ("/reports", "每日報告"),
        ):
            with self.subTest(path=path):
                html = client.get(path).get_data(as_text=True)
                self.assertIn(
                    f'class="nav-link active" href="{path if path != "/dashboard" else "/"}" aria-label="{label}" aria-current="page"',
                    html,
                )
        home = client.get("/dashboard").get_data(as_text=True)
        primary_nav = home.split('<nav class="sidebar-nav"', 1)[1].split("</nav>", 1)[0]
        self.assertNotIn('href="#', primary_nav)
        self.assertNotIn('class="mobile-nav"', home)
        self.assertIn('aria-current="page"><span class="nav-short"', primary_nav)
        ask = client.get("/ask").get_data(as_text=True)
        self.assertIn('<h1>ASK ABSORB</h1>', ask)

    def test_legacy_hash_migrator_uses_only_fixed_canonical_routes(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        expected = {
            '"#market-pulse": "/market"',
            '"#market-heatmap": "/industries"',
            '"#industry-observations": "/industries"',
            '"#stock-search": "/stocks"',
            '"#stock-events": "/stocks"',
            '"#etf-observations": "/stocks?tab=etf"',
            '"#learn": "/learn"',
            '"#daily-focus": "/"',
        }
        for mapping in expected:
            self.assertIn(mapping, script)
        self.assertIn(
            "if(!['/','/dashboard'].includes",
            script.replace('"', "'").replace(" ", ""),
        )
        self.assertNotIn("window.location.hash.slice", script)

    @patch.object(stock_app, "analyze")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_dashboard_api_returns_verified_observation_without_analysis(
        self, load_snapshot, analyze
    ):
        load_snapshot.return_value = observation_dashboard()

        response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        analyze.assert_not_called()
        self.assertEqual(payload["product_mode"], "observation")
        self.assertEqual(payload["observation_as_of"], "2026-07-15")
        self.assertEqual(
            payload["market_observation"]["advancing_count"], 1200
        )
        self.assertEqual(
            payload["industry_observations"][0]["name"], "半導體"
        )
        self.assertEqual(payload["prediction_status"], "AI 預測研究中")
        self.assertNotIn("top_picks", payload)
        self.assertNotIn("opportunities", payload)

    @patch.object(stock_app, "analyze")
    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=None
    )
    def test_dashboard_api_fails_closed_without_snapshot(
        self, _load_snapshot, analyze
    ):
        response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 503)
        analyze.assert_not_called()
        self.assertEqual(
            response.get_json()["status"], "observation_unavailable"
        )

    def test_preview_report_is_not_public_without_preview_prefix(self):
        response = stock_app.app.test_client().get("/preview/report")

        self.assertEqual(response.status_code, 404)

    @patch.object(stock_app, "analyze")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_preview_dashboard_keeps_isolated_candidate_api(
        self, load_snapshot, analyze
    ):
        analyze.return_value = {
            "price": 23150.0,
            "prob": 58,
            "trend": "多頭",
            "as_of": "2026-07-15",
            "recommendation": {},
        }
        load_snapshot.return_value = {
            "baseline_status": "initial_backtest_bootstrap",
            "inference_as_of": "2026-07-15",
            "backtest_as_of": None,
            "model_version": "lgbm-5d-v1",
            "backtest_version": None,
            "feature_schema_version": 1,
            "recommendation_policy_version": "recommendation-v1",
            "presentation": {
                "model_output_label": "模型方向分數",
                "strong_action_allowed": False,
                "performance_endorsement_allowed": False,
            },
            "sector_snapshot": {
                "sectors": {
                    "網通設備": [
                        {
                            "code": "4906",
                            "name": "正文",
                            "prob": 73.7,
                            "trend": "跌破 MA20",
                            "as_of": "2026-07-15",
                        }
                    ]
                }
            },
            "heatmap": [{"name": "網通設備", "tone": "steady"}],
            "daily_focus": ["candidate focus"],
            "top_picks": [{"code": "4906", "name": "正文"}],
        }
        with patch.object(
            stock_app, "PREVIEW_CANDIDATE_PREFIX", "previews/demo"
        ), patch.object(stock_app, "cached_opportunities", return_value=[]):
            response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["inference_as_of"], "2026-07-15")
        self.assertEqual(payload["sector_cards"][0]["leader"]["code"], "4906")
        self.assertEqual(payload["daily_focus"], ["candidate focus"])

    @patch.object(
        stock_app,
        "find_industry_peers",
        return_value={"category": "半導體", "codes": ["2454"]},
    )
    @patch.object(stock_app, "get_stock_name", return_value="聯發科")
    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_is_the_observation_workspace(
        self, fetch, _name, _peers
    ):
        fetch.return_value = quant_snapshot()

        response = stock_app.app.test_client().get("/stock/2330")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "個股觀察摘要",
            "價格與均線",
            "籌碼觀察",
            "技術指標",
            "風險事件",
            "欄位怎麼看",
            "產業同儕",
            "聯發科",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "投資金額試算",
            "支持這項建議",
            "回測",
            "勝率",
        ):
            self.assertNotIn(forbidden, html)
        self.assertIn("data-watchlist-toggle", html)
        self.assertIn("data-chart-range", html)
        self.assertIn('aria-label="個股觀察導覽"', html)
        self.assertIn('class="back-link" href="/"', html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_does_not_render_untrusted_snapshot_news(self, fetch):
        snapshot = quant_snapshot()
        snapshot["news"] = [
            {
                "title": "不安全來源",
                "link": "javascript:alert(1)",
            }
        ]
        fetch.return_value = snapshot

        html = stock_app.app.test_client().get(
            "/stock/2330"
        ).get_data(as_text=True)

        self.assertNotIn("不安全來源", html)
        self.assertNotIn('href="javascript:', html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_accepts_standard_us_ticker(self, fetch):
        fetch.return_value = quant_snapshot("AAPL", market="US")

        response = stock_app.app.test_client().get("/stock/AAPL")

        self.assertEqual(response.status_code, 200)
        fetch.assert_called_once_with("AAPL")
        self.assertIn(
            'class="back-link" href="/us"', response.get_data(as_text=True)
        )

    def test_dashboard_script_does_not_insert_api_text_with_inner_html(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(".innerHTML", script)
        self.assertIn("AbortController", script)
        self.assertNotIn('title: "五日預測"', script)

    def test_web_is_observation_only_and_old_watchlist_redirects(self):
        response = stock_app.app.test_client().get("/watchlist")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))

    @patch.object(stock_app, "analyze")
    def test_stock_summary_api_removed_with_browser_watchlist(self, analyze):
        response = stock_app.app.test_client().get(
            "/api/stock/2330/summary"
        )

        self.assertEqual(response.status_code, 404)
        analyze.assert_not_called()

    def test_line_navigation_maps_six_observation_entries(self):
        navigation = stock_app.build_line_navigation_flex(
            "https://example.com/"
        )

        self.assertEqual(navigation["type"], "carousel")
        self.assertEqual(len(navigation["contents"]), 6)
        actual_uri = {}
        actual_message = {}
        for card in navigation["contents"]:
            action = card["footer"]["contents"][0]["action"]
            title = card["body"]["contents"][0]["text"]
            if action["type"] == "uri":
                actual_uri[title] = action["uri"]
            else:
                actual_message[title] = action["text"]
        self.assertEqual(
            actual_uri,
            {
                "看大盤": "https://example.com/market",
                "看產業": "https://example.com/industries",
                "市場觀察": "https://example.com/dashboard",
            },
        )
        self.assertEqual(
            actual_message,
            {
                "查自選": "我的關注",
                "設提醒": "提醒管理",
                "查股票": "2330",
            },
        )

    def test_rich_menu_source_matches_observation_navigation(self):
        svg = Path("assets/rich-menu.svg").read_text(encoding="utf-8")

        for label in (
            "看大盤",
            "看產業",
            "查自選",
            "設提醒",
            "查股票",
            "市場觀察",
        ):
            self.assertIn(label, svg)
        for removed in (
            "找機會",
            "算報酬",
            "深度分析",
            "熱門題材與排行",
            "投入金額快速試算",
            "圖表、回測、新聞",
        ):
            self.assertNotIn(removed, svg)
        for marker in ("ABSORB", "#122643", "#ffffff", "#eaf0f7"):
            self.assertIn(marker, svg)

    def test_line_summary_card_has_one_clear_cta(self):
        card = stock_app.build_line_summary_card(
            "市場觀察",
            ["2330 台積電", "最新收盤 1000.00"],
            "查看完整觀察",
            "https://example.com/stock/2330",
        )

        self.assertEqual(len(card["footer"]["contents"]), 1)
        self.assertEqual(
            card["footer"]["contents"][0]["action"]["uri"],
            "https://example.com/stock/2330",
        )

    def test_web_shell_supports_keyboard_and_mobile_interactions(self):
        response = stock_app.app.test_client().get("/dashboard")
        html = response.get_data(as_text=True)
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )

        for marker in (
            'class="skip-link"',
            'id="main-content"',
            'aria-live="polite"',
        ):
            self.assertIn(marker, html)
        for rule in (
            ":focus-visible",
            "prefers-reduced-motion",
            "min-height:44px",
        ):
            self.assertIn(rule, css)
        self.assertIn(".dashboard-sidebar", css)
        self.assertIn('aria-controls="dashboard-sidebar"', html)
        self.assertIn('<span class="nav-label">每日報告</span>', html)

    def test_web_shell_serves_one_wordmark_font_across_devices(self):
        client = stock_app.app.test_client()
        response = client.get("/static/fonts/absorb-wordmark.woff2")
        font_payload = response.get_data()
        response.close()
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(font_payload), 1_000)
        self.assertIn('font-family:"ABSORB Wordmark"', css)
        self.assertIn('url("fonts/absorb-wordmark.woff2") format("woff2")', css)
        self.assertIn(
            'font-family:"ABSORB Wordmark","Segoe Script","Brush Script MT",cursive',
            css,
        )

    def test_web_shell_uses_softened_neutral_paper_surfaces(self):
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )
        manifest = json.loads(
            Path(stock_app.app.static_folder, "manifest.webmanifest").read_text(
                encoding="utf-8"
            )
        )

        self.assertIn("--absorb-surface:#f5f5f2", css)
        self.assertIn("--command-paper:#f7f6f2", css)
        self.assertIn("background:var(--command-paper)", css)
        self.assertNotIn("gradient", css)
        self.assertNotIn("backdrop-filter", css)
        self.assertEqual(manifest["background_color"], "#f7f6f2")

    def test_research_layout_supports_4k_and_tall_ask_workspace(self):
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        ).replace(" ", "").replace("\n", "")

        self.assertIn("--command-content-max:3200px", css)
        self.assertIn("@media(min-width:1800px)", css)
        self.assertIn("body{font-size:18px}", css)
        self.assertIn(".nav-link{font-size:17px}", css)
        self.assertIn("height:60vh", css)
        self.assertIn(".quick-ask-log{flex:1", css)
        self.assertIn(".industry-disclosure-list{", css)
        self.assertIn(".industry-disclosure.hot{", css)
        self.assertIn(".industry-disclosure.cold{", css)

    def test_browser_bundle_has_no_local_watchlist_storage(self):
        source = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        for removed in (
            "localStorage",
            "quant-watchlist",
            "data-alert-open",
            "data-alert-form",
        ):
            self.assertNotIn(removed, source)
        self.assertIn("if (!entries.length) return", source)

    def test_health_check_is_separate_from_dashboard(self):
        client = stock_app.app.test_client()

        for path in ("/health", "/healthz"):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_data(as_text=True), "ok")

    def test_stock_chart_is_clipped_and_resizes_with_its_panel(self):
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )
        js = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn(".chart-shell{overflow:hidden", css)
        self.assertIn(".stock-chart{", css)
        self.assertIn("min-height:320px", css)
        self.assertIn("function measureChartHeight", js)
        self.assertIn("Math.min(460", js)
        self.assertIn("ResizeObserver", js)


if __name__ == "__main__":
    unittest.main()
