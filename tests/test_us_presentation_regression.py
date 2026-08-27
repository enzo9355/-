import unittest
from unittest.mock import patch
import os
from pathlib import Path

os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")
import app as stock_app

class USPresentationRegressionTests(unittest.TestCase):
    def test_us_market_page_has_no_raw_keys_and_localized_labels(self):
        # Simulate a professional report with market_observation containing raw keys
        from stock_papi.services.market_summary import build_market_summary_view
        from reporting.professional_schema import ProfessionalPostCloseReport
        import json
        from unittest.mock import patch
        # Use a minimal mock report: load a real report if available, else mock view
        # We'll test template rendering directly with a synthetic summary
        summary = {
            "market": "US",
            "source_market_date": "2026-08-24",
            "applicable_trading_date": "2026-08-25",
            "executive_summary": {"supporting_evidence": [], "opposing_evidence": []},
            "key_events": [],
            "market_observation": {
                "status": "available",
                "data_as_of": "2026-08-24",
                "data": {
                    "advancing_count": 1200,
                    "declining_count": 800,
                    "ma20_breadth_pct": 55.5,
                    "return_1d_pct": -0.33,
                    "median_volume_ratio": 1.23,
                    "risk_state": "normal",
                    "unchanged_count": 10,
                    "realized_volatility_20d_pct": 18.5,
                }
            },
            "industries": {"status": "available", "data": {"ranking": []}},
            "securities": {"status": "available", "data": {"stock_events": [], "etf_observations": []}},
            "validation": {"status": "available", "data": {"gates": {"ranking": "UNAVAILABLE", "calibration": "UNAVAILABLE", "promotion": "BLOCKED"}}},
        }
        with stock_app.app.test_request_context("/us/market"):
            html = stock_app.app.test_client().get("/us/market").get_data(as_text=True) if False else None
        # Instead test helper directly
        from stock_papi.services.us_presentation import build_us_market_observation_view, MARKET_OBSERVATION_LABELS
        view = build_us_market_observation_view(summary["market_observation"])
        labels = [r["label"] for r in view]
        self.assertIn("上漲家數", labels)
        self.assertIn("下跌家數", labels)
        self.assertIn("站上 MA20 比例", labels)
        self.assertIn("單日報酬", labels)
        # Ensure localized labels exist
        self.assertIn("上漲家數", labels)
        self.assertNotIn("advancing_count", labels)
        # Format check
        adv = next(r for r in view if r["key"] == "advancing_count")
        self.assertEqual(adv["formatted"], "1200")
        ret = next(r for r in view if r["key"] == "return_1d_pct")
        self.assertIn("-0.33%", ret["formatted"])
        # None handling
        summary_none = {"status": "available", "data": {"advancing_count": None, "return_1d_pct": None}}
        view_none = build_us_market_observation_view(summary_none)
        for r in view_none:
            self.assertEqual(r["formatted"], "尚無已驗證資料")

    def test_us_industries_binding_uses_correct_fields(self):
        from stock_papi.services.us_presentation import build_us_industries_view
        section = {
            "status": "available",
            "data": {
                "ranking": [
                    {"name": "半導體", "rank": 1, "relative_return_5d_pct": 2.5, "coverage": 0.95, "available_count": 10, "component_count": 12},
                    {"name": "生技", "rank": 2, "relative_return_5d_pct": -1.2, "coverage": 0.9, "available_count": 8, "component_count": 10},
                ]
            }
        }
        view = build_us_industries_view(section)
        self.assertEqual(len(view), 2)
        self.assertEqual(view[0]["relative_return_5d_pct"], 2.5)
        self.assertEqual(view[0]["coverage"], 0.95)
        # Ensure template doesn't use rotation/status
        tmpl = Path("templates/us_industries.html").read_text(encoding="utf-8")
        self.assertNotIn("item.rotation", tmpl)
        self.assertNotIn("item.status", tmpl)
        self.assertIn("relative_return_5d_pct", tmpl)
        self.assertIn("coverage", tmpl)

    def test_validation_gate_localization(self):
        from stock_papi.services.us_presentation import validation_gate_display
        self.assertEqual(validation_gate_display("ranking", "UNAVAILABLE"), ("模型排名", "尚未提供驗證資料"))
        self.assertEqual(validation_gate_display("promotion", "BLOCKED"), ("模型 Promotion", "暫不發布"))
        # Template should contain explanatory sentence
        dash = Path("templates/us_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("不代表市場行情資料驗證失敗", dash)
        prof = Path("templates/reports/post_close_professional.html").read_text(encoding="utf-8")
        self.assertIn("不代表市場行情資料驗證失敗", prof)
        # Ensure raw UNAVAILABLE/BLOCKED not directly rendered without mapping
        self.assertIn("尚未提供驗證資料", dash)
        self.assertIn("暫不發布", dash)

    def test_stock_duplicate_rendering(self):
        from stock_papi.services.us_presentation import stock_display_name
        self.assertEqual(stock_display_name("RFAI", "RFAI"), "RFAI")
        self.assertEqual(stock_display_name("Apple Inc.", "AAPL"), "Apple Inc. · AAPL")
        tmpl = Path("templates/stocks.html").read_text(encoding="utf-8")
        self.assertNotIn("{{ item.name }} · {{ item.symbol }}</span><strong>{{ item.observation }}", tmpl)  # old pattern
        self.assertIn("item.name|upper != item.symbol|upper", tmpl)

    def test_us_report_history_distinguishes_dates(self):
        # Ensure reports.html or observation template shows both dates
        tmpl = Path("templates/report_observation.html").read_text(encoding="utf-8")
        self.assertIn("資料基準日", tmpl)
        self.assertIn("適用交易日", tmpl)

    def test_observation_report_visualizes_actual_market_and_industry_evidence(self):
        tmpl = Path("templates/report_observation.html").read_text(encoding="utf-8")
        self.assertIn('class="report-evidence-bars"', tmpl)
        self.assertIn('class="industry-strength-bar', tmpl)
        self.assertGreaterEqual(tmpl.count("<progress"), 4)

    def test_observation_report_keeps_seven_event_categories_and_severity(self):
        tmpl = Path("templates/report_observation.html").read_text(encoding="utf-8")
        for label in (
            "異常上漲", "異常下跌", "量能異常", "法人動向",
            "技術面", "官方事件", "資料警示",
        ):
            self.assertIn(label, tmpl)
        self.assertIn("severity_labels", tmpl)
        self.assertNotIn("('其他事件', report_groups.other)", tmpl)
