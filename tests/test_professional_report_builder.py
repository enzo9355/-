import unittest
import copy

from reporting.professional_builder import build_professional_post_close_artifact
from reporting.professional_schema import ProfessionalPostCloseReport, compute_content_sha256


class ProfessionalReportBuilderTests(unittest.TestCase):
    def _metadata(self):
        return {
            "schema_version": 2,
            "report_type": "post_close",
            "product_mode": "observation",
            "market": "TW",
            "source_market_date": "2026-07-17",
            "applicable_trading_date": "2026-07-20",
            "published_at": "2026-07-17T10:30:00Z",
            "data_as_of": "2026-07-17",
            "source_manifest": "quant/v1/manifests/TW-20260717T091000Z-123456789abc.json",
            "source_manifest_sha256": "a" * 64,
            "prediction_capability": {
                "mode": "research",
                "observation_enabled": True,
                "probability_allowed": False,
                "ranking_allowed": False,
                "strong_action_allowed": False,
                "performance_endorsement_allowed": False,
            },
            "content": {
                "market_observation": {
                    "return_1d_pct": -0.72,
                    "advancing_count": 520,
                    "declining_count": 812,
                    "ma20_breadth_pct": 39.7,
                    "realized_volatility_20d_pct": 18.2,
                },
                "industry_observations": [
                    {"name": "半導體", "available_count": 6, "component_count": 6, "relative_return_5d_pct": 4.31},
                    {"name": "光電", "available_count": 8, "component_count": 9, "relative_return_5d_pct": -3.20},
                ],
                "heatmap": [],
                "stock_events": [
                    {"symbol": "2330", "name": "台積電", "observation": "量能異常", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "volume_surge", "severity": "medium"},
                    {"symbol": "0000", "name": "Test1", "observation": "風險", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "rsi_overbought", "severity": "high"},
                    {"symbol": "1111", "name": "Test2", "observation": "未知", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "unknown_event", "severity": "low"}
                ],
                "etf_observations": [
                    {"symbol": "0050", "name": "台灣50", "price": 205.2, "return_1d_pct": -0.6, "return_5d_pct": 1.1}
                ],
                "daily_focus": ["市場廣度維持中性", "跌破 MA20 比例增加", "未知情緒測試文字"],
                "data_quality": {
                    "coverage": 0.982,
                    "symbol_count": 1332,
                    "failure_count": 24,
                },
            },
        }

    def test_builds_canonical_report_from_verified_observation_metadata(self):
        report = build_professional_post_close_artifact(
            self._metadata(), code_commit_sha="b" * 40
        )
        document = report.to_document()

        self.assertIsInstance(report, ProfessionalPostCloseReport)
        self.assertEqual(document["identity"]["product_tier"], "institutional")
        self.assertEqual(document["identity"]["product_mode"], "observation_with_research")
        self.assertEqual(document["executive_summary"]["strongest_industries"], ["半導體"])
        self.assertEqual(document["executive_summary"]["weakest_industries"], ["光電"])
        self.assertEqual(document["industries"]["data"]["ranking"][0]["name"], "半導體")
        self.assertEqual(document["validation"]["data"]["gates"]["promotion"], "BLOCKED")
        self.assertEqual(document["validation"]["data"]["gates"]["ranking"], "UNAVAILABLE")
        self.assertFalse(document["validation"]["data"]["probability_allowed"])
        self.assertEqual(document["quantitative_research"]["status"], "unavailable")
        self.assertEqual(document["ai_reference"]["status"], "unavailable")
        self.assertEqual(document["identity"]["content_sha256"], compute_content_sha256(document))

    def test_preserves_missing_market_values_as_none(self):
        metadata = self._metadata()
        metadata["content"]["market_observation"]["realized_volatility_20d_pct"] = None
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertIsNone(report.market.data["realized_volatility_20d_pct"])

    def test_rejects_pre_market_metadata(self):
        metadata = self._metadata()
        metadata["report_type"] = "pre_market"
        with self.assertRaisesRegex(ValueError, "post_close"):
            build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)

    def test_rejects_probability_enabled_metadata(self):
        metadata = self._metadata()
        metadata["prediction_capability"]["probability_allowed"] = True
        with self.assertRaisesRegex(ValueError, "probability"):
            build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
            
    def test_capital_flows_absent_is_unavailable(self):
        metadata = self._metadata()
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(report.capital_flows.status, "unavailable")

    def test_capital_flows_valid_is_available(self):
        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD_million",
            "foreign_net": 100,
        }
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(report.capital_flows.status, "available")
        self.assertEqual(report.capital_flows.data["foreign_net"], 100)

    def test_us_capital_flows_remain_unavailable_and_market_specific(self):
        metadata = self._metadata()
        metadata["market"] = "US"
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD_million",
            "foreign_net": 100,
        }
        report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        self.assertEqual(report.capital_flows.status, "unavailable")
        self.assertIn("美國市場機構資金流向尚未納入", report.capital_flows.reason)

    def test_capital_flows_invalid_type_is_unavailable(self):
        metadata = self._metadata()
        metadata["content"]["capital_flows"] = "some text"
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(report.capital_flows.status, "unavailable")

    def test_stock_event_classification(self):
        metadata = self._metadata()
        metadata["content"]["stock_events"].append(
            {"symbol": "9999", "name": "TestUnknownHigh", "observation": "未知異常", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "unknown_event", "severity": "high"}
        )
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        securities = report.securities.data
        
        positives = [e["symbol"] for e in securities["positive_observations"]]
        risks = [e["symbol"] for e in securities["risk_observations"]]
        high_anomalies = [e["symbol"] for e in securities["high_anomaly_observations"]]
        
        # volume_surge should be positive
        self.assertIn("2330", positives)
        self.assertNotIn("2330", risks)
        
        # rsi_overbought should be risk
        self.assertIn("0000", risks)
        self.assertNotIn("0000", positives)
        
        # unknown_event should be neither
        self.assertNotIn("1111", positives)
        self.assertNotIn("1111", risks)
        
        # high severity should be in high_anomaly if it's a valid event type
        self.assertIn("0000", high_anomalies)
        
        # unknown_event with high severity must NOT enter high_anomaly
        self.assertNotIn("9999", high_anomalies)
        self.assertNotIn("1111", high_anomalies)

    def test_trading_status_observations_stay_out_of_event_policy(self):
        metadata = self._metadata()
        metadata["content"]["trading_status_observations"] = [
            {
                "symbol": "2303",
                "name": "測試股票 2303",
                "status": "official_no_regular_trade",
                "label": "當日無正常交易",
                "observation_as_of": "2026-07-17",
                "latest_regular_price_date": "2026-07-16",
                "evidence_sha256": "c" * 64,
                "last_regular_close": 100.0,
            }
        ]

        report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        securities = report.securities.data

        self.assertEqual(
            securities["trading_status_observations"],
            metadata["content"]["trading_status_observations"],
        )
        self.assertNotIn(
            "2303", [item["symbol"] for item in securities["stock_events"]]
        )
        self.assertEqual(securities["uncategorized_event_count"], 1)
        
    def test_next_session_structured_logic(self):
        metadata = self._metadata()
        
        # test weak state
        metadata["content"]["market_observation"]["ma20_breadth_pct"] = 35.0
        metadata["content"]["market_observation"]["realized_volatility_20d_pct"] = 25.0
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        next_session = report.next_session.data
        self.assertTrue(any("偏弱" in s for s in next_session["negative"]))
        self.assertTrue(any("系統性風險" in s for s in next_session["negative"]))
        self.assertEqual(len(next_session["positive"]), 0)

        # test strong state
        metadata["content"]["market_observation"]["ma20_breadth_pct"] = 70.0
        metadata["content"]["market_observation"]["realized_volatility_20d_pct"] = 15.0
        report2 = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        next_session2 = report2.next_session.data
        self.assertTrue(any("強勢" in s for s in next_session2["positive"]))
        self.assertEqual(len(next_session2["negative"]), 0)



    def test_event_severity_and_unknown_event_type_policy(self):
        metadata = self._metadata()
        metadata["content"]["stock_events"] = [
            {"symbol": "2330", "name": "台積電", "observation": "量能異常", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "volume_surge", "severity": "medium"},
            {"symbol": "0000", "name": "Test1", "observation": "風險", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "rsi_overbought", "severity": "high"},
            {"symbol": "1111", "name": "Test2", "observation": "未知", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "unknown_event", "severity": "low"},
            {"symbol": "2222", "name": "Test3", "observation": "無效嚴重度", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "volume_surge", "severity": "super_high"},
            {"symbol": "3333", "name": "Test4", "observation": "無嚴重度", "as_of": "2026-07-17", "metric_value": 1.4, "unit": "倍", "event_type": "new_high_20d", "severity": None},
        ]
        report = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        securities = report.securities.data

        self.assertEqual(securities["policy_version"], "1.0")
        self.assertEqual(securities["uncategorized_event_count"], 1)
        self.assertEqual(securities["invalid_event_count"], 2)

        positives = [e["symbol"] for e in securities["positive_observations"]]
        risks = [e["symbol"] for e in securities["risk_observations"]]
        high_anomalies = [e["symbol"] for e in securities["high_anomaly_observations"]]

        self.assertIn("2330", positives)
        self.assertIn("0000", risks)
        self.assertIn("0000", high_anomalies)

        for s in ("1111", "2222", "3333"):
            self.assertNotIn(s, positives)
            self.assertNotIn(s, risks)
            self.assertNotIn(s, high_anomalies)

        raw_events = {e["symbol"]: e for e in securities["stock_events"]}
        self.assertEqual(raw_events["2222"]["severity"], "super_high")
        self.assertIsNone(raw_events["3333"]["severity"])

    def test_capital_flows_strict_validation(self):
        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {}
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "unavailable")

        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD_million",
            "foreign_net": 100,
            "extra_unauthorized_key": 50,
        }
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "unavailable")

        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-16",
            "unit": "TWD_million",
            "foreign_net": 100,
        }
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "unavailable")

        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD",
            "foreign_net": 100,
        }
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "unavailable")

        for bad_val in (True, False, float("nan"), float("inf"), float("-inf"), "100"):
            with self.subTest(bad_val=bad_val):
                metadata = self._metadata()
                metadata["content"]["capital_flows"] = {
                    "as_of": "2026-07-17",
                    "unit": "TWD_million",
                    "foreign_net": bad_val,
                }
                r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
                self.assertEqual(r.capital_flows.status, "unavailable")

        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD_million",
            "foreign_net": None,
            "investment_trust_net": None,
            "dealer_net": None,
        }
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "unavailable")

        metadata = self._metadata()
        metadata["content"]["capital_flows"] = {
            "as_of": "2026-07-17",
            "unit": "TWD_million",
            "foreign_net": 150.5,
            "investment_trust_net": None,
            "dealer_net": None,
        }
        r = build_professional_post_close_artifact(metadata, code_commit_sha="b" * 40)
        self.assertEqual(r.capital_flows.status, "available")
        self.assertEqual(r.capital_flows.data["foreign_net"], 150.5)
        self.assertIsNone(r.capital_flows.data["investment_trust_net"])
        self.assertIsNone(r.capital_flows.data["dealer_net"])

if __name__ == "__main__":
    unittest.main()
