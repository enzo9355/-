import tempfile
import unittest
from pathlib import Path

from tests.report_fixtures import stock_document, warmup_stock_document, write_quant_publish


class IndustryReportAnalyticsTests(unittest.TestCase):
    def test_report_filters_null_historical_features_without_dropping_prices(self):
        from reporting.industry_analytics import build_daily_report
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = warmup_stock_document("2330")
            for row in document["daily"][:20]:
                row["MARKET_RET_1"] = None
            write_quant_publish(root, [document])
            report = build_daily_report(load_report_source(root), {"半導體": ["2330"]})
        self.assertEqual(report.market.returns[60], None)
        industry = report.industries[0]
        self.assertEqual(industry.symbols, ["2330"])
        self.assertIsNone(document["daily"][-61]["MA20"])
        self.assertAlmostEqual(
            industry.returns[60],
            document["daily"][-1]["Close"] / document["daily"][-61]["Close"] - 1,
        )
        self.assertAlmostEqual(industry.returns[60], 0.5504587155963303)
        self.assertEqual(report.model_quality.pooled_oos_samples, 45)

    def _source(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        first = stock_document("2330", start_price=100)
        second = stock_document("2317", start_price=200)
        write_quant_publish(root, [first, second])
        from reporting.source_loader import load_report_source
        return temporary, load_report_source(root)

    def test_equal_weight_returns_coverage_overlap_and_exclusions(self):
        from reporting.industry_analytics import build_daily_report

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        report = build_daily_report(
            source,
            {
                "全市場": ["2330", "2317"],
                "ETF專區": ["0050"],
                "半導體": ["2330", "2317", "9999"],
                "AI 題材": ["2330"],
            },
        )

        by_name = {item.name: item for item in report.industries}
        self.assertEqual(set(by_name), {"半導體", "AI 題材"})
        expected_5d = ((169 / 164 - 1) + (269 / 264 - 1)) / 2
        self.assertAlmostEqual(by_name["半導體"].returns[5], expected_5d)
        self.assertEqual(by_name["半導體"].valid_samples[5], 2)
        self.assertEqual(by_name["半導體"].component_count, 3)
        self.assertAlmostEqual(by_name["半導體"].coverage, 2 / 3)
        self.assertEqual(by_name["半導體"].bullish_breadth, 1.0)
        self.assertIn("2330", by_name["AI 題材"].symbols)

    def test_missing_return_is_not_filled_with_zero_and_rotation_is_centralized(self):
        from reporting.industry_analytics import build_daily_report, classify_rotation

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        source.stocks[0].daily = source.stocks[0].daily[-4:]
        report = build_daily_report(source, {"短資料": [source.stocks[0].symbol]})
        item = report.industries[0]

        self.assertIsNone(item.returns[5])
        self.assertEqual(item.valid_samples[5], 0)
        self.assertEqual(classify_rotation(0.1, 0.1), "leading")
        self.assertEqual(classify_rotation(0.1, -0.1), "improving")
        self.assertEqual(classify_rotation(-0.1, 0.1), "weakening")
        self.assertEqual(classify_rotation(0.0, 0.0), "lagging")

    def test_inconsistent_market_factors_emit_data_quality_warning(self):
        from reporting.industry_analytics import build_daily_report

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        source.stocks[0].daily[-1]["MARKET_RET_5"] = 0.50
        report = build_daily_report(source, {"半導體": ["2330", "2317"]})

        self.assertTrue(any("市場因子" in warning for warning in report.warnings))
        self.assertAlmostEqual(report.market.returns[5], (0.025 + 0.50) / 2)

    def test_absolute_probability_bands_are_mutually_exclusive(self):
        from reporting.industry_analytics import build_daily_report
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_quant_publish(
                root,
                [
                    stock_document("2330", ai_probability=70),
                    stock_document("2317", ai_probability=55),
                    stock_document("2303", ai_probability=40),
                ],
            )
            report = build_daily_report(
                load_report_source(root),
                {"模型偏多": ["2330"], "中性": ["2317"], "模型偏弱": ["2303"]},
            )

        self.assertEqual([item.name for item in report.bullish_industries], ["模型偏多"])
        self.assertEqual([item.name for item in report.weak_industries], ["模型偏弱"])
        self.assertTrue(
            set(item.name for item in report.bullish_industries).isdisjoint(
                item.name for item in report.weak_industries
            )
        )
        self.assertNotIn("中性", [item.name for item in report.weak_industries])

    def test_no_weak_industry_is_not_forced_and_no_prior_is_not_zero_filled(self):
        from reporting.industry_analytics import build_daily_report

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        report = build_daily_report(source, {"半導體": ["2330", "2317"]})

        self.assertEqual(report.weak_industries, [])
        self.assertFalse(report.comparison_available)
        self.assertTrue(any("無前期報告可比較" in item for item in report.summary))
        self.assertIsNone(report.market.changes["bullish_breadth"])

    def test_previous_source_calculates_rank_probability_breadth_and_high_score_changes(self):
        from reporting.industry_analytics import build_daily_report
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as previous_dir:
            current_root = Path(current_dir)
            previous_root = Path(previous_dir)
            write_quant_publish(
                current_root,
                [
                    stock_document("2330", ai_probability=70),
                    stock_document("2317", ai_probability=50),
                ],
            )
            write_quant_publish(
                previous_root,
                [
                    stock_document("2330", ai_probability=55),
                    stock_document("2317", ai_probability=65),
                ],
            )
            industry_map = {"A 產業": ["2330"], "B 產業": ["2317"]}

            report = build_daily_report(
                load_report_source(current_root),
                industry_map,
                previous_source=load_report_source(previous_root),
            )

        by_name = {item.name: item for item in report.industries}
        self.assertTrue(report.comparison_available)
        self.assertEqual(by_name["A 產業"].rank, 1)
        self.assertEqual(by_name["A 產業"].previous_rank, 2)
        self.assertEqual(by_name["A 產業"].rank_change, 1)
        self.assertEqual(by_name["A 產業"].probability_change, 15)
        self.assertEqual(report.new_high_score_symbols, ["2330"])
        self.assertEqual(report.exited_high_score_symbols, ["2317"])
        self.assertEqual(report.market.changes["bullish_breadth"], 0.0)

    def test_rotation_neutral_band_and_market_quality_metrics(self):
        from reporting.industry_analytics import (
            build_daily_report,
            is_near_rotation_boundary,
        )

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        report = build_daily_report(source, {"半導體": ["2330", "2317"]})

        self.assertTrue(is_near_rotation_boundary(0.001, 0.01))
        self.assertTrue(is_near_rotation_boundary(-0.001, -0.01))
        self.assertFalse(is_near_rotation_boundary(0.003, 0.003))
        self.assertEqual(report.market.ma60_breadth, 1.0)
        self.assertEqual(report.market.advancing_count, 2)
        self.assertEqual(report.market.declining_count, 0)
        self.assertEqual(report.market.new_high_20d_count, 2)
        self.assertEqual(report.market.new_low_20d_count, 0)
        self.assertAlmostEqual(report.market.average_volume_ratio, 1.2)

    def test_model_quality_uses_pooled_oos_probabilities(self):
        from reporting.industry_analytics import build_daily_report

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        report = build_daily_report(source, {"半導體": ["2330", "2317"]})

        quality = report.model_quality
        self.assertEqual(quality.pooled_oos_samples, 120)
        self.assertEqual(quality.direction_accuracy, 1.0)
        self.assertAlmostEqual(quality.brier_score, 0.09)
        self.assertEqual(quality.high_score_win_rate, 1.0)
        self.assertEqual(sum(item["samples"] for item in quality.calibration_bins), 120)

    def test_watchlist_uses_consistent_ma20_and_default_risk_wording(self):
        from reporting.industry_analytics import build_daily_report

        temporary, source = self._source()
        self.addCleanup(temporary.cleanup)
        stock = next(item for item in source.stocks if item.symbol == "2330")
        stock.name = "台積電"
        stock.daily[-1]["MA20"] = stock.daily[-1]["Close"]
        report = build_daily_report(source, {"半導體": ["2330", "2317"]})
        item = next(entry for entry in report.watchlist if entry["symbol"] == "2330")

        self.assertEqual(item["name"], "台積電")
        self.assertEqual(item["trend"], "接近 MA20")
        self.assertEqual(item["foreign_net_5"], 5000)
        self.assertEqual(item["risks"], ["未觸發額外風險警示"])


if __name__ == "__main__":
    unittest.main()
