import inspect
import tempfile
import unittest
from pathlib import Path

from tests.report_fixtures import stock_document, warmup_stock_document, write_quant_publish


class IndustryReportBacktestTests(unittest.TestCase):
    def test_backtest_filters_null_signals_but_keeps_price_calendar(self):
        from reporting.config import ReportConfig
        from reporting.industry_backtest import backtest_industry
        from reporting.source_loader import StockSnapshot

        document = warmup_stock_document("2330", warmup_rows=22)
        stock = StockSnapshot.from_document(document, sha256="a" * 64, size=1)
        result = backtest_industry("半導體", [stock], ReportConfig(min_backtest_periods=2))
        self.assertEqual(len(stock.daily), 70)
        self.assertEqual(result.rebalance_dates[0].isoformat(), document["daily"][25]["Date"][:10])
        self.assertGreaterEqual(len(result.rebalance_dates), 8)

    def test_five_day_non_overlapping_oos_backtest_cost_and_cash_periods(self):
        from reporting.config import ReportConfig
        from reporting.industry_backtest import backtest_industry
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = stock_document("2330", rows=31, ai_probability=70)
            second = stock_document("2317", rows=31, ai_probability=50)
            # 最新機率是未實現預測，必須排除；中間一個再平衡期設為空手。
            for document in (first, second):
                document["daily"][15]["AI_P"] = None
            write_quant_publish(root, [first, second])
            source = load_report_source(root)

            result = backtest_industry(
                "半導體", source.stocks, ReportConfig(min_backtest_periods=2)
            )

            self.assertEqual(result.rebalance_dates, result.rebalance_dates[::1])
            gaps = [
                (right - left).days
                for left, right in zip(result.rebalance_dates, result.rebalance_dates[1:])
            ]
            self.assertTrue(all(gap >= 5 for gap in gaps))
            self.assertGreater(result.cash_period_ratio, 0)
            self.assertTrue(all(date.isoformat() < "2026-07-03" for date in result.rebalance_dates))
            gross = 115 / 110 - 1
            self.assertAlmostEqual(result.period_returns[1], gross - 0.00585)
            self.assertEqual(result.annualization_periods, 252 / 5)
            self.assertEqual(result.rebalance_periods, len(result.rebalance_dates))
            self.assertEqual(
                result.entry_periods + result.cash_periods, result.rebalance_periods
            )
            self.assertEqual(
                result.winning_periods + result.losing_periods, result.entry_periods
            )
            self.assertAlmostEqual(
                result.win_rate, result.winning_periods / result.entry_periods
            )
            self.assertIsNotNone(result.expected_return)
            self.assertIsNotNone(result.average_profit)
            self.assertIn("current_cost", result.cost_sensitivity)
            self.assertEqual(
                result.cost_sensitivity["current_cost"], result.cumulative_return
            )
            self.assertTrue(result.yearly_returns)

    def test_insufficient_samples_return_none_metrics_not_zero(self):
        from reporting.config import ReportConfig
        from reporting.industry_backtest import backtest_industry
        from reporting.source_loader import StockSnapshot

        document = stock_document("2330", rows=8)
        stock = StockSnapshot.from_document(document, sha256="a" * 64, size=1)
        result = backtest_industry(
            "半導體", [stock], ReportConfig(min_backtest_periods=5)
        )

        self.assertFalse(result.sufficient)
        self.assertIsNone(result.sharpe)
        self.assertIsNone(result.cumulative_return)
        self.assertEqual(result.sample_quality, "資料不足")

    def test_all_cash_reports_period_counts_but_not_strategy_metrics(self):
        from reporting.config import ReportConfig
        from reporting.industry_backtest import backtest_industry
        from reporting.source_loader import StockSnapshot

        document = stock_document("2330", rows=70, ai_probability=40)
        stock = StockSnapshot.from_document(document, sha256="a" * 64, size=1)
        result = backtest_industry(
            "全程空手", [stock], ReportConfig(min_backtest_periods=2)
        )

        self.assertTrue(result.all_cash)
        self.assertGreaterEqual(result.rebalance_periods, 12)
        self.assertEqual(result.entry_periods, 0)
        self.assertEqual(result.cash_periods, result.rebalance_periods)
        self.assertIsNone(result.cumulative_return)
        self.assertIsNone(result.sharpe)
        self.assertIsNone(result.win_rate)
        self.assertIsNone(result.expected_return)
        self.assertIsNone(result.profit_factor)
        self.assertEqual(result.longest_winning_streak, 0)
        self.assertEqual(result.cost_sensitivity["current_cost"], 0.0)
        self.assertEqual(result.strategy_status, "全程空手")

    def test_sample_quality_bands_mark_low_sample_metrics(self):
        from reporting.config import ReportConfig
        from reporting.industry_backtest import backtest_industry
        from reporting.source_loader import StockSnapshot

        low = StockSnapshot.from_document(
            stock_document("2330", rows=70, ai_probability=70),
            sha256="a" * 64,
            size=1,
        )
        result = backtest_industry(
            "低樣本", [low], ReportConfig(min_backtest_periods=12)
        )

        self.assertEqual(result.sample_quality, "低樣本")
        self.assertTrue(result.low_sample_warning)

    def test_backtest_chart_uses_actual_rebalance_dates(self):
        from reporting.charts import backtest_chart

        source = inspect.getsource(backtest_chart)
        self.assertIn("result.rebalance_dates", source)
        self.assertNotIn("range(len(result.drawdown_curve))", source)


if __name__ == "__main__":
    unittest.main()
