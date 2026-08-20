"""Tests for US market data fetcher and indicator computation."""

import datetime
import pandas as pd
import unittest

from stock_papi.integrations.market_data.us_market_data import (
    compute_us_technical_indicators,
    fetch_us_stock_history,
)


class USMarketDataTests(unittest.TestCase):
    def test_compute_technical_indicators(self):
        dates = pd.date_range("2026-07-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(30)],
                "High": [105.0 + i for i in range(30)],
                "Low": [95.0 + i for i in range(30)],
                "Close": [102.0 + i for i in range(30)],
                "Volume": [1000000 for _ in range(30)],
            },
            index=dates,
        )
        res = compute_us_technical_indicators(df)
        self.assertIn("MA5", res.columns)
        self.assertIn("MA20", res.columns)
        self.assertIn("MA60", res.columns)
        self.assertIn("RSI", res.columns)
        self.assertIn("MACD", res.columns)
        self.assertIn("K", res.columns)
        self.assertIn("D", res.columns)
        self.assertIn("VOL_RATIO", res.columns)
        self.assertAlmostEqual(float(res["MA5"].iloc[-1]), 129.0)  # Average of 127, 128, 129, 130, 131

    def test_fetch_us_stock_history_mock(self):
        dates = pd.date_range("2026-08-01", periods=15, freq="B")
        mock = pd.DataFrame(
            {
                "Open": [200.0 + i for i in range(15)],
                "High": [205.0 + i for i in range(15)],
                "Low": [195.0 + i for i in range(15)],
                "Close": [202.0 + i for i in range(15)],
                "Volume": [5000000 for _ in range(15)],
            },
            index=dates,
        )
        res = fetch_us_stock_history(
            "AAPL",
            target_market_date=datetime.date(2026, 8, 15),
            mock_df=mock,
        )
        self.assertLessEqual(res.index[-1], datetime.date(2026, 8, 15))
        self.assertIn("RSI", res.columns)
