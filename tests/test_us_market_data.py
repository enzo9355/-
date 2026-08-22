"""Tests for US market data fetcher and indicator computation."""

import datetime
import json
import pandas as pd
import unittest
from unittest.mock import patch

from stock_papi.integrations.market_data.us_market_data import (
    compute_us_technical_indicators,
    fetch_direct_yahoo_chart,
    fetch_us_stock_history,
    USIntegrityError,
    USSchemaError,
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

    def _placeholder_history(self, target_date, *, volume=None):
        dates = [
            target_date - datetime.timedelta(days=2),
            target_date - datetime.timedelta(days=1),
            target_date,
        ]
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, None],
                "High": [102.0, 103.0, None],
                "Low": [99.0, 100.0, None],
                "Close": [101.0, 102.0, None],
                "Volume": [1000.0, 1100.0, volume],
            },
            index=pd.to_datetime(dates),
        )
        return frame

    def test_historical_all_null_ohlcv_placeholder_is_dropped_and_counted(self):
        target = datetime.date(2026, 8, 21)
        raw = self._placeholder_history(target)
        result = fetch_us_stock_history("PLACEHOLDER", target_market_date=target, mock_df=raw)

        self.assertNotIn(target, result.index)
        self.assertEqual(result.index[-1], target - datetime.timedelta(days=1))
        self.assertEqual(result.attrs["dropped_non_observation_placeholder_count"], 1)

    def test_target_all_null_ohlcv_placeholder_is_absent_not_a_market_bar(self):
        target = datetime.date(2026, 8, 21)
        result = fetch_us_stock_history(
            "TARGETPLACEHOLDER",
            target_market_date=target,
            mock_df=self._placeholder_history(target),
        )

        self.assertLess(result.index[-1], target)
        self.assertEqual(result.attrs["dropped_non_observation_placeholder_count"], 1)

    def test_all_placeholder_history_preserves_drop_count_on_empty_result(self):
        target = datetime.date(2026, 8, 21)
        raw = self._placeholder_history(target).iloc[[-1]]
        result = fetch_us_stock_history("ONLYPLACEHOLDER", target_market_date=target, mock_df=raw)

        self.assertTrue(result.empty)
        self.assertEqual(result.attrs["dropped_non_observation_placeholder_count"], 1)

    def test_positive_volume_all_null_ohlcv_remains_schema_failure(self):
        target = datetime.date(2026, 8, 21)
        with self.assertRaises(USSchemaError):
            fetch_us_stock_history(
                "POSITIVEVOLUME",
                target_market_date=target,
                mock_df=self._placeholder_history(target, volume=7.0),
            )

    def test_direct_yahoo_array_length_mismatch_remains_schema_failure(self):
        target = datetime.date(2026, 8, 21)
        timestamp = int(
            datetime.datetime.combine(
                target,
                datetime.time(16, 0),
                tzinfo=datetime.timezone.utc,
            ).timestamp()
        )
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [timestamp, timestamp + 86400],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0],
                                    "high": [101.0, 102.0],
                                    "low": [99.0, 100.0],
                                    "close": [100.5, 101.5],
                                    "volume": [1000.0, 1100.0],
                                }
                            ]
                        },
                    }
                ]
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaises(USSchemaError):
                fetch_direct_yahoo_chart("MISMATCH", target_market_date=target)

    def test_tiny_high_low_tolerance_is_deterministic(self):
        target = datetime.date(2026, 8, 21)
        within = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [99.99995],
                "Low": [99.99995],
                "Close": [100.0],
                "Volume": [1000.0],
            },
            index=pd.to_datetime([target]),
        )
        result = fetch_us_stock_history("TOLERANCE", target_market_date=target, mock_df=within)
        self.assertEqual(result.index[-1], target)

        outside = within.copy()
        outside.loc[pd.Timestamp(target), "High"] = 99.9998
        with self.assertRaises(USIntegrityError):
            fetch_us_stock_history("TOLERANCEFAIL", target_market_date=target, mock_df=outside)
