"""Tests for US market data fetcher and indicator computation."""

import datetime
import json
import pandas as pd
import unittest
import urllib.parse
import zoneinfo
from unittest.mock import patch

from stock_papi.integrations.market_data.us_market_data import (
    compute_us_technical_indicators,
    fetch_direct_yahoo_chart,
    fetch_nasdaq_historical_chart,
    fetch_us_stock_history,
    _normalise_us_date,
    USIntegrityError,
    USSchemaError,
)


class USMarketDataTests(unittest.TestCase):
    def test_target_bound_yahoo_query_uses_new_york_exclusive_end(self):
        target = datetime.date(2026, 8, 24)
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1787587200],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0],
                                    "high": [101.0],
                                    "low": [99.0],
                                    "close": [100.5],
                                    "volume": [1000.0],
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

        requests = []

        def open_url(request, timeout):
            requests.append(request)
            return Response()

        with patch("urllib.request.urlopen", side_effect=open_url):
            fetch_direct_yahoo_chart("BRK-B", target_market_date=target)

        query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0].full_url).query)
        self.assertEqual(query["interval"], ["1d"])
        self.assertEqual(query["period1"], ["1724472000"])
        self.assertEqual(query["period2"], ["1787630400"])
        self.assertNotIn("range", query)

    def test_representative_symbols_ignore_future_bar_without_symbol_exception(self):
        target = datetime.date(2026, 8, 24)
        new_york = zoneinfo.ZoneInfo("America/New_York")

        def timestamp(session_date):
            return int(
                datetime.datetime.combine(
                    session_date,
                    datetime.time(12, 0),
                    tzinfo=new_york,
                ).timestamp()
            )

        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [timestamp(target), timestamp(target + datetime.timedelta(days=1))],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 200.0],
                                    "high": [101.0, 150.0],
                                    "low": [99.0, 149.0],
                                    "close": [100.5, 200.5],
                                    "volume": [1000.0, 1000.0],
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

        with patch("urllib.request.urlopen", side_effect=lambda request, timeout: Response()):
            for symbol in ("AKO-A", "BRK-B", "DBC"):
                with self.subTest(symbol=symbol):
                    result = fetch_us_stock_history(symbol, target_market_date=target)
                    self.assertEqual(list(result.index), [target])
                    self.assertEqual(result.attrs["ignored_future_session_row_count"], 1)

    def test_future_only_provider_bar_is_not_reused_when_target_is_missing(self):
        target = datetime.date(2026, 8, 24)
        future = target + datetime.timedelta(days=1)
        new_york = zoneinfo.ZoneInfo("America/New_York")
        timestamp = int(
            datetime.datetime.combine(
                future,
                datetime.time(12, 0),
                tzinfo=new_york,
            ).timestamp()
        )
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [200.0],
                                    "high": [201.0],
                                    "low": [199.0],
                                    "close": [200.5],
                                    "volume": [1000.0],
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
            result = fetch_us_stock_history("DBC", target_market_date=target)

        self.assertTrue(result.empty)
        self.assertEqual(result.attrs["ignored_future_session_row_count"], 1)

    def test_yfinance_fallback_uses_target_bound_start_and_end(self):
        target = datetime.date(2026, 8, 24)
        dates = pd.to_datetime([target, target + datetime.timedelta(days=1)])
        frame = pd.DataFrame(
            {
                "Open": [100.0, 200.0],
                "High": [101.0, 150.0],
                "Low": [99.0, 149.0],
                "Close": [100.5, 200.5],
                "Volume": [1000.0, 1000.0],
            },
            index=dates,
        )
        ticker = unittest.mock.Mock()
        ticker.history.return_value = frame

        with patch(
            "stock_papi.integrations.market_data.us_market_data.fetch_direct_yahoo_chart",
            side_effect=RuntimeError("synthetic direct-provider failure"),
        ), patch(
            "stock_papi.integrations.market_data.us_market_data.yf.Ticker",
            return_value=ticker,
        ):
            result = fetch_us_stock_history("DBC", target_market_date=target)

        self.assertEqual(list(result.index), [target])
        self.assertEqual(
            ticker.history.call_args.kwargs,
            {
                "start": datetime.date(2024, 8, 24),
                "end": datetime.date(2026, 8, 25),
                "auto_adjust": False,
            },
        )

    def test_new_york_session_date_conversion_remains_dst_aware(self):
        utc = datetime.timezone.utc
        self.assertEqual(
            _normalise_us_date(
                datetime.datetime(2026, 3, 6, 4, 59, tzinfo=utc), "DST"
            ),
            datetime.date(2026, 3, 5),
        )
        self.assertEqual(
            _normalise_us_date(
                datetime.datetime(2026, 3, 6, 5, 0, tzinfo=utc), "DST"
            ),
            datetime.date(2026, 3, 6),
        )
        self.assertEqual(
            _normalise_us_date(
                datetime.datetime(2026, 3, 9, 3, 59, tzinfo=utc), "DST"
            ),
            datetime.date(2026, 3, 8),
        )
        self.assertEqual(
            _normalise_us_date(
                datetime.datetime(2026, 3, 9, 4, 0, tzinfo=utc), "DST"
            ),
            datetime.date(2026, 3, 9),
        )

    def test_target_session_is_selected_before_future_live_bar_integrity_validation(self):
        target = datetime.date(2026, 8, 24)
        new_york = zoneinfo.ZoneInfo("America/New_York")

        def timestamp(session_date):
            return int(
                datetime.datetime.combine(
                    session_date,
                    datetime.time(12, 0),
                    tzinfo=new_york,
                ).timestamp()
            )

        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [timestamp(target), timestamp(target + datetime.timedelta(days=1))],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 200.0],
                                    "high": [101.0, 150.0],
                                    "low": [99.0, 149.0],
                                    "close": [100.5, 200.5],
                                    "volume": [1000.0, 1000.0],
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
            result = fetch_us_stock_history("BRK-B", target_market_date=target)

        self.assertEqual(list(result.index), [target])
        self.assertEqual(float(result.iloc[-1]["Close"]), 100.5)

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

    def test_short_history_masks_indicators_that_lack_lookback(self):
        target = datetime.date(2026, 8, 21)
        dates = pd.date_range(end=target, periods=8, freq="B")
        df = pd.DataFrame(
            {
                "Open": [10.0 + i for i in range(8)],
                "High": [10.5 + i for i in range(8)],
                "Low": [9.5 + i for i in range(8)],
                "Close": [10.25 + i for i in range(8)],
                "Volume": [1000.0 + i for i in range(8)],
            },
            index=dates,
        )
        result = compute_us_technical_indicators(df)

        self.assertFalse(pd.isna(result["MA5"].iloc[-1]))
        self.assertTrue(pd.isna(result["MA20"].iloc[-1]))
        self.assertTrue(pd.isna(result["MA60"].iloc[-1]))
        self.assertFalse(pd.isna(result["VOL_RATIO"].iloc[-1]))
        self.assertTrue(pd.isna(result["RSI"].iloc[-1]))
        self.assertTrue(pd.isna(result["MACD"].iloc[-1]))
        self.assertTrue(pd.isna(result["K"].iloc[-1]))

    def test_nasdaq_historical_fallback_requires_exact_symbol_and_preserves_target_evidence(self):
        target = datetime.date(2026, 8, 21)
        dates = [target - datetime.timedelta(days=7 - i) for i in range(8)]

        def row(date, value):
            return {
                "date": date.strftime("%m/%d/%Y"),
                "close": f"${value:.2f}",
                "volume": f"{1000 + value:.0f}",
                "open": f"${value - 0.1:.2f}",
                "high": f"${value + 0.2:.2f}",
                "low": f"${value - 0.2:.2f}",
            }

        document = {
            "data": {
                "symbol": "SNSC",
                "tradesTable": {"rows": [row(date, 10.0 + i) for i, date in enumerate(dates)]},
            }
        }
        result = fetch_nasdaq_historical_chart(
            "SNSC",
            target_market_date=target,
            fetch_json=lambda: document,
        )
        self.assertIn(target, result.index)
        self.assertEqual(result.attrs["source_schema_version"], "nasdaq-historical-v1")
        self.assertEqual(result.attrs["provider_symbol"], "SNSC")
        self.assertTrue(result.attrs["target_observation"] == "present")
        self.assertTrue(pd.isna(result["MA20"].iloc[-1]))
        self.assertTrue(pd.isna(result["RSI"].iloc[-1]))
        self.assertTrue(pd.isna(result["K"].iloc[-1]))

        wrong_identity = {"data": {**document["data"], "symbol": "WLYB"}}
        with self.assertRaises(USSchemaError):
            fetch_nasdaq_historical_chart(
                "SNSC",
                target_market_date=target,
                fetch_json=lambda: wrong_identity,
            )

    def test_nasdaq_historical_fallback_tries_etf_when_stocks_has_no_data(self):
        target = datetime.date(2026, 9, 2)
        urls = []

        def fetch_json(url):
            urls.append(url)
            if "assetclass=stocks" in url:
                return {"data": None}
            return {
                "data": {
                    "symbol": "BTCK",
                    "tradesTable": {
                        "rows": [{
                            "date": "09/02/2026",
                            "open": "24.95",
                            "high": "26.46",
                            "low": "24.95",
                            "close": "25.96",
                            "volume": "2,627",
                        }]
                    },
                }
            }

        result = fetch_nasdaq_historical_chart(
            "BTCK",
            target_market_date=target,
            fetch_json=fetch_json,
        )

        self.assertEqual(len(urls), 2)
        self.assertIn("assetclass=stocks", urls[0])
        self.assertIn("assetclass=etf", urls[1])
        self.assertEqual(result.loc[target, "Low"], 24.95)
        self.assertEqual(result.attrs["provider_asset_class"], "etf")

    def test_nasdaq_historical_fallback_does_not_promote_a_stale_date(self):
        target = datetime.date(2026, 8, 21)
        document = {
            "data": {
                "symbol": "SNSC",
                "tradesTable": {
                    "rows": [
                        {
                            "date": "08/20/2026",
                            "close": "$10.00",
                            "volume": "1000",
                            "open": "$9.90",
                            "high": "$10.20",
                            "low": "$9.80",
                        }
                    ]
                },
            }
        }
        result = fetch_nasdaq_historical_chart(
            "SNSC",
            target_market_date=target,
            fetch_json=lambda: document,
        )
        self.assertNotIn(target, result.index)
        self.assertEqual(result.attrs["target_observation"], "absent")

    def test_nasdaq_historical_fallback_rejects_incomplete_target_row(self):
        target = datetime.date(2026, 8, 21)
        document = {
            "data": {
                "symbol": "SNSC",
                "tradesTable": {
                    "rows": [
                        {
                            "date": "08/21/2026",
                            "close": "N/A",
                            "volume": "1000",
                            "open": "$9.90",
                            "high": "$10.20",
                            "low": "$9.80",
                        }
                    ]
                },
            }
        }
        with self.assertRaises(USSchemaError):
            fetch_nasdaq_historical_chart(
                "SNSC",
                target_market_date=target,
                fetch_json=lambda: document,
            )

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

    def test_direct_fetch_placeholder_count_survives_second_prepare(self):
        target = datetime.date(2026, 8, 21)
        dates = pd.date_range(end=target, periods=5, freq="B")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [101.0] * 5,
                "Low": [99.0] * 5,
                "Close": [100.5] * 5,
                "Volume": [1000.0] * 5,
            },
            index=dates,
        )
        frame.attrs["dropped_non_observation_placeholder_count"] = 2

        with patch(
            "stock_papi.integrations.market_data.us_market_data.fetch_direct_yahoo_chart",
            return_value=frame,
        ):
            result = fetch_us_stock_history(
                "DIRECTCOUNT",
                target_market_date=target,
            )

        self.assertEqual(result.attrs["dropped_non_observation_placeholder_count"], 2)

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
