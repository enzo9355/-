"""Tests for US stock market universe from SEC exchange listings."""

import datetime
import tempfile
import unittest

from stock_papi.integrations.market_data.us_universe import (
    get_us_symbols,
    parse_sec_us_universe,
    validate_us_ticker,
)


class USUniverseTests(unittest.TestCase):
    def test_validate_us_ticker(self):
        self.assertEqual(validate_us_ticker("AAPL"), "AAPL")
        self.assertEqual(validate_us_ticker("BRK-B"), "BRK-B")
        self.assertEqual(validate_us_ticker("nvda"), "NVDA")
        self.assertEqual(validate_us_ticker("SPY"), "SPY")

        with self.assertRaises(ValueError):
            validate_us_ticker("")
        with self.assertRaises(ValueError):
            validate_us_ticker("../traversal")
        with self.assertRaises(ValueError):
            validate_us_ticker("A" * 15)
        with self.assertRaises(ValueError):
            validate_us_ticker("INVALID/SLASH")

    def test_parse_sec_us_universe(self):
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [1045810, "NVIDIA CORP", "NVDA", "Nasdaq"],
                [1067983, "BERKSHIRE HATHAWAY INC", "BRK.B", "NYSE"],
                [1234567, "Some Penny Stock", "OTCXYZ", "OTC"],
                [9999999, "Bitcoin Trust Fund", "BTCF", "Nasdaq"],
            ],
        }
        symbols = parse_sec_us_universe(doc)
        self.assertEqual(symbols, ["AAPL", "BRK-B", "NVDA"])
        self.assertNotIn("OTCXYZ", symbols)  # OTC excluded
        self.assertNotIn("BTCF", symbols)  # Crypto excluded

    def test_get_us_symbols_cache_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_doc = {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1, "Apple Inc.", "AAPL", "Nasdaq"],
                    [2, "Microsoft Corp", "MSFT", "Nasdaq"],
                ],
            }
            # First fetch (creates cache)
            symbols = get_us_symbols(
                tmpdir,
                fetch_json=lambda: sample_doc,
                now=datetime.datetime(2026, 8, 20, tzinfo=datetime.timezone.utc),
            )
            self.assertEqual(symbols, ["AAPL", "MSFT"])

            # Second fetch with failing source (uses cache)
            def failing_fetch():
                raise RuntimeError("network down")

            symbols2 = get_us_symbols(
                tmpdir,
                fetch_json=failing_fetch,
                now=datetime.datetime(2026, 8, 20, tzinfo=datetime.timezone.utc),
            )
            self.assertEqual(symbols2, ["AAPL", "MSFT"])

    def test_derivative_security_type_exclusions(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Apple Inc.", "AAPL", "Nasdaq"],
                [2, "SPDR S&P 500 ETF Trust", "SPY", "NYSE"],
                [3, "Some Acquisition Warrant", "AACIW", "Nasdaq"],
                [4, "Some SPAC Unit", "AAC-UN", "Nasdaq"],
                [5, "Some Company Preferred Stock", "ACP-PA", "NYSE"],
                [6, "Some Subscription Right", "AESPR", "Nasdaq"],
            ],
        }
        # Equity observation scope excludes derivative instruments
        equity_bd = parse_sec_us_universe_with_metadata(doc, scope="EQUITY_OBSERVATION")
        self.assertEqual(equity_bd.symbols, ["AAPL", "SPY"])
        self.assertEqual(equity_bd.excluded_derivative_count, 4)
        self.assertEqual(equity_bd.derivative_breakdown["WARRANT"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["UNIT"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["PREFERRED"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["RIGHT"], 1)

        # ALL_LISTED scope includes derivatives
        all_bd = parse_sec_us_universe_with_metadata(doc, scope="ALL_LISTED")
        self.assertEqual(len(all_bd.symbols), 6)
        self.assertEqual(all_bd.excluded_derivative_count, 0)
