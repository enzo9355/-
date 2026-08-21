"""Adversarial regression tests for US institutional-grade error classification and Manifest v4."""

import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from stock_papi.batch.us_official_post_close_cli import (
    _fetch_and_classify_symbol,
    ObservationResult,
    run_us_post_close,
)
from stock_papi.integrations.market_data.us_market_data import fetch_us_stock_history
from stock_papi.integrations.market_data.us_trading_status import (
    create_us_status_evidence,
    validate_us_status_evidence,
)
from stock_papi.integrations.market_data.us_universe import USUniverseBreakdown


class TestUSAdversarialFailures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.target_date = datetime.date(2026, 8, 19)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_valid_df(self, symbol="AAPL", date=None):
        date = date or self.target_date
        dates = [d.date() for d in pd.date_range(end=date, periods=30, freq="B")]
        df = pd.DataFrame(
            {
                "Open": [150.0] * len(dates),
                "High": [155.0] * len(dates),
                "Low": [149.0] * len(dates),
                "Close": [152.0] * len(dates),
                "Volume": [1000000.0] * len(dates),
                "MA5": [152.0] * len(dates),
                "MA20": [152.0] * len(dates),
                "MA60": [152.0] * len(dates),
                "VOL_RATIO": [1.0] * len(dates),
                "RSI": [50.0] * len(dates),
                "MACD": [0.0] * len(dates),
                "MACD_SIGNAL": [0.0] * len(dates),
                "MACD_OSC": [0.0] * len(dates),
                "K": [50.0] * len(dates),
                "D": [50.0] * len(dates),
            },
            index=dates,
        )
        df.index.name = "Date"
        return df

    def test_ohlc_integrity_rejection_no_silent_repair(self):
        """High < Open or Low > High must raise ValueError (OP_FAIL) rather than silent repair."""
        dates = pd.date_range(end=self.target_date, periods=10, freq="B")
        # Corrupt High < Open
        bad_df = pd.DataFrame(
            {
                "Open": [150.0] * len(dates),
                "High": [140.0] * len(dates),  # Impossible High < Open
                "Low": [130.0] * len(dates),
                "Close": [135.0] * len(dates),
                "Volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        with self.assertRaises(ValueError) as ctx:
            fetch_us_stock_history("BAD", mock_df=bad_df)
        self.assertIn("integrity violation", str(ctx.exception).lower())

    def test_99_good_1_network_error_fails_closed(self):
        """99% good + 1% network error must FAIL CLOSED (operational failure)."""
        symbols = [f"SYM{i:03d}" for i in range(100)]
        breakdown = USUniverseBreakdown(
            configured_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
        )

        def mock_fetch(sym, target_market_date=None, mock_df=None):
            if sym == "SYM099":
                raise ConnectionError("Network timeout to Yahoo Finance")
            return self._make_valid_df(sym, target_market_date)

        with patch("stock_papi.batch.us_official_post_close_cli.get_us_universe_breakdown", return_value=breakdown),              patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=mock_fetch):
            with self.assertRaises(RuntimeError) as ctx:
                run_us_post_close(self.root, self.target_date)
            self.assertIn("operational symbol failures encountered", str(ctx.exception).lower())

    def test_99_good_1_parser_error_fails_closed(self):
        """99% good + 1% parser error must FAIL CLOSED."""
        symbols = [f"SYM{i:03d}" for i in range(100)]
        breakdown = USUniverseBreakdown(
            configured_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
        )

        def mock_fetch(sym, target_market_date=None, mock_df=None):
            if sym == "SYM050":
                raise ValueError("US price schema is incomplete for SYM050")
            return self._make_valid_df(sym, target_market_date)

        with patch("stock_papi.batch.us_official_post_close_cli.get_us_universe_breakdown", return_value=breakdown),              patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=mock_fetch):
            with self.assertRaises(RuntimeError) as ctx:
                run_us_post_close(self.root, self.target_date)
            self.assertIn("operational symbol failures encountered", str(ctx.exception).lower())

    def test_99_good_1_genuine_unavailable_passes_gate(self):
        """99% good + 1% genuine unavailable (no trades) must PASS with operational_failure_count=0."""
        symbols = [f"SYM{i:03d}" for i in range(100)]
        breakdown = USUniverseBreakdown(
            configured_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
        )

        def mock_fetch(sym, target_market_date=None, mock_df=None):
            if sym == "SYM001":
                return pd.DataFrame()  # Empty dataframe = genuine unavailable
            return self._make_valid_df(sym, target_market_date)

        with patch("stock_papi.batch.us_official_post_close_cli.get_us_universe_breakdown", return_value=breakdown),              patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=mock_fetch):
            promoted = run_us_post_close(self.root, self.target_date)
            self.assertIsNotNone(promoted)

            manifest_doc = json.loads((self.root / "publish" / "quant" / "v1" / "latest-US.json").read_text(encoding="utf-8"))
            man_rel = manifest_doc["manifest"]
            full_manifest = json.loads((self.root / "publish" / "quant" / "v1" / man_rel).read_text(encoding="utf-8"))

            self.assertEqual(full_manifest["active_universe_count"], 100)
            self.assertEqual(full_manifest["observation_count"], 99)
            self.assertEqual(full_manifest["regular_price_symbol_count"], 99)
            self.assertEqual(full_manifest["verified_non_price_symbol_count"], 0)
            self.assertEqual(full_manifest["unavailable_count"], 1)
            self.assertEqual(full_manifest["unavailable_symbols"], ["SYM001"])
            self.assertEqual(full_manifest["operational_failure_count"], 0)
            self.assertEqual(full_manifest["operational_failed_symbols"], [])
            self.assertAlmostEqual(full_manifest["observation_coverage"], 0.99)

    def test_exactly_95_percent_coverage_fails_closed(self):
        """95.0% exact coverage must FAIL (strictly >95% required)."""
        symbols = [f"SYM{i:03d}" for i in range(100)]
        breakdown = USUniverseBreakdown(
            configured_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
        )

        def mock_fetch(sym, target_market_date=None, mock_df=None):
            idx = int(sym[3:])
            if idx >= 95:  # 5 symbols unavailable -> 95/100 = 95.0%
                return pd.DataFrame()
            return self._make_valid_df(sym, target_market_date)

        with patch("stock_papi.batch.us_official_post_close_cli.get_us_universe_breakdown", return_value=breakdown),              patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=mock_fetch):
            with self.assertRaises(RuntimeError) as ctx:
                run_us_post_close(self.root, self.target_date)
            self.assertIn("fails strict >95% publishable threshold", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
