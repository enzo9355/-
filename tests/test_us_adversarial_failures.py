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
from stock_papi.integrations.market_data.us_market_data import (
    USIntegrityError,
    USSchemaError,
    fetch_direct_yahoo_chart,
    fetch_us_stock_history,
)
from stock_papi.integrations.market_data.us_trading_status import (
    USStatusOperationalError,
    create_us_status_evidence,
    validate_us_status_evidence,
)
from stock_papi.integrations.market_data.us_universe import USUniverseBreakdown


class TestUSAdversarialFailures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.target_date = datetime.date(2026, 8, 19)
        self.status_patch = patch(
            "stock_papi.batch.us_official_post_close_cli.get_us_trading_status_snapshot",
            return_value={},
        )
        self.status_patch.start()
        self.addCleanup(self.status_patch.stop)

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
        """High < Open or Low > High must raise USIntegrityError (OP_FAIL) rather than silent repair."""
        from stock_papi.integrations.market_data.us_market_data import USIntegrityError
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
        with self.assertRaises(USIntegrityError) as ctx:
            fetch_us_stock_history("BAD", mock_df=bad_df)
        self.assertIn("integrity violation", str(ctx.exception).lower())

    def test_missing_target_observation_remains_legitimate_absence(self):
        """A payload with no target-date record may become M, not an operational failure."""
        dates = pd.date_range(end=self.target_date - datetime.timedelta(days=1), periods=10, freq="B")
        history = pd.DataFrame(
            {
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [102.0] * len(dates),
                "Volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        with patch(
            "stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history",
            return_value=history,
        ):
            result = _fetch_and_classify_symbol(
                self.root,
                "MISSING",
                self.target_date,
                security_evidence_by_symbol={
                    "MISSING": {
                        "symbol": "MISSING",
                        "security_type": "COMMON_EQUITY",
                        "eligible": True,
                        "source": "nasdaqtrader:nasdaqlisted",
                        "source_identity": "nasdaqtrader:nasdaqlisted:test",
                        "evidence_sha256": "a" * 64,
                    }
                },
            )
        self.assertEqual(result.kind, "M")
        self.assertEqual(result.reason_code, "provider_healthy_no_target_observation")
        self.assertEqual(result.provider_result["status"], "healthy")
        self.assertEqual(result.provider_result["latest_regular_price_date"], self.target_date.replace(day=18).isoformat())
        self.assertEqual(result.security_evidence["security_type"], "COMMON_EQUITY")

    def test_target_row_with_null_ohlc_is_schema_failure(self):
        dates = pd.date_range(end=self.target_date, periods=10, freq="B")
        bad = self._make_valid_df(date=self.target_date).loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        bad.loc[self.target_date, "Close"] = None
        with self.assertRaises(USSchemaError):
            fetch_us_stock_history("NULLTARGET", target_market_date=self.target_date, mock_df=bad)

    def test_target_row_with_non_numeric_ohlc_is_schema_failure(self):
        bad = self._make_valid_df(date=self.target_date).loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        bad["High"] = bad["High"].astype(object)
        bad.loc[self.target_date, "High"] = "not-a-number"
        with self.assertRaises(USSchemaError):
            fetch_us_stock_history("TEXTTARGET", target_market_date=self.target_date, mock_df=bad)

    def test_target_row_high_below_open_is_integrity_failure(self):
        bad = self._make_valid_df(date=self.target_date).loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        bad.loc[self.target_date, "High"] = bad.loc[self.target_date, "Open"] - 1
        with self.assertRaises(USIntegrityError):
            fetch_us_stock_history("HIGHLOW", target_market_date=self.target_date, mock_df=bad)

    def test_target_row_low_above_close_is_integrity_failure(self):
        bad = self._make_valid_df(date=self.target_date).loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        bad.loc[self.target_date, "Low"] = bad.loc[self.target_date, "Close"] + 1
        with self.assertRaises(USIntegrityError):
            fetch_us_stock_history("LOWHIGH", target_market_date=self.target_date, mock_df=bad)

    def test_conflicting_duplicate_target_rows_are_integrity_failure(self):
        one = self._make_valid_df(date=self.target_date).loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        two = one.copy()
        two.loc[self.target_date, "Close"] = 153.0
        conflicting = pd.concat([one, two]).sort_index()
        with self.assertRaises(USIntegrityError):
            fetch_us_stock_history("DUPLICATE", target_market_date=self.target_date, mock_df=conflicting)

    def test_direct_yahoo_target_payload_malformed_bar_is_not_dropped(self):
        timestamp = int(
            datetime.datetime.combine(
                self.target_date,
                datetime.time(16, 0),
                tzinfo=datetime.timezone.utc,
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
                                    "open": [None],
                                    "high": [105.0],
                                    "low": [95.0],
                                    "close": [102.0],
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
            with self.assertRaises(USSchemaError):
                fetch_direct_yahoo_chart("NULLTARGET", target_market_date=self.target_date)

    def test_required_official_status_source_failure_blocks_whole_batch(self):
        symbols = ["AAPL"]
        breakdown = USUniverseBreakdown(
            configured_listed_count=1,
            eligible_listed_count=1,
            active_universe_count=1,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            excluded_derivative_count=0,
            derivative_breakdown={},
            terminated_delisted_count=None,
            exchange_counts={"NASDAQ": 1},
            symbols=symbols,
            exclusions_by_symbol={},
        )
        with patch(
            "stock_papi.batch.us_official_post_close_cli.get_us_universe_breakdown",
            return_value=breakdown,
        ), patch(
            "stock_papi.batch.us_official_post_close_cli.get_us_trading_status_snapshot",
            side_effect=USStatusOperationalError("timeout"),
        ), patch(
            "stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history"
        ) as fetch:
            with self.assertRaises(RuntimeError) as ctx:
                run_us_post_close(self.root, self.target_date)
        self.assertIn("official trading-status source failed", str(ctx.exception))
        fetch.assert_not_called()

    def test_99_good_1_network_error_fails_closed(self):
        """99% good + 1% network error must FAIL CLOSED (operational failure)."""
        symbols = [f"SYM{i:03d}" for i in range(100)]
        breakdown = USUniverseBreakdown(
            configured_listed_count=100,
            eligible_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            excluded_derivative_count=0,
            derivative_breakdown={},
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
            exclusions_by_symbol={},
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
            eligible_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            excluded_derivative_count=0,
            derivative_breakdown={},
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
            exclusions_by_symbol={},
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
            eligible_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            excluded_derivative_count=0,
            derivative_breakdown={},
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
            exclusions_by_symbol={},
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
            eligible_listed_count=100,
            active_universe_count=100,
            excluded_exchange_count=0,
            excluded_crypto_count=0,
            excluded_invalid_count=0,
            excluded_derivative_count=0,
            derivative_breakdown={},
            terminated_delisted_count=0,
            exchange_counts={"NASDAQ": 100},
            symbols=symbols,
            exclusions_by_symbol={},
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

    def test_typed_exceptions_error_classification(self):
        """Typed exceptions must classify strictly into OP_FAIL vs M without broad ValueError fallback."""
        from stock_papi.integrations.market_data.us_market_data import (
            USObservationUnavailable,
            USProviderOperationalError,
            USRateLimitError,
            USSchemaError,
            USIntegrityError,
        )

        # 1. USObservationUnavailable -> M
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=USObservationUnavailable("no data")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "M")

        # 2. USProviderOperationalError -> OP_FAIL
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=USProviderOperationalError("timeout")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "OP_FAIL")
            self.assertEqual(res.error_type, "USProviderOperationalError")

        # 3. USRateLimitError -> OP_FAIL
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=USRateLimitError("429")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "OP_FAIL")
            self.assertEqual(res.error_type, "USRateLimitError")

        # 4. USSchemaError -> OP_FAIL
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=USSchemaError("missing cols")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "OP_FAIL")
            self.assertEqual(res.error_type, "USSchemaError")

        # 5. USIntegrityError -> OP_FAIL
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=USIntegrityError("High < Open")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "OP_FAIL")
            self.assertEqual(res.error_type, "USIntegrityError")

        # 6. Unknown ValueError / KeyError / OSError -> OP_FAIL
        with patch("stock_papi.batch.us_official_post_close_cli.fetch_us_stock_history", side_effect=ValueError("unknown format error")):
            res = _fetch_and_classify_symbol(self.root, "TEST", self.target_date)
            self.assertEqual(res.kind, "OP_FAIL")


if __name__ == "__main__":
    unittest.main()
