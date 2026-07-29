import gzip
import datetime
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import local_quant

from local_quant import (
    TAIPEI,
    build_stock_snapshot,
    ensure_layout,
    get_us_symbols,
    load_checkpoint,
    parse_sec_us_universe,
    run_market_batch,
    save_checkpoint,
    write_stock_artifact,
)
from stock_papi.integrations.market_data.provider import FinMindFetchError


class LocalQuantBatchTests(unittest.TestCase):
    def test_sec_us_universe_filters_exchange_crypto_and_unsafe_tickers(self):
        document = {
            "fields": ["exchange", "ticker", "name", "cik"],
            "data": [
                ["Nasdaq", "aapl", "Apple Inc.", 1],
                ["NYSE", "BRK-B", "Berkshire Hathaway", 2],
                ["CBOE", "XYZ", "Example Fund", 3],
                ["OTC", "OTCM", "OTC Company", 4],
                ["Nasdaq", "BTCX", "Example Bitcoin Trust", 5],
                ["NYSE", "AAPL.B", "Unsafe Symbol", 6],
                ["Nasdaq", "AAPL", "Duplicate", 7],
            ],
        }

        self.assertEqual(
            parse_sec_us_universe(document),
            ["AAPL", "BRK-B", "XYZ"],
        )

    def test_us_universe_uses_daily_cache_and_stale_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            document = {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[1, "Apple Inc.", "AAPL", "Nasdaq"]],
            }
            calls = []

            first = get_us_symbols(
                root,
                fetch_json=lambda: calls.append("fetch") or document,
                now=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
            )
            same_day = get_us_symbols(
                root,
                fetch_json=lambda: self.fail("same-day cache should avoid fetch"),
                now=datetime.datetime(2026, 7, 5, 7, tzinfo=TAIPEI),
            )
            stale = get_us_symbols(
                root,
                fetch_json=lambda: (_ for _ in ()).throw(TimeoutError("offline")),
                fetch_nasdaq=lambda: (_ for _ in ()).throw(TimeoutError("offline")),
                now=datetime.datetime(2026, 7, 6, 6, tzinfo=TAIPEI),
            )

            self.assertEqual(first, ["AAPL"])
            self.assertEqual(same_day, ["AAPL"])
            self.assertEqual(stale, ["AAPL"])
            self.assertEqual(calls, ["fetch"])
            cache = json.loads(
                (root / "raw" / "us-universe.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(cache), {"as_of", "source", "symbols"})

    def test_us_universe_fails_safely_without_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            with self.assertRaisesRegex(RuntimeError, "US universe is unavailable"):
                get_us_symbols(
                    root,
                    fetch_json=lambda: (_ for _ in ()).throw(TimeoutError("secret")),
                    fetch_nasdaq=lambda: (_ for _ in ()).throw(TimeoutError("secret")),
                    now=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                )

    def test_nasdaq_us_universe_filters_test_crypto_and_unsafe_symbols(self):
        listed = """Symbol|Security Name|Market Category|Test Issue|ETF
AAPL|Apple Inc. Common Stock|Q|N|N
TEST|Test Security|Q|Y|N
BTCX|Example Bitcoin Trust|G|N|Y
BAD/WS|Unsafe Warrant|S|N|N
File Creation Time: 07082026||||
"""
        other = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Class B|N|BRK.B|N|N|BRK.B
FAKE|Test Security|A|FAKE|N|Y|FAKE
File Creation Time: 07082026||||||
"""

        self.assertEqual(
            local_quant.parse_nasdaq_us_universe(listed, other),
            ["AAPL", "BRK-B"],
        )

    def test_us_universe_falls_back_to_nasdaq_without_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            listed = "Symbol|Security Name|Test Issue\nAAPL|Apple Inc.|N\n"
            other = "ACT Symbol|Security Name|Test Issue|NASDAQ Symbol\n"

            symbols = get_us_symbols(
                root,
                fetch_json=lambda: (_ for _ in ()).throw(PermissionError("403")),
                fetch_nasdaq=lambda: (listed, other),
                now=datetime.datetime(2026, 7, 8, 6, tzinfo=TAIPEI),
            )

            self.assertEqual(symbols, ["AAPL"])
            cache = json.loads(
                (root / "raw" / "us-universe.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cache["symbols"], ["AAPL"])
            self.assertIn("nasdaqtrader.com", cache["source"])

    def test_stock_artifact_is_atomic_gzip_json_with_fixed_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)

            target = write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-03", "rows": 500, "latest": {"prob": 61.2}},
            )

            with gzip.open(target, "rt", encoding="utf-8") as stream:
                document = json.load(stream)
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["market"], "TW")
            self.assertEqual(document["symbol"], "2330")
            self.assertFalse(target.with_suffix(target.suffix + ".tmp").exists())

    def test_stock_artifact_rejects_invalid_symbols_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            for symbol in ("../2330", "AAPL", "2330/evil"):
                with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                    write_stock_artifact(root, "TW", symbol, {"value": 1.0})
            with self.assertRaises(ValueError):
                write_stock_artifact(root, "TW", "2330", {"value": math.nan})

    def test_us_stock_artifact_uses_separate_market_path_and_safe_symbol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)

            target = write_stock_artifact(
                root, "US", "BRK-B", {"as_of": "2026-07-03"}
            )

            self.assertEqual(
                target,
                root / "artifacts" / "stocks" / "US" / "BRK-B.json.gz",
            )
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["market"], "US")
            for symbol in ("../AAPL", "AAPL/evil", "AAPL.B", "AAPL\\evil"):
                with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                    write_stock_artifact(root, "US", symbol, {"value": 1})

    def test_taiwan_and_us_checkpoints_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)

            save_checkpoint(root, {"market": "TW", "next_index": 200}, market="TW")
            save_checkpoint(root, {"market": "US", "next_index": 50}, market="US")

            self.assertEqual(load_checkpoint(root, market="TW")["next_index"], 200)
            self.assertEqual(load_checkpoint(root, market="US")["next_index"], 50)
            self.assertTrue((root / "checkpoints" / "progress.json").exists())
            self.assertTrue((root / "checkpoints" / "progress-US.json").exists())

    def test_market_batch_isolates_failure_and_saves_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)

            def analyze(symbol):
                if symbol == "2317":
                    raise TimeoutError("secret upstream response")
                return {"as_of": "2026-07-03"}

            summary = run_market_batch(
                root,
                "TW",
                ["2330", "2317", "2454"],
                analyze,
                limit=2,
                now_fn=lambda: datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                delay=0,
            )

            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["failed"], [{"symbol": "2317", "error": "TimeoutError"}])
            checkpoint = load_checkpoint(root)
            self.assertEqual(checkpoint["next_index"], 2)
            self.assertNotIn("secret upstream response", json.dumps(checkpoint))

    def test_market_batch_fails_fast_and_preserves_provider_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            calls = []
            warnings = []

            def analyze(symbol):
                calls.append(symbol)
                if symbol == "2330":
                    return {"as_of": "2026-07-23"}
                error = FinMindFetchError(
                    "quota_or_rate_limit",
                    "TaiwanStockPrice",
                    symbol,
                    "2026-07-01",
                    "2026-07-23",
                    http_status=402,
                    exception_type="HTTPError",
                    blocked_until=2000,
                    retry_after_seconds=3600,
                )
                warnings.append(error.safe_message)
                raise error

            with self.assertRaises(FinMindFetchError):
                run_market_batch(
                    root,
                    "TW",
                    ["2330", "2317", "2454"],
                    analyze,
                    now_fn=lambda: datetime.datetime(
                        2026, 7, 23, 17, 14, 3, tzinfo=TAIPEI
                    ),
                    delay=0,
                    enforce_window=False,
                    batch_identity={
                        "target_market_date": "2026-07-23",
                        "product_mode": "observation",
                    },
                )

            self.assertEqual(calls, ["2330", "2317"])
            self.assertEqual(len(warnings), 1)
            checkpoint = load_checkpoint(root)
            self.assertEqual(checkpoint["stage"], "market_batch")
            self.assertEqual(checkpoint["provider"], "FinMind")
            self.assertEqual(checkpoint["dataset"], "TaiwanStockPrice")
            self.assertEqual(checkpoint["category"], "quota_or_rate_limit")
            self.assertEqual(checkpoint["http_status"], 402)
            self.assertEqual(checkpoint["blocked_until"], 2000)
            self.assertEqual(checkpoint["first_failed_symbol"], "2317")
            self.assertEqual(checkpoint["next_index"], 2)
            self.assertEqual(checkpoint["attempted_count"], 2)
            self.assertEqual(checkpoint["successful_count"], 1)
            self.assertEqual(checkpoint["failed_count"], 1)
            self.assertEqual(checkpoint["provider_failure_count"], 1)
            self.assertEqual(
                checkpoint["timestamp"],
                "2026-07-23T17:14:03+08:00",
            )
            self.assertIn(
                "category=quota_or_rate_limit",
                checkpoint["safe_message"],
            )
            serialized = json.dumps(checkpoint)
            self.assertNotIn("token", serialized.lower())
            self.assertNotIn("password", serialized.lower())
            self.assertFalse(
                list((root / "publish" / "quant" / "v1").glob("manifests/*.json"))
            )

    def test_provider_failure_keeps_prior_failures_and_consistent_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 2,
                    "failed": [
                        {"symbol": "2330", "error": "TimeoutError"},
                        {"symbol": "2317", "error": "ValueError"},
                        {"symbol": "2330", "error": "DuplicateMustBeRemoved"},
                    ],
                },
            )

            def analyze(symbol):
                if symbol in {"2330", "2317"}:
                    raise ValueError("raw upstream credential must not persist")
                raise FinMindFetchError(
                    "quota_or_rate_limit",
                    "TaiwanStockPrice",
                    symbol,
                    "2026-07-01",
                    "2026-07-23",
                    http_status=429,
                    exception_type="HTTPError",
                    blocked_until=2000,
                    retry_after_seconds=17,
                )

            with self.assertRaises(FinMindFetchError):
                run_market_batch(
                    root,
                    "TW",
                    ["2330", "2317", "2454"],
                    analyze,
                    now_fn=lambda: datetime.datetime(
                        2026, 7, 23, 17, 14, 3, tzinfo=TAIPEI
                    ),
                    delay=0,
                    enforce_window=False,
                )

            checkpoint = load_checkpoint(root)
            failed = checkpoint["failed"]
            symbols = [item["symbol"] for item in failed]
            self.assertEqual(symbols, ["2330", "2317", "2454"])
            self.assertEqual(len(symbols), len(set(symbols)))
            self.assertEqual(checkpoint["failed_count"], len(failed))
            self.assertEqual(checkpoint["failed_count"], 3)
            self.assertEqual(checkpoint["provider_failure_count"], 1)
            self.assertEqual(checkpoint["first_failed_symbol"], "2454")
            self.assertEqual(checkpoint["next_index"], 3)
            self.assertGreaterEqual(
                checkpoint["attempted_count"],
                checkpoint["successful_count"] + len(failed),
            )
            serialized = json.dumps(checkpoint).lower()
            self.assertNotIn("raw upstream credential", serialized)
            self.assertNotIn("token", serialized)
            self.assertNotIn("password", serialized)

    def test_provider_failure_resumes_retry_then_next_main_symbol_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            first_calls = []

            def first_analyze(symbol):
                first_calls.append(symbol)
                if symbol == "2330":
                    return {"as_of": "2026-07-23"}
                raise FinMindFetchError(
                    "quota_or_rate_limit",
                    "TaiwanStockPrice",
                    symbol,
                    "2026-07-01",
                    "2026-07-23",
                    http_status=429,
                    exception_type="HTTPError",
                    blocked_until=2000,
                    retry_after_seconds=17,
                )

            with self.assertRaises(FinMindFetchError):
                run_market_batch(
                    root,
                    "TW",
                    ["2330", "2317", "2454"],
                    first_analyze,
                    limit=2,
                    now_fn=lambda: datetime.datetime(
                        2026, 7, 23, 17, 14, 3, tzinfo=TAIPEI
                    ),
                    delay=0,
                    enforce_window=False,
                )

            first_checkpoint = load_checkpoint(root)
            self.assertEqual(first_calls, ["2330", "2317"])
            self.assertEqual(
                [item["symbol"] for item in first_checkpoint["failed"]],
                ["2317"],
            )
            self.assertEqual(first_checkpoint["next_index"], 2)

            resumed_calls = []
            summary = run_market_batch(
                root,
                "TW",
                ["2330", "2317", "2454"],
                lambda symbol: resumed_calls.append(symbol)
                or {"as_of": "2026-07-23"},
                limit=2,
                now_fn=lambda: datetime.datetime(
                    2026, 7, 23, 17, 15, 3, tzinfo=TAIPEI
                ),
                delay=0,
                enforce_window=False,
            )

            self.assertEqual(resumed_calls, ["2317", "2454"])
            self.assertEqual(resumed_calls.count("2317"), 1)
            self.assertEqual(resumed_calls.count("2454"), 1)
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], [])
            self.assertEqual(summary["next_index"], 3)
            final_checkpoint = load_checkpoint(root)
            self.assertEqual(final_checkpoint["next_index"], 3)
            self.assertEqual(final_checkpoint["failed"], [])

    def test_retry_provider_failure_does_not_advance_main_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 2,
                    "failed": [{"symbol": "2317", "error": "TimeoutError"}],
                },
            )

            def fail_provider_retry(symbol):
                raise FinMindFetchError(
                    "timeout",
                    "TaiwanStockPrice",
                    symbol,
                    "2026-07-01",
                    "2026-07-23",
                    exception_type="Timeout",
                )

            with self.assertRaises(FinMindFetchError):
                run_market_batch(
                    root,
                    "TW",
                    ["2330", "2317", "2454"],
                    fail_provider_retry,
                    limit=1,
                    now_fn=lambda: datetime.datetime(
                        2026, 7, 23, 17, 14, 3, tzinfo=TAIPEI
                    ),
                    delay=0,
                    enforce_window=False,
                )

            checkpoint = load_checkpoint(root)
            self.assertEqual(checkpoint["next_index"], 2)
            self.assertEqual(
                [item["symbol"] for item in checkpoint["failed"]],
                ["2317"],
            )

    def test_market_batch_retries_previous_failures_before_new_symbols(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 2,
                    "failed": [{"symbol": "2317", "error": "TimeoutError"}],
                },
            )
            calls = []

            summary = run_market_batch(
                root,
                "TW",
                ["2330", "2317", "2454"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-03"},
                limit=2,
                now_fn=lambda: datetime.datetime(2026, 7, 6, 6, tzinfo=TAIPEI),
                delay=0,
            )

            self.assertEqual(calls, ["2317", "2454"])
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], [])
            checkpoint = load_checkpoint(root)
            self.assertEqual(checkpoint["next_index"], 3)
            self.assertEqual(checkpoint["failed"], [])

    def test_market_batch_restarts_when_explicit_target_identity_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 1,
                    "failed": [],
                    "updated_at": "2026-07-14T17:00:00+08:00",
                    "batch_identity": {"target_market_date": "2026-07-14"},
                },
            )
            calls = []
            identity = {"target_market_date": "2026-07-15"}

            result = run_market_batch(
                root,
                "TW",
                ["2330", "2317"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-15"},
                limit=1,
                now_fn=lambda: datetime.datetime(2026, 7, 15, 17, tzinfo=TAIPEI),
                delay=0,
                enforce_window=False,
                batch_identity=identity,
            )

            self.assertEqual(calls, ["2330"])
            self.assertEqual(result["next_index"], 1)
            self.assertEqual(load_checkpoint(root)["batch_identity"], identity)

    def test_market_batch_keeps_one_copy_of_a_failed_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 1,
                    "failed": [{"symbol": "2330", "error": "TimeoutError"}],
                },
            )

            summary = run_market_batch(
                root,
                "TW",
                ["2330", "2317"],
                lambda _symbol: (_ for _ in ()).throw(ConnectionError("private")),
                limit=1,
                now_fn=lambda: datetime.datetime(2026, 7, 6, 6, tzinfo=TAIPEI),
                delay=0,
            )

            self.assertEqual(summary["next_index"], 1)
            self.assertEqual(
                summary["failed"],
                [{"symbol": "2330", "error": "ConnectionError"}],
            )
            self.assertEqual(load_checkpoint(root)["failed"], summary["failed"])

    def test_market_batch_does_not_restart_an_unpublished_completed_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 1,
                    "failed": [],
                    "cycle_completed_on": "2026-07-05",
                },
            )
            calls = []

            summary = run_market_batch(
                root,
                "TW",
                ["2330"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-03"},
                now_fn=lambda: datetime.datetime(2026, 7, 6, 6, tzinfo=TAIPEI),
                delay=0,
            )

            self.assertEqual(calls, [])
            self.assertEqual(summary["next_index"], 1)

    def test_market_batch_resumes_and_stops_before_drain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            calls = []
            times = iter(
                [
                    datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                    datetime.datetime(2026, 7, 5, 9, 20, tzinfo=TAIPEI),
                ]
            )

            first = run_market_batch(
                root,
                "TW",
                ["2330", "2317"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-03"},
                now_fn=lambda: next(times),
                delay=0,
            )
            self.assertEqual(calls, ["2330"])
            self.assertEqual(first["next_index"], 1)

            second = run_market_batch(
                root,
                "TW",
                ["2330", "2317"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-03"},
                now_fn=lambda: datetime.datetime(2026, 7, 6, 6, tzinfo=TAIPEI),
                delay=0,
            )
            self.assertEqual(calls, ["2330", "2317"])
            self.assertEqual(second["next_index"], 2)
            checkpoint = load_checkpoint(root)
            checkpoint["published_cycle_on"] = checkpoint["cycle_completed_on"]
            save_checkpoint(root, checkpoint)

            third = run_market_batch(
                root,
                "TW",
                ["2330", "2317"],
                lambda symbol: calls.append(symbol) or {"as_of": "2026-07-04"},
                limit=1,
                now_fn=lambda: datetime.datetime(2026, 7, 7, 6, tzinfo=TAIPEI),
                delay=0,
            )
            self.assertEqual(calls, ["2330", "2317", "2330"])
            self.assertEqual(third["next_index"], 1)

    def test_market_batch_does_not_advance_checkpoint_when_disk_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)

            with (
                patch("local_quant._write_gzip_json_atomic", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                run_market_batch(
                    root,
                    "TW",
                    ["2330"],
                    lambda _symbol: {"as_of": "2026-07-03"},
                    now_fn=lambda: datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                    delay=0,
                )

            self.assertEqual(load_checkpoint(root), {})

    def test_taiwan_snapshot_reuses_existing_pipeline_and_keeps_daily_records(self):
        calls = []
        frame = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Volume": [1000, 1200],
            },
            index=pd.to_datetime(["2026-07-02", "2026-07-03"]),
        )
        frame.index.name = "Date"

        def get_data(symbol, days):
            calls.append((symbol, days))
            return frame.copy()

        def calc_all(data):
            data["RSI"] = [50.0, 55.0]
            return data

        def run_ai_engine(data):
            data["AI_P"] = [58.0, 61.2]
            return {"accuracy": 57.5, "trades": 12}

        pipeline = SimpleNamespace(
            get_data=get_data,
            calc_all=calc_all,
            run_ai_engine=run_ai_engine,
            get_stock_name=lambda _symbol: "台積電",
            PREDICTION_HORIZON=5,
        )

        payload = build_stock_snapshot(pipeline, "TW", "2330")

        self.assertEqual(calls, [("2330", 730)])
        self.assertEqual(payload["as_of"], "2026-07-03")
        self.assertEqual(payload["rows"], 2)
        self.assertEqual(payload["latest"]["AI_P"], 61.2)
        self.assertEqual(payload["backtest"]["trades"], 12)
        self.assertEqual(payload["model_version"], "lgbm-5d-v1")
        self.assertEqual(len(payload["daily"]), 2)
        self.assertEqual(payload["daily"][-1]["Date"], "2026-07-03T00:00:00.000")

    def test_taiwan_snapshot_rejects_missing_history_or_backtest(self):
        empty = SimpleNamespace(
            get_data=lambda _symbol, _days: pd.DataFrame(),
            calc_all=lambda data: data,
            run_ai_engine=lambda _data: None,
            get_stock_name=lambda symbol: symbol,
            PREDICTION_HORIZON=5,
        )
        with self.assertRaises(ValueError):
            build_stock_snapshot(empty, "TW", "2330")

    def test_taiwan_snapshot_rejects_target_market_date_mismatch(self):
        frame = pd.DataFrame(
            {"Close": [100.0], "AI_P": [63.0]},
            index=pd.to_datetime(["2026-07-16"]),
        )
        frame.index.name = "Date"
        pipeline = SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: data,
            run_ai_engine=lambda _data: {"accuracy": 55.0},
            get_stock_name=lambda symbol: symbol,
            PREDICTION_HORIZON=5,
        )

        with self.assertRaisesRegex(ValueError, "target market date mismatch"):
            build_stock_snapshot(
                pipeline,
                "TW",
                "2330",
                target_market_date=datetime.date(2026, 7, 17),
            )

    def test_taiwan_status_snapshot_preserves_last_regular_price_date(self):
        from stock_papi.integrations.market_data.tw_trading_status import evidence_sha256

        frame = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000.0]},
            index=pd.to_datetime(["2026-07-16"]),
        )
        frame.index.name = "Date"
        pipeline = SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: data,
            get_stock_name=lambda symbol: symbol,
        )
        status = {
            "schema_version": 1,
            "status": "official_no_regular_trade",
            "market": "TW",
            "exchange": "TPEx",
            "symbol": "2330",
            "target_market_date": "2026-07-17",
            "source_id": "tpex_price",
            "payload_sha256": "a" * 64,
            "raw_row_sha256": "b" * 64,
            "raw_fields": {"open": "---", "high": "---", "low": "---", "close": "---", "volume": "0"},
            "parser_version": "tw-official-historical-parser-v3",
        }
        status["evidence_sha256"] = evidence_sha256(status)

        payload = build_stock_snapshot(
            pipeline,
            "TW",
            "2330",
            target_market_date=datetime.date(2026, 7, 17),
            observation_only=True,
            trading_status=status,
        )

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["as_of"], "2026-07-16")
        self.assertEqual(payload["latest_regular_price_date"], "2026-07-16")
        self.assertEqual(payload["observation_as_of"], "2026-07-17")
        self.assertEqual(payload["observation_kind"], "official_no_regular_trade")
        self.assertEqual(payload["trading_status"], status)
        self.assertEqual(len(payload["daily"]), 1)

    def test_taiwan_snapshot_fast_lane_uses_promoted_backtest_without_walk_forward(self):
        frame = pd.DataFrame(
            {"Close": [100.0], "AI_P": [None]},
            index=pd.to_datetime(["2026-07-16"]),
        )
        frame.index.name = "Date"
        calls = []

        def latest_inference(data):
            calls.append("latest")
            data.loc[data.index[-1], "AI_P"] = 64.5
            return {"model_version": "lgbm-5d-v1", "probability": 64.5}

        pipeline = SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: data,
            run_latest_inference=latest_inference,
            run_ai_engine=lambda _data: self.fail("walk-forward must not run"),
            get_stock_name=lambda symbol: symbol,
            PREDICTION_HORIZON=5,
        )
        promoted = {
            "model_version": "lgbm-5d-v1",
            "feature_schema_version": 1,
            "recommendation_policy_version": "recommendation-v1",
            "cutoff": "2026-07-09",
            "accuracy": 55.0,
            "candidate_sha256": "a" * 64,
            "promoted_at": "2026-07-16T10:00:00Z",
            "gates": {
                gate: True
                for gate in (
                    "parity",
                    "leakage",
                    "calibration",
                    "schema",
                    "security",
                    "quality",
                )
            },
        }

        payload = build_stock_snapshot(
            pipeline,
            "TW",
            "2330",
            target_market_date=datetime.date(2026, 7, 16),
            promoted_backtest=promoted,
        )

        self.assertEqual(calls, ["latest"])
        self.assertEqual(payload["latest"]["AI_P"], 64.5)
        self.assertEqual(payload["backtest"], promoted)
        self.assertTrue(payload["backtest_compatibility"]["strong_action_allowed"])

    def test_taiwan_snapshot_can_run_explicit_degraded_bootstrap_without_backtest_metrics(self):
        frame = pd.DataFrame(
            {"Close": [100.0, 101.0], "MA20": [99.0, 99.5], "AI_P": [None, None]},
            index=pd.to_datetime(["2026-07-15", "2026-07-16"]),
        )
        frame.index.name = "Date"
        pipeline = SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: data,
            run_latest_inference=lambda data: data.__setitem__("AI_P", [62.0]) or {
                "model_version": "lgbm-5d-v1",
                "probability": 62.0,
            },
            run_ai_engine=lambda _data: self.fail("bootstrap must not walk forward"),
            get_stock_name=lambda symbol: symbol,
            PREDICTION_HORIZON=5,
        )
        payload = build_stock_snapshot(
            pipeline,
            "TW",
            "2330",
            target_market_date=datetime.date(2026, 7, 15),
            degraded_bootstrap=True,
        )
        self.assertEqual(payload["backtest"], {})
        self.assertEqual(payload["as_of"], "2026-07-15")
        self.assertEqual(len(payload["daily"]), 1)
        self.assertEqual(
            payload["backtest_compatibility"]["reason"],
            "initial_backtest_bootstrap",
        )
        self.assertFalse(payload["backtest_compatibility"]["strong_action_allowed"])

    def test_taiwan_observation_snapshot_skips_prediction_and_removes_model_columns(self):
        frame = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "MA20": [99.0, 99.5],
                "AI_P": [88.0, 99.0],
                "FUTURE_RET_5": [0.1, 0.2],
                "T": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-07-15", "2026-07-16"]),
        )
        frame.index.name = "Date"
        pipeline = SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: data,
            run_latest_inference=lambda _data: self.fail(
                "Observation source must not run latest inference"
            ),
            run_ai_engine=lambda _data: self.fail(
                "Observation source must not run walk-forward backtest"
            ),
            get_stock_name=lambda symbol: symbol,
            PREDICTION_HORIZON=5,
        )

        payload = build_stock_snapshot(
            pipeline,
            "TW",
            "2330",
            target_market_date=datetime.date(2026, 7, 16),
            observation_only=True,
        )

        self.assertEqual(payload["model_version"], "observation-source-v1")
        self.assertEqual(payload["backtest"], {})
        self.assertNotIn("feature_schema_version", payload)
        self.assertNotIn("recommendation_policy_version", payload)
        for row in payload["daily"]:
            self.assertNotIn("AI_P", row)
            self.assertNotIn("FUTURE_RET_5", row)
            self.assertNotIn("T", row)

    def test_us_snapshot_reuses_existing_pipeline(self):
        frame = pd.DataFrame(
            {"Close": [100.0], "AI_P": [63.0]},
            index=pd.to_datetime(["2026-07-03"]),
        )
        frame.index.name = "Date"
        calls = []
        pipeline = SimpleNamespace(
            get_data=lambda symbol, days: calls.append((symbol, days)) or frame.copy(),
            calc_all=lambda data: data,
            run_ai_engine=lambda _data: {"accuracy": 55.0},
            get_stock_name=lambda symbol: f"美股 {symbol}",
            PREDICTION_HORIZON=5,
        )

        payload = build_stock_snapshot(pipeline, "US", "AAPL")

        self.assertEqual(calls, [("AAPL", 730)])
        self.assertEqual(payload["name"], "美股 AAPL")
        self.assertEqual(payload["latest"]["AI_P"], 63.0)

    def test_pipeline_loader_keeps_cloud_secrets_out_of_local_app_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_paths = []
            fake_yfinance = SimpleNamespace(
                set_tz_cache_location=lambda path: cache_paths.append(path)
            )
            pipeline = object()

            def import_app(name):
                self.assertEqual(name, "app")
                self.assertNotIn("GEMINI_API_KEY", os.environ)
                self.assertNotIn("GCP_PROJECT_ID", os.environ)
                return pipeline

            environment = {
                "GEMINI_API_KEY": "must-not-reach-local-app",
                "GCP_PROJECT_ID": "must-not-reach-local-app",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.dict(sys.modules, {"yfinance": fake_yfinance}),
                patch.object(local_quant.importlib, "import_module", side_effect=import_app),
            ):
                result = local_quant.load_stock_pipeline(root)
                self.assertEqual(os.environ["GEMINI_API_KEY"], environment["GEMINI_API_KEY"])
                self.assertEqual(os.environ["GCP_PROJECT_ID"], environment["GCP_PROJECT_ID"])

            self.assertIs(result, pipeline)
            self.assertEqual(cache_paths, [str(root / "cache" / "yfinance")])


if __name__ == "__main__":
    unittest.main()
