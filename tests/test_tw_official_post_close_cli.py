import datetime
import gzip
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock

import pandas as pd

from stock_papi.batch.calendar import TWSE_CALENDAR_URL, TradingCalendarSet
from stock_papi.batch.tw_official_post_close_cli import (
    _load_calendar_set,
    _required_trading_dates,
    run,
)
from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialDailySnapshot,
    OfficialRequestBudget,
)
from stock_papi.integrations.market_data.tw_official_historical import (
    OfficialSnapshotSeries,
)

TARGET = datetime.date(2026, 7, 24)


def daily_snapshot(value, price_symbols=("2330", "2303")):
    price = {
        symbol: MappingProxyType({
            "date": value.isoformat(), "stock_id": symbol,
            "open": 1.0, "max": 1.0, "min": 1.0,
            "close": 1.0, "Trading_Volume": 1.0,
        })
        for symbol in price_symbols
    }
    return OfficialDailySnapshot(
        target_date=value,
        price_by_symbol=MappingProxyType(price),
        institutional_by_symbol=MappingProxyType({}),
        margin_by_symbol=MappingProxyType({}),
        source_results=MappingProxyType({}),
        manifest_sha256=("a" if value == TARGET else "b") * 64,
        request_count=6,
        request_budget=OfficialRequestBudget(6, 12, 6, 0, True, "capacity_proven"),
        source_mode="tw_official_bulk_v2",
        source_schema_version="tw-official-historical-v1",
    )


def snapshot_series(dates=(TARGET,), price_symbols=("2330", "2303")):
    snapshots = {value: daily_snapshot(value, price_symbols) for value in dates}
    digest = hashlib.sha256(
        json.dumps(
            [(value.isoformat(), snapshots[value].manifest_sha256) for value in dates]
        ).encode()
    ).hexdigest()
    return OfficialSnapshotSeries(
        target_date=max(dates),
        snapshots=MappingProxyType(snapshots),
        manifest_sha256=digest,
        request_count=6 * len(dates),
        request_budget=OfficialRequestBudget(
            6 * len(dates), 12 * len(dates), 6 * len(dates), 0,
            True, "capacity_proven",
        ),
    )


def write_calendar(root, year=2026):
    path = Path(root) / f"TW-{year}.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "market": "TW",
        "year": year,
        "source_url": TWSE_CALENDAR_URL,
        "fetched_at": f"{year}-01-01T00:00:00+00:00",
        "source_sha256": "c" * 64,
        "valid_from": f"{year}-01-01",
        "valid_to": f"{year}-12-31",
        "closed_dates": [],
        "special_open_dates": [],
    }), encoding="utf-8")
    return path


def write_artifact(root, symbol, as_of="2026-07-23"):
    path = Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "market": "TW",
        "symbol": symbol,
        "as_of": as_of,
        "daily": [{
            "Date": f"{as_of}T00:00:00.000",
            "Open": 1.0, "High": 1.0, "Low": 1.0,
            "Close": 1.0, "Volume": 1.0,
        }],
    }
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(document).encode())


class Pipeline:
    pd = pd
    industry_map = {"全市場": ["2330", "2303"]}

    @staticmethod
    def fetch_finmind_dataset(*_args):
        raise AssertionError("original FinMind fetch must not run")


class TWOfficialPostCloseCLITests(unittest.TestCase):
    def test_calendar_lists_every_missing_trading_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendars = _load_calendar_set([write_calendar(temporary)])
            dates = _required_trading_dates(
                calendars,
                earliest_latest_date=datetime.date(2026, 7, 16),
                target_market_date=TARGET,
            )
        self.assertEqual(
            dates,
            (
                datetime.date(2026, 7, 17),
                datetime.date(2026, 7, 20),
                datetime.date(2026, 7, 21),
                datetime.date(2026, 7, 22),
                datetime.date(2026, 7, 23),
                TARGET,
            ),
        )

    def test_catchup_over_ten_sessions_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendars = _load_calendar_set([write_calendar(temporary)])
            with self.assertRaises(ValueError):
                _required_trading_dates(
                    calendars,
                    earliest_latest_date=datetime.date(2026, 7, 1),
                    target_market_date=TARGET,
                )

    def test_prefetches_series_enriches_identity_and_restores_patches(self):
        pipeline = Pipeline()
        original_fetch = pipeline.fetch_finmind_dataset
        observed = {}
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2303", "2330"]
        module.load_stock_pipeline = lambda _root: pipeline

        def original_batch(
            root, market, symbols, analyze, *args, batch_identity=None, **kwargs
        ):
            observed["identity"] = batch_identity
            return {
                "next_index": 0, "failed": [],
                "pending": [], "excluded": [],
            }

        module.run_market_batch = original_batch
        module.build_stock_snapshot = (
            lambda _pipeline, market, symbol, *args, **kwargs: {"symbol": symbol}
        )

        def local_main(argv):
            self.assertIn("--observation-only", argv)
            module.run_market_batch(
                Path("x"), "TW", ["2303", "2330"], lambda _symbol: {},
                batch_identity={
                    "target_market_date": TARGET.isoformat(),
                    "product_mode": "observation",
                },
            )
            payload = module.build_stock_snapshot(pipeline, "TW", "2330")
            observed["lineage"] = payload["source_lineage"]
            self.assertNotEqual(pipeline.fetch_finmind_dataset, original_fetch)
            return 0

        module.main = local_main
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        builder = Mock(return_value=snapshot_series())
        try:
            with tempfile.TemporaryDirectory() as temporary:
                for symbol in ("2303", "2330"):
                    write_artifact(temporary, symbol)
                calendar = write_calendar(temporary)
                result = run(
                    root=Path(temporary),
                    target_market_date=TARGET,
                    calendar_artifacts=[calendar],
                    limit=5000,
                    delay=0.5,
                    series_builder=builder,
                )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old

        self.assertEqual(result, 0)
        builder.assert_called_once()
        called_dates = builder.call_args.args[1]
        self.assertEqual(called_dates, (TARGET,))
        self.assertEqual(
            observed["identity"]["source_mode"], "tw_official_bulk_v2"
        )
        self.assertEqual(
            observed["identity"]["official_series_manifest_sha256"],
            snapshot_series().manifest_sha256,
        )
        self.assertEqual(
            observed["identity"]["official_snapshot_dates"],
            ["2026-07-24"],
        )
        self.assertEqual(
            observed["lineage"]["source_mode"], "tw_official_bulk_v2"
        )
        self.assertEqual(pipeline.fetch_finmind_dataset, original_fetch)
        self.assertIs(module.run_market_batch, original_batch)

    def test_source_failure_occurs_before_local_main_or_batch(self):
        pipeline = Pipeline()
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2330", "2303"]
        module.load_stock_pipeline = lambda _root: pipeline
        module.main = Mock(return_value=0)
        module.run_market_batch = Mock()
        module.build_stock_snapshot = Mock()
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        try:
            with tempfile.TemporaryDirectory() as temporary:
                for symbol in ("2303", "2330"):
                    write_artifact(temporary, symbol)
                calendar = write_calendar(temporary)
                with self.assertRaises(RuntimeError):
                    run(
                        root=Path(temporary),
                        target_market_date=TARGET,
                        calendar_artifacts=[calendar],
                        limit=1,
                        delay=0,
                        series_builder=lambda *_args: (
                            _ for _ in ()
                        ).throw(RuntimeError("source unavailable")),
                    )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old
        module.main.assert_not_called()
        module.run_market_batch.assert_not_called()

    def test_historical_coverage_failure_occurs_before_network(self):
        pipeline = Pipeline()
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2330", "2303"]
        module.load_stock_pipeline = lambda _root: pipeline
        module.main = Mock(return_value=0)
        module.run_market_batch = Mock()
        module.build_stock_snapshot = Mock()
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        builder = Mock(return_value=snapshot_series())
        try:
            with tempfile.TemporaryDirectory() as temporary:
                write_artifact(temporary, "2330")
                calendar = write_calendar(temporary)
                with self.assertRaises(RuntimeError):
                    run(
                        root=Path(temporary),
                        target_market_date=TARGET,
                        calendar_artifacts=[calendar],
                        limit=1,
                        delay=0,
                        series_builder=builder,
                    )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old
        builder.assert_not_called()
        module.main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
