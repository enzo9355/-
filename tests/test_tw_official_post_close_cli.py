import datetime
import csv
import gzip
import hashlib
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock, patch

import pandas as pd

import local_quant

from stock_papi.batch.calendar import TWSE_CALENDAR_URL, TradingCalendarSet
from stock_papi.batch import tw_official_post_close_cli as cli
from stock_papi.batch.tw_official_post_close_cli import (
    _load_calendar_set,
    _patched_pipeline,
    _plan_recovery_stage,
    _required_symbols_by_exchange,
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
from stock_papi.integrations.market_data.tw_trading_status import evidence_sha256
from stock_papi.quant.tw_incremental import (
    IncrementalHistoryError,
    OfficialCompatFetcher,
    load_incremental_artifact,
)
from stock_papi.quant.tw_legacy_reconciliation import (
    LegacyArtifactBackupStore,
    resolve_truncated_daily_history,
)

TARGET = datetime.date(2026, 7, 24)
BASELINE = datetime.date(2026, 7, 16)
FULL_SERIES_DATES = (
    BASELINE,
    datetime.date(2026, 7, 17),
    datetime.date(2026, 7, 20),
    datetime.date(2026, 7, 21),
    datetime.date(2026, 7, 22),
    datetime.date(2026, 7, 23),
    TARGET,
)
EXCLUSION_FIELDS = [
    "Symbol", "Name", "ExclusionDate", "ConsecutiveFailures",
    "State", "Type", "Reason", "OperatorAction",
]


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
        source_schema_version="tw-official-historical-v2",
    )


def snapshot_series(dates=(TARGET,), price_symbols=("2330", "2303")):
    snapshots = {value: daily_snapshot(value, price_symbols) for value in dates}
    manifest_document = {
        "source_mode": "tw_official_bulk_v2",
        "source_schema_version": "tw-official-historical-v2",
        "target_date": max(dates).isoformat(),
        "snapshots": [
            {
                "date": value.isoformat(),
                "manifest_sha256": snapshots[value].manifest_sha256,
            }
            for value in dates
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            manifest_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
        source_schema_version="tw-official-historical-v2",
    )


def status_snapshot_series():
    status = {
        "schema_version": 1,
        "status": "official_no_regular_trade",
        "market": "TW",
        "exchange": "TWSE",
        "symbol": "2303",
        "target_market_date": TARGET.isoformat(),
        "source_id": "twse_price",
        "payload_sha256": "d" * 64,
        "raw_row_sha256": "e" * 64,
        "raw_fields": {"symbol": "2303", "name": "聯電", "open": "--", "high": "--", "low": "--", "close": "--", "volume": "0"},
        "parser_version": "tw-official-historical-parser-v3",
    }
    status["evidence_sha256"] = evidence_sha256(status)
    snapshot = daily_snapshot(TARGET, ("2330",))
    snapshot = OfficialDailySnapshot(
        **{
            **snapshot.__dict__,
            "source_schema_version": "tw-official-historical-v3",
            "trading_status_by_symbol": MappingProxyType({
                "2303": MappingProxyType(status)
            }),
        }
    )
    document = {
        "source_mode": snapshot.source_mode,
        "source_schema_version": snapshot.source_schema_version,
        "target_date": TARGET.isoformat(),
        "snapshots": [{"date": TARGET.isoformat(), "manifest_sha256": snapshot.manifest_sha256}],
    }
    return OfficialSnapshotSeries(
        target_date=TARGET,
        snapshots=MappingProxyType({TARGET: snapshot}),
        manifest_sha256=hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        request_count=8,
        request_budget=OfficialRequestBudget(8, 16, 8, 0, True, "capacity_proven"),
        source_schema_version="tw-official-historical-v3",
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


def write_artifact(
    root, symbol, as_of="2026-07-23", source_lineage=None, *, lineage_present=False
):
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
    if lineage_present or source_lineage is not None:
        document["source_lineage"] = source_lineage
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(document).encode())
    return path


def write_payload_artifact(root, symbol, payload):
    path = Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload, market="TW", symbol=str(symbol))
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(document, separators=(",", ":")).encode())
    return path


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def write_checkpoint(root, document):
    path = Path(root) / "checkpoints" / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def write_exclusions(root, rows, fields=EXCLUSION_FIELDS):
    path = Path(root) / "checkpoints" / "exclusion_list-TW.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def exclusion_row(symbol, state="Excluded", action=""):
    return {
        "Symbol": symbol,
        "Name": symbol,
        "ExclusionDate": TARGET.isoformat(),
        "ConsecutiveFailures": "1",
        "State": state,
        "Type": "delisted",
        "Reason": "test",
        "OperatorAction": action,
    }
def reconciliation_evidence(original_sha, series):
    dates = [value.isoformat() for value in series.dates]
    manifests = [
        {"date": value.isoformat(), "manifest_sha256": snapshot.manifest_sha256}
        for value, snapshot in series.snapshots.items()
    ]
    return {
        "schema_version": 2,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": original_sha,
        "legacy_artifact_as_of": BASELINE.isoformat(),
        "official_source_mode": series.source_mode,
        "official_source_schema_version": series.source_schema_version,
        "official_series_manifest_sha256": series.manifest_sha256,
        "official_snapshot_dates": dates,
        "official_snapshot_manifests": manifests,
        "overlap_dates": [BASELINE.isoformat()],
        "price_replaced_dates": [BASELINE.isoformat()],
        "price_preserved_no_official_row_dates": [],
        "institutional_replaced_dates": [],
        "institutional_preserved_no_official_row_dates": [BASELINE.isoformat()],
        "margin_replaced_dates": [],
        "margin_preserved_no_official_row_dates": [BASELINE.isoformat()],
        "date_evidence": [{
            "date": BASELINE.isoformat(),
            "price_action": "replaced_official",
            "institutional_action": "preserved_legacy_no_official_row",
            "margin_action": "preserved_legacy_no_official_row",
        }],
    }


def official_lineage(symbol, series):
    return {
        "source_mode": series.source_mode,
        "source_schema_version": series.source_schema_version,
        "symbol": symbol,
        "target_market_date": series.target_date.isoformat(),
        "historical_as_of": series.target_date.isoformat(),
        "historical_artifact_sha256": "c" * 64,
        "official_target_price_available": (
            symbol in series.snapshots[series.target_date].price_by_symbol
        ),
        "official_snapshot_dates": [value.isoformat() for value in series.dates],
        "official_snapshot_manifests": [
            {
                "date": value.isoformat(),
                "manifest_sha256": snapshot.manifest_sha256,
            }
            for value, snapshot in series.snapshots.items()
        ],
        "official_series_manifest_sha256": series.manifest_sha256,
    }


def write_official_artifact(root, symbol, series, evidence):
    path = write_artifact(root, symbol, TARGET.isoformat())
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    document["source_lineage"] = {
        "source_mode": series.source_mode,
        "source_schema_version": series.source_schema_version,
        "symbol": symbol,
        "target_market_date": TARGET.isoformat(),
        "historical_as_of": BASELINE.isoformat(),
        "historical_artifact_sha256": evidence["legacy_artifact_sha256"],
        "official_target_price_available": True,
        "official_snapshot_dates": evidence["official_snapshot_dates"],
        "official_snapshot_manifests": evidence["official_snapshot_manifests"],
        "official_series_manifest_sha256": series.manifest_sha256,
        "legacy_reconciliation": evidence,
    }
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(document, separators=(",", ":")).encode())
    return path


class Pipeline:
    pd = pd
    industry_map = {"全市場": ["2330", "2303"]}

    @staticmethod
    def fetch_finmind_dataset(*_args):
        raise AssertionError("original FinMind fetch must not run")


class TWOfficialPostCloseCLITests(unittest.TestCase):
    def test_universe_exchange_partition_uses_catalog_metadata_not_symbols(self):
        registry = {
            "2330": types.SimpleNamespace(data_source="twse"),
            "6488": types.SimpleNamespace(data_source="tpex"),
        }

        partition = _required_symbols_by_exchange(
            ["6488", "2330"], registry=registry
        )

        self.assertEqual(partition, {"TWSE": {"2330"}, "TPEx": {"6488"}})
        with self.assertRaisesRegex(RuntimeError, "exchange metadata"):
            _required_symbols_by_exchange(["9999"], registry=registry)

    def _run_fake(
        self,
        root,
        *,
        reconcile=False,
        recover=False,
        series=None,
        status=0,
        final_dates=None,
        failed=None,
        checkpoint_changes=None,
        writer_call=None,
        symbols=("2303", "2330"),
        builder=None,
        final_status_symbols=(),
        recovery_symbol_allowlist=None,
    ):
        root = Path(root)
        pipeline = Pipeline()
        module = types.ModuleType("local_quant")
        module.OBSERVATION_SOURCE_VERSION = "observation-source-v1"
        module.get_taiwan_symbols = lambda _pipeline: list(symbols)
        module.load_stock_pipeline = lambda _root: pipeline
        module.build_stock_snapshot = (
            lambda _pipeline, market, symbol, *args, **kwargs: {"symbol": symbol}
        )
        module.write_stock_artifact = (
            lambda data_root, market, symbol, payload: write_artifact(
                data_root, symbol, payload.get("as_of", TARGET.isoformat())
            )
        )
        observed = {}
        module.publish_market_snapshot = lambda *args, **kwargs: (
            observed.update(
                published_args=args,
                published_kwargs=kwargs,
            )
            or Path(root) / "publish" / "quant" / "v1" / "latest-TW.json"
        )

        def original_batch(
            data_root,
            market,
            batch_symbols,
            analyze,
            *args,
            batch_identity=None,
            **kwargs,
        ):
            observed["identity"] = batch_identity
            if writer_call is not None:
                module.write_stock_artifact(
                    data_root, "TW", writer_call, {"as_of": TARGET.isoformat()}
                )
            for symbol, as_of in (final_dates or {}).items():
                if as_of is None:
                    write_artifact(data_root, symbol).unlink()
                else:
                    path = write_artifact(
                        data_root,
                        symbol,
                        as_of,
                        (
                            None
                            if chosen_series.source_schema_version
                            == "tw-official-historical-v3"
                            else official_lineage(symbol, chosen_series)
                        ),
                    )
                    if chosen_series.source_schema_version == "tw-official-historical-v3":
                        with gzip.open(path, "rt", encoding="utf-8") as stream:
                            document = json.load(stream)
                        document.update(
                            schema_version=2,
                            target_market_date=TARGET.isoformat(),
                            observation_as_of=TARGET.isoformat(),
                            latest_regular_price_date=as_of,
                            observation_kind="regular_price",
                            source_lineage=pipeline.fetch_finmind_dataset.lineage_for(symbol),
                        )
                        with path.open("wb") as raw:
                            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                                stream.write(json.dumps(document, separators=(",", ":")).encode())
            for symbol in final_status_symbols:
                path = Path(data_root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    document = json.load(stream)
                status_evidence = pipeline.fetch_finmind_dataset.status_for(symbol)
                document.update(
                    schema_version=2,
                    target_market_date=TARGET.isoformat(),
                    observation_as_of=TARGET.isoformat(),
                    latest_regular_price_date=document["as_of"],
                    observation_kind=status_evidence["status"],
                    trading_status_evidence=status_evidence,
                    source_lineage=pipeline.fetch_finmind_dataset.lineage_for(symbol),
                )
                with path.open("wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                        stream.write(json.dumps(document, separators=(",", ":")).encode())
            state = {
                "stage": "market_batch",
                "market": "TW",
                "next_index": len(batch_symbols),
                "failed": list(failed or []),
                "batch_identity": batch_identity,
            }
            state.update(checkpoint_changes or {})
            write_checkpoint(data_root, state)
            return state

        module.run_market_batch = original_batch

        def local_main(argv):
            if status:
                return status
            module.run_market_batch(
                root,
                "TW",
                list(symbols),
                lambda _symbol: {},
                batch_identity={
                    "target_market_date": TARGET.isoformat(),
                    "product_mode": "observation",
                    "source_version": module.OBSERVATION_SOURCE_VERSION,
                },
            )
            return 0

        module.main = local_main
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        chosen_series = series or snapshot_series()
        chosen_builder = builder or Mock(return_value=chosen_series)
        try:
            result = run(
                root=root,
                target_market_date=TARGET,
                calendar_artifacts=[write_calendar(root)],
                limit=5000,
                delay=0,
                series_builder=chosen_builder,
                reconcile_legacy_overlaps=reconcile,
                recover_truncated_history=recover,
                recovery_symbol_allowlist=recovery_symbol_allowlist,
            )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old
        return result, observed, chosen_builder, module

    @staticmethod
    def _official_payload(fetcher, symbol):
        price = fetcher(
            "TaiwanStockPrice", symbol, BASELINE.isoformat(), TARGET.isoformat()
        )
        fetcher(
            "TaiwanStockInstitutionalInvestorsBuySell",
            symbol,
            BASELINE.isoformat(),
            TARGET.isoformat(),
        )
        fetcher(
            "TaiwanStockMarginPurchaseShortSale",
            symbol,
            BASELINE.isoformat(),
            TARGET.isoformat(),
        )
        daily = [
            {
                "Date": f"{row.date}T00:00:00.000",
                "Open": float(row.open),
                "High": float(row.max),
                "Low": float(row.min),
                "Close": float(row.close),
                "Volume": float(row.Trading_Volume),
            }
            for row in price.itertuples(index=False)
        ]
        return {
            "schema_version": 1,
            "as_of": daily[-1]["Date"][:10],
            "daily": daily,
        }

    def _run_recovery_runtime(
        self,
        root,
        *,
        series,
        reconcile,
        recover,
    ):
        root = Path(root)
        pipeline = Pipeline()
        events = []
        self._recovery_runtime_events = events
        module = types.ModuleType("local_quant")
        module.OBSERVATION_SOURCE_VERSION = "observation-source-v1"
        module.get_taiwan_symbols = lambda _pipeline: ["2330"]
        module.load_stock_pipeline = lambda _root: pipeline

        def original_build(pipeline_arg, market, symbol, *_args, **_kwargs):
            events.append("build")
            return self._official_payload(
                pipeline_arg.fetch_finmind_dataset, str(symbol)
            )

        def original_writer(data_root, market, symbol, payload, *_args, **_kwargs):
            self.assertEqual(market, "TW")
            events.append("write")
            return write_payload_artifact(data_root, symbol, payload)

        def original_batch(
            data_root,
            market,
            symbols,
            analyze,
            *_args,
            batch_identity=None,
            **_kwargs,
        ):
            failures = []
            for index, symbol in enumerate(symbols, start=1):
                try:
                    payload = module.build_stock_snapshot(
                        pipeline,
                        market,
                        str(symbol),
                        target_market_date=TARGET,
                        observation_only=True,
                    )
                    module.write_stock_artifact(data_root, market, str(symbol), payload)
                except Exception as exc:
                    failures.append({"symbol": str(symbol), "error": type(exc).__name__})
                checkpoint = {
                    "stage": "market_batch",
                    "market": market,
                    "next_index": index,
                    "failed": list(failures),
                    "batch_identity": batch_identity,
                }
                write_checkpoint(data_root, checkpoint)
                events.append("checkpoint")
            return checkpoint

        module.build_stock_snapshot = original_build
        module.write_stock_artifact = original_writer
        module.run_market_batch = original_batch
        module.publish_market_snapshot = lambda *_args, **_kwargs: events.append("publish")

        def local_main(_argv):
            module.run_market_batch(
                root,
                "TW",
                ["2330"],
                lambda _symbol: {},
                batch_identity={
                    "target_market_date": TARGET.isoformat(),
                    "product_mode": "observation",
                    "source_version": module.OBSERVATION_SOURCE_VERSION,
                },
            )
            return 0

        module.main = local_main
        previous = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        try:
            with patch.object(
                cli,
                "_required_symbols_by_exchange",
                return_value={"TWSE": {"2330"}, "TPEx": set()},
            ):
                result = run(
                    root=root,
                    target_market_date=TARGET,
                    calendar_artifacts=[write_calendar(root)],
                    limit=1,
                    delay=0,
                    series_builder=Mock(return_value=series),
                    reconcile_legacy_overlaps=reconcile,
                    recover_truncated_history=recover,
                )
        finally:
            if previous is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = previous
        return result, events

    def _prepare_truncated_direct_artifact(self, root, series):
        root = Path(root)
        original = write_artifact(root, "2330", BASELINE.isoformat())
        direct_fetcher = OfficialCompatFetcher(
            root,
            series,
            pd=pd,
            legacy_overlap_policy="replace_verified_legacy",
        )
        direct_payload = self._official_payload(direct_fetcher, "2330")
        direct_lineage = direct_fetcher.lineage_for(
            "2330", persisted_daily=direct_payload["daily"]
        )
        store = LegacyArtifactBackupStore(
            root,
            target_date=TARGET,
            series_manifest_sha256=series.manifest_sha256,
        )
        self.assertEqual(
            store.backup_before_write(
                symbol="2330",
                artifact_path=original,
                evidence=direct_lineage["legacy_reconciliation"],
            ),
            "write",
        )
        direct_payload["daily"] = [direct_payload["daily"][-1]]
        direct_path = write_payload_artifact(
            root,
            "2330",
            {**direct_payload, "source_lineage": direct_lineage},
        )
        store.mark_applied(symbol="2330", artifact_path=direct_path)
        return direct_path, hashlib.sha256(direct_path.read_bytes()).hexdigest()

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

    def test_segment_plan_uses_observation_date_and_preserves_request_bound(self):
        target = datetime.date(2026, 7, 29)
        with tempfile.TemporaryDirectory() as temporary:
            calendars = _load_calendar_set([write_calendar(temporary)])
            first = types.SimpleNamespace(
                observation_by_symbol=MappingProxyType({
                    "1111": datetime.date(2026, 7, 8),
                    "2222": datetime.date(2026, 7, 16),
                    "3333": datetime.date(2026, 7, 24),
                })
            )
            stage_target, stage_symbols, baseline = _plan_recovery_stage(
                calendars,
                first,
                symbols=["1111", "2222", "3333"],
                target_market_date=target,
                reconcile_legacy_overlaps=True,
            )
            second = types.SimpleNamespace(
                observation_by_symbol=MappingProxyType({
                    "1111": stage_target,
                    "2222": stage_target,
                    "3333": datetime.date(2026, 7, 24),
                })
            )
            final_target, final_symbols, final_baseline = _plan_recovery_stage(
                calendars,
                second,
                symbols=["1111", "2222", "3333"],
                target_market_date=target,
                reconcile_legacy_overlaps=True,
            )

        self.assertEqual(stage_target, datetime.date(2026, 7, 21))
        self.assertEqual(stage_symbols, ["1111", "2222"])
        self.assertEqual(baseline, datetime.date(2026, 7, 8))
        self.assertEqual(final_target, target)
        self.assertEqual(final_symbols, ["1111", "2222", "3333"])
        self.assertEqual(final_baseline, stage_target)

    def test_cli_runs_partial_stages_without_publishing_them(self):
        target = datetime.date(2026, 7, 29)
        pipeline = Pipeline()
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2303", "2330"]
        module.load_stock_pipeline = lambda _root: pipeline
        first = types.SimpleNamespace(
            latest_by_symbol=MappingProxyType({
                "2303": datetime.date(2026, 7, 8),
                "2330": datetime.date(2026, 7, 24),
            }),
            observation_by_symbol=MappingProxyType({
                "2303": datetime.date(2026, 7, 8),
                "2330": datetime.date(2026, 7, 24),
            }),
            unavailable_symbols=(),
        )
        second = types.SimpleNamespace(
            latest_by_symbol=first.latest_by_symbol,
            observation_by_symbol=MappingProxyType({
                "2303": datetime.date(2026, 7, 21),
                "2330": datetime.date(2026, 7, 24),
            }),
            unavailable_symbols=(),
        )
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        try:
            with tempfile.TemporaryDirectory() as temporary, patch.object(
                cli, "audit_artifact_dates", side_effect=[first, second]
            ), patch.object(
                cli, "_run_stage", side_effect=[(0, set()), (0, set())]
            ) as stage:
                result = run(
                    root=Path(temporary),
                    target_market_date=target,
                    calendar_artifacts=[write_calendar(temporary)],
                    limit=5000,
                    delay=0,
                    reconcile_legacy_overlaps=True,
                )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old

        self.assertEqual(result, 0)
        self.assertEqual(stage.call_count, 2)
        partial = stage.call_args_list[0].kwargs
        final = stage.call_args_list[1].kwargs
        self.assertEqual(partial["target_market_date"], datetime.date(2026, 7, 21))
        self.assertEqual(partial["symbols"], ["2303"])
        self.assertFalse(partial["publish"])
        self.assertEqual(final["target_market_date"], target)
        self.assertEqual(final["symbols"], ["2303", "2330"])
        self.assertTrue(final["publish"])

    def test_cli_reconciliation_flag_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendar = write_calendar(temporary)
            arguments = [
                "--root", temporary,
                "--target-market-date", TARGET.isoformat(),
                "--calendar-artifact", str(calendar),
            ]
            with patch.object(cli, "run", return_value=0) as run_mock:
                self.assertEqual(cli.main(arguments), 0)
                self.assertFalse(
                    run_mock.call_args.kwargs["reconcile_legacy_overlaps"]
                )
                self.assertEqual(
                    cli.main(arguments + ["--reconcile-legacy-overlaps"]), 0
                )
                self.assertTrue(
                    run_mock.call_args.kwargs["reconcile_legacy_overlaps"]
                )

    def test_cli_recovery_flag_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendar = write_calendar(temporary)
            arguments = [
                "--root", temporary,
                "--target-market-date", TARGET.isoformat(),
                "--calendar-artifact", str(calendar),
            ]
            with patch.object(cli, "run", return_value=0) as run_mock:
                self.assertEqual(cli.main(arguments), 0)
                self.assertIs(
                    run_mock.call_args.kwargs["recover_truncated_history"], False
                )
                self.assertEqual(
                    cli.main(arguments + ["--recover-truncated-history"]), 0
)
                self.assertIs(
                    run_mock.call_args.kwargs["recover_truncated_history"], True
                )

    def test_allowlist_loader_validates_symbols_comments_and_sha_identity(self):
        from stock_papi.batch.tw_official_post_close_cli import (
            _load_recovery_symbol_allowlist,
            _universe_sha256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "allow.txt"
            symbols = ["0051", "0050", "# header", "", "0053"]
            path.write_text("\n".join(symbols), encoding="utf-8")
            canonical = ["0050", "0051", "0053"]
            expected = _universe_sha256(canonical)
            loaded = _load_recovery_symbol_allowlist(
                path, expected_sha256=expected
            )
            self.assertEqual(loaded, {"0050", "0051", "0053"})
            with self.assertRaises(ValueError):
                _load_recovery_symbol_allowlist(
                    path, expected_sha256="0" * 64
                )
            bad = Path(temporary) / "bad.txt"
            bad.write_text("0050\nnotasymbol\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_recovery_symbol_allowlist(bad, expected_sha256=None)
            dup = Path(temporary) / "dup.txt"
            dup.write_text("0050\n0050\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_recovery_symbol_allowlist(dup, expected_sha256=None)

    def test_cli_recovery_symbol_allowlist_is_threaded_to_run(self):
        from stock_papi.batch.tw_official_post_close_cli import _universe_sha256
        with tempfile.TemporaryDirectory() as temporary:
            calendar = write_calendar(temporary)
            allow = Path(temporary) / "allow.txt"
            allow.write_text("0051\n# header\n0050\n", encoding="utf-8")
            expected = _universe_sha256(["0050", "0051"])
            arguments = [
                "--root", temporary,
                "--target-market-date", TARGET.isoformat(),
                "--calendar-artifact", str(calendar),
                "--recover-truncated-history",
                "--recovery-symbol-allowlist", str(allow),
                "--recovery-allowlist-sha256", expected,
            ]
            with patch.object(cli, "run", return_value=0) as run_mock:
                self.assertEqual(cli.main(arguments), 0)
                self.assertEqual(
                    run_mock.call_args.kwargs["recovery_symbol_allowlist"],
                    {"0050", "0051"},
                )

    def test_cli_recovery_allowlist_requires_identity_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendar = write_calendar(temporary)
            allow = Path(temporary) / "allow.txt"
            allow.write_text("0050\n", encoding="utf-8")
            arguments = [
                "--root", temporary,
                "--target-market-date", TARGET.isoformat(),
                "--calendar-artifact", str(calendar),
                "--recover-truncated-history",
                "--recovery-symbol-allowlist", str(allow),
            ]
            with patch.object(cli, "run", return_value=0):
                self.assertEqual(cli.main(arguments), 2)

    def test_cli_default_path_never_constructs_resolver_or_touches_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with patch.object(
                cli,
                "resolve_truncated_daily_history",
                create=True,
            ) as resolver:
                result, _observed, _builder, _module = self._run_fake(
                    temporary,
                    recover=False,
                    final_dates={
                        "2303": TARGET.isoformat(), "2330": TARGET.isoformat()
                    },
                )
        self.assertEqual(result, 0)
        resolver.assert_not_called()
        self.assertFalse((Path(temporary) / "quarantine").exists())

    def test_cli_wires_recovery_resolver_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with (
                patch.object(
                    cli,
                    "resolve_truncated_daily_history",
                    create=True,
                    return_value=None,
                ) as resolver,
                patch.object(
                    cli, "OfficialCompatFetcher", wraps=OfficialCompatFetcher
                ) as constructor,
            ):
                result, _observed, _builder, _module = self._run_fake(
                    temporary,
                    recover=True,
                    final_dates={
                        "2303": TARGET.isoformat(), "2330": TARGET.isoformat()
},
                )
                recovery_resolver = constructor.call_args.kwargs[
                    "recovery_resolver"
                ]
                recovery_resolver("2330", object())
        self.assertEqual(result, 0)
        resolver.assert_called_once()

    def test_cli_recovery_allowlist_gates_fallback_to_allowed_symbols_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            calls = []

            def tracking_resolver(root, symbol, artifact):
                calls.append(symbol)
                return None

            with (
                patch.object(
                    cli,
                    "resolve_truncated_daily_history",
                    create=True,
                    side_effect=tracking_resolver,
                ),
                patch.object(
                    cli, "OfficialCompatFetcher", wraps=OfficialCompatFetcher
                ) as constructor,
            ):
                result, _observed, _builder, _module = self._run_fake(
                    temporary,
                    recover=True,
                    symbols=("2303", "2330"),
                    recovery_symbol_allowlist={"2330"},
                    final_dates={
                        "2303": TARGET.isoformat(), "2330": TARGET.isoformat()
                    },
                )
                gated_resolver = constructor.call_args.kwargs[
                    "recovery_resolver"
                ]
                self.assertIsNone(gated_resolver("2303", object()))
                gated_resolver("2330", object())
        self.assertEqual(result, 0)
        self.assertEqual(calls, ["2330"])

    def test_cli_checkpoint_identity_rejects_changed_recovery_mode(self):
        series = snapshot_series()
        audit = types.SimpleNamespace(
            latest_date_counts=MappingProxyType({BASELINE.isoformat(): 1}),
            unavailable_symbols=(),
        )
        common = {
            "target_market_date": TARGET.isoformat(),
            "product_mode": "observation",
        }
        disabled = cli._enrich_batch_identity(
            common,
            series=series,
            audit=audit,
            symbols=["2330"],
            reconcile_legacy_overlaps=False,
            recover_truncated_history=False,
        )
        enabled = cli._enrich_batch_identity(
            common,
            series=series,
            audit=audit,
            symbols=["2330"],
            reconcile_legacy_overlaps=False,
            recover_truncated_history=True,
        )

        self.assertIs(disabled["recover_truncated_history"], False)
        self.assertIs(enabled["recover_truncated_history"], True)
        self.assertNotEqual(disabled, enabled)

    def test_both_recovery_and_reconcile_flags_keep_legacy_artifact_in_existing_reconciliation_flow(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        for explicit_null in (False, True):
            with self.subTest(explicit_null=explicit_null), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_artifact(
                    root,
                    "2330",
                    BASELINE.isoformat(),
                    source_lineage=None,
                    lineage_present=explicit_null,
                )
                resolved = []

                def resolver(data_root, symbol, artifact):
                    value = resolve_truncated_daily_history(data_root, symbol, artifact)
                    resolved.append(value)
                    return value

                with patch.object(
                    cli,
                    "resolve_truncated_daily_history",
                    create=True,
                    side_effect=resolver,
                ):
                    result, events = self._run_recovery_runtime(
                        root,
                        series=series,
                        reconcile=True,
                        recover=True,
                    )
                artifact = load_incremental_artifact(root, "2330")
                lineage_value = artifact.document["source_lineage"]
                store = LegacyArtifactBackupStore(
                    root,
                    target_date=TARGET,
                    series_manifest_sha256=series.manifest_sha256,
                )
                manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))

                self.assertEqual(result, 0)
                self.assertEqual(resolved, [None])
                self.assertEqual(events.count("write"), 1)
                self.assertEqual(
                    lineage_value["legacy_reconciliation"]["overlap_dates"],
                    [BASELINE.isoformat()],
                )
                self.assertEqual(manifest["entries"]["2330"]["status"], "applied")

    def test_both_flags_direct_recovery_rotates_history_writes_once_and_reruns_stably(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _direct_path, direct_sha = self._prepare_truncated_direct_artifact(
                root, series
            )
            first_result, first_events = self._run_recovery_runtime(
                root,
                series=series,
                reconcile=True,
                recover=True,
            )
            first_artifact = load_incremental_artifact(root, "2330")
            first_lineage = first_artifact.document["source_lineage"]
            first_bytes = canonical_bytes({
                "daily": first_artifact.document["daily"],
                "daily_history_recovery": first_lineage["daily_history_recovery"],
            })
            second_result, second_events = self._run_recovery_runtime(
                root,
                series=series,
                reconcile=True,
                recover=True,
            )
            second_artifact = load_incremental_artifact(root, "2330")
            second_lineage = second_artifact.document["source_lineage"]

        self.assertEqual(first_result, 0)
        self.assertEqual(first_events.count("write"), 1)
        self.assertNotIn("legacy_reconciliation", first_lineage)
        self.assertEqual(
            first_lineage["legacy_reconciliation_history"][0][
                "reconciled_artifact_sha256"
            ],
            direct_sha,
        )
        self.assertTrue(
            OfficialCompatFetcher._valid_official_lineage(
                first_lineage, first_artifact
            )
        )
        self.assertEqual(second_result, 0)
        self.assertEqual(second_events.count("write"), 1)
        self.assertEqual(
            first_bytes,
            canonical_bytes({
                "daily": second_artifact.document["daily"],
                "daily_history_recovery": second_lineage["daily_history_recovery"],
            }),
        )

    def test_recovery_failure_blocks_assert_complete_and_publication(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = write_artifact(root, "2330", BASELINE.isoformat())
            before = artifact.read_bytes()
            with patch.object(
                cli,
                "resolve_truncated_daily_history",
                create=True,
                side_effect=IncrementalHistoryError("recovery failed"),
            ), self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                self._run_recovery_runtime(
                    root,
                    series=series,
                    reconcile=False,
                    recover=True,
                )
            checkpoint = json.loads(
                (root / "checkpoints" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact.read_bytes(), before)
            self.assertNotIn("write", self._recovery_runtime_events)
            self.assertNotIn("publish", self._recovery_runtime_events)
            self.assertEqual(checkpoint["next_index"], 1)
            self.assertEqual(
                checkpoint["failed"], [{"symbol": "2330", "error": "IncrementalHistoryError"}]
            )

    def _prepare_terminal_gate(
        self, root, series, price_symbols, lineage_symbol="2330", artifact_lineage=None
    ):
        root = Path(root)
        write_artifact(
            root,
            lineage_symbol,
            TARGET.isoformat(),
            (
                artifact_lineage
                if artifact_lineage is not None
                else official_lineage(lineage_symbol, series)
            ),
        )
        expected_identity = cli._enrich_batch_identity(
            {
                "target_market_date": TARGET.isoformat(),
                "product_mode": "observation",
                "source_version": local_quant.OBSERVATION_SOURCE_VERSION,
            },
            series=series,
            audit=Mock(latest_date_counts={}, unavailable_symbols=[]),
            symbols=[lineage_symbol],
            reconcile_legacy_overlaps=False,
            recover_truncated_history=False,
        )
        write_checkpoint(
            root,
            {
                "stage": "market_batch",
                "market": "TW",
                "next_index": 1,
                "failed": [],
                "batch_identity": expected_identity,
            },
        )
        return expected_identity

    def test_assert_complete_scopes_raw_snapshot_symbols_to_universe(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330", "700001"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_identity = self._prepare_terminal_gate(root, series, ("2330", "700001"))
            cli._assert_complete(
                root,
                symbols=["2330"],
                target_market_date=TARGET,
                expected_identity=expected_identity,
                official_series=series,
                applied_reconciliation_artifacts={},
            )

    def test_assert_complete_rerun_with_same_series_is_deterministic(self):
        first = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330", "700001"))
        second = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330", "700001"))
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        for date in FULL_SERIES_DATES:
            self.assertEqual(
                first.snapshots[date].manifest_sha256,
                second.snapshots[date].manifest_sha256,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_identity = self._prepare_terminal_gate(root, first, ("2330", "700001"))
            for series in (first, second):
                cli._assert_complete(
                    root,
                    symbols=["2330"],
                    target_market_date=TARGET,
                    expected_identity=expected_identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_assert_complete_rejects_lineage_from_different_series(self):
        current = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        stale = snapshot_series((TARGET,), price_symbols=("2330",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_identity = self._prepare_terminal_gate(
                root,
                current,
                ("2330",),
                artifact_lineage=official_lineage("2330", stale),
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=["2330"],
                    target_market_date=TARGET,
                    expected_identity=expected_identity,
                    official_series=current,
                    applied_reconciliation_artifacts={},
                )

    def test_assert_complete_rejects_wrong_target_market_date_lineage(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage = dict(
                official_lineage("2330", series),
                target_market_date=BASELINE.isoformat(),
            )
            expected_identity = self._prepare_terminal_gate(
                root, series, ("2330",), artifact_lineage=lineage
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=["2330"],
                    target_market_date=TARGET,
                    expected_identity=expected_identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_assert_complete_rejects_wrong_source_mode_or_schema(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        for field, value in (
            ("source_mode", "legacy_finmind"),
            ("source_schema_version", "tw-official-historical-v1"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lineage = dict(official_lineage("2330", series), **{field: value})
                expected_identity = self._prepare_terminal_gate(
                    root, series, ("2330",), artifact_lineage=lineage
                )
                with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                    cli._assert_complete(
                        root,
                        symbols=["2330"],
                        target_market_date=TARGET,
                        expected_identity=expected_identity,
                        official_series=series,
                        applied_reconciliation_artifacts={},
                    )

    def test_assert_complete_rejects_checkpoint_identity_from_other_series(self):
        series = snapshot_series(FULL_SERIES_DATES, price_symbols=("2330",))
        other = snapshot_series((TARGET,), price_symbols=("2330",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_identity = self._prepare_terminal_gate(root, series, ("2330",))
            incompatible_identity = dict(
                expected_identity,
                official_series_manifest_sha256=other.manifest_sha256,
            )
            write_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 1,
                    "failed": [],
                    "batch_identity": incompatible_identity,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=["2330"],
                    target_market_date=TARGET,
                    expected_identity=expected_identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_terminal_validation_failure_blocks_publication_and_propagates(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, "2303", "2026-07-23")
            write_artifact(temporary, "2330", "2026-07-23")
            observed = {}
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                _result, observed, _builder, _module = self._run_fake(
                    temporary,
                    final_dates={"2330": TARGET.isoformat()},
                )
            self.assertNotIn("published_args", observed)
            self.assertNotIn("published_kwargs", observed)


    def test_cli_default_path_remains_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with patch.object(
                cli, "OfficialCompatFetcher", wraps=OfficialCompatFetcher
            ) as constructor:
                result, observed, _builder, _module = self._run_fake(
                    temporary,
                    final_dates={"2303": TARGET.isoformat(), "2330": TARGET.isoformat()},
                )
        self.assertEqual(result, 0)
        self.assertEqual(
            constructor.call_args.kwargs["legacy_overlap_policy"], "strict"
        )
        self.assertNotIn("legacy_overlap_policy", observed["identity"])
        self.assertIn("historical_latest_date_counts", observed["identity"])

    def test_cli_reconciliation_prepends_baseline_overlap_date(self):
        series = snapshot_series((datetime.date(2026, 7, 23), TARGET))
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with patch.object(
                LegacyArtifactBackupStore,
                "assert_current_state_complete",
                return_value=None,
            ):
                result, observed, builder, _module = self._run_fake(
                    temporary,
                    reconcile=True,
                    series=series,
                    final_dates={
                        "2303": TARGET.isoformat(),
                        "2330": TARGET.isoformat(),
                    },
                )
        self.assertEqual(result, 0)
        self.assertEqual(
            builder.call_args.args[1], (datetime.date(2026, 7, 23), TARGET)
        )
        self.assertEqual(
            observed["identity"]["legacy_overlap_policy"],
            "replace_verified_legacy",
        )
        self.assertNotIn("historical_latest_date_counts", observed["identity"])
        self.assertNotIn("historical_unavailable_count", observed["identity"])

    def test_reconciliation_window_counts_baseline_in_session_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            calendars = _load_calendar_set([write_calendar(temporary)])
            with self.assertRaises(ValueError):
                cli._reconciliation_trading_dates(
                    calendars,
                    baseline_date=datetime.date(2026, 7, 10),
                    target_market_date=TARGET,
                )

    def test_cli_resume_reuses_discovered_baseline_and_series_identity(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(
                    temporary,
                    symbol,
                    TARGET.isoformat(),
                    official_lineage(symbol, series),
                )
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=(series.manifest_sha256, BASELINE),
            ), patch.object(
                LegacyArtifactBackupStore,
                "assert_current_state_complete",
                return_value=None,
            ):
                result, _observed, builder, _module = self._run_fake(
                    temporary,
                    reconcile=True,
                    series=series,
                    final_dates={},
                )
        self.assertEqual(result, 0)
        self.assertEqual(builder.call_args.args[1], FULL_SERIES_DATES)

    def test_cli_resume_uses_earlier_current_audit_baseline(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, "2303", BASELINE.isoformat())
            write_artifact(temporary, "2330", TARGET.isoformat())
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=(series.manifest_sha256, datetime.date(2026, 7, 20)),
            ), patch.object(
                LegacyArtifactBackupStore,
                "assert_current_state_complete",
                return_value=None,
            ):
                result, _observed, builder, _module = self._run_fake(
                    temporary,
                    reconcile=True,
                    series=series,
                    final_dates={
                        "2303": TARGET.isoformat(),
                        "2330": TARGET.isoformat(),
                    },
                )
        self.assertEqual(result, 0)
        self.assertEqual(builder.call_args.args[1], FULL_SERIES_DATES)

    def test_cli_resume_rejects_changed_series_identity(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol, TARGET.isoformat())
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=("f" * 64, BASELINE),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "series does not match resume state"
                ):
                    self._run_fake(
                        temporary,
                        reconcile=True,
                        series=series,
                        final_dates={},
                    )

    def test_cli_reconciliation_patches_and_restores_writer(self):
        events = []
        local = types.SimpleNamespace()
        local.load_stock_pipeline = lambda _root: None
        local.run_market_batch = lambda *_args, **_kwargs: None
        local.build_stock_snapshot = lambda *_args, **_kwargs: {}
        local.write_stock_artifact = (
            lambda *_args, **_kwargs: events.append("writer") or Path("written")
        )
        originals = dict(vars(local))
        pipeline = Pipeline()
        original_fetch = pipeline.fetch_finmind_dataset
        fetcher = Mock()
        fetcher.lineage_for.return_value = {"legacy_reconciliation": {"x": 1}}
        store = Mock()
        store.backup_before_write.side_effect = (
            lambda **_kwargs: events.append("backup") or "write"
        )
        store.mark_applied.side_effect = (
            lambda **_kwargs: events.append("applied") or Path("written")
        )
        with _patched_pipeline(
            local, pipeline, fetcher, snapshot_series(), Mock(), backup_store=store
        ):
            local.write_stock_artifact(Path("root"), "TW", "2330", {})
        self.assertEqual(events, ["backup", "writer", "applied"])
        self.assertIs(pipeline.fetch_finmind_dataset, original_fetch)
        for name in (
            "load_stock_pipeline", "run_market_batch", "build_stock_snapshot",
            "write_stock_artifact",
        ):
            self.assertIs(getattr(local, name), originals[name])

    def test_cli_scopes_and_restores_segment_universe(self):
        original_symbols = lambda _pipeline: ["2303", "2330"]
        local = types.SimpleNamespace(
            get_taiwan_symbols=original_symbols,
            load_stock_pipeline=lambda _root: None,
            run_market_batch=lambda *_args, **_kwargs: None,
            build_stock_snapshot=lambda *_args, **_kwargs: {},
        )
        pipeline = Pipeline()
        with _patched_pipeline(
            local,
            pipeline,
            Mock(),
            snapshot_series(),
            Mock(),
            symbols=["2303"],
        ):
            self.assertEqual(local.get_taiwan_symbols(pipeline), ["2303"])
        self.assertIs(local.get_taiwan_symbols, original_symbols)

    def test_cli_reconciliation_recalculates_through_existing_calc_all_before_write(self):
        pipeline = Pipeline()
        pipeline.calc_all = lambda: 42
        local = types.SimpleNamespace(
            load_stock_pipeline=lambda _root: pipeline,
            run_market_batch=lambda *_args, **_kwargs: None,
            build_stock_snapshot=lambda value, *_args, **_kwargs: {
                "derived": value.calc_all()
            },
        )
        written = {}
        local.write_stock_artifact = (
            lambda _root, _market, _symbol, payload: written.update(payload)
            or Path("written")
        )
        fetcher = Mock()
        fetcher.lineage_for.return_value = {}
        with _patched_pipeline(
            local, pipeline, fetcher, snapshot_series(), Mock(), backup_store=None
        ):
            payload = local.build_stock_snapshot(pipeline, "TW", "2330")
            local.write_stock_artifact(Path("root"), "TW", "2330", payload)
        self.assertEqual(written["derived"], 42)

    def test_cli_injects_verified_status_into_status_symbol_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            pipeline = Pipeline()
            observed = {}
            local = types.SimpleNamespace(
                load_stock_pipeline=lambda _root: pipeline,
                run_market_batch=lambda *_args, **_kwargs: None,
                build_stock_snapshot=lambda *_args, **kwargs: observed.update(kwargs) or {},
            )
            fetcher = OfficialCompatFetcher(
                Path(temporary), status_snapshot_series(), pd=pd
            )
            with _patched_pipeline(
                local,
                pipeline,
                fetcher,
                status_snapshot_series(),
                Mock(),
            ):
                local.build_stock_snapshot(
                    pipeline,
                    "TW",
                    "2303",
                    target_market_date=TARGET,
                    observation_only=True,
                )

        self.assertEqual(
            observed["trading_status"]["status"],
            "official_no_regular_trade",
        )

    def test_cli_publishes_verified_status_after_terminal_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, "2303", "2026-07-23")
            write_artifact(temporary, "2330", "2026-07-23")
            write_exclusions(temporary, [exclusion_row("2303")])

            result, observed, _builder, _module = self._run_fake(
                temporary,
                series=status_snapshot_series(),
                final_dates={"2330": TARGET.isoformat()},
                final_status_symbols=("2303",),
            )

        self.assertEqual(result, 0)
        self.assertEqual(observed["published_args"][1:3], ("TW", ["2303", "2330"]))
        self.assertEqual(observed["published_kwargs"]["target_market_date"], TARGET)
        self.assertNotIn("2303", observed["published_kwargs"]["failed_symbols"])

    def test_cli_scopes_downstream_operational_failures_to_universe(self):
        base = snapshot_series((TARGET,), price_symbols=("2330",))
        snapshot = base.snapshots[TARGET]
        snapshot = OfficialDailySnapshot(
            **{
                **snapshot.__dict__,
                "terminated_by_symbol": MappingProxyType({"9999": MappingProxyType({})}),
            }
        )
        series = OfficialSnapshotSeries(
            target_date=base.target_date,
            snapshots=MappingProxyType({TARGET: snapshot}),
            manifest_sha256=base.manifest_sha256,
            request_count=base.request_count,
            request_budget=base.request_budget,
            source_mode=base.source_mode,
            source_schema_version=base.source_schema_version,
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, "2330", TARGET.isoformat())
            write_exclusions(
                temporary,
                [exclusion_row("8888")],
            )
            result, observed, _builder, _module = self._run_fake(
                temporary,
                series=series,
                symbols=("2330",),
                final_dates={"2330": TARGET.isoformat()},
            )

        self.assertEqual(result, 0)
        self.assertEqual(observed["published_kwargs"]["failed_symbols"], [])

    def test_cli_restores_all_patches_when_assignment_or_pipeline_fails(self):
        class RejectOnce(types.SimpleNamespace):
            armed = False

            def __setattr__(self, name, value):
                if name == "write_stock_artifact" and self.armed:
                    object.__setattr__(self, "armed", False)
                    raise RuntimeError("assignment failed")
                object.__setattr__(self, name, value)

        local = RejectOnce(
            load_stock_pipeline=lambda _root: None,
            run_market_batch=lambda *_args, **_kwargs: None,
            build_stock_snapshot=lambda *_args, **_kwargs: {},
            write_stock_artifact=lambda *_args, **_kwargs: None,
        )
        originals = dict(vars(local))
        local.armed = True
        pipeline = Pipeline()
        original_fetch = pipeline.fetch_finmind_dataset
        with self.assertRaisesRegex(RuntimeError, "assignment failed"):
            with _patched_pipeline(
                local,
                pipeline,
                Mock(),
                snapshot_series(),
                Mock(),
                backup_store=Mock(),
            ):
                pass
        self.assertIs(pipeline.fetch_finmind_dataset, original_fetch)
        for name in (
            "load_stock_pipeline", "run_market_batch", "build_stock_snapshot",
            "write_stock_artifact",
        ):
            self.assertIs(getattr(local, name), originals[name])

    def test_cli_reconciliation_does_not_bootstrap_missing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, "2330")
            with self.assertRaises(RuntimeError):
                self._run_fake(temporary, reconcile=True)
            self.assertFalse((Path(temporary) / "quarantine").exists())

    def test_no_price_reconciliation_backup_is_created_before_write(self):
        events = []
        local = types.SimpleNamespace(
            load_stock_pipeline=lambda _root: None,
            run_market_batch=lambda *_args, **_kwargs: None,
            build_stock_snapshot=lambda *_args, **_kwargs: {},
            write_stock_artifact=lambda *_args, **_kwargs: events.append("writer")
            or Path("written"),
        )
        fetcher = Mock()
        fetcher.lineage_for.return_value = {"legacy_reconciliation": {"x": 1}}
        store = Mock()
        store.backup_before_write.side_effect = (
            lambda **_kwargs: events.append("backup") or "write"
        )
        with _patched_pipeline(
            local, Pipeline(), fetcher, snapshot_series(), Mock(), backup_store=store
        ):
            local.write_stock_artifact(Path("root"), "TW", "2330", {})
        self.assertEqual(events[:2], ["backup", "writer"])

    def test_cli_writer_failure_leaves_backup_complete(self):
        local = types.SimpleNamespace(
            load_stock_pipeline=lambda _root: None,
            run_market_batch=lambda *_args, **_kwargs: None,
            build_stock_snapshot=lambda *_args, **_kwargs: {},
            write_stock_artifact=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("writer failed")
            ),
        )
        fetcher = Mock()
        fetcher.lineage_for.return_value = {"legacy_reconciliation": {"x": 1}}
        store = Mock()
        store.backup_before_write.return_value = "write"
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            with _patched_pipeline(
                local,
                Pipeline(),
                fetcher,
                snapshot_series(),
                Mock(),
                backup_store=store,
            ):
                local.write_stock_artifact(Path("root"), "TW", "2330", {})
        store.backup_before_write.assert_called_once()
        store.mark_applied.assert_not_called()

    def test_cli_refuses_success_when_checkpoint_has_active_symbol_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with self.assertRaisesRegex(
                RuntimeError, "TW official observation recovery is incomplete"
            ):
                self._run_fake(
                    temporary,
                    final_dates={
                        "2303": TARGET.isoformat(), "2330": TARGET.isoformat()
                    },
                    failed=[{"symbol": "2330", "error": "RuntimeError"}],
                )

    def test_cli_refuses_success_when_checkpoint_is_partial_or_wrong_identity(self):
        cases = (
            {"next_index": 1},
            {"batch_identity": {"target_market_date": "2026-07-23"}},
            {"stage": "wrong"},
            {"market": "US"},
            {"failed": "invalid"},
        )
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                for symbol in ("2303", "2330"):
                    write_artifact(temporary, symbol)
                with self.assertRaisesRegex(
                    RuntimeError, "TW official observation recovery is incomplete"
                ):
                    self._run_fake(
                        temporary,
                        final_dates={
                            "2303": TARGET.isoformat(),
                            "2330": TARGET.isoformat(),
                        },
                        checkpoint_changes=changes,
                    )

    def test_cli_refuses_success_when_active_artifact_is_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                self._run_fake(
                    temporary, final_dates={"2303": TARGET.isoformat()}
                )

    def test_cli_refuses_success_when_active_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                self._run_fake(
                    temporary,
                    final_dates={"2303": TARGET.isoformat(), "2330": None},
                )

    def test_cli_refuses_success_when_active_artifact_is_future_dated(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                self._run_fake(
                    temporary,
                    final_dates={
                        "2303": TARGET.isoformat(), "2330": "2026-07-25"
                    },
                )

    def test_cli_allows_excluded_symbol_to_remain_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            write_exclusions(temporary, [exclusion_row("2330")])
            result, *_rest = self._run_fake(
                temporary, final_dates={"2303": TARGET.isoformat()}
            )
        self.assertEqual(result, 0)

    def test_cli_allows_checkpoint_failure_only_for_excluded_symbol(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            write_exclusions(temporary, [exclusion_row("2330")])
            result, *_rest = self._run_fake(
                temporary,
                final_dates={"2303": TARGET.isoformat()},
                failed=[{"symbol": "2330", "error": "RuntimeError"}],
            )
        self.assertEqual(result, 0)

    def test_cli_refuses_malformed_raw_exclusion_state(self):
        cases = (
            ([{"Symbol": "2330"}], ["Symbol"]),
            ([exclusion_row("../2330")], EXCLUSION_FIELDS),
            ([exclusion_row("2330"), exclusion_row("2330")], EXCLUSION_FIELDS),
            ([exclusion_row("2330", "Unknown")], EXCLUSION_FIELDS),
            ([exclusion_row("2330", action="Approve")], EXCLUSION_FIELDS),
        )
        for rows, fields in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temporary:
                for symbol in ("2303", "2330"):
                    write_artifact(temporary, symbol)
                write_exclusions(temporary, rows, fields=fields)
                with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                    self._run_fake(
                        temporary,
                        final_dates={
                            "2303": TARGET.isoformat(),
                            "2330": TARGET.isoformat(),
                        },
                    )

    def test_cli_refuses_unreadable_or_short_raw_exclusion_rows(self):
        payloads = (
            b"\xff\xfe\x00",
            (",".join(EXCLUSION_FIELDS) + "\n2330\n").encode(),
        )
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                for symbol in ("2303", "2330"):
                    write_artifact(temporary, symbol)
                path = Path(temporary) / "checkpoints" / "exclusion_list-TW.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                    self._run_fake(
                        temporary,
                        final_dates={
                            "2303": TARGET.isoformat(),
                            "2330": TARGET.isoformat(),
                        },
                    )

    def test_cli_returns_success_when_all_active_artifacts_match_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            result, *_rest = self._run_fake(
                temporary,
                final_dates={
                    "2303": TARGET.isoformat(), "2330": TARGET.isoformat()
                },
            )
        self.assertEqual(result, 0)

    def test_cli_accepts_only_hash_bound_target_status_for_stale_price_artifact(self):
        series = status_snapshot_series()
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            result, *_rest = self._run_fake(
                temporary,
                series=series,
                final_dates={"2330": TARGET.isoformat()},
                final_status_symbols={"2303"},
            )

        self.assertEqual(result, 0)

    def test_cli_post_run_checks_reconciliation_state_before_artifact_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            with patch.object(cli, "LegacyArtifactBackupStore") as store_type:
                store_type.discover_resume.return_value = None
                store_type.return_value.assert_current_state_complete = Mock(
                    side_effect=RuntimeError("reconciliation state is incomplete")
                )
                with self.assertRaisesRegex(
                    RuntimeError, "reconciliation state is incomplete"
                ):
                    self._run_fake(
                        temporary,
                        reconcile=True,
                        final_dates={
                            "2303": TARGET.isoformat(),
                            "2330": TARGET.isoformat(),
                        },
                    )

    def test_cli_refuses_target_date_active_artifact_without_official_lineage(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol, TARGET.isoformat())
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=(series.manifest_sha256, BASELINE),
            ), patch.object(
                LegacyArtifactBackupStore,
                "assert_current_state_complete",
                return_value=frozenset(),
            ):
                with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                    self._run_fake(
                        temporary,
                        reconcile=True,
                        series=series,
                        final_dates={},
                    )

    def test_cli_strict_refuses_target_date_artifact_without_official_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol, TARGET.isoformat())
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                self._run_fake(temporary, final_dates={})

    def test_cli_final_loader_compares_applied_manifest_sha(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(
                root,
                "2303",
                TARGET.isoformat(),
                official_lineage("2303", series),
            )
            legacy = write_artifact(root, "2330", BASELINE.isoformat())
            value = reconciliation_evidence(
                hashlib.sha256(legacy.read_bytes()).hexdigest(), series
            )
            write_official_artifact(root, "2330", series, value)
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=(series.manifest_sha256, BASELINE),
            ), patch.object(
                LegacyArtifactBackupStore,
                "assert_current_state_complete",
                return_value={"2330": "f" * 64},
            ):
                with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                    self._run_fake(
                        root,
                        reconcile=True,
                        series=series,
                        final_dates={},
                    )

    def test_cli_returns_nonzero_local_quant_status_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol)
            result, *_rest = self._run_fake(temporary, status=7)
        self.assertEqual(result, 7)

    def test_cli_repairs_last_artifact_post_write_pre_apply_without_rewrite(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = write_artifact(root, "2330", BASELINE.isoformat())
            original_sha = hashlib.sha256(legacy.read_bytes()).hexdigest()
            evidence = reconciliation_evidence(original_sha, series)
            store = LegacyArtifactBackupStore(
                root,
                target_date=TARGET,
                series_manifest_sha256=series.manifest_sha256,
            )
            store.backup_before_write(
                symbol="2330", artifact_path=legacy, evidence=evidence
            )
            target = write_official_artifact(root, "2330", series, evidence)
            write_artifact(
                root,
                "2303",
                TARGET.isoformat(),
                official_lineage("2303", series),
            )
            before_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            before_mtime = target.stat().st_mtime_ns
            time.sleep(0.01)
            result, _observed, builder, _module = self._run_fake(
                root,
                reconcile=True,
                series=series,
                writer_call="2330",
            )
            after_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            after_mtime = target.stat().st_mtime_ns
            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(builder.call_args.args[1], FULL_SERIES_DATES)
        self.assertEqual(after_sha, before_sha)
        self.assertEqual(after_mtime, before_mtime)
        self.assertEqual(manifest["entries"]["2330"]["status"], "applied")

    def test_prefetches_series_enriches_identity_and_restores_patches(self):
        pipeline = Pipeline()
        series = snapshot_series()
        original_fetch = pipeline.fetch_finmind_dataset
        observed = {}
        module = types.ModuleType("local_quant")
        module.OBSERVATION_SOURCE_VERSION = "observation-source-v1"
        module.get_taiwan_symbols = lambda _pipeline: ["2303", "2330"]
        module.load_stock_pipeline = lambda _root: pipeline

        def original_batch(
            root, market, symbols, analyze, *args, batch_identity=None, **kwargs
        ):
            observed["identity"] = batch_identity
            for symbol in symbols:
                write_artifact(
                    root,
                    symbol,
                    TARGET.isoformat(),
                    official_lineage(symbol, series),
                )
            state = {
                "stage": "market_batch",
                "market": "TW",
                "next_index": len(symbols),
                "failed": [],
                "batch_identity": batch_identity,
            }
            write_checkpoint(root, state)
            return state

        module.run_market_batch = original_batch
        module.build_stock_snapshot = (
            lambda _pipeline, market, symbol, *args, **kwargs: {"symbol": symbol}
        )

        module.publish_market_snapshot = (
            lambda *_args, **_kwargs: observed.update(published=True)
        )

        def local_main(argv):
            self.assertIn("--observation-only", argv)
            data_root = Path(argv[argv.index("--root") + 1])
            module.run_market_batch(
                data_root, "TW", ["2303", "2330"], lambda _symbol: {},
                batch_identity={
                    "target_market_date": TARGET.isoformat(),
                    "product_mode": "observation",
                    "source_version": module.OBSERVATION_SOURCE_VERSION,
                },
            )
            payload = module.build_stock_snapshot(pipeline, "TW", "2330")
            observed["lineage"] = payload["source_lineage"]
            self.assertNotEqual(pipeline.fetch_finmind_dataset, original_fetch)
            return 0

        module.main = local_main
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        builder = Mock(return_value=series)
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
                        series_builder=lambda *_args, **_kwargs: (
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

    def test_coverage_boundary_thresholds(self):
        calendar_doc = {
            "schema_version": 1,
            "market": "TW",
            "year": 2026,
            "source_url": TWSE_CALENDAR_URL,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "source_sha256": "c" * 64,
            "valid_from": "2026-01-01",
            "valid_to": "2026-12-31",
            "closed_dates": [],
            "special_open_dates": [],
        }
        calendar = TradingCalendarSet.from_documents([calendar_doc])
        symbols_1000 = [f"{i:04d}" for i in range(1000)]
        target_date = TARGET

        # Helper mock audit
        class MockAudit:
            def __init__(self, available_count, total_count):
                available = symbols_1000[:available_count]
                unavailable = symbols_1000[available_count:total_count]
                self.latest_by_symbol = {s: target_date for s in available}
                self.observation_by_symbol = {s: target_date for s in available}
                self.unavailable_symbols = unavailable

        # 100.0% coverage: 1000/1000 available (0 unavailable) -> PASS
        cli._assert_audit_publishable(
            MockAudit(1000, 1000),
            symbols=symbols_1000,
            target_market_date=target_date,
            calendars=calendar,
        )

        # 99.9% coverage: 999/1000 available (1 unavailable, 0.1%) -> PASS
        cli._assert_audit_publishable(
            MockAudit(999, 1000),
            symbols=symbols_1000,
            target_market_date=target_date,
            calendars=calendar,
        )

        # 98.5% coverage: 985/1000 available (15 unavailable, 1.5%) -> PASS
        cli._assert_audit_publishable(
            MockAudit(985, 1000),
            symbols=symbols_1000,
            target_market_date=target_date,
            calendars=calendar,
        )

        # 95.1% coverage: 951/1000 available (49 unavailable, 4.9%) -> PASS
        cli._assert_audit_publishable(
            MockAudit(951, 1000),
            symbols=symbols_1000,
            target_market_date=target_date,
            calendars=calendar,
        )

        # Exactly 95.0% coverage: 950/1000 available (50 unavailable, 5.0%) -> FAIL (strict > 95%)
        with self.assertRaisesRegex(RuntimeError, "historical artifact coverage is not publishable"):
            cli._assert_audit_publishable(
                MockAudit(950, 1000),
                symbols=symbols_1000,
                target_market_date=target_date,
                calendars=calendar,
            )

        # 94.9% coverage: 949/1000 available (51 unavailable, 5.1%) -> FAIL
        with self.assertRaisesRegex(RuntimeError, "historical artifact coverage is not publishable"):
            cli._assert_audit_publishable(
                MockAudit(949, 1000),
                symbols=symbols_1000,
                target_market_date=target_date,
                calendars=calendar,
            )

    def test_plan_recovery_stage_prefilters_excluded_symbols(self):
        calendar_doc = {
            "schema_version": 1,
            "market": "TW",
            "year": 2026,
            "source_url": TWSE_CALENDAR_URL,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "source_sha256": "c" * 64,
            "valid_from": "2026-01-01",
            "valid_to": "2026-12-31",
            "closed_dates": [],
            "special_open_dates": [],
        }
        calendar = TradingCalendarSet.from_documents([calendar_doc])
        symbols = ["2330", "2303", "4130", "4987", "6806"]
        excluded = {"4130", "4987", "6806"}

        class MockAudit:
            latest_by_symbol = {
                "2330": datetime.date(2026, 7, 24),
                "2303": datetime.date(2026, 7, 20),
                "4130": datetime.date(2026, 7, 16),
                "4987": datetime.date(2026, 7, 16),
                "6806": datetime.date(2026, 7, 16),
            }
            observation_by_symbol = dict(latest_by_symbol)
            unavailable_symbols = []

        # When excluded_symbols are filtered, baseline should be 2026-07-20 (2303), NOT 2026-07-16
        stage_target, stage_symbols, baseline = _plan_recovery_stage(
            calendar,
            MockAudit(),
            symbols=symbols,
            target_market_date=datetime.date(2026, 8, 31),
            reconcile_legacy_overlaps=False,
            excluded_symbols=excluded,
        )
        self.assertEqual(baseline, datetime.date(2026, 7, 20))
        self.assertLess(stage_target, datetime.date(2026, 8, 31))
        self.assertEqual(set(stage_symbols), {"2330", "2303"})
        self.assertNotIn("4130", stage_symbols)
        self.assertNotIn("4987", stage_symbols)
        self.assertNotIn("6806", stage_symbols)

    def test_stage_coverage_uses_full_market_universe_denominator(self):
        # 2026-08-04 regression scenario: stage subset has 12 symbols with 3 missing prices (25% of stage),
        # but full market has 2076 symbols with 2042 prices (98.36% coverage > 95%).
        full_market = [f"{i:04d}" for i in range(2076)]
        covered_market = set(full_market[:2042])
        stage_symbols = full_market[:12]

        series = snapshot_series(dates=(BASELINE,), price_symbols=covered_market)
        snapshot_0804 = series.snapshots[BASELINE]

        # Stage with full_market_symbols should PASS because 2042/2076 = 98.36% > 95%
        # If denominator was stage_symbols (12), 9/12 = 75% would fail!
        market_universe = set(full_market)
        covered = set(snapshot_0804.price_by_symbol)
        missing = market_universe - covered
        self.assertLess(len(missing) / len(market_universe), 0.05)

    def _gate_fixture(self, root, observed_count, total_count, failures=None):
        root = Path(root)
        symbols = [f"{i:04d}" for i in range(total_count)]
        observed = symbols[:observed_count]
        series = snapshot_series(dates=(TARGET,), price_symbols=tuple(observed))
        for symbol in observed:
            write_artifact(
                root, symbol, TARGET.isoformat(), official_lineage(symbol, series)
            )
        expected_identity = cli._enrich_batch_identity(
            {
                "target_market_date": TARGET.isoformat(),
                "product_mode": "observation",
                "source_version": local_quant.OBSERVATION_SOURCE_VERSION,
            },
            series=series,
            audit=Mock(latest_date_counts={}, unavailable_symbols=[]),
            symbols=symbols,
            reconcile_legacy_overlaps=False,
            recover_truncated_history=False,
        )
        write_checkpoint(
            root,
            {
                "stage": "market_batch",
                "market": "TW",
                "next_index": total_count,
                "failed": list(failures or []),
                "batch_identity": expected_identity,
            },
        )
        return symbols, series, expected_identity

    def test_assert_complete_exact_95_percent_coverage_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symbols, series, identity = self._gate_fixture(
                root,
                observed_count=19,
                total_count=20,
                failures=[{"symbol": "0019", "error": "_UnavailableObservationError"}],
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=symbols,
                    target_market_date=TARGET,
                    expected_identity=identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_assert_complete_above_95_percent_with_unavailable_partition_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symbols, series, identity = self._gate_fixture(
                root,
                observed_count=20,
                total_count=21,
                failures=[{"symbol": "0020", "error": "_UnavailableObservationError"}],
            )
            cli._assert_complete(
                root,
                symbols=symbols,
                target_market_date=TARGET,
                expected_identity=identity,
                official_series=series,
                applied_reconciliation_artifacts={},
            )

    def test_assert_complete_rejects_operational_failure_in_unavailable_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symbols, series, identity = self._gate_fixture(
                root,
                observed_count=20,
                total_count=21,
                failures=[{"symbol": "0020", "error": "OSError"}],
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=symbols,
                    target_market_date=TARGET,
                    expected_identity=identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_assert_complete_rejects_operational_failure_in_observed_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symbols, series, identity = self._gate_fixture(
                root,
                observed_count=20,
                total_count=21,
                failures=[
                    {"symbol": "0005", "error": "OSError"},
                    {"symbol": "0020", "error": "_UnavailableObservationError"},
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is incomplete"):
                cli._assert_complete(
                    root,
                    symbols=symbols,
                    target_market_date=TARGET,
                    expected_identity=identity,
                    official_series=series,
                    applied_reconciliation_artifacts={},
                )

    def test_patched_builder_classifies_data_absence_as_unavailable(self):
        class RaisingPipeline:
            pd = pd
            fetch_finmind_dataset = None

            @staticmethod
            def calc_all(_frame):
                return _frame

            def get_stock_name(self, _symbol):
                return "測試標的"

        def original_build(_pipeline_arg, market, symbol, *args, **kwargs):
            if symbol == "2303":
                raise ValueError("point-in-time price history is unavailable")
            if symbol == "2330":
                raise ValueError("calculated history is unavailable")
            raise AssertionError("unreachable")

        fake_module = types.SimpleNamespace(
            build_stock_snapshot=original_build,
            load_stock_pipeline=lambda _root: RaisingPipeline(),
            run_market_batch=lambda *_args, **_kwargs: None,
            get_taiwan_symbols=lambda _pipeline: ["2303", "2330"],
            load_exclusion_list=lambda _root, _market: (set(), set(), [], 0),
        )
        with _patched_pipeline(
            fake_module,
            RaisingPipeline(),
            Mock(),
            snapshot_series(dates=(TARGET,), price_symbols=("2330",)),
            Mock(),
        ):
            with self.assertRaisesRegex(
                ValueError, "point-in-time price history is unavailable"
            ) as unavailable:
                fake_module.build_stock_snapshot(
                    RaisingPipeline(), "TW", "2303", target_market_date=TARGET
                )
            self.assertIsInstance(
                unavailable.exception, cli._UnavailableObservationError
            )
            with self.assertRaises(ValueError) as operational:
                fake_module.build_stock_snapshot(
                    RaisingPipeline(), "TW", "2330", target_market_date=TARGET
                )
            self.assertNotIsInstance(
                operational.exception, cli._UnavailableObservationError
            )


if __name__ == "__main__":
    unittest.main()
