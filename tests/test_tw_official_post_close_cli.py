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

from stock_papi.batch.calendar import TWSE_CALENDAR_URL, TradingCalendarSet
from stock_papi.batch import tw_official_post_close_cli as cli
from stock_papi.batch.tw_official_post_close_cli import (
    _load_calendar_set,
    _patched_pipeline,
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
from stock_papi.quant.tw_incremental import OfficialCompatFetcher
from stock_papi.quant.tw_legacy_reconciliation import LegacyArtifactBackupStore

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
    return path


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
        "schema_version": 1,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": original_sha,
        "legacy_artifact_as_of": BASELINE.isoformat(),
        "official_source_mode": series.source_mode,
        "official_source_schema_version": series.source_schema_version,
        "official_series_manifest_sha256": series.manifest_sha256,
        "official_snapshot_dates": dates,
        "official_snapshot_manifests": manifests,
        "replaced_dates": [BASELINE.isoformat()],
        "price_replaced_dates": [BASELINE.isoformat()],
        "institutional_replaced_dates": [],
        "margin_replaced_dates": [],
        "date_evidence": [{
            "date": BASELINE.isoformat(),
            "price_replaced": True,
            "institutional_replaced": False,
            "margin_replaced": False,
        }],
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
    def _run_fake(
        self,
        root,
        *,
        reconcile=False,
        series=None,
        status=0,
        final_dates=None,
        failed=None,
        checkpoint_changes=None,
        writer_call=None,
        symbols=("2303", "2330"),
        builder=None,
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
                    write_artifact(data_root, symbol, as_of)
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
            )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old
        return result, observed, chosen_builder, module

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
            result, observed, builder, _module = self._run_fake(
                temporary,
                reconcile=True,
                series=series,
                final_dates={"2303": TARGET.isoformat(), "2330": TARGET.isoformat()},
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

    def test_cli_reconciliation_counts_baseline_in_session_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol, "2026-07-10")
            with self.assertRaises(ValueError):
                self._run_fake(temporary, reconcile=True)

    def test_cli_resume_reuses_discovered_baseline_and_series_identity(self):
        series = snapshot_series(FULL_SERIES_DATES)
        with tempfile.TemporaryDirectory() as temporary:
            for symbol in ("2303", "2330"):
                write_artifact(temporary, symbol, TARGET.isoformat())
            with patch.object(
                LegacyArtifactBackupStore,
                "discover_resume",
                return_value=(series.manifest_sha256, BASELINE),
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
            write_artifact(root, "2303", TARGET.isoformat())
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
                write_artifact(root, symbol, TARGET.isoformat())
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
