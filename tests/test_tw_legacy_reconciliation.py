import datetime
import copy
import dataclasses
import gzip
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import stock_papi.quant.tw_legacy_reconciliation as reconciliation
from stock_papi.quant.tw_legacy_reconciliation import (
    LegacyArtifactBackupStore,
    LegacyReconciliationError,
)
from stock_papi.quant.tw_incremental import (
    OfficialCompatFetcher,
    load_incremental_artifact,
)


TARGET = datetime.date(2026, 7, 24)
BASELINE = datetime.date(2026, 7, 16)
SERIES_SHA = "56cc6940752cb667e5225ece40685236148cef14dd7e6b3b7c3a50a6079b941a"


def artifact_path(root, symbol="2330"):
    return Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"


def write_artifact(root, document, symbol="2330"):
    path = artifact_path(root, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload.update(schema_version=1, market="TW", symbol=symbol)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)
    return path, encoded


def legacy_document(value=BASELINE):
    return {
        "as_of": value.isoformat(),
        "daily": [{
            "Date": f"{value.isoformat()}T00:00:00.000",
            "Open": 1.0,
            "High": 2.0,
            "Low": 0.5,
            "Close": 1.5,
            "Volume": 7.0,
            "InstitutionalNet": 11.0,
            "ForeignNet": 9.0,
            "MarginBalance": 22.0,
            "ShortBalance": 3.0,
        }],
    }


def evidence(original_sha, replaced_date=BASELINE):
    date_text = replaced_date.isoformat()
    return {
        "schema_version": 2,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": original_sha,
        "legacy_artifact_as_of": date_text,
        "official_source_mode": "tw_official_bulk_v2",
        "official_source_schema_version": "tw-official-historical-v2",
        "official_series_manifest_sha256": SERIES_SHA,
        "official_snapshot_dates": [date_text, TARGET.isoformat()],
        "official_snapshot_manifests": [
            {"date": date_text, "manifest_sha256": "a" * 64},
            {"date": TARGET.isoformat(), "manifest_sha256": "b" * 64},
        ],
        "overlap_dates": [date_text],
        "price_replaced_dates": [date_text],
        "price_preserved_no_official_row_dates": [],
        "institutional_replaced_dates": [date_text],
        "institutional_preserved_no_official_row_dates": [],
        "margin_replaced_dates": [date_text],
        "margin_preserved_no_official_row_dates": [],
        "date_evidence": [{
            "date": date_text,
            "price_action": "replaced_official",
            "institutional_action": "replaced_official",
            "margin_action": "replaced_official",
        }],
    }


def no_price_evidence(original_sha, overlap_date=BASELINE):
    date_text = overlap_date.isoformat()
    return {
        "schema_version": 2,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": original_sha,
        "legacy_artifact_as_of": date_text,
        "official_source_mode": "tw_official_bulk_v2",
        "official_source_schema_version": "tw-official-historical-v2",
        "official_series_manifest_sha256": SERIES_SHA,
        "official_snapshot_dates": [date_text, TARGET.isoformat()],
        "official_snapshot_manifests": [
            {"date": date_text, "manifest_sha256": "a" * 64},
            {"date": TARGET.isoformat(), "manifest_sha256": "b" * 64},
        ],
        "overlap_dates": [date_text],
        "price_replaced_dates": [],
        "price_preserved_no_official_row_dates": [date_text],
        "institutional_replaced_dates": [date_text],
        "institutional_preserved_no_official_row_dates": [],
        "margin_replaced_dates": [],
        "margin_preserved_no_official_row_dates": [date_text],
        "date_evidence": [{
            "date": date_text,
            "price_action": "preserved_legacy_no_official_row",
            "institutional_action": "replaced_official",
            "margin_action": "preserved_legacy_no_official_row",
        }],
    }


def official_document(reconciliation):
    return {
        "as_of": TARGET.isoformat(),
        "daily": [{
            "Date": f"{TARGET.isoformat()}T00:00:00.000",
            "Open": 1100.0,
            "High": 1120.0,
            "Low": 1090.0,
            "Close": 1110.0,
            "Volume": 1000.0,
            "InstitutionalNet": 90.0,
            "ForeignNet": 80.0,
            "MarginBalance": 5000.0,
            "ShortBalance": 200.0,
        }],
        "source_lineage": {
            "source_mode": "tw_official_bulk_v2",
            "source_schema_version": "tw-official-historical-v2",
            "symbol": "2330",
            "target_market_date": TARGET.isoformat(),
            "historical_as_of": BASELINE.isoformat(),
            "historical_artifact_sha256": reconciliation[
                "legacy_artifact_sha256"
            ],
            "official_target_price_available": True,
            "official_snapshot_dates": copy.deepcopy(
                reconciliation["official_snapshot_dates"]
            ),
            "official_snapshot_manifests": copy.deepcopy(
                reconciliation["official_snapshot_manifests"]
            ),
            "official_series_manifest_sha256": SERIES_SHA,
            "legacy_reconciliation": reconciliation,
        },
    }


def read_manifest(root):
    path = (
        Path(root)
        / "quarantine"
        / "tw-recovery"
        / "legacy-reconciliation"
        / "v2"
        / TARGET.isoformat()
        / SERIES_SHA
        / "manifest.json"
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


class LegacyArtifactBackupStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path, self.decoded = write_artifact(self.root, legacy_document())
        self.original = self.path.read_bytes()
        self.original_sha = hashlib.sha256(self.original).hexdigest()
        self.evidence = evidence(self.original_sha)
        self.store = LegacyArtifactBackupStore(
            self.root,
            target_date=TARGET,
            series_manifest_sha256=SERIES_SHA,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def backup(self):
        return self.store.backup_before_write(
            symbol="2330",
            artifact_path=self.path,
            evidence=self.evidence,
        )

    def write_official(self):
        return write_artifact(
            self.root,
            official_document(self.evidence),
        )[0]

    def prepare_verified_reader(self):
        self.backup()
        target = self.write_official()
        result_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        manifest_path, manifest = read_manifest(self.root)
        entry = manifest["entries"]["2330"]
        return (
            manifest_path,
            manifest,
            self.store.backup_root / entry["backup_path"],
            result_sha,
        )

    @staticmethod
    def compressed_document(document):
        decoded = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return gzip.compress(decoded, mtime=0), decoded

    def recovery_artifact(self):
        return load_incremental_artifact(self.root, "2330")

    def prepare_direct_recovery(self):
        self.prepare_verified_reader()
        return self.recovery_artifact()

    def write_history_lineage(self, *, result_sha, reconciliation_value=None):
        document = official_document(self.evidence)
        selected = copy.deepcopy(reconciliation_value or self.evidence)
        document["source_lineage"].pop("legacy_reconciliation")
        item = {
            "schema_version": 2,
            "symbol": "2330",
            "reconciled_artifact_sha256": result_sha,
            "reconciliation": selected,
        }
        item["history_sha256"] = OfficialCompatFetcher._canonical_json_sha256(item)
        document["source_lineage"]["legacy_reconciliation_history"] = [item]
        write_artifact(self.root, document)
        return document, item

    def test_missing_or_null_lineage_is_legacy_and_not_recovery_eligible(self):
        for lineage in (None, object()):
            with self.subTest(lineage=lineage is None):
                document = legacy_document()
                if lineage is None:
                    document["source_lineage"] = None
                write_artifact(self.root, document)
                artifact = self.recovery_artifact()

                from stock_papi.quant.tw_legacy_reconciliation import (
                    resolve_truncated_daily_history,
                )

                self.assertIsNone(
                    resolve_truncated_daily_history(self.root, "2330", artifact)
                )

    def test_present_invalid_lineage_fails_closed_for_recovery(self):
        document = official_document(self.evidence)
        document["source_lineage"]["target_market_date"] = "not-a-date"
        write_artifact(self.root, document)
        artifact = self.recovery_artifact()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        with self.assertRaises(LegacyReconciliationError):
            resolve_truncated_daily_history(self.root, "2330", artifact)

    def test_valid_official_lineage_without_reconciliation_returns_none(self):
        document = official_document(self.evidence)
        document["source_lineage"].pop("legacy_reconciliation")
        write_artifact(self.root, document)
        artifact = self.recovery_artifact()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        self.assertIsNone(
            resolve_truncated_daily_history(self.root, "2330", artifact)
        )

    def test_resolver_binds_direct_result_sha_and_exact_snapshot_date(self):
        artifact = self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        result = resolve_truncated_daily_history(self.root, "2330", artifact)

        self.assertEqual(result.input_artifact_sha256, artifact.compressed_sha256)
        self.assertEqual(result.original_artifact_sha256, self.original_sha)
        self.assertEqual(result.expected_result_sha256, artifact.compressed_sha256)
        self.assertEqual(result.backup_target_market_date, TARGET)
        self.assertEqual(result.backup_series_manifest_sha256, SERIES_SHA)

    def test_resolver_binds_historical_result_sha_and_exact_snapshot_date(self):
        self.prepare_verified_reader()
        direct_result_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.write_history_lineage(result_sha=direct_result_sha)
        artifact = self.recovery_artifact()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        result = resolve_truncated_daily_history(self.root, "2330", artifact)

        self.assertEqual(result.input_artifact_sha256, artifact.compressed_sha256)
        self.assertEqual(result.expected_result_sha256, direct_result_sha)
        self.assertEqual(result.backup_target_market_date, TARGET)

    def test_resolver_rejects_missing_or_multiple_distinct_backups(self):
        self.prepare_verified_reader()
        manifest_path, manifest = read_manifest(self.root)
        manifest_path.unlink()
        artifact = self.recovery_artifact()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        with self.assertRaises(LegacyReconciliationError):
            resolve_truncated_daily_history(self.root, "2330", artifact)

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        document = official_document(self.evidence)
        second = copy.deepcopy(self.evidence)
        second["legacy_artifact_sha256"] = "d" * 64
        document["source_lineage"]["legacy_reconciliation_history"] = [{
            "schema_version": 2,
            "symbol": "2330",
            "reconciled_artifact_sha256": result_sha,
            "reconciliation": second,
            "history_sha256": "0" * 64,
        }]
        write_artifact(self.root, document)
        artifact = dataclasses.replace(
            self.recovery_artifact(), compressed_sha256=result_sha
        )
        backup_document = json.loads(self.decoded.decode("utf-8"))
        first_entry = manifest["entries"]["2330"]
        second_entry = dict(first_entry, original_sha256="d" * 64)
        with (
            patch.object(OfficialCompatFetcher, "_valid_official_lineage", return_value=True),
            patch.object(
                LegacyArtifactBackupStore,
                "read_original_document",
                side_effect=[
                    (backup_document, first_entry),
                    (backup_document, second_entry),
                ],
            ),
            self.assertRaises(LegacyReconciliationError),
        ):
            resolve_truncated_daily_history(self.root, "2330", artifact)

    def test_resolver_deduplicates_identical_repeated_history_bindings(self):
        self.prepare_verified_reader()
        result_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        document = official_document(self.evidence)
        item = {
            "schema_version": 2,
            "symbol": "2330",
            "reconciled_artifact_sha256": result_sha,
            "reconciliation": copy.deepcopy(self.evidence),
        }
        item["history_sha256"] = OfficialCompatFetcher._canonical_json_sha256(item)
        document["source_lineage"]["legacy_reconciliation_history"] = [
            copy.deepcopy(item)
        ]
        write_artifact(self.root, document)
        artifact = dataclasses.replace(
            self.recovery_artifact(), compressed_sha256=result_sha
        )

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        original_reader = LegacyArtifactBackupStore.read_original_document

        def tracked_reader(store, **kwargs):
            return original_reader(store, **kwargs)

        with (
            patch.object(OfficialCompatFetcher, "_valid_official_lineage", return_value=True),
            patch.object(
                LegacyArtifactBackupStore,
                "read_original_document",
                autospec=True,
                side_effect=tracked_reader,
            ) as reader,
        ):
            result = resolve_truncated_daily_history(self.root, "2330", artifact)

        self.assertEqual(reader.call_count, 2)
        self.assertEqual(result.original_artifact_sha256, self.original_sha)
        self.assertEqual(result.expected_result_sha256, result_sha)

    def test_resolver_fails_closed_when_second_identical_binding_cannot_be_verified(self):
        self.prepare_verified_reader()
        result_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        document = official_document(self.evidence)
        item = {
            "schema_version": 2,
            "symbol": "2330",
            "reconciled_artifact_sha256": result_sha,
            "reconciliation": copy.deepcopy(self.evidence),
        }
        item["history_sha256"] = OfficialCompatFetcher._canonical_json_sha256(item)
        document["source_lineage"]["legacy_reconciliation_history"] = [item]
        write_artifact(self.root, document)
        artifact = dataclasses.replace(
            self.recovery_artifact(), compressed_sha256=result_sha
        )

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        original_reader = LegacyArtifactBackupStore.read_original_document
        calls = 0

        def second_read_fails(store, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise LegacyReconciliationError("second identical binding is unreadable")
            return original_reader(store, **kwargs)

        with (
            patch.object(OfficialCompatFetcher, "_valid_official_lineage", return_value=True),
            patch.object(
                LegacyArtifactBackupStore,
                "read_original_document",
                autospec=True,
                side_effect=second_read_fails,
            ) as reader,
            self.assertRaises(LegacyReconciliationError),
        ):
            resolve_truncated_daily_history(self.root, "2330", artifact)

        self.assertEqual(reader.call_count, 2)

    def test_resolver_rejects_same_object_sha_with_conflicting_authorization_bindings(self):
        self.prepare_verified_reader()
        result_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        conflicting = copy.deepcopy(self.evidence)
        conflicting["official_snapshot_dates"][-1] = "2026-07-23"
        conflicting["official_snapshot_manifests"][-1]["date"] = "2026-07-23"
        document = official_document(self.evidence)
        item = {
            "schema_version": 2,
            "symbol": "2330",
            "reconciled_artifact_sha256": result_sha,
            "reconciliation": conflicting,
            "history_sha256": "0" * 64,
        }
        document["source_lineage"]["legacy_reconciliation_history"] = [item]
        write_artifact(self.root, document)
        artifact = dataclasses.replace(
            self.recovery_artifact(), compressed_sha256=result_sha
        )
        backup_document = json.loads(self.decoded.decode("utf-8"))
        entry = read_manifest(self.root)[1]["entries"]["2330"]

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        with (
            patch.object(OfficialCompatFetcher, "_valid_official_lineage", return_value=True),
            patch.object(
                LegacyArtifactBackupStore,
                "read_original_document",
                return_value=(backup_document, entry),
            ),
            self.assertRaises(LegacyReconciliationError),
        ):
            resolve_truncated_daily_history(self.root, "2330", artifact)

    def test_merge_rejects_duplicate_dates_and_overlap_ohlcv_conflict(self):
        self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        merge = reconciliation._merge_recovery_daily
        row = legacy_document()["daily"][0]
        with self.assertRaises(LegacyReconciliationError):
            merge([row, dict(row)], [row])
        with self.assertRaises(LegacyReconciliationError):
            merge([dict(row, Close=9.0)], [row])

    def test_merge_rejects_bool_nan_and_infinite_ohlcv(self):
        self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        merge = reconciliation._merge_recovery_daily
        row = legacy_document()["daily"][0]
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(LegacyReconciliationError):
                    merge([dict(row, Open=value)], [])

    def test_merge_rejects_non_prefix_backup_only_rows(self):
        self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        merge = reconciliation._merge_recovery_daily
        active = official_document(self.evidence)["daily"]
        backup = [
            legacy_document()["daily"][0],
            dict(active[0], Date="2026-07-25T00:00:00.000"),
        ]
        with self.assertRaises(LegacyReconciliationError):
            merge(active, backup)

    def test_merge_keeps_current_whole_row_when_ohlcv_matches(self):
        self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        merge = reconciliation._merge_recovery_daily
        backup = legacy_document()["daily"]
        current = dict(backup[0], InstitutionalNet=999.0)
        merged, restored = merge([current], backup)

        self.assertEqual(merged, (current,))
        self.assertEqual(restored, ())

    def test_resolver_returns_full_merge_and_candidates_without_range_filter(self):
        artifact = self.prepare_direct_recovery()

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        result = resolve_truncated_daily_history(self.root, "2330", artifact)

        self.assertEqual(
            [row["Date"][:10] for row in result.merged_daily],
            [BASELINE.isoformat(), TARGET.isoformat()],
        )
        self.assertEqual(
            [row["Date"][:10] for row in result.restored_candidates],
            [BASELINE.isoformat()],
        )
        self.assertEqual(
            [row["Date"][:10] for row in result.backup_daily],
            [BASELINE.isoformat()],
        )

    def test_resolver_preserves_selected_reconciliation_evidence_for_receipt_finalization(self):
        nested_legacy = legacy_document()
        nested_legacy["daily"][0]["Nested"] = {"value": "backup"}
        self.path, self.decoded = write_artifact(self.root, nested_legacy)
        self.original = self.path.read_bytes()
        self.original_sha = hashlib.sha256(self.original).hexdigest()
        self.evidence = evidence(self.original_sha)
        artifact = self.prepare_direct_recovery()
        artifact.document["daily"][0]["Nested"] = {"value": "active"}
        stored_receipt = {"schema_version": 1, "nested": {"date": TARGET.isoformat()}}
        artifact.document["source_lineage"]["daily_history_recovery"] = stored_receipt

        from stock_papi.quant.tw_legacy_reconciliation import (
            resolve_truncated_daily_history,
        )

        original_reader = LegacyArtifactBackupStore.read_original_document
        source_backup_document = None
        source_entry = None

        def reader_with_nested_entry(store, **kwargs):
            nonlocal source_backup_document, source_entry
            source_backup_document, source_entry = original_reader(store, **kwargs)
            source_entry["nested"] = {"value": "entry"}
            return source_backup_document, source_entry

        with patch.object(
            LegacyArtifactBackupStore,
            "read_original_document",
            autospec=True,
            side_effect=reader_with_nested_entry,
        ):
            result = resolve_truncated_daily_history(self.root, "2330", artifact)

        with self.assertRaises(TypeError):
            result.reconciliation["mode"] = "tampered"
        with self.assertRaises(TypeError):
            result.existing_receipt["schema_version"] = 2
        with self.assertRaises(TypeError):
            result.reconciliation["official_snapshot_manifests"][0]["date"] = "tampered"
        with self.assertRaises(TypeError):
            result.existing_receipt["nested"]["date"] = "tampered"
        with self.assertRaises(TypeError):
            result.backup_manifest_entry["nested"]["value"] = "tampered"
        for rows in (
            result.merged_daily,
            result.restored_candidates,
            result.backup_daily,
        ):
            with self.subTest(rows=rows):
                with self.assertRaises(TypeError):
                    rows[0] = {}
                with self.assertRaises(AttributeError):
                    rows.append({})
                with self.assertRaises(TypeError):
                    rows[0]["Nested"]["value"] = "tampered"
        stored_receipt["nested"]["date"] = "tampered"
        artifact.document["daily"][0]["Nested"]["value"] = "tampered"
        artifact.document["source_lineage"]["legacy_reconciliation"][
            "official_snapshot_manifests"
        ][0]["date"] = "tampered"
        source_backup_document["daily"][0]["Nested"]["value"] = "tampered"
        source_entry["nested"]["value"] = "tampered"
        self.assertEqual(
            result.existing_receipt["nested"]["date"], TARGET.isoformat()
        )
        self.assertEqual(result.merged_daily[0]["Nested"]["value"], "backup")
        self.assertEqual(result.merged_daily[1]["Nested"]["value"], "active")
        self.assertEqual(result.backup_manifest_entry["nested"]["value"], "entry")
        self.assertEqual(
            result.reconciliation["official_snapshot_manifests"][0]["date"],
            BASELINE.isoformat(),
        )

    def test_verified_reader_reads_object_once_and_parses_same_bytes(self):
        manifest_path, manifest, object_path, result_sha = self.prepare_verified_reader()
        expected = json.loads(self.decoded.decode("utf-8"))
        object_reads = []
        original_read = reconciliation._read_bytes

        def read_once(path, **kwargs):
            if Path(path) == object_path:
                object_reads.append(path)
                return (self.original, gzip.compress(b'{"market":"US"}', mtime=0))[len(object_reads) - 1]
            return original_read(path, **kwargs)

        with patch.object(reconciliation, "_read_bytes", side_effect=read_once):
            document, entry = self.store.read_original_document(
                symbol="2330",
                original_sha256=manifest["entries"]["2330"]["original_sha256"],
                expected_result_sha256=result_sha,
            )

        self.assertEqual(object_reads, [object_path])
        self.assertEqual(document, expected)
        entry["symbol"] = "2303"
        self.assertEqual(read_manifest(self.root)[1]["entries"]["2330"]["symbol"], "2330")

    def test_verified_reader_rejects_changed_bytes_before_decode(self):
        _manifest_path, manifest, object_path, result_sha = self.prepare_verified_reader()
        original_read = reconciliation._read_bytes
        changed = bytearray(self.original)
        changed[len(changed) // 2] ^= 1

        def changed_object(path, **kwargs):
            if Path(path) == object_path:
                return bytes(changed)
            return original_read(path, **kwargs)

        with (
            patch.object(reconciliation, "_read_bytes", side_effect=changed_object),
            patch.object(reconciliation, "_decode_gzip", side_effect=AssertionError),
            self.assertRaises(LegacyReconciliationError),
        ):
            self.store.read_original_document(
                symbol="2330",
                original_sha256=manifest["entries"]["2330"]["original_sha256"],
                expected_result_sha256=result_sha,
            )

    def test_verified_reader_binds_all_sizes_hash_gzip_and_path_checks(self):
        manifest_path, manifest, object_path, result_sha = self.prepare_verified_reader()
        original_manifest = json.dumps(manifest).encode("utf-8")
        original_object = object_path.read_bytes()

        def replace_object(entry, raw, decoded):
            original_sha = hashlib.sha256(raw).hexdigest()
            entry.update(
                original_sha256=original_sha,
                original_size=len(raw),
                original_uncompressed_size=len(decoded),
                backup_path=f"objects/{original_sha}.json.gz",
            )
            path = self.store.backup_root / entry["backup_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

        cases = (
            ("compressed size", lambda entry: entry.update(original_size=entry["original_size"] + 1)),
            ("invalid gzip", lambda entry: replace_object(entry, b"not-gzip", b"not-gzip")),
            (
                "gzip expansion",
                lambda entry: replace_object(
                    entry,
                    gzip.compress(b"x" * (reconciliation.MAX_UNCOMPRESSED_BYTES + 1), mtime=0),
                    b"x" * (reconciliation.MAX_UNCOMPRESSED_BYTES + 1),
                ),
            ),
            (
                "uncompressed size",
                lambda entry: entry.update(
                    original_uncompressed_size=entry["original_uncompressed_size"] + 1
                ),
            ),
            ("backup path", lambda entry: entry.update(backup_path="../escape.json.gz")),
            ("entry symbol", lambda entry: entry.update(symbol="2303")),
            ("entry status", lambda entry: entry.update(status="backup_complete")),
            ("overlap dates", lambda entry: entry.update(overlap_dates=[])),
            ("result sha", lambda entry: entry.update(new_sha256="f" * 64)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                manifest_path.write_bytes(original_manifest)
                object_path.write_bytes(original_object)
                current = json.loads(original_manifest)
                entry = current["entries"]["2330"]
                mutate(entry)
                manifest_path.write_text(json.dumps(current), encoding="utf-8")
                with self.assertRaises(LegacyReconciliationError):
                    self.store.read_original_document(
                        symbol="2330",
                        original_sha256=entry["original_sha256"],
                        expected_result_sha256=result_sha,
                    )

    def test_verified_reader_rejects_symbol_market_or_daily_identity_mismatch(self):
        manifest_path, manifest, object_path, result_sha = self.prepare_verified_reader()
        original_manifest = json.dumps(manifest).encode("utf-8")
        original_object = object_path.read_bytes()
        original_document = json.loads(self.decoded.decode("utf-8"))

        def write_document(document):
            raw, decoded = self.compressed_document(document)
            entry = manifest["entries"]["2330"]
            original_sha = hashlib.sha256(raw).hexdigest()
            entry.update(
                original_sha256=original_sha,
                original_size=len(raw),
                original_uncompressed_size=len(decoded),
                backup_path=f"objects/{original_sha}.json.gz",
            )
            path = self.store.backup_root / entry["backup_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

        def reverse_daily_dates(document):
            document["daily"].append(
                dict(document["daily"][0], Date="2026-07-15T00:00:00.000")
            )
            document["as_of"] = "2026-07-15"

        cases = (
            ("boolean schema version", lambda document: document.update(schema_version=True)),
            ("market", lambda document: document.update(market="US")),
            ("symbol", lambda document: document.update(symbol="2303")),
            ("declared date", lambda document: document.update(as_of="2026-07-15")),
            (
                "duplicate daily date",
                lambda document: document["daily"].append(dict(document["daily"][0])),
            ),
            ("reverse daily dates", reverse_daily_dates),
            ("boolean OHLCV", lambda document: document["daily"][0].update(Open=True)),
            ("nonfinite OHLCV", lambda document: document["daily"][0].update(Close=float("nan"))),
            ("oversized OHLCV", lambda document: document["daily"][0].update(Volume=10**400)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                manifest_path.write_bytes(original_manifest)
                object_path.write_bytes(original_object)
                manifest = json.loads(original_manifest)
                document = copy.deepcopy(original_document)
                mutate(document)
                write_document(document)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                entry = manifest["entries"]["2330"]
                with self.assertRaises(LegacyReconciliationError):
                    self.store.read_original_document(
                        symbol="2330",
                        original_sha256=entry["original_sha256"],
                        expected_result_sha256=result_sha,
                    )

    def test_verified_reader_rejects_symlink_and_windows_reparse_components(self):
        manifest_path, manifest, object_path, result_sha = self.prepare_verified_reader()
        original_sha = manifest["entries"]["2330"]["original_sha256"]
        with (
            patch.object(
                reconciliation,
                "_is_reparse",
                side_effect=lambda path: Path(path) == object_path.parent,
            ),
            self.assertRaises(LegacyReconciliationError),
        ):
            self.store.read_original_document(
                symbol="2330",
                original_sha256=original_sha,
                expected_result_sha256=result_sha,
            )
        real = object_path.with_name("object-real.json.gz")
        os.replace(object_path, real)
        try:
            os.symlink(real, object_path)
        except OSError:
            os.replace(real, object_path)
            with (
                patch.object(
                    reconciliation,
                    "_is_reparse",
                    side_effect=lambda path: Path(path) == object_path,
                ),
                self.assertRaises(LegacyReconciliationError),
            ):
                self.store.read_original_document(
                    symbol="2330",
                    original_sha256=original_sha,
                    expected_result_sha256=result_sha,
                )
            return
        with self.assertRaises(LegacyReconciliationError):
            self.store.read_original_document(
                symbol="2330",
                original_sha256=original_sha,
                expected_result_sha256=result_sha,
            )

    def test_backup_is_written_before_artifact_replacement(self):
        self.assertEqual(self.backup(), "write")
        manifest_path, manifest = read_manifest(self.root)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(manifest["entries"]["2330"]["status"], "backup_complete")
        self.assertTrue(manifest_path.is_file())

    def test_no_entry_without_evidence_is_passthrough(self):
        self.assertEqual(
            self.store.backup_before_write(
                symbol="2330", artifact_path=self.path, evidence=None
            ),
            "passthrough",
        )
        self.assertFalse(self.store.manifest_path.exists())
        self.assertFalse(self.store.backup_root.exists())
        self.assertIsNone(
            LegacyArtifactBackupStore.discover_resume(
                self.root,
                target_date=TARGET,
            )
        )

    def test_constructor_and_symbol_identity_fail_closed(self):
        for target_date, series_sha in (
            ("2026-07-24", SERIES_SHA),
            (TARGET, "g" * 64),
        ):
            with self.subTest(target_date=target_date, series_sha=series_sha):
                with self.assertRaises((TypeError, ValueError)):
                    LegacyArtifactBackupStore(
                        self.root,
                        target_date=target_date,
                        series_manifest_sha256=series_sha,
                    )
        with self.assertRaises(LegacyReconciliationError):
            self.store.backup_before_write(
                symbol="../2330",
                artifact_path=self.path,
                evidence=self.evidence,
            )

    def test_backup_object_is_content_addressed_and_immutable(self):
        self.backup()
        manifest_path, _manifest = read_manifest(self.root)
        target = manifest_path.parent / "objects" / f"{self.original_sha}.json.gz"
        self.assertEqual(target.read_bytes(), self.original)
        before = target.stat().st_mtime_ns
        self.assertEqual(self.backup(), "write")
        self.assertEqual(target.read_bytes(), self.original)
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_backup_manifest_records_original_and_new_hashes(self):
        self.backup()
        target = self.write_official()
        new_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        _path, manifest = read_manifest(self.root)
        entry = manifest["entries"]["2330"]
        self.assertEqual(entry["original_sha256"], self.original_sha)
        self.assertEqual(entry["status"], "applied")
        self.assertEqual(entry["new_sha256"], new_sha)
        self.assertEqual(
            self.store.assert_current_state_complete(),
            {"2330": new_sha},
        )

    def test_backup_manifest_records_compressed_and_uncompressed_sizes(self):
        self.backup()
        _path, manifest = read_manifest(self.root)
        entry = manifest["entries"]["2330"]
        self.assertEqual(entry["original_size"], len(self.original))
        self.assertEqual(entry["original_uncompressed_size"], len(self.decoded))

    def test_backup_rejects_invalid_gzip_bounds_or_evidence_sha(self):
        cases = (
            b"",
            b"not-gzip",
            gzip.compress(b"x" * (20 * 1024 * 1024 + 1), mtime=0),
        )
        for raw in cases:
            with self.subTest(size=len(raw)):
                self.path.write_bytes(raw)
                invalid_evidence = evidence(hashlib.sha256(raw).hexdigest())
                with self.assertRaises(LegacyReconciliationError):
                    self.store.backup_before_write(
                        symbol="2330",
                        artifact_path=self.path,
                        evidence=invalid_evidence,
                    )
        self.path.write_bytes(self.original)
        wrong_sha = dict(self.evidence, legacy_artifact_sha256="f" * 64)
        with self.assertRaises(LegacyReconciliationError):
            self.store.backup_before_write(
                symbol="2330",
                artifact_path=self.path,
                evidence=wrong_sha,
            )

    def test_backup_rejects_malformed_reconciliation_evidence(self):
        mutations = [
            ("schema_version", 1),
            ("official_source_schema_version", "unknown"),
            ("official_snapshot_dates", [TARGET.isoformat(), BASELINE.isoformat()]),
            ("official_snapshot_manifests", []),
            ("price_preserved_no_official_row_dates", [BASELINE.isoformat()]),
            ("institutional_replaced_dates", [TARGET.isoformat()]),
            ("margin_replaced_dates", [TARGET.isoformat()]),
            ("date_evidence", []),
        ]
        for field, invalid in mutations:
            with self.subTest(field=field):
                value = copy.deepcopy(self.evidence)
                value[field] = invalid
                with self.assertRaisesRegex(
                    LegacyReconciliationError, "evidence is invalid"
                ):
                    self.store.backup_before_write(
                        symbol="2330",
                        artifact_path=self.path,
                        evidence=value,
                    )

    def test_backup_retry_is_idempotent(self):
        self.backup()
        manifest_path, first = read_manifest(self.root)
        first_mtime = manifest_path.stat().st_mtime_ns
        time.sleep(0.01)
        self.assertEqual(self.backup(), "write")
        _path, second = read_manifest(self.root)
        self.assertEqual(second, first)
        self.assertEqual(manifest_path.stat().st_mtime_ns, first_mtime)

    def test_concurrent_backups_do_not_lose_manifest_entries(self):
        second_path, _ = write_artifact(
            self.root, legacy_document(), symbol="2303"
        )
        second_evidence = evidence(
            hashlib.sha256(second_path.read_bytes()).hexdigest()
        )
        second_store = LegacyArtifactBackupStore(
            self.root,
            target_date=TARGET,
            series_manifest_sha256=SERIES_SHA,
        )
        first_in_write = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        errors = []
        real_write = LegacyArtifactBackupStore._write_manifest

        def delayed_write(store, document):
            if threading.current_thread().name == "first-backup":
                first_in_write.set()
                if not release_first.wait(5):
                    raise AssertionError("timed out waiting to release first write")
            return real_write(store, document)

        def backup(store, symbol, path, value):
            try:
                if symbol == "2303":
                    second_started.set()
                store.backup_before_write(
                    symbol=symbol, artifact_path=path, evidence=value
                )
            except BaseException as exc:
                errors.append(exc)

        with patch.object(
            LegacyArtifactBackupStore, "_write_manifest", delayed_write
        ):
            first = threading.Thread(
                target=backup,
                args=(self.store, "2330", self.path, self.evidence),
                name="first-backup",
            )
            second = threading.Thread(
                target=backup,
                args=(second_store, "2303", second_path, second_evidence),
                name="second-backup",
            )
            first.start()
            self.assertTrue(first_in_write.wait(5))
            second.start()
            self.assertTrue(second_started.wait(5))
            time.sleep(0.05)
            self.assertTrue(second.is_alive())
            release_first.set()
            first.join(5)
            second.join(5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        _path, manifest = read_manifest(self.root)
        self.assertEqual(set(manifest["entries"]), {"2303", "2330"})

    def test_backup_complete_state_can_resume(self):
        self.backup()
        target = self.write_official()
        new_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        self.assertEqual(
            self.store.backup_before_write(
                symbol="2330", artifact_path=target, evidence=None
            ),
            "noop",
        )
        _path, manifest = read_manifest(self.root)
        self.assertEqual(manifest["entries"]["2330"]["status"], "applied")
        self.assertEqual(manifest["entries"]["2330"]["new_sha256"], new_sha)

    def test_applied_state_is_noop_without_mtime_change(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        before_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        before_mtime = target.stat().st_mtime_ns
        time.sleep(0.01)
        self.assertEqual(
            self.store.backup_before_write(
                symbol="2330", artifact_path=target, evidence=None
            ),
            "noop",
        )
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), before_sha)
        self.assertEqual(target.stat().st_mtime_ns, before_mtime)

    def test_post_run_gate_rejects_non_applied_manifest_entry(self):
        self.backup()
        with self.assertRaises(LegacyReconciliationError):
            self.store.assert_current_state_complete()

    def test_post_run_gate_rejects_applied_artifact_state_mismatch(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        write_artifact(self.root, legacy_document(datetime.date(2026, 7, 17)))
        with self.assertRaises(LegacyReconciliationError):
            self.store.assert_current_state_complete()

    def test_backup_state_conflict_fails_closed(self):
        self.backup()
        write_artifact(self.root, legacy_document(datetime.date(2026, 7, 17)))
        with self.assertRaisesRegex(
            LegacyReconciliationError,
            "legacy reconciliation state conflict for TW:2330",
        ):
            self.store.backup_before_write(
                symbol="2330", artifact_path=self.path, evidence=None
            )

    def test_mark_applied_rejects_wrong_expected_result_or_path(self):
        self.backup()
        cases = (
            {},
            {"target_market_date": "2026-07-23"},
            {"official_series_manifest_sha256": "f" * 64},
            {"legacy_reconciliation": dict(
                self.evidence,
                legacy_artifact_sha256="f" * 64,
            )},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                document = official_document(self.evidence)
                if not changes:
                    document.pop("source_lineage")
                elif "legacy_reconciliation" in changes:
                    document["source_lineage"]["legacy_reconciliation"] = changes[
                        "legacy_reconciliation"
                    ]
                else:
                    document["source_lineage"].update(changes)
                target = write_artifact(self.root, document)[0]
                with self.assertRaises(LegacyReconciliationError):
                    self.store.mark_applied(symbol="2330", artifact_path=target)
        outside = self.root / "outside.json.gz"
        outside.write_bytes(self.original)
        with self.assertRaises(LegacyReconciliationError):
            self.store.mark_applied(symbol="2330", artifact_path=outside)

    def test_mark_applied_rejects_malformed_full_official_lineage(self):
        cases = (
            {"source_schema_version": "unknown-schema"},
            {"official_snapshot_manifests": [
                {"date": BASELINE.isoformat(), "manifest_sha256": "c" * 64},
                {"date": TARGET.isoformat(), "manifest_sha256": "b" * 64},
            ]},
        )
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as root:
                path, _ = write_artifact(root, legacy_document())
                original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                value = evidence(original_sha)
                store = LegacyArtifactBackupStore(
                    Path(root),
                    target_date=TARGET,
                    series_manifest_sha256=SERIES_SHA,
                )
                store.backup_before_write(
                    symbol="2330", artifact_path=path, evidence=value
                )
                document = official_document(value)
                document["source_lineage"].update(changes)
                target = write_artifact(root, document)[0]
                with self.assertRaises(LegacyReconciliationError):
                    store.mark_applied(symbol="2330", artifact_path=target)

    def test_backup_result_rejects_outer_inner_identity_mismatch(self):
        self.backup()
        document = official_document(self.evidence)
        document["source_lineage"]["historical_artifact_sha256"] = "f" * 64
        target = write_artifact(self.root, document)[0]
        with self.assertRaises(LegacyReconciliationError):
            self.store.mark_applied(symbol="2330", artifact_path=target)

    def test_mark_applied_requires_valid_incremental_artifact(self):
        self.backup()
        malformed = official_document(self.evidence)
        malformed["as_of"] = "2026-07-23"
        target = write_artifact(self.root, malformed)[0]
        with self.assertRaises(LegacyReconciliationError):
            self.store.mark_applied(symbol="2330", artifact_path=target)

    def test_existing_object_conflict_fails_closed(self):
        object_path = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / TARGET.isoformat()
            / SERIES_SHA
            / "objects"
            / f"{self.original_sha}.json.gz"
        )
        object_path.parent.mkdir(parents=True)
        object_path.write_bytes(b"conflict")
        with self.assertRaises(LegacyReconciliationError):
            self.backup()
        self.assertEqual(object_path.read_bytes(), b"conflict")

    def test_manifest_original_identity_cannot_change(self):
        self.backup()
        write_artifact(self.root, legacy_document(datetime.date(2026, 7, 17)))
        changed = self.path.read_bytes()
        changed_evidence = evidence(
            hashlib.sha256(changed).hexdigest(),
            datetime.date(2026, 7, 17),
        )
        with self.assertRaises(LegacyReconciliationError):
            self.store.backup_before_write(
                symbol="2330",
                artifact_path=self.path,
                evidence=changed_evidence,
            )

    def test_malformed_manifest_or_backup_path_fails_closed(self):
        self.backup()
        manifest_path, manifest = read_manifest(self.root)
        manifest["entries"]["2330"]["backup_path"] = "../../escape.json.gz"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(LegacyReconciliationError):
            self.backup()

    def test_every_manifest_identity_field_is_validated(self):
        self.backup()
        manifest_path, original = read_manifest(self.root)
        mutations = (
            lambda value: value.update(schema_version=1),
            lambda value: value.update(target_market_date="2026-07-23"),
            lambda value: value.update(official_series_manifest_sha256="f" * 64),
            lambda value: value["entries"]["2330"].update(symbol="2303"),
            lambda value: value["entries"]["2330"].update(status="unknown"),
            lambda value: value["entries"]["2330"].update(original_sha256="f" * 64),
            lambda value: value["entries"]["2330"].update(original_size=0),
            lambda value: value["entries"]["2330"].update(
                original_uncompressed_size=0
            ),
            lambda value: value["entries"]["2330"].update(overlap_dates=[]),
            lambda value: value["entries"]["2330"].update(new_sha256="f" * 64),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                malformed = copy.deepcopy(original)
                mutate(malformed)
                manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
                with self.assertRaises(LegacyReconciliationError):
                    self.backup()
        manifest_path.write_text(json.dumps(original), encoding="utf-8")

    def test_path_escape_or_symlink_is_rejected(self):
        escaped = self.root / "outside.json.gz"
        escaped.write_bytes(self.original)
        with self.assertRaises(LegacyReconciliationError):
            self.store.backup_before_write(
                symbol="2330",
                artifact_path=escaped,
                evidence=self.evidence,
            )

    def test_symlink_path_component_is_rejected(self):
        real = self.root / "original-real.json.gz"
        os.replace(self.path, real)
        try:
            os.symlink(real, self.path)
        except OSError:
            os.replace(real, self.path)
            self.skipTest("filesystem does not permit symlink creation")
        with self.assertRaises(LegacyReconciliationError):
            self.store.backup_before_write(
                symbol="2330",
                artifact_path=self.path,
                evidence=self.evidence,
            )

    def test_resume_discovery_restores_original_baseline(self):
        self.backup()
        self.assertEqual(
            LegacyArtifactBackupStore.discover_resume(
                self.root,
                target_date=TARGET,
            ),
            (SERIES_SHA, BASELINE),
        )

    def test_resume_discovery_rejects_applied_artifact_sha_mismatch(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        write_artifact(self.root, legacy_document(datetime.date(2026, 7, 17)))
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            )

    def test_resume_discovery_rejects_applied_artifact_missing_lineage(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        document = official_document(self.evidence)
        document.pop("source_lineage")
        write_artifact(self.root, document)
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            )

    def test_resume_discovery_rejects_applied_artifact_tampered_reconciliation(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        document = official_document(copy.deepcopy(self.evidence))
        document["source_lineage"]["legacy_reconciliation"][
            "legacy_artifact_sha256"
        ] = "f" * 64
        write_artifact(self.root, document)
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            )

    def test_resume_discovery_rejects_backup_complete_unknown_artifact(self):
        self.backup()
        write_artifact(self.root, legacy_document(datetime.date(2026, 7, 17)))
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            )

    def test_resume_discovery_accepts_valid_applied_artifact_read_only(self):
        self.backup()
        target = self.write_official()
        self.store.mark_applied(symbol="2330", artifact_path=target)
        manifest_path, manifest = read_manifest(self.root)
        object_path = manifest_path.parent / manifest["entries"]["2330"]["backup_path"]
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (target, manifest_path, object_path)
        }
        self.assertEqual(
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            ),
            (SERIES_SHA, BASELINE),
        )
        self.assertEqual(
            before,
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (target, manifest_path, object_path)
            },
        )

    def test_resume_discovery_accepts_post_write_pre_apply_read_only(self):
        self.backup()
        target = self.write_official()
        manifest_path, before_manifest = read_manifest(self.root)
        before = (target.read_bytes(), target.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)
        self.assertEqual(
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            ),
            (SERIES_SHA, BASELINE),
        )
        self.assertEqual(read_manifest(self.root)[1], before_manifest)
        self.assertEqual(
            (target.read_bytes(), target.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns),
            before,
        )

    def test_no_price_reconciliation_resume_is_idempotent(self):
        value = no_price_evidence(self.original_sha)
        self.assertTrue(
            OfficialCompatFetcher._valid_reconciliation(value, target_date=TARGET)
        )
        self.assertEqual(
            self.store.backup_before_write(
                symbol="2330", artifact_path=self.path, evidence=value
            ),
            "write",
        )
        manifest_path, first = read_manifest(self.root)
        before_mtime = manifest_path.stat().st_mtime_ns
        self.assertEqual(
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            ),
            (SERIES_SHA, BASELINE),
        )
        self.assertEqual(
            LegacyArtifactBackupStore.discover_resume(
                self.root, target_date=TARGET
            ),
            (SERIES_SHA, BASELINE),
        )
        self.assertEqual(read_manifest(self.root)[1], first)
        self.assertEqual(manifest_path.stat().st_mtime_ns, before_mtime)

    def test_resume_discovery_rejects_multiple_series(self):
        self.backup()
        second = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / TARGET.isoformat()
            / ("f" * 64)
        )
        second.mkdir(parents=True)
        (_path, manifest) = read_manifest(self.root)
        manifest["official_series_manifest_sha256"] = "f" * 64
        (second / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        second_object = second / "objects" / f"{self.original_sha}.json.gz"
        second_object.parent.mkdir()
        second_object.write_bytes(self.original)
        with self.assertRaisesRegex(
            LegacyReconciliationError,
            "multiple legacy reconciliation series",
        ):
            LegacyArtifactBackupStore.discover_resume(
                self.root,
                target_date=TARGET,
            )

    def test_resume_discovery_rejects_malformed_series_manifest(self):
        directory = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / TARGET.isoformat()
            / SERIES_SHA
        )
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root,
                target_date=TARGET,
            )

    def test_resume_discovery_rejects_old_manifest_schema_without_write(self):
        directory = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v1"
            / TARGET.isoformat()
            / SERIES_SHA
        )
        directory.mkdir(parents=True)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        before = (manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns)
        with self.assertRaises(LegacyReconciliationError):
            LegacyArtifactBackupStore.discover_resume(
                self.root,
                target_date=TARGET,
            )
        self.assertEqual(
            (manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns), before
        )


if __name__ == "__main__":
    unittest.main()
