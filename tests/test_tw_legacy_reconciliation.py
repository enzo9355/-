import datetime
import copy
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

from stock_papi.quant.tw_legacy_reconciliation import (
    LegacyArtifactBackupStore,
    LegacyReconciliationError,
)
from stock_papi.quant.tw_incremental import OfficialCompatFetcher


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
        self.assertFalse((self.root / "quarantine").exists())

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
