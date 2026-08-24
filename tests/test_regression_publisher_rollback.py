"""Transactional rollback tests for regression-aware report publication."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import reporting.publisher as publisher_module
from reporting.exceptions import ReportPublishError
from reporting.professional_schema import ProfessionalPostCloseReport
from reporting.publisher import _write_atomic as real_write_atomic
from reporting.publisher import _write_immutable as real_write_immutable
from reporting.publisher import publish_report_v2
from reporting.publish_lock import report_v2_publish_lock
from reporting.regression_schema import serialize_regression_artifact
from tests import test_canonical_publisher_integrity as canonical_helpers
from tests.regression_fixtures import make_artifact_document, rehash_artifact_document


class TestRegressionPublisherRollback(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        helper = canonical_helpers.CanonicalPublisherIntegrityTests()
        self.report_document = helper._base_report_doc()
        self.metadata = helper._base_metadata_doc(self.report_document)
        self.report = ProfessionalPostCloseReport.from_document(self.report_document)
        self.artifact = make_artifact_document()
        self.artifact["identity"]["source_manifest"] = self.report.identity.source_manifest
        self.artifact["identity"]["source_manifest_sha256"] = self.report.identity.source_manifest_sha256
        rehash_artifact_document(self.artifact)
        self.regression_bytes = serialize_regression_artifact(self.artifact)
        self.regression_sha = hashlib.sha256(self.regression_bytes).hexdigest()
        self.publish_dir = self.root / "publish" / "reports" / "v2"
        self.lock_path = self.publish_dir / ".publish-transaction-lock"
        self.index_path = self.publish_dir / "index-TW.json"
        self.latest_path = self.publish_dir / "latest-TW-post_close.json"
        self.previous_index = json.dumps(
            {
                "schema_version": 2,
                "kind": "absorb-report-index",
                "market": "TW",
                "updated_at": "2026-07-16T10:30:00Z",
                "reports": [],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        self.previous_latest = b'{"previous":true}'
        self.index_path.parent.mkdir(parents=True)
        self.index_path.write_bytes(self.previous_index)
        self.latest_path.write_bytes(self.previous_latest)

    def tearDown(self):
        self.temp.cleanup()

    def publish(self):
        return publish_report_v2(
            self.root,
            copy.deepcopy(self.metadata),
            professional_report=self.report,
            regression_artifact=copy.deepcopy(self.artifact),
        )

    def assert_previous_pointers_restored(self):
        self.assertEqual(self.index_path.read_bytes(), self.previous_index)
        self.assertEqual(self.latest_path.read_bytes(), self.previous_latest)

    def fail_write_for(self, fragment, *, after_write=False):
        def writer(path, content):
            if fragment in path.as_posix():
                if after_write:
                    real_write_atomic(path, content)
                raise OSError(f"injected {fragment} failure")
            return real_write_atomic(path, content)
        return writer

    def fail_immutable_write_for(self, fragment):
        def writer(path, content, conflict_message):
            if fragment in path.as_posix():
                raise OSError(f"injected {fragment} failure")
            return real_write_immutable(path, content, conflict_message)

        return writer

    def race_after_exists_check(self, target, content):
        real_exists = Path.exists
        raced = False

        def exists(path):
            nonlocal raced
            if path == target and not raced:
                raced = True
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                return False
            return real_exists(path)

        return exists

    def test_regression_canonical_metadata_index_and_latest_failures_roll_back(self):
        stages = (
            "objects/regression/",
            "objects/canonical/",
            "metadata/",
            "index-TW.json",
            "latest-TW-post_close.json",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory() as directory:
                    original_root = self.root
                    self.root = Path(directory)
                    self.publish_dir = self.root / "publish" / "reports" / "v2"
                    self.index_path = self.publish_dir / "index-TW.json"
                    self.latest_path = self.publish_dir / "latest-TW-post_close.json"
                    self.index_path.parent.mkdir(parents=True)
                    self.index_path.write_bytes(self.previous_index)
                    self.latest_path.write_bytes(self.previous_latest)
                    try:
                        target = (
                            "reporting.publisher._write_immutable"
                            if stage in {"objects/regression/", "objects/canonical/", "metadata/"}
                            else "reporting.publisher._write_atomic"
                        )
                        side_effect = (
                            self.fail_immutable_write_for(stage)
                            if target.endswith("_write_immutable")
                            else self.fail_write_for(stage)
                        )
                        with mock.patch(target, side_effect=side_effect):
                            with self.assertRaises(Exception):
                                self.publish()
                        self.assert_previous_pointers_restored()
                        self.assertFalse(any((self.publish_dir / "objects" / "canonical").glob("*.json")))
                        self.assertFalse(any((self.publish_dir / "objects" / "regression").glob("*.json")))
                        self.assertFalse(any((self.publish_dir / "metadata").glob("*.json")))
                    finally:
                        self.root = original_root

    def test_failure_after_latest_replace_restores_previous_latest_and_index(self):
        with mock.patch(
            "reporting.publisher._write_atomic",
            side_effect=self.fail_write_for("latest-TW-post_close.json", after_write=True),
        ):
            with self.assertRaises(Exception):
                self.publish()
        self.assert_previous_pointers_restored()

    def test_lock_is_acquired_before_previous_pointer_snapshot(self):
        real_exists = Path.exists
        pointer_reads = []

        def exists(path):
            if path in {self.index_path, self.latest_path}:
                self.assertTrue(real_exists(self.lock_path))
                pointer_reads.append(path)
            return real_exists(path)

        with mock.patch.object(Path, "exists", autospec=True, side_effect=exists), mock.patch(
            "reporting.publisher._publish_report_v2_impl",
            return_value=self.latest_path,
        ):
            self.assertEqual(self.publish(), self.latest_path)

        self.assertEqual(pointer_reads, [self.index_path, self.latest_path])
        self.assertFalse(self.lock_path.exists())

    def test_contending_transaction_cannot_interleave_with_successful_publish(self):
        real_impl = publisher_module._publish_report_v2_impl
        first_published = threading.Event()
        release_first = threading.Event()
        first_thread = None
        second_thread = None
        first_errors = []
        second_errors = []
        second_completed = threading.Event()
        impl_threads = []

        def controlled_impl(*args, **kwargs):
            result = real_impl(*args, **kwargs)
            impl_threads.append(threading.current_thread())
            if threading.current_thread() is first_thread:
                first_published.set()
                if not release_first.wait(10):
                    raise RuntimeError("timed out holding publish transaction lock")
            return result

        def publish_first():
            try:
                self.publish()
            except BaseException as exc:
                first_errors.append(exc)

        def publish_second():
            try:
                self.publish()
            except BaseException as exc:
                second_errors.append(exc)
            finally:
                second_completed.set()

        with mock.patch(
            "reporting.publisher._publish_report_v2_impl",
            side_effect=controlled_impl,
        ):
            first_thread = threading.Thread(target=publish_first)
            first_thread.start()
            try:
                self.assertTrue(first_published.wait(10))
                successful_state = {
                    path.relative_to(self.publish_dir): path.read_bytes()
                    for path in self.publish_dir.rglob("*")
                    if path.is_file() and self.lock_path not in path.parents
                }

                second_thread = threading.Thread(target=publish_second)
                second_thread.start()
                self.assertFalse(
                    second_completed.wait(0.25),
                    "contending publisher did not wait for the transaction lock",
                )

                contended_state = {
                    path.relative_to(self.publish_dir): path.read_bytes()
                    for path in self.publish_dir.rglob("*")
                    if path.is_file() and self.lock_path not in path.parents
                }
                self.assertEqual(contended_state, successful_state)
                self.assertEqual(impl_threads, [first_thread])
            finally:
                release_first.set()
                first_thread.join(10)
                if second_thread is not None:
                    second_thread.join(10)

            self.assertFalse(first_thread.is_alive())
            self.assertIsNotNone(second_thread)
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(first_errors, [])
            self.assertEqual(second_errors, [])

        self.assertEqual(len(impl_threads), 2)
        self.assertFalse(self.lock_path.exists())

    def test_lock_released_after_successful_rollback(self):
        with mock.patch(
            "reporting.publisher._write_atomic",
            side_effect=self.fail_write_for("index-TW.json"),
        ):
            with self.assertRaises(OSError):
                self.publish()

        self.assert_previous_pointers_restored()
        self.assertFalse(self.lock_path.exists())

    def test_regression_readback_mismatch_rolls_back_new_object(self):
        original_read = Path.read_bytes

        def corrupted_read(path):
            payload = original_read(path)
            if "objects/regression/" in path.as_posix():
                return payload[:-1] + bytes([payload[-1] ^ 1])
            return payload

        with mock.patch.object(Path, "read_bytes", corrupted_read):
            with self.assertRaises(ReportPublishError):
                self.publish()
        self.assert_previous_pointers_restored()
        self.assertFalse(any((self.publish_dir / "objects" / "regression").glob("*.json")))

    def test_rollback_does_not_delete_object_recreated_after_owned_delete(self):
        regression_path = self.publish_dir / "objects" / "regression" / f"{self.regression_sha}.json"
        competitor = b"recreated by another writer"
        original_read = Path.read_bytes
        original_unlink = Path.unlink
        recreated = False

        def corrupted_read(path):
            payload = original_read(path)
            if path == regression_path:
                return payload[:-1] + bytes([payload[-1] ^ 1])
            return payload

        def recreate_after_delete(path, *args, **kwargs):
            nonlocal recreated
            result = original_unlink(path, *args, **kwargs)
            if path == regression_path and not recreated:
                recreated = True
                path.write_bytes(competitor)
            return result

        with mock.patch.object(Path, "read_bytes", corrupted_read), mock.patch.object(
            Path,
            "unlink",
            recreate_after_delete,
        ):
            with self.assertRaises(ReportPublishError):
                self.publish()

        self.assertTrue(regression_path.exists())
        self.assertEqual(regression_path.read_bytes(), competitor)
        self.assert_previous_pointers_restored()

    def test_regression_content_hash_mismatch_writes_nothing(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["presentation"]["headline"] += " tampered"
        with self.assertRaises(ReportPublishError):
            publish_report_v2(
                self.root,
                copy.deepcopy(self.metadata),
                professional_report=self.report,
                regression_artifact=artifact,
            )
        self.assert_previous_pointers_restored()
        self.assertFalse(any((self.publish_dir / "objects" / "regression").glob("*.json")))
        self.assertFalse(any((self.publish_dir / "objects" / "canonical").glob("*.json")))
        self.assertFalse(any((self.publish_dir / "metadata").glob("*.json")))

    def test_existing_identical_regression_object_is_reused_and_never_deleted(self):
        regression_path = self.publish_dir / "objects" / "regression" / f"{self.regression_sha}.json"
        regression_path.parent.mkdir(parents=True)
        regression_path.write_bytes(self.regression_bytes)
        with mock.patch(
            "reporting.publisher._write_atomic",
            side_effect=self.fail_write_for("index-TW.json"),
        ):
            with self.assertRaises(Exception):
                self.publish()
        self.assertEqual(regression_path.read_bytes(), self.regression_bytes)
        self.assert_previous_pointers_restored()

    def test_same_content_race_is_reused_and_survives_later_rollback(self):
        regression_path = self.publish_dir / "objects" / "regression" / f"{self.regression_sha}.json"
        with mock.patch.object(
            Path,
            "exists",
            autospec=True,
            side_effect=self.race_after_exists_check(regression_path, self.regression_bytes),
        ), mock.patch(
            "reporting.publisher._write_atomic",
            side_effect=self.fail_write_for("index-TW.json"),
        ):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(regression_path.read_bytes(), self.regression_bytes)
        self.assert_previous_pointers_restored()

    def test_different_content_race_fails_closed_and_preserves_competitor(self):
        regression_path = self.publish_dir / "objects" / "regression" / f"{self.regression_sha}.json"
        competitor = b"concurrent writer bytes"
        with mock.patch.object(
            Path,
            "exists",
            autospec=True,
            side_effect=self.race_after_exists_check(regression_path, competitor),
        ):
            with self.assertRaisesRegex(ReportPublishError, "immutable regression"):
                self.publish()
        self.assertEqual(regression_path.read_bytes(), competitor)
        self.assert_previous_pointers_restored()

    def test_concurrent_immutable_writers_use_distinct_temps_and_single_owner(self):
        writer = getattr(publisher_module, "_write_immutable", None)
        self.assertIsNotNone(writer)
        path = self.publish_dir / "objects" / "regression" / "race.json"
        payload = b"same bytes"
        real_link = os.link
        temporary_paths = []

        def race(source, target):
            temporary_paths.append(Path(source))
            if len(temporary_paths) == 1:
                self.assertTrue(writer(Path(target), payload, "immutable conflict"))
            return real_link(source, target)

        with mock.patch.object(publisher_module.os, "link", side_effect=race):
            self.assertFalse(writer(path, payload, "immutable conflict"))

        self.assertEqual(len(temporary_paths), 2)
        self.assertEqual(len(set(temporary_paths)), 2)
        self.assertEqual(path.read_bytes(), payload)

    def test_atomic_pointer_writes_use_unique_temporary_paths(self):
        temporary_paths = []

        def record(source, _target):
            temporary_paths.append(Path(source))

        with mock.patch.object(publisher_module.os, "replace", side_effect=record):
            real_write_atomic(self.index_path, b"one")
            real_write_atomic(self.index_path, b"two")

        self.assertEqual(len(temporary_paths), 2)
        self.assertEqual(len(set(temporary_paths)), 2)

    def test_metadata_conflict_rolls_back_new_regression_and_canonical_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            clean_root = Path(directory)
            latest = publish_report_v2(
                clean_root,
                copy.deepcopy(self.metadata),
                professional_report=self.report,
                regression_artifact=copy.deepcopy(self.artifact),
            )
            metadata_relative = json.loads(latest.read_text(encoding="utf-8"))["metadata"]
        conflict = self.publish_dir / metadata_relative
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"conflict")
        with self.assertRaisesRegex(ReportPublishError, "metadata conflict"):
            self.publish()
        self.assertEqual(conflict.read_bytes(), b"conflict")
        self.assert_previous_pointers_restored()
        self.assertFalse(any((self.publish_dir / "objects" / "regression").glob("*.json")))
        self.assertFalse(any((self.publish_dir / "objects" / "canonical").glob("*.json")))

    def test_cleanup_failure_is_reported_after_restoring_previous_pointers(self):
        original_unlink = Path.unlink

        def fail_regression_cleanup(path, *args, **kwargs):
            if "objects/regression/" in path.as_posix() and path.suffix == ".json":
                raise OSError("cleanup")
            return original_unlink(path, *args, **kwargs)

        with mock.patch(
            "reporting.publisher._write_atomic",
            side_effect=self.fail_write_for("index-TW.json"),
        ), mock.patch.object(Path, "unlink", fail_regression_cleanup):
            with self.assertRaisesRegex(ReportPublishError, "rollback"):
                self.publish()
        self.assert_previous_pointers_restored()
        self.assertFalse(self.lock_path.exists())

    def test_rollback_only_deletes_paths_created_by_this_publication(self):
        unrelated = self.publish_dir / "objects" / "regression" / "unrelated.json"

        def writer(path, content):
            if path.name == "index-TW.json":
                unrelated.parent.mkdir(parents=True, exist_ok=True)
                unrelated.write_bytes(b"concurrent writer")
                raise OSError("injected index failure")
            return real_write_atomic(path, content)

        with mock.patch("reporting.publisher._write_atomic", side_effect=writer):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(unrelated.read_bytes(), b"concurrent writer")
        self.assert_previous_pointers_restored()


if __name__ == "__main__":
    unittest.main()
