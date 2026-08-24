import datetime
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.batch.backtest_store import (
    REQUIRED_PROMOTION_GATES,
    BacktestStore,
    BacktestStoreError,
    assess_backtest_compatibility,
)
from stock_papi.batch.backtest_worker import FullBacktestWorker
from stock_papi.batch.runtime import acquire_job_lock


UTC = datetime.timezone.utc


def candidate():
    return {
        "schema_version": 1,
        "market": "TW",
        "dataset_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
        "dataset_sha256": "a" * 64,
        "model_version": "lgbm-5d-v1",
        "feature_schema_version": 1,
        "recommendation_policy_version": "recommendation-v1",
        "cutoff": "2026-07-14",
        "data_start": "2024-01-02",
        "data_end": "2026-07-14",
        "fold_count": 6,
        "five_session_gap": True,
        "oos_observations": 180,
        "oos_predictions_path": "backtests/v1/oos/" + "b" * 64 + ".json.gz",
        "oos_predictions_sha256": "b" * 64,
        "metrics": {"accuracy": 55.0, "brier": 0.24},
        "generated_at": "2026-07-14T10:00:00Z",
        "git_sha": "a" * 40,
    }


def promoted_candidate(model_version="lgbm-5d-v1"):
    document = candidate()
    document.update(
        model_version=model_version,
        candidate_sha256="c" * 64,
        promoted_at="2026-07-14T11:00:00Z",
        gates={gate: True for gate in REQUIRED_PROMOTION_GATES},
    )
    return document


class BacktestWorkerTests(unittest.TestCase):
    def test_completed_checkpoint_verifier_binds_every_source_identity_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = FullBacktestWorker(
                root,
                dataset_manifest="quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                dataset_sha256="a" * 64,
                model_version="lgbm-5d-v1",
                feature_schema_version=1,
                cutoff=datetime.date(2026, 7, 14),
                items=("2330", "2317"),
            )
            worker.run(
                lambda _item: None,
                now=datetime.datetime(2026, 7, 14, 12, tzinfo=UTC),
            )
            checkpoint_path = worker.checkpoint_path
            completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertTrue(worker.verify_completed_checkpoint())

            mutations = {
                "schema_version": 2,
                "job_type": "daily_prediction",
                "dataset_manifest": "quant/v1/manifests/TW-20260715T090000Z-bbbbbbbbbbbb.json",
                "dataset_sha256": "b" * 64,
                "model_version": "lgbm-5d-v2",
                "feature_schema_version": 2,
                "cutoff": "2026-07-15",
                "items_sha256": "c" * 64,
                "item_count": 1,
                "next_index": 1,
                "completed_items": ["2330"],
            }
            for field, value in mutations.items():
                with self.subTest(field=field):
                    tampered = dict(completed)
                    tampered[field] = value
                    checkpoint_path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaises(BacktestStoreError):
                        worker.verify_completed_checkpoint()
            incomplete = dict(completed, status="running")
            checkpoint_path.write_text(json.dumps(incomplete), encoding="utf-8")
            self.assertFalse(worker.verify_completed_checkpoint())
            checkpoint_path.write_text(json.dumps(completed), encoding="utf-8")

    def test_worker_resumes_same_dataset_across_days_and_rejects_dataset_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "dataset_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                "dataset_sha256": "a" * 64,
                "model_version": "lgbm-5d-v1",
                "feature_schema_version": 1,
                "cutoff": datetime.date(2026, 7, 14),
                "items": ("2330", "2317"),
            }
            calls = []
            first = FullBacktestWorker(root, **kwargs).run(
                lambda item: calls.append(item),
                max_items=1,
                now=datetime.datetime(2026, 7, 14, 12, tzinfo=UTC),
            )
            self.assertFalse(
                FullBacktestWorker(root, **kwargs).verify_completed_checkpoint()
            )
            second = FullBacktestWorker(root, **kwargs).run(
                lambda item: calls.append(item),
                now=datetime.datetime(2026, 7, 15, 12, tzinfo=UTC),
            )

            self.assertEqual(calls, ["2330", "2317"])
            self.assertEqual(first["next_index"], 1)
            self.assertEqual(second["status"], "completed")
            changed = dict(kwargs, dataset_sha256="b" * 64)
            with self.assertRaises(BacktestStoreError):
                FullBacktestWorker(root, **changed).run(
                    lambda _item: self.fail("changed dataset must not run"),
                    now=datetime.datetime(2026, 7, 16, 12, tzinfo=UTC),
                )

    def test_explicit_rotation_archives_self_consistent_completed_stale_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = FullBacktestWorker(
                root,
                dataset_manifest="quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                dataset_sha256="a" * 64,
                model_version="lgbm-5d-v1",
                feature_schema_version=1,
                cutoff=datetime.date(2026, 7, 14),
                items=("2330", "2317"),
            )
            old.run(
                lambda _item: None,
                now=datetime.datetime(2026, 7, 14, 12, tzinfo=UTC),
            )
            old_document = json.loads(old.checkpoint_path.read_text(encoding="utf-8"))
            current = FullBacktestWorker(
                root,
                dataset_manifest="quant/v1/manifests/TW-20260824T090000Z-bbbbbbbbbbbb.json",
                dataset_sha256="b" * 64,
                model_version="lgbm-5d-v1",
                feature_schema_version=1,
                cutoff=datetime.date(2026, 8, 24),
                items=("2330", "2317"),
            )

            self.assertTrue(current.has_archivable_stale_completed_checkpoint())
            calls = []
            result = current.run(
                calls.append,
                archive_stale_completed=True,
                now=datetime.datetime(2026, 8, 25, 12, tzinfo=UTC),
            )

            self.assertEqual(calls, ["2330", "2317"])
            self.assertEqual(result["status"], "completed")
            archives = list(old.checkpoint_path.parent.glob("archive/completed-*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8")), old_document)
            active = json.loads(old.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(active["dataset_sha256"], "b" * 64)

    def test_worker_yields_to_daily_lock_before_next_item(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked_at = datetime.datetime(2026, 7, 14, 12, tzinfo=UTC)
            daily = acquire_job_lock(
                root,
                "daily_prediction",
                datetime.date(2026, 7, 14),
                now=checked_at,
                pid=100,
                token="d" * 32,
            )
            worker = FullBacktestWorker(
                root,
                dataset_manifest="quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                dataset_sha256="a" * 64,
                model_version="lgbm-5d-v1",
                feature_schema_version=1,
                cutoff=datetime.date(2026, 7, 14),
                items=("2330",),
            )

            result = worker.run(
                lambda _item: self.fail("worker must yield before item"),
                now=checked_at,
            )

            self.assertEqual(result["status"], "yielded")
            self.assertEqual(result["yield_reason"], "daily_pipeline_active")
            daily.release()

    def test_candidate_is_immutable_and_not_visible_as_latest_before_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = BacktestStore(Path(temporary), "TW")
            digest = store.write_candidate(candidate())

            self.assertTrue(store.candidate_path(digest).exists())
            self.assertIsNone(store.load_latest())
            changed = candidate()
            changed["metrics"] = {"accuracy": 99.0, "brier": 0.01}
            with self.assertRaises(BacktestStoreError):
                store.write_candidate(changed, candidate_id=digest)

    def test_promotion_requires_every_gate_and_updates_latest_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = BacktestStore(Path(temporary), "TW")
            digest = store.write_candidate(candidate())
            gates = {gate: True for gate in REQUIRED_PROMOTION_GATES}
            gates["leakage"] = False
            with self.assertRaises(BacktestStoreError):
                store.promote(digest, gates=gates, promoted_at=datetime.datetime.now(UTC))
            self.assertIsNone(store.load_latest())

            gates["leakage"] = True
            promoted = store.promote(
                digest,
                gates=gates,
                promoted_at=datetime.datetime(2026, 7, 14, 11, tzinfo=UTC),
            )

            self.assertEqual(store.load_latest()["candidate_sha256"], digest)
            self.assertEqual(promoted["model_version"], "lgbm-5d-v1")

    def test_version_mismatch_caps_confidence_and_blocks_strong_action(self):
        compatible = assess_backtest_compatibility(
            promoted_candidate(), expected_model_version="lgbm-5d-v1"
        )
        mismatch = assess_backtest_compatibility(
            promoted_candidate(), expected_model_version="lgbm-5d-v2"
        )

        self.assertTrue(compatible["strong_action_allowed"])
        self.assertEqual(compatible["confidence_cap"], "normal")
        self.assertFalse(mismatch["strong_action_allowed"])
        self.assertEqual(mismatch["confidence_cap"], "low")
        self.assertEqual(mismatch["reason"], "model_version_mismatch")
        feature_mismatch = assess_backtest_compatibility(
            promoted_candidate(),
            expected_model_version="lgbm-5d-v1",
            expected_feature_schema_version=2,
        )
        self.assertEqual(feature_mismatch["mismatch_fields"], ["feature_schema_version"])


if __name__ == "__main__":
    unittest.main()
