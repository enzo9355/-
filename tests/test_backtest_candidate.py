import datetime
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.batch.backtest_candidate import build_candidate
from stock_papi.batch.backtest_store import BacktestStore, BacktestStoreError
from stock_papi.batch.runtime import job_namespace


UTC = datetime.timezone.utc


class BacktestCandidateTests(unittest.TestCase):
    def _write_fixture(self, root, *, completed=True, include_oos=True, market="TW"):
        dataset_sha = "a" * 64
        job_type = "full_backtest" if market == "TW" else "full_backtest_us"
        namespace = job_namespace(root, job_type)
        namespace.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": 1,
            "job_type": job_type,
            "status": "completed" if completed else "running",
            "next_index": 1 if completed else 0,
            "item_count": 1,
            "completed_items": [("2330" if market == "TW" else "AAPL")] if completed else [],
            "dataset_manifest": f"quant/v1/manifests/{market}-20260714T090000Z-aaaaaaaaaaaa.json",
            "dataset_sha256": dataset_sha,
            "model_version": "lgbm-5d-v1",
            "feature_schema_version": 1,
            "cutoff": "2026-07-14",
        }
        if market == "US":
            checkpoint["market"] = market
        namespace.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
        result_root = namespace.output / dataset_sha / "symbols"
        result_root.mkdir(parents=True, exist_ok=True)
        predictions = [
            {
                "source_market_date": f"2026-05-{day:02d}",
                "probability": 0.6 if day % 2 else 0.4,
                "predicted_return": 0.008 if day % 2 else -0.008,
                "future_return": 0.01 if day % 2 else -0.01,
                "direction": 1 if day % 2 else 0,
                "fold_index": 0,
            }
            for day in range(1, 31)
        ]
        result = {
            "dataset_manifest": checkpoint["dataset_manifest"],
            "dataset_sha256": dataset_sha,
            "model_version": checkpoint["model_version"],
            "cutoff": checkpoint["cutoff"],
            "backtest": {"five_session_gap": True, "fold_count": 1},
            "oos_predictions": predictions if include_oos else [],
        }
        (result_root / ("2330.json" if market == "TW" else "AAPL.json")).write_text(json.dumps(result), encoding="utf-8")

    def test_builds_immutable_candidate_without_promoting_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            candidate = build_candidate(
                root,
                git_sha="b" * 40,
                now=datetime.datetime(2026, 7, 15, 1, tzinfo=UTC),
            )

            store = BacktestStore(root, "TW")
            self.assertTrue(store.candidate_path(candidate["candidate_sha256"]).exists())
            self.assertEqual(candidate["recommendation_policy_version"], "recommendation-v1")
            self.assertAlmostEqual(candidate["metrics"]["price_mae"], 0.002)
            self.assertLess(candidate["metrics"]["price_mae_ratio"], 1.0)
            self.assertIsNone(store.load_latest())
            oos_path = store.root / "oos" / f"{candidate['oos_predictions_sha256']}.json.gz"
            with gzip.open(oos_path, "rt", encoding="utf-8") as stream:
                self.assertEqual(len(json.load(stream)["predictions"]), 30)

    def test_refuses_incomplete_or_missing_oos_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, completed=False)
            with self.assertRaisesRegex(BacktestStoreError, "not complete"):
                build_candidate(root, git_sha="b" * 40)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, include_oos=False)
            with self.assertRaisesRegex(BacktestStoreError, "OOS evidence"):
                build_candidate(root, git_sha="b" * 40)

    def test_builds_us_candidate_in_us_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, market="US")
            candidate = build_candidate(root, market="US", git_sha="b" * 40)

            self.assertEqual(candidate["market"], "US")
            self.assertIsNone(BacktestStore(root, "US").load_latest())

if __name__ == "__main__":
    unittest.main()
