import datetime
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.batch.runtime import (
    DAILY_JOB_TYPES,
    JOB_TYPES,
    JobLockError,
    acquire_job_lock,
    job_namespace,
    yield_full_backtest_to_daily,
)


UTC = datetime.timezone.utc


class BatchRuntimeTests(unittest.TestCase):
    def test_daily_observation_has_an_isolated_priority_lock(self):
        self.assertIn("daily_observation", JOB_TYPES)
        self.assertIn("full_backtest_us", JOB_TYPES)
        self.assertIn("daily_observation", DAILY_JOB_TYPES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked_at = datetime.datetime(2026, 7, 14, 9, tzinfo=UTC)
            observation = acquire_job_lock(
                root,
                "daily_observation",
                datetime.date(2026, 7, 14),
                now=checked_at,
                pid=100,
                token="a" * 32,
            )

            yielded = yield_full_backtest_to_daily(
                root,
                boundary="symbol",
                save_checkpoint=lambda _reason: None,
            )

            self.assertTrue(yielded)
            observation.release()

    def test_six_jobs_have_disjoint_lock_checkpoint_status_and_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespaces = [job_namespace(Path(temporary), job) for job in JOB_TYPES]

        for field in ("lock", "checkpoint", "status", "output"):
            values = [getattr(namespace, field) for namespace in namespaces]
            self.assertEqual(len(values), len(set(values)))

    def test_lock_is_exclusive_owner_released_and_jobs_do_not_block_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked_at = datetime.datetime(2026, 7, 14, 9, tzinfo=UTC)
            backtest = acquire_job_lock(
                root,
                "full_backtest",
                datetime.date(2026, 7, 14),
                now=checked_at,
                pid=100,
                token="a" * 32,
            )
            daily = acquire_job_lock(
                root,
                "daily_prediction",
                datetime.date(2026, 7, 14),
                now=checked_at,
                pid=101,
                token="b" * 32,
            )
            with self.assertRaises(JobLockError):
                acquire_job_lock(
                    root,
                    "daily_prediction",
                    datetime.date(2026, 7, 14),
                    now=checked_at,
                    pid=102,
                    token="c" * 32,
                )

            document = json.loads(daily.path.read_text(encoding="utf-8"))
            document["token"] = "d" * 32
            daily.path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(JobLockError):
                daily.release()
            document["token"] = daily.token
            daily.path.write_text(json.dumps(document), encoding="utf-8")
            daily.release()
            backtest.release()

    def test_stale_lock_is_archived_before_reacquire(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = datetime.datetime(2026, 7, 14, 1, tzinfo=UTC)
            old = acquire_job_lock(
                root,
                "upload",
                datetime.date(2026, 7, 14),
                now=start,
                pid=100,
                token="a" * 32,
            )

            replacement = acquire_job_lock(
                root,
                "upload",
                datetime.date(2026, 7, 14),
                now=start + datetime.timedelta(hours=7),
                pid=101,
                token="b" * 32,
                stale_after=datetime.timedelta(hours=6),
            )

            archives = list(old.path.parent.glob("upload.lock.stale.*.json"))
            self.assertEqual(len(archives), 1)
            archived = json.loads(archives[0].read_text(encoding="utf-8"))
            self.assertEqual(archived["token"], old.token)
            replacement.release()

    def test_full_backtest_saves_checkpoint_and_yields_at_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked_at = datetime.datetime(2026, 7, 14, 9, tzinfo=UTC)
            daily = acquire_job_lock(
                root,
                "post_close_report",
                datetime.date(2026, 7, 14),
                now=checked_at,
                pid=101,
                token="b" * 32,
            )
            checkpoints = []

            yielded = yield_full_backtest_to_daily(
                root,
                boundary="fold",
                save_checkpoint=lambda reason: checkpoints.append(reason),
            )

            self.assertTrue(yielded)
            self.assertEqual(checkpoints, ["daily_pipeline_active"])
            daily.release()
            self.assertFalse(
                yield_full_backtest_to_daily(
                    root,
                    boundary="symbol",
                    save_checkpoint=lambda _reason: self.fail("unexpected checkpoint"),
                )
            )

    def test_yield_rejects_non_boundary_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                yield_full_backtest_to_daily(
                    Path(temporary), boundary="row", save_checkpoint=lambda _reason: None
                )


if __name__ == "__main__":
    unittest.main()
