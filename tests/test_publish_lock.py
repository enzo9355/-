"""Cross-process report-v2 publication lock tests."""

from contextlib import redirect_stdout
from io import StringIO
import json
import multiprocessing
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from reporting.exceptions import ReportPublishError
from reporting.publish_lock import report_v2_publish_lock


LOCK_NAME = ".publish-transaction-lock"


def _hold_publish_lock(root, ready, release):
    with redirect_stdout(StringIO()):
        with report_v2_publish_lock(Path(root)):
            ready.set()
            if not release.wait(10):
                raise RuntimeError("timed out waiting to release publish lock")


def _publish_market(root, market, ready, release, attempting, completed, failed):
    import reporting.publisher as publisher
    from tests.test_report_schema_v2 import metadata

    document = metadata("pre_market")
    document["market"] = market
    document["source_manifest"] = (
        f"quant/v1/manifests/{market}-20260714T090000Z-aaaaaaaaaaaa.json"
    )
    original = publisher._publish_report_v2_impl

    def gated(*args, **kwargs):
        if ready is not None:
            ready.set()
            if not release.wait(10):
                raise RuntimeError("timed out holding report publisher")
        return original(*args, **kwargs)

    publisher._publish_report_v2_impl = gated
    try:
        attempting.set()
        with redirect_stdout(StringIO()):
            publisher.publish_report_v2(Path(root), document)
    except BaseException:
        failed.set()
        raise
    finally:
        completed.set()


class ReportV2PublishLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lock_path = self.root / "publish" / "reports" / "v2" / LOCK_NAME

    def tearDown(self):
        self.temp.cleanup()

    def test_lock_wait_is_bounded_and_stale_lock_remains_fail_closed(self):
        self.lock_path.mkdir(parents=True)
        owner_file = self.lock_path / "owner-token"
        owner_file.write_text("stale-owner", encoding="ascii")

        started = time.monotonic()
        captured = StringIO()
        with redirect_stdout(captured):
            with self.assertRaisesRegex(ReportPublishError, "timed out"):
                with report_v2_publish_lock(
                    self.root, wait_seconds=0.12, poll_seconds=0.01
                ):
                    self.fail("stale lock was bypassed")
        elapsed = time.monotonic() - started

        self.assertGreaterEqual(elapsed, 0.10)
        self.assertLess(elapsed, 1.0)
        self.assertTrue(self.lock_path.is_dir())
        self.assertEqual(owner_file.read_text(encoding="ascii"), "stale-owner")
        events = [json.loads(line)["event"] for line in captured.getvalue().splitlines()]
        self.assertEqual(events, ["waiting", "timeout"])

    def test_tw_and_us_publishers_serialize_across_real_processes(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        tw_attempting = context.Event()
        us_attempting = context.Event()
        tw_completed = context.Event()
        us_completed = context.Event()
        tw_failed = context.Event()
        us_failed = context.Event()
        tw_process = context.Process(
            target=_publish_market,
            args=(
                str(self.root), "TW", ready, release, tw_attempting,
                tw_completed, tw_failed,
            ),
        )
        us_process = context.Process(
            target=_publish_market,
            args=(
                str(self.root), "US", None, release, us_attempting,
                us_completed, us_failed,
            ),
        )
        tw_process.start()
        try:
            self.assertTrue(ready.wait(10))
            us_process.start()
            self.assertTrue(us_attempting.wait(10))
            self.assertFalse(
                us_completed.wait(0.25),
                "US publisher did not wait for the TW transaction",
            )
            release.set()
            self.assertTrue(tw_completed.wait(10))
            self.assertTrue(us_completed.wait(10))
        finally:
            release.set()
            for process in (tw_process, us_process):
                if process.pid is None:
                    continue
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(10)

        self.assertEqual(tw_process.exitcode, 0)
        self.assertEqual(us_process.exitcode, 0)
        self.assertFalse(tw_failed.is_set())
        self.assertFalse(us_failed.is_set())
        report_root = self.root / "publish" / "reports" / "v2"
        self.assertTrue((report_root / "index-TW.json").is_file())
        self.assertTrue((report_root / "index-US.json").is_file())
        self.assertFalse(self.lock_path.exists())

    def test_wait_acquire_and_release_are_observable(self):
        captured = StringIO()
        with redirect_stdout(captured):
            with report_v2_publish_lock(self.root):
                pass

        events = [json.loads(line) for line in captured.getvalue().splitlines()]
        self.assertEqual([item["event"] for item in events], ["acquired", "released"])
        self.assertEqual(events[0]["kind"], "absorb-pipeline-lock")
        self.assertEqual(events[0]["scope"], "report_v2_publish")
        self.assertIsInstance(events[0]["waited_milliseconds"], int)

    def test_closed_stdout_does_not_change_lock_or_publication_semantics(self):
        class ClosedStdout:
            def write(self, _value):
                raise BrokenPipeError("consumer closed the pipe")

            def flush(self):
                raise BrokenPipeError("consumer closed the pipe")

        with mock.patch.object(sys, "stdout", ClosedStdout()):
            with report_v2_publish_lock(self.root) as receipt:
                self.assertTrue(receipt["acquired"])

        self.assertTrue(receipt["released"])
        self.assertFalse(self.lock_path.exists())

    def test_real_closed_pipe_keeps_success_exit_code_and_releases_lock(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        holder = context.Process(
            target=_hold_publish_lock,
            args=(str(self.root), ready, release),
        )
        completed_marker = self.root / "publisher-completed"
        script = (
            "import sys; "
            "from pathlib import Path; "
            "from reporting.publish_lock import report_v2_publish_lock; "
            "root = Path(sys.argv[1]); marker = Path(sys.argv[2]); "
            "manager = report_v2_publish_lock(root); "
            "manager.__enter__(); marker.write_text('completed', encoding='ascii'); "
            "manager.__exit__(None, None, None)"
        )
        publisher = None

        holder.start()
        try:
            self.assertTrue(ready.wait(10))
            publisher = subprocess.Popen(
                [sys.executable, "-c", script, str(self.root), str(completed_marker)],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            first_line = publisher.stdout.readline()
            self.assertEqual(json.loads(first_line)["event"], "waiting")
            publisher.stdout.close()
            release.set()
            publisher.wait(timeout=20)
            stderr = publisher.stderr.read()
            publisher.stderr.close()
        finally:
            release.set()
            holder.join(10)
            if holder.is_alive():
                holder.terminate()
                holder.join(10)
            if publisher is not None and publisher.poll() is None:
                publisher.terminate()
                publisher.wait(10)

        self.assertEqual(holder.exitcode, 0)
        self.assertEqual(publisher.returncode, 0, stderr)
        self.assertEqual(completed_marker.read_text(encoding="ascii"), "completed")
        self.assertFalse(self.lock_path.exists())
        self.assertNotIn("Exception ignored", stderr)

    def test_batch_publisher_emits_structured_lock_receipts_with_default_logging(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        holder = context.Process(
            target=_hold_publish_lock,
            args=(str(self.root), ready, release),
        )
        publisher = None
        script = (
            "import json, sys; "
            "from pathlib import Path; "
            "from stock_papi.batch.cli import _publish_v2_receipt; "
            "from tests.test_report_schema_v2 import metadata; "
            "root = Path(sys.argv[1]); "
            "_publish_v2_receipt(root, metadata('pre_market')); "
            "print(json.dumps({'batch_result': 'ok'}), flush=True)"
        )

        holder.start()
        try:
            self.assertTrue(ready.wait(10))
            publisher = subprocess.Popen(
                [sys.executable, "-c", script, str(self.root)],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            first_line_queue = queue.Queue()
            reader = threading.Thread(
                target=lambda: first_line_queue.put(publisher.stdout.readline()),
                daemon=True,
            )
            reader.start()
            first_line = first_line_queue.get(timeout=10)
            self.assertEqual(json.loads(first_line)["event"], "waiting")
            release.set()
            stdout_tail, stderr = publisher.communicate(timeout=20)
            stdout = first_line + stdout_tail
        finally:
            release.set()
            holder.join(10)
            if holder.is_alive():
                holder.terminate()
                holder.join(10)
            if publisher is not None and publisher.poll() is None:
                publisher.terminate()
                publisher.wait(10)

        self.assertEqual(holder.exitcode, 0)
        self.assertEqual(publisher.returncode, 0, stderr)
        documents = [json.loads(line) for line in stdout.splitlines() if line]
        events = [
            item for item in documents
            if item.get("kind") == "absorb-pipeline-lock"
            and item.get("scope") == "report_v2_publish"
        ]
        self.assertEqual(
            [item["event"] for item in events],
            ["waiting", "acquired", "released"],
        )
        self.assertGreater(events[1]["waited_milliseconds"], 0)
        self.assertEqual(
            events[2]["waited_milliseconds"],
            events[1]["waited_milliseconds"],
        )
        self.assertEqual(documents[-1], {"batch_result": "ok"})

    def test_success_uses_unique_owner_tokens_and_releases_lock(self):
        owner_tokens = []
        for _ in range(2):
            with report_v2_publish_lock(self.root):
                owner_files = list(self.lock_path.iterdir())
                self.assertEqual(len(owner_files), 1)
                owner_token = owner_files[0].read_text(encoding="ascii")
                self.assertRegex(owner_token, r"\A[0-9a-f]{64}\Z")
                owner_tokens.append(owner_token)
            self.assertFalse(self.lock_path.exists())

        self.assertNotEqual(owner_tokens[0], owner_tokens[1])

    def test_lock_is_released_when_body_fails(self):
        with self.assertRaisesRegex(ValueError, "body failed"):
            with report_v2_publish_lock(self.root):
                raise ValueError("body failed")

        self.assertFalse(self.lock_path.exists())

    def test_foreign_owner_lock_cannot_be_removed(self):
        manager = report_v2_publish_lock(self.root)
        manager.__enter__()
        owner_file = next(self.lock_path.iterdir())
        owner_file.write_text("foreign-owner", encoding="ascii")

        with self.assertRaisesRegex(ReportPublishError, "ownership"):
            manager.__exit__(None, None, None)

        self.assertTrue(self.lock_path.is_dir())
        self.assertEqual(owner_file.read_text(encoding="ascii"), "foreign-owner")

    def test_corrupt_owner_lock_cannot_be_removed(self):
        manager = report_v2_publish_lock(self.root)
        manager.__enter__()
        owner_file = next(self.lock_path.iterdir())
        owner_file.write_bytes(b"\xff")

        with self.assertRaisesRegex(ReportPublishError, "ownership"):
            manager.__exit__(None, None, None)

        self.assertTrue(self.lock_path.is_dir())
        self.assertEqual(owner_file.read_bytes(), b"\xff")

    def test_stale_lock_fails_closed_and_is_not_removed(self):
        self.lock_path.mkdir(parents=True)
        owner_file = self.lock_path / "owner-token"
        owner_file.write_text("stale-owner", encoding="ascii")

        with self.assertRaisesRegex(ReportPublishError, "timed out"):
            with report_v2_publish_lock(
                self.root, wait_seconds=0.05, poll_seconds=0.01
            ):
                self.fail("stale lock was removed automatically")

        self.assertTrue(self.lock_path.is_dir())
        self.assertEqual(owner_file.read_text(encoding="ascii"), "stale-owner")

    def test_lock_cleanup_failure_is_reported_and_lock_remains_fail_closed(self):
        real_rmdir = Path.rmdir

        def fail_lock_rmdir(path):
            if path == self.lock_path:
                raise OSError("injected lock cleanup failure")
            return real_rmdir(path)

        captured = StringIO()
        with redirect_stdout(captured):
            with mock.patch.object(Path, "rmdir", fail_lock_rmdir):
                with self.assertRaisesRegex(ReportPublishError, "release"):
                    with report_v2_publish_lock(self.root):
                        pass

        self.assertTrue(self.lock_path.is_dir())
        events = [json.loads(line)["event"] for line in captured.getvalue().splitlines()]
        self.assertEqual(events, ["acquired", "release_failed"])
        with self.assertRaisesRegex(ReportPublishError, "timed out"):
            with report_v2_publish_lock(
                self.root, wait_seconds=0.05, poll_seconds=0.01
            ):
                self.fail("cleanup failure lock was bypassed")


if __name__ == "__main__":
    unittest.main()
