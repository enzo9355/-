import datetime
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_batch_calendar import calendar_document

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def taipei_today():
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTimeOffset]::UtcNow, 'Taipei Standard Time').Date.ToString('yyyy-MM-dd')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("unable to determine Taipei today")
    return datetime.date.fromisoformat(completed.stdout.strip())


def write_calendars(data_root):
    years = (taipei_today().year - 1, taipei_today().year, taipei_today().year + 1)
    directory = Path(data_root) / "publish" / "calendars" / "v1"
    directory.mkdir(parents=True, exist_ok=True)
    for year in years:
        (directory / f"TW-{year}.json").write_text(
            json.dumps(calendar_document(year), ensure_ascii=False),
            encoding="utf-8",
        )


def derived_session():
    from stock_papi.batch.calendar import TradingCalendarSet

    today = taipei_today()
    documents = [calendar_document(year) for year in (today.year - 1, today.year, today.year + 1)]
    calendars = TradingCalendarSet.from_documents(documents)
    return calendars.latest_session_on_or_before(today - datetime.timedelta(days=1))


class TwObservationRecoveryTests(unittest.TestCase):
    def _stage_scripts(self, temporary, data_root):
        staging = Path(temporary) / "scripts"
        staging.mkdir(parents=True, exist_ok=True)
        source = (SCRIPTS / "run_tw_observation_recovery.ps1").read_text(
            encoding="utf-8"
        )
        replacements = (
            ("'D:\\AbsorbData', 'D:\\StockPapiData'",
             f"'{data_root}'", 1),
            (
                "$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path",
                f"$RepoRoot = '{str(ROOT).replace(chr(39), chr(39) + chr(39))}'",
                1,
            ),
        )
        for old, new, expected in replacements:
            count = source.count(old)
            if count != expected:
                raise AssertionError(
                    f"unsafe substitution count for {old}: expected {expected}, got {count}"
                )
            source = source.replace(old, new)
        (staging / "run_tw_observation_recovery.ps1").write_text(
            source, encoding="utf-8-sig"
        )
        for helper in ("python_runtime.ps1", "catch_up_latest_completed_session.ps1"):
            (staging / helper).write_bytes(
                (SCRIPTS / helper).read_bytes()
            )
        return staging / "run_tw_observation_recovery.ps1"

    def _environment(self, fake_bin):
        environment = os.environ.copy()
        environment["ABSORB_PYTHON_EXE"] = sys.executable
        environment["PATH"] = f"{fake_bin};{environment.get('PATH', '')}"
        return environment

    def _write_pointer(self, data_root, source_date):
        pointer_dir = Path(data_root) / "publish" / "reports" / "v2"
        pointer_dir.mkdir(parents=True, exist_ok=True)
        (pointer_dir / "latest-TW-post_close.json").write_text(
            json.dumps(
                {
                    "report_type": "post_close",
                    "source_market_date": source_date,
                    "applicable_trading_date": "2099-01-01",
                }
            ),
            encoding="utf-8",
        )

    def _fake_gcloud(self, fake_bin, source_date):
        fake = Path(fake_bin) / "gcloud.cmd"
        fake.write_text(
            "@echo off\r\n"
            "if \"%~1\"==\"storage\" if \"%~2\"==\"cat\" (\r\n"
            f"  echo {{\"source_market_date\": \"{source_date}\", \"report_type\": \"post_close\"}}\r\n"
            "  exit /b 0\r\n"
            ")\r\n"
            "exit /b 1\r\n",
            encoding="utf-8",
        )

    def test_recovery_noops_when_pointers_are_current(self):
        expected = derived_session()
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            fake_bin = Path(temporary) / "fake-bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            script = self._stage_scripts(temporary, str(data_root))
            write_calendars(data_root)
            self._write_pointer(data_root, expected.isoformat())
            self._fake_gcloud(fake_bin, expected.isoformat())

            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-DataRoot",
                    str(data_root),
                ],
                env=self._environment(str(fake_bin)),
                capture_output=True,
                text=True,
                timeout=180,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(expected.isoformat(), completed.stdout)
        self.assertIn("no-op", completed.stdout)

    def test_recovery_invokes_supported_catchup_for_exact_stale_session(self):
        expected = derived_session()
        stale = expected - datetime.timedelta(days=7)
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            fake_bin = Path(temporary) / "fake-bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            script = self._stage_scripts(temporary, str(data_root))
            write_calendars(data_root)
            self._write_pointer(data_root, stale.isoformat())
            self._fake_gcloud(fake_bin, stale.isoformat())

            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-DataRoot",
                    str(data_root),
                ],
                env=self._environment(str(fake_bin)),
                capture_output=True,
                text=True,
                timeout=180,
            )

        self.assertNotEqual(completed.returncode, 0)
        output = completed.stdout + completed.stderr
        self.assertIn(expected.isoformat(), output)
        self.assertIn("invoking supported catch-up", output)
        self.assertIn("Data root is not allowlisted", output)


if __name__ == "__main__":
    unittest.main()
