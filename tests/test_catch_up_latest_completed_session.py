import datetime
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.batch.catch_up_latest_completed_session import (
    CatchUpContractError,
    LivePointerDate,
    classify_live_pointer_dates,
    validate_target_session,
)


def calendar_set(*, special_open_dates=()):
    document = {
        "schema_version": 1,
        "market": "TW",
        "year": 2026,
        "source_url": "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule",
        "fetched_at": "2026-01-01T00:00:00Z",
        "source_sha256": "a" * 64,
        "valid_from": "2026-01-01",
        "valid_to": "2026-12-31",
        "closed_dates": ["2026-01-01"],
        "special_open_dates": list(special_open_dates),
    }
    return TradingCalendarSet.from_documents([document])


class CatchUpLatestCompletedSessionTests(unittest.TestCase):
    TODAY = datetime.date(2026, 8, 16)
    TARGET = datetime.date(2026, 8, 14)

    def test_latest_completed_session_is_accepted(self):
        result = validate_target_session(
            calendar_set(), target_date=self.TARGET, local_today=self.TODAY
        )
        self.assertEqual(result["latest_completed_session"], "2026-08-14")

    def test_arbitrary_older_session_is_rejected(self):
        with self.assertRaisesRegex(CatchUpContractError, "latest completed"):
            validate_target_session(
                calendar_set(),
                target_date=datetime.date(2026, 8, 13),
                local_today=self.TODAY,
            )

    def test_future_target_is_rejected(self):
        with self.assertRaisesRegex(CatchUpContractError, "strictly before"):
            validate_target_session(
                calendar_set(),
                target_date=datetime.date(2026, 8, 17),
                local_today=self.TODAY,
            )

    def test_non_session_target_is_rejected(self):
        with self.assertRaisesRegex(CatchUpContractError, "not a TW trading session"):
            validate_target_session(
                calendar_set(),
                target_date=datetime.date(2026, 8, 15),
                local_today=self.TODAY,
            )

    def test_later_special_open_session_is_rejected(self):
        calendars = calendar_set(special_open_dates=["2026-08-15"])
        with self.assertRaisesRegex(CatchUpContractError, "latest completed"):
            validate_target_session(
                calendars, target_date=self.TARGET, local_today=self.TODAY
            )

    def test_live_newer_than_target_is_rejected(self):
        pointers = [
            LivePointerDate("quant", datetime.date(2026, 8, 15)),
            LivePointerDate("dashboard", datetime.date(2026, 8, 15)),
        ]
        with self.assertRaisesRegex(CatchUpContractError, "newer"):
            classify_live_pointer_dates(
                pointers, target_date=self.TARGET, local_today=self.TODAY
            )

    def test_live_future_is_rejected(self):
        pointers = [
            LivePointerDate("quant", datetime.date(2026, 8, 17)),
            LivePointerDate("dashboard", datetime.date(2026, 8, 17)),
        ]
        with self.assertRaisesRegex(CatchUpContractError, "future"):
            classify_live_pointer_dates(
                pointers, target_date=self.TARGET, local_today=self.TODAY
            )

    def test_partial_live_pointer_state_is_rejected(self):
        pointers = [
            LivePointerDate("quant", datetime.date(2026, 8, 14)),
            LivePointerDate("dashboard", datetime.date(2026, 8, 7)),
        ]
        with self.assertRaisesRegex(CatchUpContractError, "date-coherent"):
            classify_live_pointer_dates(
                pointers, target_date=self.TARGET, local_today=self.TODAY
            )

    def test_matching_live_target_is_idempotent(self):
        pointers = [
            LivePointerDate("quant", self.TARGET),
            LivePointerDate("dashboard", self.TARGET),
        ]
        result = classify_live_pointer_dates(
            pointers, target_date=self.TARGET, local_today=self.TODAY
        )
        self.assertEqual(result["mode"], "idempotent")


class CatchUpScriptContractTests(unittest.TestCase):
    SCRIPT = (
        Path(__file__).parents[1]
        / "scripts"
        / "catch_up_latest_completed_session.ps1"
    )

    def test_dedicated_script_is_fail_closed_and_reuses_release_guards(self):
        source = self.SCRIPT.read_text(encoding="utf-8")
        release_common = (
            self.SCRIPT.parent / "observation_release_common.ps1"
        ).read_text(encoding="utf-8")
        uploader = (self.SCRIPT.parent / "upload_local_quant.ps1").read_text(
            encoding="utf-8"
        )
        for required in (
            "[Parameter(Mandatory)]",
            "run_tw_post_close_pipeline.ps1",
            "Invoke-NativeProcessCaptured",
            "Invoke-PowerShellScript",
            "-ExecutionPolicy",
            "-File",
            "capture_observation_lkg.ps1",
            "upload_local_quant.ps1",
            "-LkgReceiptPath",
            "-RequireReportV2",
            "-RequireDashboard",
            "-ObservationOnly",
            "Get-GcloudObjectState",
            "Assert-GcloudFileMatches",
            "pending",
            "Assert-NoPendingPointerJournals",
            "Assert-IdentityMatch",
            "Taipei Standard Time",
            "tw-catch-up.lock",
            "Global\\ABSORB-TW-Observation-Writer",
            "Live pointer state changed before LKG capture",
            "idempotent",
            "TargetDate",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertIn("Invoke-GcloudConditionalCopy", release_common)
        self.assertIn("Invoke-GcloudConditionalCopy", uploader)
        self.assertNotIn(
            "run_tw_post_close_pipeline.ps1') -DataRoot $DataRoot "
            "-TargetDate $TargetDate -PublishObservation",
            source,
        )
        task_wrapper = (
            Path(__file__).parents[1]
            / "scripts"
            / "invoke_pipeline_task.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("Global\\ABSORB-TW-Observation-Writer", task_wrapper)
        for job in (
            "'TW-PostClose'",
            "'TW-PreMarket'",
            "'WeeklyModel'",
            "'ReportUploadRecovery'",
        ):
            self.assertIn(job, task_wrapper)
        self.assertIn("WaitOne(0)", task_wrapper)

    def test_normal_historical_publish_guard_remains_unchanged(self):
        source = (
            Path(__file__).parents[1]
            / "scripts"
            / "run_tw_post_close_pipeline.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if ($HistoricalTargetDate -and $PublishObservation)", source
        )
        self.assertIn(
            "Historical TargetDate cannot publish observation products", source
        )

    def test_report_index_delta_gate_executes_against_real_powershell_function(self):
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        source_path = self.SCRIPT.resolve()
        old_entry = {
            "report_type": "post_close",
            "product_mode": "observation",
            "source_market_date": "2026-08-07",
            "applicable_trading_date": "2026-08-10",
            "metadata": "metadata/" + "a" * 64 + ".json",
            "metadata_sha256": "a" * 64,
            "title": "old",
        }
        target_entry = {
            "report_type": "post_close",
            "product_mode": "observation",
            "source_market_date": "2026-08-14",
            "applicable_trading_date": "2026-08-17",
            "metadata": "metadata/" + "b" * 64 + ".json",
            "metadata_sha256": "b" * 64,
            "title": "target",
        }
        captured = {
            "schema_version": 2,
            "kind": "absorb-report-index",
            "market": "TW",
            "reports": [old_entry],
        }
        accepted_local = dict(captured)
        accepted_local["reports"] = [target_entry, old_entry]
        rejected_local = dict(captured)
        rejected_local["reports"] = [
            target_entry,
            old_entry,
            {
                **old_entry,
                "source_market_date": "2026-08-06",
                "metadata": "metadata/" + "c" * 64 + ".json",
                "metadata_sha256": "c" * 64,
            },
        ]

        def ps_literal(value):
            return "'" + str(value).replace("'", "''") + "'"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captured_path = root / "captured.json"
            accepted_path = root / "accepted.json"
            rejected_path = root / "rejected.json"
            captured_path.write_text(json.dumps(captured), encoding="utf-8")
            accepted_path.write_text(json.dumps(accepted_local), encoding="utf-8")
            rejected_path.write_text(json.dumps(rejected_local), encoding="utf-8")
            harness = f"""
$ErrorActionPreference = 'Stop'
$source = [IO.File]::ReadAllText({ps_literal(source_path)})
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
foreach ($name in @(
    'ConvertTo-CanonicalJsonValue',
    'Get-CanonicalJson',
    'Assert-ObservationReportIndexCatchUpDelta'
)) {{
    $function = $ast.Find({{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
    }}, $true)
    if ($null -eq $function) {{ throw "function not found: $name" }}
    . ([scriptblock]::Create($function.Extent.Text))
}}
function Read-JsonWithinRoot {{
    param([string]$Path)
    return [pscustomobject]@{{
        path = $Path
        document = (Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json)
    }}
}}
$TargetDate = '2026-08-14'
$captured = Get-Content -LiteralPath {ps_literal(captured_path)} -Raw -Encoding utf8 | ConvertFrom-Json
$local = Get-Content -LiteralPath {ps_literal(accepted_path)} -Raw -Encoding utf8 | ConvertFrom-Json
$pointers = [pscustomobject]@{{
    reports_index = [pscustomobject]@{{ path = {ps_literal(accepted_path)} }}
    reports_latest = [pscustomobject]@{{
        identity = [pscustomobject]@{{
            metadata = 'metadata/{'b' * 64}.json'
            metadata_sha256 = '{'b' * 64}'
            applicable_trading_date = '2026-08-17'
        }}
    }}
}}
Assert-ObservationReportIndexCatchUpDelta -CapturedIndex $captured -LocalPointers $pointers
$rejected = Get-Content -LiteralPath {ps_literal(rejected_path)} -Raw -Encoding utf8 | ConvertFrom-Json
$pointers.reports_index.path = {ps_literal(rejected_path)}
try {{
    Assert-ObservationReportIndexCatchUpDelta -CapturedIndex $captured -LocalPointers $pointers
    throw 'unauthorized delta was accepted'
}}
catch [System.Exception] {{
    if ($_.Exception.Message -notmatch 'unauthorized entry delta|changed a captured entry') {{ throw }}
}}
Write-Output 'DELTA_HARNESS_OK'
"""
            harness_path = root / "harness.ps1"
            harness_path.write_text(harness, encoding="utf-8")
            completed = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(harness_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"PowerShell harness failed: {completed.stdout}\n{completed.stderr}",
        )
        self.assertIn("DELTA_HARNESS_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
