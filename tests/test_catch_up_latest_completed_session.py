import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reporting.observation_v2 import build_post_close_observation_metadata
from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.batch.catch_up_latest_completed_session import (
    CatchUpContractError,
    LivePointerDate,
    classify_live_pointer_dates,
    validate_target_session,
)
from stock_papi.batch.observation_products import (
    promote_observation_candidate,
    write_observation_candidate,
)
from tests.test_observation_report_v2 import dashboard as valid_observation_dashboard


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
            "Get-LocalObservationPromotionResume",
            "resumed-existing-local-promotion",
            "local_promotion",
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

    def test_atomic_evidence_writer_accepts_a_new_path(self):
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        source_path = self.SCRIPT.resolve()
        common_path = (self.SCRIPT.parent / "observation_release_common.ps1").resolve()

        def ps_literal(value):
            return "'" + str(value).replace("'", "''") + "'"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = f"""
$ErrorActionPreference = 'Stop'
. {ps_literal(common_path)}
$source = [IO.File]::ReadAllText({ps_literal(source_path)})
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
$function = $ast.Find({{
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Write-AtomicJson'
}}, $true)
if ($null -eq $function) {{ throw 'function not found: Write-AtomicJson' }}
. ([scriptblock]::Create($function.Extent.Text))
$root = {ps_literal(root)}
[IO.Directory]::CreateDirectory($root) | Out-Null
$path = Join-Path $root 'new-evidence.json'
Write-AtomicJson -Path $path -Root $root -Document ([ordered]@{{ kind = 'test'; value = 1 }})
if (-not [IO.File]::Exists($path)) {{ throw 'new evidence file was not created' }}
$existingFile = Join-Path $root 'existing-evidence.json'
[IO.File]::WriteAllText($existingFile, 'original')
$existingFileRejected = $false
try {{
    Write-AtomicJson -Path $existingFile -Root $root -Document ([ordered]@{{ kind = 'collision' }})
}}
catch {{
    $existingFileRejected = $true
}}
if (-not $existingFileRejected) {{ throw 'existing evidence file was accepted' }}
if ((Get-Content -LiteralPath $existingFile -Raw -Encoding utf8) -ne 'original') {{
    throw 'existing evidence file was changed'
}}
$existingDirectory = Join-Path $root 'existing-evidence-directory'
[IO.Directory]::CreateDirectory($existingDirectory) | Out-Null
$existingDirectoryRejected = $false
try {{
    Write-AtomicJson -Path $existingDirectory -Root $root -Document ([ordered]@{{ kind = 'collision' }})
}}
catch {{
    $existingDirectoryRejected = $true
}}
if (-not $existingDirectoryRejected) {{ throw 'existing evidence directory was accepted' }}
Write-Output 'ATOMIC_NEW_PATH_OK'
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
        self.assertIn("ATOMIC_NEW_PATH_OK", completed.stdout)

    def test_local_promotion_resume_requires_exact_candidate_binding(self):
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        source_path = self.SCRIPT.resolve()
        common_path = (self.SCRIPT.parent / "observation_release_common.ps1").resolve()

        def ps_literal(value):
            return "'" + str(value).replace("'", "''") + "'"

        manifest_relative = "manifests/TW-20260815T130142Z-aaaaaaaaaaaa.json"
        manifest_sha = "b" * 64
        target_date = "2026-08-14"
        applicable_date = "2026-08-17"
        dashboard = {
            "schema_version": 2,
            "kind": "absorb-observation-dashboard",
            "product_mode": "observation",
            "market": "TW",
            "observation_as_of": target_date,
            "generated_at": "2026-08-15T13:01:42.239000Z",
            "source_manifest": f"quant/v1/{manifest_relative}",
            "source_manifest_sha256": manifest_sha,
            "value": "same-dashboard",
        }
        report = {
            "schema_version": 2,
            "product_mode": "observation",
            "report_type": "post_close",
            "market": "TW",
            "source_market_date": target_date,
            "applicable_trading_date": applicable_date,
            "published_at": "2026-08-15T13:01:42.239000Z",
            "data_as_of": target_date,
            "source_manifest": f"quant/v1/{manifest_relative}",
            "source_manifest_sha256": manifest_sha,
            "model_versions": {},
            "title": "target",
            "summary": [],
            "content": {"value": "same-report"},
        }
        old_entry = {
            "report_type": "post_close",
            "product_mode": "observation",
            "source_market_date": "2026-08-07",
            "applicable_trading_date": "2026-08-10",
            "metadata": "metadata/" + "d" * 64 + ".json",
            "metadata_sha256": "d" * 64,
            "title": "old",
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dashboard_bytes = json.dumps(
                dashboard, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            dashboard_sha = hashlib.sha256(dashboard_bytes).hexdigest()
            report_bytes = json.dumps(
                report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            canonical_document = {
                "schema_version": 1,
                "kind": "absorb-professional-post-close-report",
                "identity": {
                    "schema_version": 1,
                    "report_type": "post_close",
                    "product_tier": "institutional",
                    "product_mode": "observation_with_research",
                    "market": "TW",
                    "source_market_date": target_date,
                    "applicable_trading_date": applicable_date,
                    "published_at": report["published_at"],
                    "generated_at": report["published_at"],
                    "source_manifest": f"quant/v1/{manifest_relative}",
                    "source_manifest_sha256": manifest_sha,
                    "content_sha256": "f" * 64,
                    "report_id": "TW-20260814-post-close-institutional",
                    "generator_version": "test",
                    "code_commit_sha": "e" * 40,
                    "model_version": None,
                    "feature_schema_version": "test-v1",
                    "recommendation_policy_version": "test-v1",
                },
            }
            canonical_bytes = json.dumps(
                canonical_document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()
            local_metadata = dict(report)
            local_metadata["content_sha256"] = "c" * 64
            local_metadata["professional_report"] = {
                "object": f"objects/canonical/{canonical_sha}.json",
                "sha256": canonical_sha,
                "content_sha256": "f" * 64,
                "schema_version": 1,
                "generator_version": "test",
                "code_commit_sha": "e" * 40,
            }
            metadata_bytes = json.dumps(
                local_metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
            dashboard_root = root / "publish" / "dashboard" / "v1"
            report_root = root / "publish" / "reports" / "v2"
            (dashboard_root / "objects").mkdir(parents=True)
            (report_root / "metadata").mkdir(parents=True)
            (report_root / "objects" / "canonical").mkdir(parents=True)
            (dashboard_root / "objects" / f"{dashboard_sha}.json").write_bytes(
                dashboard_bytes
            )
            (report_root / "objects" / "canonical" / f"{canonical_sha}.json").write_bytes(
                canonical_bytes
            )
            (report_root / "metadata" / f"{metadata_sha}.json").write_bytes(
                metadata_bytes
            )
            (dashboard_root / "latest-TW.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-observation-dashboard",
                        "product_mode": "observation",
                        "market": "TW",
                        "observation_as_of": target_date,
                        "generated_at": dashboard["generated_at"],
                        "source_manifest": f"quant/v1/{manifest_relative}",
                        "source_manifest_sha256": manifest_sha,
                        "path": f"objects/{dashboard_sha}.json",
                        "sha256": dashboard_sha,
                        "size": len(dashboard_bytes),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (report_root / "latest-TW-post_close.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-report",
                        "market": "TW",
                        "report_type": "post_close",
                        "product_mode": "observation",
                        "source_market_date": target_date,
                        "applicable_trading_date": applicable_date,
                        "published_at": report["published_at"],
                        "metadata": f"metadata/{metadata_sha}.json",
                        "metadata_sha256": metadata_sha,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            target_entry = {
                "report_type": "post_close",
                "product_mode": "observation",
                "source_market_date": target_date,
                "applicable_trading_date": applicable_date,
                "published_at": report["published_at"],
                "data_as_of": report["data_as_of"],
                "model_versions": report["model_versions"],
                "summary": report["summary"],
                "metadata": f"metadata/{metadata_sha}.json",
                "metadata_sha256": metadata_sha,
                "content_sha256": local_metadata["content_sha256"],
                "title": "target",
            }
            (report_root / "index-TW.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-report-index",
                        "market": "TW",
                        "reports": [target_entry, old_entry],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            captured_path = root / "captured.json"
            captured_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-report-index",
                        "market": "TW",
                        "reports": [old_entry],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path = root / "candidate"
            candidate_path.mkdir()
            (candidate_path / "dashboard-snapshot.json").write_bytes(dashboard_bytes)
            (candidate_path / "post-close-report-v2.json").write_bytes(report_bytes)
            candidate_manifest = {
                "schema_version": 1,
                "kind": "absorb-observation-candidate",
                "product_mode": "observation",
                "observation_as_of": target_date,
                "files": {
                    "dashboard-snapshot.json": {
                        "sha256": dashboard_sha,
                        "size": len(dashboard_bytes),
                    },
                    "post-close-report-v2.json": {
                        "sha256": hashlib.sha256(report_bytes).hexdigest(),
                        "size": len(report_bytes),
                    },
                },
            }
            (candidate_path / "candidate.json").write_text(
                json.dumps(candidate_manifest), encoding="utf-8"
            )

            def write_json(path, document):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document), encoding="utf-8")

            harness = f"""
$ErrorActionPreference = 'Stop'
. {ps_literal(common_path)}
$source = [IO.File]::ReadAllText({ps_literal(source_path)})
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
foreach ($name in @(
    'ConvertTo-CanonicalDate',
    'Read-JsonWithinRoot',
    'ConvertTo-CanonicalJsonValue',
    'Get-CanonicalJson',
    'Assert-ObservationReportIndexCatchUpDelta',
    'Get-LocalObservationPointers',
    'Get-LocalObservationPromotionResume',
    'Assert-LocalObservationFormalValidation'
)) {{
    $function = $ast.Find({{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
    }}, $true)
    if ($null -eq $function) {{ throw "function not found: $name" }}
    . ([scriptblock]::Create($function.Extent.Text))
}}
function Invoke-PythonJson {{
    param([string[]]$Arguments)
    return [pscustomobject]@{{ mode = 'validated' }}
}}
$DataRoot = {ps_literal(root)}
$TargetDate = '2026-08-14'
$ParsedTargetDate = [DateTime]::ParseExact($TargetDate, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture).Date
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$manifestRelative = {ps_literal(manifest_relative)}
$manifestSha = {ps_literal(manifest_sha)}
$quant = [pscustomobject]@{{
    latest_path = (Join-Path $DataRoot 'publish\\quant\\v1\\latest-TW.json')
    manifest_relative = $manifestRelative
    manifest_sha256 = $manifestSha
    identity = [ordered]@{{ source_date = $TargetDate; manifest = $manifestRelative; manifest_sha256 = $manifestSha }}
}}
$candidateManifest = Get-Content -LiteralPath {ps_literal(candidate_path / 'candidate.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidateDashboard = Get-Content -LiteralPath {ps_literal(candidate_path / 'dashboard-snapshot.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidateReport = Get-Content -LiteralPath {ps_literal(candidate_path / 'post-close-report-v2.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidate = [pscustomobject]@{{ path = {ps_literal(candidate_path)}; manifest = $candidateManifest; dashboard = $candidateDashboard; report = $candidateReport }}
$captured = Get-Content -LiteralPath {ps_literal(captured_path)} -Raw -Encoding utf8 | ConvertFrom-Json
$resumed = Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured
if ($null -eq $resumed) {{ throw 'valid local promotion was not resumed' }}
if ([string]$resumed.reports_latest.identity.metadata_sha256 -ne {ps_literal(metadata_sha)}) {{ throw 'wrong resumed report metadata' }}
$badCandidate = $candidate | ConvertTo-Json -Depth 50 | ConvertFrom-Json
$badCandidate.dashboard.value = 'different-dashboard'
$rejected = $false
try {{
    Get-LocalObservationPromotionResume -Quant $quant -Candidate $badCandidate -CapturedIndex $captured | Out-Null
}}
catch {{
    if ($_.Exception.Message -notmatch 'dashboard object') {{ throw }}
    $rejected = $true
}}
if (-not $rejected) {{ throw 'candidate dashboard mismatch was accepted' }}
$dashboardLatestPath = Join-Path $DataRoot 'publish\\dashboard\\v1\\latest-TW.json'
$dashboardLatestOriginal = [IO.File]::ReadAllText($dashboardLatestPath)
$dashboardPointer = $dashboardLatestOriginal | ConvertFrom-Json
$dashboardPointer.size = [int]$dashboardPointer.size + 1
[IO.File]::WriteAllText($dashboardLatestPath, ($dashboardPointer | ConvertTo-Json -Depth 20))
$rejected = $false
try {{
    Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null
}}
catch {{
    if ($_.Exception.Message -notmatch 'dashboard') {{ throw }}
    $rejected = $true
}}
[IO.File]::WriteAllText($dashboardLatestPath, $dashboardLatestOriginal)
if (-not $rejected) {{ throw 'dashboard pointer size mismatch was accepted' }}
$indexPath = Join-Path $DataRoot 'publish\\reports\\v2\\index-TW.json'
$indexOriginal = [IO.File]::ReadAllText($indexPath)
$indexPointer = $indexOriginal | ConvertFrom-Json
$indexPointer.reports[0].PSObject.Properties.Remove('summary')
[IO.File]::WriteAllText($indexPath, ($indexPointer | ConvertTo-Json -Depth 50))
$rejected = $false
try {{
    Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null
}}
catch {{
    if ($_.Exception.Message -notmatch 'report index') {{ throw }}
    $rejected = $true
}}
[IO.File]::WriteAllText($indexPath, $indexOriginal)
if (-not $rejected) {{ throw 'report index field mismatch was accepted' }}
$canonicalPath = Join-Path $DataRoot 'publish\\reports\\v2\\objects\\canonical\\{canonical_sha}.json'
[IO.File]::Delete($canonicalPath)
$rejected = $false
try {{
    Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null
}}
catch {{ $rejected = $true }}
if (-not $rejected) {{ throw 'missing canonical object was accepted' }}
Write-Output 'LOCAL_PROMOTION_RESUME_OK'
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
        self.assertIn("LOCAL_PROMOTION_RESUME_OK", completed.stdout)

    def test_local_promotion_resume_runs_real_formal_validator_in_ps5(self):
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        source_path = self.SCRIPT.resolve()
        source_root = source_path.parent.parent
        common_path = (self.SCRIPT.parent / "observation_release_common.ps1").resolve()
        python_executable = Path(sys.executable).resolve()
        target_date = "2026-08-14"
        manifest_relative = "manifests/TW-20260815T130142Z-aaaaaaaaaaaa.json"
        manifest_sha = "b" * 64

        class ResumeCalendar:
            def next_session(self, value):
                return datetime.date(2026, 8, 17)

        dashboard = valid_observation_dashboard()
        dashboard.update(
            {
                "observation_as_of": target_date,
                "generated_at": "2026-08-15T13:01:42.239000Z",
                "source_manifest": f"quant/v1/{manifest_relative}",
                "source_manifest_sha256": manifest_sha,
            }
        )
        metadata = build_post_close_observation_metadata(
            dashboard, ResumeCalendar()
        )

        def ps_literal(value):
            return "'" + str(value).replace("'", "''") + "'"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = write_observation_candidate(root, metadata, dashboard)
            promote_observation_candidate(root, candidate_path)
            captured_path = root / "captured.json"
            captured_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-report-index",
                        "market": "TW",
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )
            harness = f"""
$ErrorActionPreference = 'Stop'
. {ps_literal(common_path)}
$source = [IO.File]::ReadAllText({ps_literal(source_path)})
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
foreach ($name in @(
    'ConvertTo-CanonicalDate',
    'Read-JsonWithinRoot',
    'ConvertTo-CanonicalJsonValue',
    'Get-CanonicalJson',
    'Assert-ObservationReportIndexCatchUpDelta',
    'Get-LocalObservationPointers',
    'Get-LocalObservationPromotionResume',
    'Assert-LocalObservationFormalValidation',
    'Invoke-PythonJson'
)) {{
    $function = $ast.Find({{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
    }}, $true)
    if ($null -eq $function) {{ throw "function not found: $name" }}
    . ([scriptblock]::Create($function.Extent.Text))
}}
$PythonExe = {ps_literal(python_executable)}
$env:PYTHONPATH = {ps_literal(source_root)}
$DataRoot = {ps_literal(root)}
$TargetDate = '2026-08-14'
$ParsedTargetDate = [DateTime]::ParseExact($TargetDate, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture).Date
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$manifestRelative = {ps_literal(manifest_relative)}
$manifestSha = {ps_literal(manifest_sha)}
$quant = [pscustomobject]@{{
    latest_path = (Join-Path $DataRoot 'publish\\quant\\v1\\latest-TW.json')
    manifest_relative = $manifestRelative
    manifest_sha256 = $manifestSha
    identity = [ordered]@{{ source_date = $TargetDate; manifest = $manifestRelative; manifest_sha256 = $manifestSha }}
}}
$candidateManifest = Get-Content -LiteralPath {ps_literal(candidate_path / 'candidate.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidateDashboard = Get-Content -LiteralPath {ps_literal(candidate_path / 'dashboard-snapshot.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidateReport = Get-Content -LiteralPath {ps_literal(candidate_path / 'post-close-report-v2.json')} -Raw -Encoding utf8 | ConvertFrom-Json
$candidate = [pscustomobject]@{{ path = {ps_literal(candidate_path)}; manifest = $candidateManifest; dashboard = $candidateDashboard; report = $candidateReport }}
$captured = Get-Content -LiteralPath {ps_literal(captured_path)} -Raw -Encoding utf8 | ConvertFrom-Json
$resumed = Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured
if ($null -eq $resumed) {{ throw 'valid local promotion was not resumed' }}
Write-Output 'REAL_FORMAL_VALIDATOR_OK'

$badCandidate = $candidate | ConvertTo-Json -Depth 50 | ConvertFrom-Json
$badCandidate.report.published_at = '2026-08-15T13:01:42.240000Z'
$rejected = $false
try {{ Get-LocalObservationPromotionResume -Quant $quant -Candidate $badCandidate -CapturedIndex $captured | Out-Null }}
catch {{ if ($_.Exception.Message -notmatch 'report latest') {{ throw }}; $rejected = $true }}
if (-not $rejected) {{ throw 'candidate report latest mismatch was accepted' }}

$reportLatestPath = Join-Path $DataRoot 'publish\\reports\\v2\\latest-TW-post_close.json'
$reportLatestOriginal = [IO.File]::ReadAllText($reportLatestPath)
$reportLatest = $reportLatestOriginal | ConvertFrom-Json
$reportLatest.published_at = '2026-08-15T13:01:42.240000Z'
[IO.File]::WriteAllText($reportLatestPath, ($reportLatest | ConvertTo-Json -Depth 20))
$rejected = $false
try {{ Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null }}
catch {{ if ($_.Exception.Message -notmatch 'report latest') {{ throw }}; $rejected = $true }}
[IO.File]::WriteAllText($reportLatestPath, $reportLatestOriginal)
if (-not $rejected) {{ throw 'tampered report latest pointer was accepted' }}

$reportLatest = $reportLatestOriginal | ConvertFrom-Json
$reportLatest | Add-Member -NotePropertyName extra -NotePropertyValue 'unexpected'
[IO.File]::WriteAllText($reportLatestPath, ($reportLatest | ConvertTo-Json -Depth 20))
$rejected = $false
try {{ Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null }}
catch {{ if ($_.Exception.Message -notmatch 'report latest') {{ throw }}; $rejected = $true }}
[IO.File]::WriteAllText($reportLatestPath, $reportLatestOriginal)
if (-not $rejected) {{ throw 'extra report latest field was accepted' }}

$dashboardLatestPath = Join-Path $DataRoot 'publish\\dashboard\\v1\\latest-TW.json'
$dashboardLatestOriginal = [IO.File]::ReadAllText($dashboardLatestPath)
$dashboardLatest = $dashboardLatestOriginal | ConvertFrom-Json
$dashboardLatest.generated_at = '2026-08-15T13:01:42.240000Z'
[IO.File]::WriteAllText($dashboardLatestPath, ($dashboardLatest | ConvertTo-Json -Depth 20))
$rejected = $false
try {{ Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null }}
catch {{ if ($_.Exception.Message -notmatch 'dashboard') {{ throw }}; $rejected = $true }}
[IO.File]::WriteAllText($dashboardLatestPath, $dashboardLatestOriginal)
if (-not $rejected) {{ throw 'tampered dashboard pointer was accepted' }}

$canonicalPath = Get-ChildItem -LiteralPath (Join-Path $DataRoot 'publish\\reports\\v2\\objects\\canonical') -Filter '*.json' | Select-Object -First 1
if ($null -eq $canonicalPath) {{ throw 'canonical report fixture was not created' }}
[IO.File]::Delete($canonicalPath.FullName)
$rejected = $false
try {{ Get-LocalObservationPromotionResume -Quant $quant -Candidate $candidate -CapturedIndex $captured | Out-Null }}
catch {{ $rejected = $true }}
if (-not $rejected) {{ throw 'missing canonical report was accepted' }}
Write-Output 'REAL_FORMAL_TAMPER_GATES_OK'
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
                cwd=source_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"PowerShell real-validator harness failed: {completed.stdout}\n{completed.stderr}",
        )
        self.assertIn("REAL_FORMAL_VALIDATOR_OK", completed.stdout)
        self.assertIn("REAL_FORMAL_TAMPER_GATES_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
