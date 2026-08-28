"""Phase 4C 至 Phase 5 腳本與文件的安全結構測試。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROLLBACK = REPOSITORY_ROOT / "scripts" / "manual_rollback.ps1"
CUTOVER = REPOSITORY_ROOT / "scripts" / "verify_cutover.ps1"


class ReleaseOperationsTests(unittest.TestCase):
    def test_manual_rollback_is_allowlisted_conditional_and_non_destructive(self) -> None:
        source = MANUAL_ROLLBACK.read_text(encoding="utf-8")

        self.assertIn("SupportsShouldProcess", source)
        self.assertIn("line-stock-bot-498908-quant-snapshots", source)
        self.assertIn("--if-generation-match=", source)
        self.assertIn("Get-FileHash", source)
        self.assertIn("latest-$Market.json", source)
        self.assertNotIn("storage', 'rm'", source)
        self.assertNotIn("--recursive", source)

    def test_cutover_is_read_only_and_fails_closed(self) -> None:
        source = CUTOVER.read_text(encoding="utf-8")

        for required in (
            "quality_gate -ne 'PASS'",
            "source_hashes",
            "uniformBucketLevelAccess",
            "publicAccessPrevention",
            "get-iam-policy",
            "stock-papi-line-channel-access-token",
            "'D:\\AbsorbData'",
            "'ABSORB-LocalQuant'",
            "latest-$Market.json",
            "overall = if ($Ready) { 'READY' } else { 'BLOCKED' }",
        ):
            self.assertIn(required, source)
        self.assertNotIn("storage', 'rm'", source)
        self.assertNotIn("run', 'deploy'", source)

    def test_cutover_supports_current_gcloud_bucket_security_schema(self) -> None:
        source = CUTOVER.read_text(encoding="utf-8")

        for required in (
            "uniform_bucket_level_access",
            "public_access_prevention",
            "lifecycle_config.rule",
            "uniformBucketLevelAccess",
            "publicAccessPrevention",
            "lifecycle.rule",
        ):
            self.assertIn(required, source)
        self.assertIn("$LifecycleRules = @(", source)
        self.assertNotIn("$LifecycleRules = if (", source)

    def test_cutover_quant_pointer_supports_tw_v3_and_tw_us_v4(self) -> None:
        source = CUTOVER.read_text(encoding="utf-8")
        pointer = source[
            source.index("function Test-MarketPointer"):
            source.index("function Test-LocalOperations")
        ]

        for required in (
            "$Latest.schema_version -notin @(2, 3, 4)",
            "Manifest v3 is TW-only",
            "Get-ObservationManifestCoverage",
            "ExpectedSha256",
            "generated_at",
        ):
            self.assertIn(required, pointer)
        self.assertIn("-not $Latest.manifest.EndsWith(", pointer)
        self.assertIn("$Latest.manifest_sha256.Substring(0, 12)", pointer)
        for required in (
            "MaximumMarketAgeDays",
            "Test-ObservationMarketSymbol",
            "target = [ordered]@",
            "project = $Project",
        ):
            self.assertIn(required, source)

    def test_manual_quant_rollback_supports_schema_v3_without_weakening_hash_gate(self) -> None:
        source = MANUAL_ROLLBACK.read_text(encoding="utf-8")
        quant = source[source.index("if ($LkgManifest"):]

        for required in (
            "$Current.schema_version -notin @(2, 3, 4)",
            "Manifest v3/v4 is TW-only",
            "observation_coverage",
            "regular_price_coverage",
            "operational_failure_rate",
            "ManifestHash.Substring(0, 12)",
            "schema_version = [int]$Manifest.schema_version",
            "sample_data",
            "Test-MarketSymbol",
            "PreviousErrorActionPreference",
        ):
            self.assertIn(required, quant)
        self.assertNotIn("if ($Manifest.schema_version -ne $Current.schema_version)", quant)
        self.assertIn("Assert-QuantManifest -Manifest $Manifest -Market $Market", quant)

    def test_manual_quant_manifest_fixture_accepts_us_ticker_and_rejects_sample(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is required for executable rollback fixtures")

        source = MANUAL_ROLLBACK.read_text(encoding="utf-8")
        helper_start = source.index("function Test-JsonInteger")
        helper_end = source.index("try {", helper_start)
        helper = source[helper_start:helper_end]
        digest = "a" * 64
        valid = {
            "schema_version": 2,
            "market": "US",
            "generated_at": "2026-08-08T07:13:53.141589Z",
            "market_as_of": "2026-08-07",
            "universe_count": 1,
            "symbol_count": 1,
            "failure_count": 0,
            "failure_rate": 0.0,
            "coverage": 1.0,
            "failed_symbols": [],
            "symbols": {
                "BRK-B": {
                    "as_of": "2026-08-07",
                    "model_version": "lgbm-5d-v1",
                    "path": f"objects/{digest}.json.gz",
                    "sha256": digest,
                    "size": 100,
                    "uncompressed_size": 1000,
                }
            },
        }
        cases = {
            "valid": valid,
            "sample": {**valid, "sample_data": True},
            "bad_symbol": {
                **valid,
                "symbols": {"brk-b": valid["symbols"]["BRK-B"]},
            },
            "stale_date": {
                **valid,
                "symbols": {
                    "BRK-B": {
                        **valid["symbols"]["BRK-B"],
                        "as_of": "2000-01-01",
                    }
                },
            },
        }
        script = f"""
$ErrorActionPreference = 'Stop'
{helper}
$Cases = @'
{json.dumps(cases, ensure_ascii=False)}
'@ | ConvertFrom-Json
$Results = foreach ($Property in $Cases.PSObject.Properties) {{
    try {{
        Assert-QuantManifest -Manifest $Property.Value -Market 'US'
        "$($Property.Name)=PASS"
    }} catch {{
        "$($Property.Name)=FAIL"
    }}
}}
$Results -join ';'
"""
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        results = dict(item.split("=", 1) for item in completed.stdout.strip().split(";"))
        self.assertEqual(results["valid"], "PASS")
        self.assertEqual(results["sample"], "FAIL")
        self.assertEqual(results["bad_symbol"], "FAIL")
        self.assertEqual(results["stale_date"], "FAIL")

    def test_manual_rollback_wrappers_handle_ps51_native_stderr(self) -> None:
        powershell = shutil.which("powershell")
        if powershell is None:
            self.skipTest("Windows PowerShell 5.1 is required for wrapper regression")

        source = MANUAL_ROLLBACK.read_text(encoding="utf-8")
        quant_start = source.index("function Invoke-Gcloud")
        quant_end = source.index("function Get-JsonFile", quant_start)
        quant_wrapper = source[quant_start:quant_end]
        script = f"""
$ErrorActionPreference = 'Stop'
$Gcloud = 'cmd.exe'
{quant_wrapper}
$null = Invoke-Gcloud @('/c', 'echo warning 1>&2')
'PASS'
"""
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS", completed.stdout)

    def test_manual_rollback_observation_wrapper_restores_error_preference(self) -> None:
        source = MANUAL_ROLLBACK.read_text(encoding="utf-8")
        observation = source[
            source.index("function Invoke-ObservationGcloud"):
            source.index("$ReceiptRoot")
        ]

        self.assertIn("$PreviousErrorActionPreference = $ErrorActionPreference", observation)
        self.assertIn("$ErrorActionPreference = 'SilentlyContinue'", observation)
        self.assertIn("$ErrorActionPreference = $PreviousErrorActionPreference", observation)

    def test_cutover_handles_ps51_native_progress_but_checks_exit_code(self) -> None:
        source = CUTOVER.read_text(encoding="utf-8")
        invoke_gcloud = source[
            source.index("function Invoke-Gcloud"):
            source.index("function Invoke-Checked")
        ]

        self.assertIn(
            "$PreviousErrorActionPreference = $ErrorActionPreference",
            invoke_gcloud,
        )
        self.assertIn("$ErrorActionPreference = 'SilentlyContinue'", invoke_gcloud)
        self.assertIn(
            "$ErrorActionPreference = $PreviousErrorActionPreference",
            invoke_gcloud,
        )
        self.assertIn("if ($ExitCode -ne 0)", invoke_gcloud)

    def test_required_runbook_and_handover_documents_exist(self) -> None:
        documents = {
            "runbook_incident_response.md": "手動回滾",
            "architecture_overview.md": "回測六層",
            "deployment_guide.md": "Secret Manager",
            "release_blockers_and_risks.md": "Cutover 停止條件",
        }
        for name, expected_text in documents.items():
            content = (REPOSITORY_ROOT / "docs" / name).read_text(encoding="utf-8")
            self.assertIn(expected_text, content)


if __name__ == "__main__":
    unittest.main()
