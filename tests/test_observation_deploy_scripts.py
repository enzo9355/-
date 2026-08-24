"""Observation Production 部署、驗證與回滾腳本的安全結構測試。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = REPOSITORY_ROOT / "scripts" / "deploy_observation_production.ps1"
VERIFY = REPOSITORY_ROOT / "scripts" / "verify_cutover.ps1"
ROLLBACK = REPOSITORY_ROOT / "scripts" / "manual_rollback.ps1"
COMMON = REPOSITORY_ROOT / "scripts" / "observation_release_common.ps1"
CHECKLIST = REPOSITORY_ROOT / "docs" / "absorb-cutover-checklist.md"


class ObservationDeployScriptTests(unittest.TestCase):
    def _run_smoke_harness(self, *, us_final_path="/stock/AAPL"):
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            self.skipTest("Windows PowerShell is unavailable")
        deploy = str(DEPLOY.resolve()).replace("'", "''")
        final_path = us_final_path.replace("'", "''")
        harness = rf"""
$ErrorActionPreference = 'Stop'
$script:ForbiddenPredictionKeys = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
foreach ($key in @(
    'forecast_probability', 'probability', 'ranking_score',
    'model_version', 'backtest_version', 'recommendation'
)) {{ [void]$script:ForbiddenPredictionKeys.Add($key) }}
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('{deploy}', [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ throw 'deploy script did not parse' }}
foreach ($name in @('Assert-NoPredictionKeys', 'Get-TextSha256', 'Invoke-ObservationSmoke')) {{
    $definition = @($ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }}, $true))[0]
    if ($null -eq $definition) {{ throw "missing function: $name" }}
    Invoke-Expression $definition.Extent.Text
}}
$script:Requests = New-Object System.Collections.Generic.List[string]
function New-SmokeResponse {{
    param([string]$Content, [string]$FinalUri)
    return [pscustomobject]@{{
        StatusCode = 200
        Content = $Content
        BaseResponse = [pscustomobject]@{{
            ResponseUri = [Uri]$FinalUri
            RequestMessage = [pscustomobject]@{{ RequestUri = [Uri]$FinalUri }}
        }}
    }}
}}
function Invoke-WebRequest {{
    param(
        [string]$Uri,
        [switch]$UseBasicParsing,
        [int]$MaximumRedirection,
        [int]$TimeoutSec
    )
    [void]$script:Requests.Add($Uri)
    $request = [Uri]$Uri
    $key = $request.PathAndQuery
    $content = switch -Regex ($key) {{
        '^/health/data$' {{ '{{"service":{{"status":"ok"}},"markets":{{"TW":{{"status":"current"}},"US":{{"status":"current"}}}}}}'; break }}
        '^/api/dashboard$' {{ '{{"product_mode":"observation","market_observation":{{}},"industry_observations":[],"data_quality":{{}}}}'; break }}
        '^/$' {{ '<html><body data-market="TW"><link href="/static/app.css?v=abcdef123456"><script src="/static/app.js?v=abcdef123456"></script></body></html>'; break }}
        '^/reports$' {{ '<body data-market="TW"><a href="/reports/2026-08-21/post-close">post</a><a href="/reports/2026-08-21/pre-market">pre</a></body>'; break }}
        '^/reports/2026-08-21/post-close$' {{ '<body data-market="TW"><div class="professional-report">market-actuals-title</div></body>'; break }}
        '^/reports/2026-08-21/pre-market$' {{ '<body data-market="TW"><div id="overnight-title"></div></body>'; break }}
        '^/reports/us$' {{ '<body data-market="US"><a href="/reports/us/2026-08-21/post-close">us</a></body>'; break }}
        '^/reports/us/2026-08-21/post-close$' {{ '<body data-market="US"><div class="professional-report">AAPL</div></body>'; break }}
        '^/search\?market=TW&q=2330$' {{ return New-SmokeResponse '<body data-market="TW"><h1>2330</h1></body>' 'https://candidate.example/stock/2330' }}
        '^/search\?market=US&q=AAPL$' {{ return New-SmokeResponse '<body data-market="US"><h1>AAPL</h1></body>' 'https://candidate.example{final_path}' }}
        '^/static/app\.(?:css|js)\?v=abcdef123456$' {{ 'asset'; break }}
        '^/us(?:/.*)?$' {{ '<body data-market="US">AAPL</body>'; break }}
        '^/stock/AAPL$' {{ '<body data-market="US">AAPL</body>'; break }}
        '^/(?:health|market|industries|stocks|ask|learn|market-map|stock/2330)$' {{ '<body data-market="TW">2330</body>'; break }}
        default {{ throw "unexpected smoke request: $key" }}
    }}
    return New-SmokeResponse ([string]$content) $Uri
}}
$results = @(Invoke-ObservationSmoke -BaseUrl 'https://candidate.example')
@{{ requests = @($script:Requests); results = $results }} | ConvertTo-Json -Depth 8 -Compress
"""
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                harness,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_path_allowlist_read_is_not_suppressed_by_whatif(self) -> None:
        source = COMMON.read_text(encoding="utf-8")
        path_guard = source[
            source.index("function Assert-PathWithinRoot"):source.index(
                "function Invoke-GcloudCaptured"
            )
        ]

        self.assertIn("[IO.Path]::GetFullPath", path_guard)
        self.assertIn("[IO.File]::GetAttributes", path_guard)
        self.assertIn("[IO.Directory]::GetParent", path_guard)
        self.assertNotIn("Resolve-Path", path_guard)
        self.assertNotIn("Get-Item", path_guard)

    def test_deploy_is_no_traffic_first_and_explicitly_fail_closed(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        for required in (
            "SupportsShouldProcess",
            "ABSORB_PREDICTION_MODE=research",
            "ABSORB_OBSERVATION_ENABLED=true",
            "ABSORB_PREDICTION_PROBABILITY_ENABLED=false",
            "ABSORB_PREDICTION_RANKING_ENABLED=false",
            "ABSORB_PREDICTION_STRONG_ACTIONS_ENABLED=false",
            "ABSORB_PREDICTION_PERFORMANCE_ENDORSEMENT_ENABLED=false",
            "ABSORB_PREVIEW_CANDIDATE_PREFIX",
            "PREVIEW_CANDIDATE_PREFIX",
            "--no-traffic",
            "Invoke-ObservationSmoke",
            "Invoke-ObservationCutoverVerification",
            "update-traffic",
        ):
            self.assertIn(required, source)

        smoke_call = "$CandidateSmoke = Invoke-ObservationSmoke"
        verify_call = "$Receipt.cutover_verification = Invoke-ObservationCutoverVerification"
        traffic_call = '"--to-revisions=$CandidateRevision=100"'
        self.assertLess(source.index("--no-traffic"), source.index(smoke_call))
        self.assertLess(source.index(smoke_call), source.index(verify_call))
        self.assertLess(source.index(verify_call), source.index(traffic_call))
        self.assertNotIn("--set-env-vars", source)
        self.assertNotIn("--set-secrets", source)

    def test_whatif_does_not_disable_read_only_gcloud_preflight(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")
        invoke_gcloud = source[
            source.index("function Invoke-Gcloud"):source.index(
                "function Get-Service"
            )
        ]

        self.assertIn("$PreviousWhatIfPreference = $WhatIfPreference", invoke_gcloud)
        self.assertIn("$WhatIfPreference = $false", invoke_gcloud)
        self.assertIn("$WhatIfPreference = $PreviousWhatIfPreference", invoke_gcloud)
        preflight = source[:source.index("if (-not $PSCmdlet.ShouldProcess(")]
        self.assertIn("[IO.File]::ReadAllBytes", preflight)
        self.assertNotIn("Get-Content", preflight)
        self.assertNotIn("Get-FileHash", preflight)

    def test_traffic_preflight_sums_ordered_receipt_entries_explicitly(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        self.assertIn("$PreviousTrafficPercent = (", source)
        self.assertIn("ForEach-Object { [int]$_['percent'] }", source)
        self.assertIn("$PreviousTrafficPercent -ne 100", source)
        self.assertIn(
            'ForEach-Object { "$($_[\'revision\'])=$($_[\'percent\'])" }',
            source,
        )
        self.assertNotIn("Measure-Object -Property percent -Sum", source)

    def test_deploy_receipt_captures_previous_state_and_endpoint_evidence(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        for required in (
            "absorb-observation-deployment",
            "previous_service",
            "previous_revision",
            "previous_traffic",
            "previous_environment",
            "observation_lkg_receipt",
            "candidate_revision",
            "candidate_url",
            "traffic_applied",
            "'/health'",
            "'/'",
            "'/market'",
            "'/industries'",
            "'/stocks'",
            "'/ask'",
            "'/learn'",
            "'/api/dashboard'",
            "'/reports'",
            "'/market-map'",
            "'/stock/2330'",
            "'/health/data'",
            "'/us'",
            "'/us/market'",
            "'/us/industries'",
            "'/us/stocks'",
            "'/reports/us'",
            "'/stock/AAPL'",
        ):
            self.assertIn(required, source)

    def test_deployment_receipt_binds_the_checked_out_commit_and_rollback_revision(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        self.assertIn("rev-parse HEAD", source)
        self.assertIn("source_commit = $Commit", source)
        self.assertIn("previous_revision", source)
        self.assertLess(source.index("rev-parse HEAD"), source.index("source_commit = $Commit"))

    def test_candidate_revision_provenance_is_read_back_before_smoke_or_traffic(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")
        for required in (
            "rev-parse 'HEAD^{tree}'",
            "ABSORB_SOURCE_COMMIT=$Commit",
            "ABSORB_SOURCE_TREE=$SourceTree",
            "absorb-source-commit=$Commit",
            "absorb-source-tree=$SourceTree",
            "Get-Revision -Revision $CandidateRevision",
            "CandidateInfo.status.imageDigest",
            "status.containerStatuses[0].imageDigest",
            "Candidate revision provenance verification failed",
            "Deployment source changed while candidate was building",
            "candidate_provenance = [ordered]@{",
            "image_digest = $ImageDigest",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        provenance_gate = source.index("Candidate revision provenance verification failed")
        self.assertLess(
            provenance_gate,
            source.index("$CandidateSmoke = Invoke-ObservationSmoke"),
        )
        self.assertLess(provenance_gate, source.index('"--to-revisions=$CandidateRevision=100"'))
        self.assertGreaterEqual(source.count("status --porcelain"), 2)
        self.assertGreaterEqual(source.count("rev-parse HEAD"), 2)
        self.assertGreaterEqual(source.count("rev-parse 'HEAD^{tree}'"), 2)

    def test_smoke_and_cutover_verification_forbid_prediction_payloads(self) -> None:
        deploy = DEPLOY.read_text(encoding="utf-8")
        verify = VERIFY.read_text(encoding="utf-8")

        forbidden = (
            "forecast_probability",
            "probability",
            "ranking_score",
            "model_version",
            "backtest_version",
            "recommendation",
        )
        for key in forbidden:
            self.assertIn(key, deploy)
            self.assertIn(key, verify)

        for required in (
            "ObservationOnly",
            "product_mode",
            "observation",
            "ABSORB_PREDICTION_MODE",
            "ABSORB_PREVIEW_CANDIDATE_PREFIX",
            "dashboard/v1/latest-TW.json",
            "reports/v2/index-TW.json",
            "/api/dashboard",
        ):
            self.assertIn(required, verify)

        self.assertNotIn("run', 'deploy'", verify)
        self.assertNotIn("storage', 'rm'", verify)

    def test_observation_manifest_gate_supports_schema_v3_without_weakening_gate(self) -> None:
        source = VERIFY.read_text(encoding="utf-8")
        helper_start = source.index("function Get-ObservationManifestCoverage")
        helper_end = source.index("function Test-ObservationDashboardPointer")
        helper = source[helper_start:helper_end]

        for required in (
            "$SchemaVersion -eq 3",
            "observation_coverage",
            "operational_failure_rate",
            "sample_data",
            "MinimumCoverage",
            "Test-ObservationJsonInteger",
            "regular_price_coverage",
            "expected_non_price_symbols",
            "operational_failed_symbols",
            "counts do not match coverage",
        ):
            self.assertIn(required, helper)
        self.assertIn(
            "Get-ObservationManifestCoverage `\n        -Manifest $SourceEvidence.document",
            source,
        )
        self.assertIn("ExpectedObservationAsOf", source)
        self.assertIn("ExpectedSha256", source)
        self.assertNotIn("$SourceEvidence.document.coverage", source[helper_end:])

    def test_observation_manifest_gate_executes_v2_v3_fixture_matrix(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is required for executable verifier fixtures")

        source = VERIFY.read_text(encoding="utf-8")
        helper_start = source.index("function Test-ObservationJsonInteger")
        helper_end = source.index("function Test-ObservationDashboardPointer")
        helper = source[helper_start:helper_end]

        regular_sha = "a" * 64
        status_sha = "b" * 64
        evidence_sha = "c" * 64
        target_date = "2026-08-07"
        regular_entry = {
            "as_of": target_date,
            "latest_regular_price_date": target_date,
            "model_version": "observation-source-v1",
            "observation_as_of": target_date,
            "observation_kind": "regular_price",
            "path": f"objects/{regular_sha}.json.gz",
            "sha256": regular_sha,
            "size": 100,
            "uncompressed_size": 1000,
        }
        status_entry = {
            "as_of": "2026-08-06",
            "evidence_sha256": evidence_sha,
            "latest_regular_price_date": "2026-08-06",
            "model_version": "observation-source-v1",
            "observation_as_of": target_date,
            "observation_kind": "official_no_regular_trade",
            "path": f"objects/{status_sha}.json.gz",
            "sha256": status_sha,
            "size": 100,
            "uncompressed_size": 1000,
        }
        valid_v3 = {
            "schema_version": 3,
            "market": "TW",
            "generated_at": "2026-08-08T07:13:53.141589Z",
            "target_market_date": target_date,
            "observation_as_of": target_date,
            "universe_count": 2,
            "observation_count": 2,
            "regular_price_symbol_count": 1,
            "expected_non_price_symbol_count": 1,
            "operational_failure_count": 0,
            "regular_price_denominator": 1,
            "regular_price_coverage": 1.0,
            "observation_coverage": 1.0,
            "operational_failure_rate": 0.0,
            "expected_non_price_symbols": {
                "1538": {
                    "status": "official_no_regular_trade",
                    "evidence_sha256": evidence_sha,
                    "artifact_sha256": status_sha,
                    "latest_regular_price_date": "2026-08-06",
                }
            },
            "operational_failed_symbols": [],
            "symbols": {"2330": regular_entry, "1538": status_entry},
        }
        valid_v2 = {
            "schema_version": 2,
            "market": "TW",
            "generated_at": "2026-08-08T07:13:53.141589Z",
            "market_as_of": target_date,
            "universe_count": 1,
            "symbol_count": 1,
            "failure_count": 0,
            "failure_rate": 0.0,
            "coverage": 1.0,
            "failed_symbols": [],
            "symbols": {"2330": copy.deepcopy(regular_entry)},
        }
        valid_v2["symbols"]["2330"].pop("observation_as_of")
        valid_v2["symbols"]["2330"].pop("observation_kind")

        cases = {
            "valid_v2": valid_v2,
            "valid_v3": valid_v3,
        }
        for name, mutation in (
            ("missing_symbols", lambda item: item.pop("symbols")),
            ("fractional_count", lambda item: item.__setitem__("universe_count", 2.5)),
            ("bad_partition", lambda item: item.__setitem__("regular_price_symbol_count", 2)),
            ("bad_status", lambda item: item["expected_non_price_symbols"]["1538"].__setitem__("status", "regular_price")),
            ("bad_date", lambda item: item.__setitem__("target_market_date", "2026-08-06")),
            ("bad_path_hash", lambda item: item["symbols"]["2330"].__setitem__("path", "objects/" + "d" * 64 + ".json.gz")),
        ):
            invalid = copy.deepcopy(valid_v3)
            mutation(invalid)
            cases[name] = invalid

        script = f"""
$ErrorActionPreference = 'Stop'
$MinimumCoverage = 0.95
{helper}
$Cases = @'
{json.dumps(cases, ensure_ascii=False)}
'@ | ConvertFrom-Json
$Results = foreach ($Property in $Cases.PSObject.Properties) {{
    try {{
        $null = Get-ObservationManifestCoverage -Manifest $Property.Value -ExpectedObservationAsOf '{target_date}'
        \"$($Property.Name)=PASS\"
    }} catch {{
        \"$($Property.Name)=FAIL\"
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
        self.assertEqual(results["valid_v2"], "PASS")
        self.assertEqual(results["valid_v3"], "PASS")
        for name in (
            "missing_symbols",
            "fractional_count",
            "bad_partition",
            "bad_status",
            "bad_date",
            "bad_path_hash",
        ):
            self.assertEqual(results[name], "FAIL")

    def test_generic_manifest_gate_accepts_us_v2_and_rejects_stale_v3(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is required for executable verifier fixtures")

        source = VERIFY.read_text(encoding="utf-8")
        helper_start = source.index("function Test-ObservationJsonInteger")
        helper_end = source.index("function Test-ObservationDashboardPointer")
        helper = source[helper_start:helper_end]
        target_date = "2026-08-07"
        symbols = {}
        for index in range(19):
            symbol = f"A{index}"
            digest = f"{index + 1:064x}"
            symbols[symbol] = {
                "as_of": target_date,
                "model_version": "lgbm-5d-v1",
                "path": f"objects/{digest}.json.gz",
                "sha256": digest,
                "size": 100,
                "uncompressed_size": 1000,
            }
        valid_us_v2 = {
            "schema_version": 2,
            "market": "US",
            "generated_at": "2026-08-08T07:13:53.141589Z",
            "market_as_of": target_date,
            "universe_count": 20,
            "symbol_count": 19,
            "failure_count": 1,
            "failure_rate": 0.05,
            "coverage": 0.95,
            "failed_symbols": ["BRK-B"],
            "symbols": symbols,
        }
        stale_symbol_v2 = {
            **valid_us_v2,
            "symbols": {
                symbol: {**entry, "as_of": "2000-01-01"}
                for symbol, entry in symbols.items()
            },
        }
        stale_v3 = {
            "schema_version": 3,
            "market": "TW",
            "generated_at": "2000-01-02T00:00:00Z",
            "target_market_date": "2000-01-01",
            "observation_as_of": "2000-01-01",
            "universe_count": 1,
            "observation_count": 1,
            "regular_price_symbol_count": 1,
            "expected_non_price_symbol_count": 0,
            "operational_failure_count": 0,
            "regular_price_denominator": 1,
            "regular_price_coverage": 1.0,
            "observation_coverage": 1.0,
            "operational_failure_rate": 0.0,
            "expected_non_price_symbols": {},
            "operational_failed_symbols": [],
            "symbols": {
                "2330": {
                    "as_of": "2000-01-01",
                    "latest_regular_price_date": "2000-01-01",
                    "model_version": "observation-source-v1",
                    "observation_as_of": "2000-01-01",
                    "observation_kind": "regular_price",
                    "path": "objects/" + "a" * 64 + ".json.gz",
                    "sha256": "a" * 64,
                    "size": 100,
                    "uncompressed_size": 1000,
                }
            },
        }
        cases = {
            "valid_us_v2": valid_us_v2,
            "stale_symbol_v2": stale_symbol_v2,
            "stale_v3": stale_v3,
        }
        script = f"""
$ErrorActionPreference = 'Stop'
$MinimumCoverage = 0.95
{helper}
$Cases = @'
{json.dumps(cases, ensure_ascii=False)}
'@ | ConvertFrom-Json
$Results = @()
        try {{
            $null = Get-ObservationManifestCoverage `
                -Manifest $Cases.valid_us_v2 `
        -ExpectedMarket 'US' `
        -FailureThreshold 0.25 `
        -ExpectedModelVersion ''
    $Results += 'valid_us_v2=PASS'
        }} catch {{ $Results += 'valid_us_v2=FAIL' }}
        try {{
            $null = Get-ObservationManifestCoverage `
                -Manifest $Cases.stale_symbol_v2 `
                -ExpectedMarket 'US' `
                -FailureThreshold 0.25 `
                -ExpectedModelVersion ''
            $Results += 'stale_symbol_v2=PASS'
        }} catch {{ $Results += 'stale_symbol_v2=FAIL' }}
try {{
    $null = Get-ObservationManifestCoverage `
        -Manifest $Cases.stale_v3 `
        -ExpectedObservationAsOf '2000-01-01' `
        -MaximumMarketAgeDays 7
    $Results += 'stale_v3=PASS'
}} catch {{ $Results += 'stale_v3=FAIL' }}
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
        self.assertEqual(results["valid_us_v2"], "PASS")
        self.assertEqual(results["stale_symbol_v2"], "FAIL")
        self.assertEqual(results["stale_v3"], "FAIL")

    def test_dashboard_source_gate_enforces_manifest_freshness(self) -> None:
        source = VERIFY.read_text(encoding="utf-8")
        dashboard = source[
            source.index("function Test-ObservationDashboardPointer"):
            source.index("function Test-ObservationReportPointers")
        ]

        self.assertIn("-MaximumMarketAgeDays $MaximumMarketAgeDays", dashboard)

    def test_cutover_binds_cloud_run_http_to_production_traffic(self) -> None:
        source = VERIFY.read_text(encoding="utf-8")

        for required in (
            "$ServiceInfo.status.traffic",
            "percent -eq 100",
            "$CloudRunTrafficEvidence",
            "Get-CloudRunRevision",
            "spec.serviceAccountName",
            "$CloudRunActiveRevision",
            "BaseUrl does not match Cloud Run service URL",
            "revision = $ActiveTraffic[0].revisionName",
            "url = [string]$ServiceInfo.status.url",
            "CloudRunTrafficEvidence.url -ne $ServiceUrl",
            "AfterTrafficEvidence",
            "Cloud Run traffic or URL changed during HTTP smoke",
        ):
            self.assertIn(required, source)
        self.assertNotIn("spec.template.spec.containers[0].env", source)

    def test_cutover_reads_mutable_gcs_objects_at_captured_generation(self) -> None:
        source = VERIFY.read_text(encoding="utf-8")

        for required in (
            '"$Uri#$($Metadata.generation)"',
            "AfterMetadata",
            "[string]$AfterMetadata.generation -ne [string]$Metadata.generation",
            "generation = [string]$Metadata.generation",
        ):
            self.assertIn(required, source)

    def test_no_traffic_smoke_requests_both_canonical_report_types(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        for required in (
            "CanonicalReportPaths",
            "post-close",
            "pre-market",
            "Observation report link is unavailable",
            "Observation report smoke failed",
        ):
            self.assertIn(required, source)

    def test_no_traffic_smoke_covers_dual_market_identity_and_data_health(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")
        smoke = source[
            source.index("function Invoke-ObservationSmoke"):
            source.index("function Invoke-ObservationCutoverVerification")
        ]
        for path in (
            "'/health/data'",
            "'/us'",
            "'/us/market'",
            "'/us/industries'",
            "'/us/stocks'",
            "'/reports/us'",
            "'/stock/AAPL'",
        ):
            with self.subTest(path=path):
                self.assertIn(path, smoke)
        for contract in (
            "Data health endpoint is missing TW or US identity",
            'data-market="US"',
            'data-market="TW"',
            "US canonical report link is unavailable",
            "US canonical report market identity smoke failed",
            "[int]$Response.StatusCode -ne 200",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, smoke)

    def test_no_traffic_smoke_executes_tw_and_us_search_redirect_contract(self) -> None:
        completed = self._run_smoke_harness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        evidence = json.loads(completed.stdout)
        self.assertIn(
            "https://candidate.example/search?market=TW&q=2330",
            evidence["requests"],
        )
        self.assertIn(
            "https://candidate.example/search?market=US&q=AAPL",
            evidence["requests"],
        )
        search_results = [
            item for item in evidence["results"]
            if str(item.get("path", "")).startswith("/search?")
        ]
        self.assertEqual(
            [item["final_url"] for item in search_results],
            [
                "https://candidate.example/stock/2330",
                "https://candidate.example/stock/AAPL",
            ],
        )

    def test_no_traffic_smoke_rejects_wrong_search_final_url(self) -> None:
        completed = self._run_smoke_harness(us_final_path="/us")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("search final URL", completed.stdout + completed.stderr)

    def test_no_traffic_smoke_verifies_revisioned_static_assets(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")

        self.assertIn("/static/app.css?v=", source)
        self.assertIn("/static/app.js?v=", source)
        self.assertIn("asset_version", source)

    def test_deploy_failure_rolls_back_traffic_without_mutating_existing_pointers(self) -> None:
        source = DEPLOY.read_text(encoding="utf-8")
        failure_handler = source[source.index("} catch {"):]

        self.assertIn("Restore-PreviousTraffic", failure_handler)
        self.assertNotIn("rollback_observation.ps1", failure_handler)
        self.assertNotIn("applied_generation", failure_handler)

    def test_manual_rollback_can_restore_cloud_run_and_observation_pointers(self) -> None:
        source = ROLLBACK.read_text(encoding="utf-8")

        for required in (
            "ObservationDeploymentReceipt",
            "absorb-observation-deployment",
            "previous_traffic",
            "update-traffic",
            "rollback_observation.ps1",
            "observation_lkg_receipt",
        ):
            self.assertIn(required, source)
        self.assertNotIn("--recursive", source)

    def test_manual_observation_rollback_compensates_and_writes_recovery_receipt(self) -> None:
        source = ROLLBACK.read_text(encoding="utf-8")
        observation = source[:source.index("if ($LkgManifest")]

        for required in (
            "Restore-ObservationCandidateTraffic",
            "Write-ObservationRecoveryReceipt",
            "manual-rollback-recovery-",
            "traffic_compensation",
            "pointer_rollback_attempted",
            "candidate traffic was restored",
            "traffic compensation failed",
        ):
            self.assertIn(required, observation)

    def test_manual_rollback_preflight_handles_whatif_and_ordered_traffic(self) -> None:
        source = ROLLBACK.read_text(encoding="utf-8")
        observation = source[:source.index("if ($LkgManifest")]
        invoke_gcloud = observation[
            observation.index("function Invoke-ObservationGcloud"):
            observation.index("$ReceiptRoot")
        ]

        self.assertIn("$PreviousWhatIfPreference = $WhatIfPreference", invoke_gcloud)
        self.assertIn("$WhatIfPreference = $false", invoke_gcloud)
        self.assertIn("$WhatIfPreference = $PreviousWhatIfPreference", invoke_gcloud)
        self.assertIn("$PreviousTrafficPercent = (", observation)
        self.assertIn("ForEach-Object { [int]$_['percent'] }", observation)
        self.assertIn("$PreviousTrafficPercent -ne 100", observation)
        self.assertIn(
            'ForEach-Object { "$($_[\'revision\'])=$($_[\'percent\'])" }',
            observation,
        )
        self.assertIn(
            "$_.revisionName -eq [string]$Expected['revision']",
            observation,
        )
        self.assertIn(
            "[int]$_.percent -eq [int]$Expected['percent']",
            observation,
        )
        self.assertNotIn("Measure-Object -Property percent -Sum", observation)

    def test_cutover_checklist_documents_order_and_stop_conditions(self) -> None:
        source = CHECKLIST.read_text(encoding="utf-8")

        for required in (
            "Observation Production",
            "capture_observation_lkg.ps1",
            "deploy_observation_production.ps1",
            "no-traffic",
            "verify_cutover.ps1",
            "manual_rollback.ps1",
            "prediction fields",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
