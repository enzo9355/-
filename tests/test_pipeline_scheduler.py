import datetime
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

UTC = datetime.timezone.utc


class PipelineSchedulerTests(unittest.TestCase):
    @staticmethod
    def _powershell_function_import(script):
        escaped = str(Path(script).resolve()).replace("'", "''")
        return f"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('{escaped}', [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ throw 'PowerShell source did not parse' }}
foreach ($definition in @($ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }}, $true))) {{
    Invoke-Expression $definition.Extent.Text
}}
"""

    def test_historical_target_with_publish_fails_before_data_access(self):
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "run_tw_post_close_pipeline.ps1"
        )
        environment = os.environ.copy()
        environment["ABSORB_PYTHON_EXE"] = (
            r"C:\absorb-missing-python-for-guard-test.exe"
        )
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
                r"D:\StockPapiData",
                "-TargetDate",
                "2026-08-07",
                "-PublishObservation",
            ],
            cwd=str(script.parents[1]),
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "Historical TargetDate cannot publish observation products",
            output,
        )
        post_close = script.read_text(encoding="utf-8")
        self.assertLess(
            post_close.index("$HistoricalTargetDate"),
            post_close.index("$RepoRoot"),
        )

    def test_new_tasks_are_separate_resilient_limited_and_secret_free(self):
        scripts = Path(__file__).parents[1] / "scripts"
        script = (scripts / "install_pipeline_tasks.ps1").read_text(encoding="utf-8")
        for name in (
            "ABSORB-TW-PostClose",
            "ABSORB-TW-PreMarket",
            "ABSORB-FullBacktest",
            "ABSORB-US-Daily",
            "ABSORB-WeeklyModel",
            "ABSORB-ReportUploadRecovery",
        ):
            self.assertIn(name, script)
        for setting in (
            "StartWhenAvailable = $true",
            "WakeToRun = $true",
            "MultipleInstances IgnoreNew",
            "RestartCount 3",
            "RunLevel Limited",
        ):
            self.assertIn(setting, script)
        for secret in (
            "LINE_CHANNEL_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "Bearer",
        ):
            self.assertNotIn(secret, script)
        self.assertNotIn("Unregister-ScheduledTask", script)
        for wrapper in (
            "run_tw_post_close_pipeline.ps1",
            "run_tw_pre_market_pipeline.ps1",
            "run_full_backtest.ps1",
            "run_us_daily.ps1",
            "run_weekly_model.ps1",
            "upload_local_quant.ps1",
        ):
            with self.subTest(wrapper=wrapper):
                self.assertTrue((scripts / wrapper).is_file())
                self.assertIn(
                    wrapper,
                    (scripts / "invoke_pipeline_task.ps1").read_text(
                        encoding="utf-8"
                    ),
                )
        self.assertIn("invoke_pipeline_task.ps1", script)
        self.assertIn("Task wrapper not found", script)
        self.assertIn("New-ScheduledTaskTrigger -Weekly", script)
        self.assertNotIn("RepeatMinutes=1", script)
        self.assertIn(r"D:\AbsorbData", script)
        wrapper_source = (scripts / "invoke_pipeline_task.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("-RequireReportV2", wrapper_source)
        self.assertIn("-RequireDashboard", wrapper_source)
        self.assertIn("@('-MaxItems', '500')", wrapper_source)
        self.assertIn(
            "'TW-PostClose' = @{ Script = 'run_tw_post_close_pipeline.ps1'; "
            "Arguments = @('-PublishObservation') }",
            wrapper_source,
        )
        post_close = (scripts / "run_tw_post_close_pipeline.ps1").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            post_close.index("calendar-check"),
            post_close.index("stock_papi.batch.tw_official_post_close_cli"),
        )
        self.assertIn("stock_papi.batch.observation_products_cli", post_close)
        self.assertIn("stock_papi.batch.tw_official_post_close_cli", post_close)
        self.assertIn("[switch]$PublishObservation", post_close)
        self.assertIn("[switch]$ReconcileLegacyOverlaps", post_close)
        self.assertIn(
            "$QuantArguments += '--reconcile-legacy-overlaps'", post_close
        )
        self.assertIn("if (-not $PublishObservation) { exit 0 }", post_close)
        self.assertNotIn("local_quant.py", post_close)
        self.assertNotIn("'--observation-only'", post_close)
        self.assertIn("$ExplicitTargetDate = $PSBoundParameters.ContainsKey('TargetDate')", post_close)
        self.assertIn("if ($ExplicitTargetDate)", post_close)
        self.assertIn("'--source-validation-date'", post_close)
        self.assertIn(
            "$HistoricalTargetDate = $ParsedTargetDate.Date -lt [DateTime]::Today",
            post_close,
        )
        self.assertIn(
            "if ($HistoricalTargetDate -and $PublishObservation)",
            post_close,
        )
        self.assertIn(
            "Historical TargetDate cannot publish observation products",
            post_close,
        )
        self.assertLess(
            post_close.index("$HistoricalTargetDate"),
            post_close.index("calendar-check"),
        )
        self.assertNotIn("AllowDegradedBootstrap", post_close)
        self.assertIn("-RequireDashboard", post_close)
        self.assertIn("$ParsedTargetDate.Year", post_close)
        self.assertIn("@(($Year - 1), ($Year + 1))", post_close)
        self.assertIn("$Year - 1", post_close)
        self.assertIn("$Year + 1", post_close)
        self.assertGreaterEqual(post_close.count("'--calendar-artifact'"), 2)
        self.assertLess(
            post_close.index("$CalendarArguments"),
            post_close.index("$QuantArguments"),
        )
        self.assertLess(
            post_close.index("$QuantArguments"),
            post_close.index("$CandidateArguments"),
        )
        pre_market = (scripts / "run_tw_pre_market_pipeline.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("$Latest.product_mode -ne 'observation'", pre_market)

        runtime_helper = scripts / "python_runtime.ps1"
        self.assertTrue(runtime_helper.is_file())
        for pipeline_name, source in (
            ("post_close", post_close),
            ("pre_market", pre_market),
        ):
            with self.subTest(pipeline=pipeline_name):
                self.assertIn(
                    ". (Join-Path $PSScriptRoot 'python_runtime.ps1')",
                    source,
                )
                self.assertIn("Resolve-AbsorbPythonExecutable", source)
                self.assertIn("Assert-AbsorbPythonRuntime", source)
                self.assertIn("[IO.Path]::PathSeparator", source)
                self.assertIn("@($RepoRoot, (Join-Path $RepoRoot '.deps'))", source)
                self.assertNotIn("codex-runtimes", source)
                self.assertNotIn("$BundledPython", source)

        for wrapper_name in ("run_us_daily.ps1", "run_full_backtest.ps1"):
            source = (scripts / wrapper_name).read_text(encoding="utf-8")
            with self.subTest(wrapper=wrapper_name):
                self.assertIn(
                    ". (Join-Path $PSScriptRoot 'python_runtime.ps1')", source
                )
                self.assertIn("Resolve-AbsorbPythonExecutable", source)
                self.assertIn(
                    "-RequiredImports @('stock_papi', 'yfinance')", source
                )
                self.assertNotIn("codex-runtimes", source)
                self.assertNotIn("$BundledPython", source)

    def test_post_close_source_market_date_has_fail_closed_manifest_fallback(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "run_tw_post_close_pipeline.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("$SourceMarketDate", script)
        for required in (
            "$LatestSchema = [int]$Latest.schema_version",
            "Manifest path is not allowlisted",
            "Manifest hash mismatch",
            "Source market date does not match TargetDate",
            "Get-Date -Date",
        ):
            with self.subTest(required=required):
                self.assertIn(required, script)
        fallback = script[script.index("$SourceMarketDate") : script.index("$CandidateArguments")]
        market_as_of = fallback.index("market_as_of")
        observation_as_of = fallback.index("observation_as_of")
        target_market_date = fallback.index("target_market_date")
        self.assertLess(market_as_of, observation_as_of)
        self.assertLess(observation_as_of, target_market_date)
        self.assertIn("'--source-market-date', $SourceMarketDate", script)
        self.assertNotIn("'--source-market-date', $Manifest.market_as_of", script)

    def test_observation_only_upload_scopes_pointers_to_tw_daily_reports(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "upload_local_quant.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("} elseif ($ObservationOnly) {", script)
        self.assertIn("@('TW')", script)
        self.assertIn("if (-not $ObservationOnly -and", script)
        self.assertIn("Get-ObservationReportV2Types", script)
        self.assertIn("latest-TW-pre_market.json", script)
        self.assertIn("[switch]$ReportV2Only", script)
        self.assertIn("$Markets = if ($ReportV2Only)", script)
        self.assertIn("if (-not $ReportV2Only)", script)
        self.assertIn("foreach ($Type in $ReportV2Types)", script)
        self.assertNotIn("foreach ($Market in @('TW', 'US'))", script)

        common = (
            Path(__file__).parents[1]
            / "scripts"
            / "observation_release_common.ps1"
        )
        completed = subprocess.run(
            [
                r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    f". '{common}'; "
                    "$types=@(Get-ObservationReportV2Types -ObservationOnly); "
                    "if (($types -join '|') -ne 'post_close|pre_market') { exit 46 }"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_full_backtest_logs_nonfatal_python_warnings_but_keeps_exit_code(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "run_full_backtest.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$env:ComSpec", source)
        self.assertIn("2>&1", source)
        self.assertIn("$ExitCode = $LASTEXITCODE", source)

    def test_mutex_plan_is_market_specific_and_waits_boundedly_under_contention(self):
        """Execute the production mutex helpers without launching a pipeline."""
        script = (
            Path(__file__).parents[1] / "scripts" / "invoke_pipeline_task.ps1"
        ).resolve()
        harness = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('{str(script).replace("'", "''")}', [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ throw 'invoke_pipeline_task.ps1 did not parse' }}
foreach ($name in @('Get-AbsorbPipelineMutexPlan', 'Enter-AbsorbPipelineMutexPlan', 'Exit-AbsorbPipelineMutexPlan')) {{
    $definition = @($ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }}, $true))[0]
    if ($null -eq $definition) {{ throw "required mutex helper was not found: $name" }}
    Invoke-Expression $definition.Extent.Text
}}
$twPlan = @(Get-AbsorbPipelineMutexPlan -Job 'TW-PreMarket')
$usPlan = @(Get-AbsorbPipelineMutexPlan -Job 'US-PreMarket')
$postClosePlan = @(Get-AbsorbPipelineMutexPlan -Job 'TW-PostClose')
$recoveryPlan = @(Get-AbsorbPipelineMutexPlan -Job 'ReportUploadRecovery')
if ($twPlan.Count -ne 1 -or $twPlan[0].scope -ne 'market' -or $twPlan[0].market -ne 'TW') {{ throw 'TW premarket mutex plan is incorrect' }}
if ($usPlan.Count -ne 1 -or $usPlan[0].scope -ne 'market' -or $usPlan[0].market -ne 'US') {{ throw 'US premarket mutex plan is incorrect' }}
if (($postClosePlan.scope -join '|') -ne 'market|publication') {{ throw 'market-to-publication lock ordering is incorrect' }}
if (($recoveryPlan.scope -join '|') -ne 'publication') {{ throw 'recovery should only hold the publication lock' }}
$twName = 'Local\\ABSORB-Task3-TW-' + [Guid]::NewGuid().ToString('N')
$usName = 'Local\\ABSORB-Task3-US-' + [Guid]::NewGuid().ToString('N')
$publicationName = 'Local\\ABSORB-Task3-Publication-' + [Guid]::NewGuid().ToString('N')
$twPlan[0].mutex_name = $twName
$usPlan[0].mutex_name = $usName
($postClosePlan | Where-Object {{ $_.scope -eq 'publication' }})[0].mutex_name = $publicationName
$recoveryPlan[0].mutex_name = $publicationName
$twPlan[0].wait_milliseconds = 100
$recoveryPlan[0].wait_milliseconds = 100
function Start-TestMutexHolder {{
    param([string]$MutexName)
    $token = [Guid]::NewGuid().ToString('N')
    $ready = Join-Path ([IO.Path]::GetTempPath()) "absorb-task3-$token.ready"
    $release = Join-Path ([IO.Path]::GetTempPath()) "absorb-task3-$token.release"
    $job = Start-Job -ScriptBlock {{
        param($Name, $ReadyPath, $ReleasePath)
        $mutex = [Threading.Mutex]::new($false, $Name)
        try {{
            if (-not $mutex.WaitOne(5000)) {{ throw 'test holder could not acquire mutex' }}
            [IO.File]::WriteAllText($ReadyPath, 'ready')
            while (-not [IO.File]::Exists($ReleasePath)) {{ Start-Sleep -Milliseconds 10 }}
        }} finally {{
            try {{ $mutex.ReleaseMutex() }} catch {{ }}
            $mutex.Dispose()
        }}
    }} -ArgumentList $MutexName, $ready, $release
    for ($index = 0; $index -lt 200 -and -not [IO.File]::Exists($ready); $index++) {{ Start-Sleep -Milliseconds 10 }}
    if (-not [IO.File]::Exists($ready)) {{
        Receive-Job -Job $job -Keep | Out-String | Write-Error
        throw 'test holder did not establish mutex contention'
    }}
    return [pscustomobject]@{{ job = $job; ready = $ready; release = $release }}
}}
function Stop-TestMutexHolder {{
    param($Holder)
    if ($null -eq $Holder) {{ return }}
    [IO.File]::WriteAllText($Holder.release, 'release')
    Wait-Job -Job $Holder.job -Timeout 5 | Out-Null
    Remove-Job -Job $Holder.job -Force
    Remove-Item -LiteralPath $Holder.ready, $Holder.release -Force -ErrorAction SilentlyContinue
}}
$twHolder = Start-TestMutexHolder -MutexName $twName
$publicationHolder = Start-TestMutexHolder -MutexName $publicationName
try {{
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {{
        Enter-AbsorbPipelineMutexPlan -Plan $twPlan | Out-Null
        throw 'TW contention was not rejected'
    }} catch [TimeoutException] {{
        if ($watch.ElapsedMilliseconds -lt 70 -or $watch.ElapsedMilliseconds -gt 1500) {{ throw 'TW wait was not bounded' }}
    }} finally {{ $watch.Stop() }}
    $usLeases = @(Enter-AbsorbPipelineMutexPlan -Plan $usPlan)
    try {{
        if ($usLeases.Count -ne 1 -or -not $usLeases[0].receipt.acquired) {{ throw 'US work was incorrectly blocked by TW contention' }}
    }} finally {{ Exit-AbsorbPipelineMutexPlan -Leases $usLeases }}
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {{
        Enter-AbsorbPipelineMutexPlan -Plan $recoveryPlan | Out-Null
        throw 'publication contention was not rejected'
    }} catch [TimeoutException] {{
        if ($watch.ElapsedMilliseconds -lt 70 -or $watch.ElapsedMilliseconds -gt 1500) {{ throw 'publication wait was not bounded' }}
    }} finally {{ $watch.Stop() }}
}} finally {{
    Stop-TestMutexHolder -Holder $twHolder
    Stop-TestMutexHolder -Holder $publicationHolder
}}
@{{ tw_mutex = $twPlan[0].mutex_name; us_mutex = $usPlan[0].mutex_name; publication_scope = $recoveryPlan[0].scope }} | ConvertTo-Json -Compress
"""
        completed = subprocess.run(
            [
                "powershell.exe",
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
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertNotEqual(receipt["tw_mutex"], receipt["us_mutex"])
        self.assertEqual(receipt["publication_scope"], "publication")

    def test_us_daily_and_us_post_close_serialize_while_tw_premarket_is_independent(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "invoke_pipeline_task.ps1"
        ).resolve()
        harness = self._powershell_function_import(script) + r"""
$ErrorActionPreference = 'Stop'
$dailyPlan = @(Get-AbsorbPipelineMutexPlan -Job 'US-Daily')
$postClosePlan = @(Get-AbsorbPipelineMutexPlan -Job 'US-PostClose')
$twPlan = @(Get-AbsorbPipelineMutexPlan -Job 'TW-PreMarket')
if ($dailyPlan.Count -lt 1 -or $dailyPlan[0].scope -ne 'market' -or $dailyPlan[0].market -ne 'US') { throw 'US-Daily lacks the US market lock' }
if ($dailyPlan[0].mutex_name -ne $postClosePlan[0].mutex_name) { throw 'US jobs do not share the US market lock' }
if (($dailyPlan.scope -join '|') -ne 'market|publication') { throw 'US-Daily lock order is not market then publication' }
$sharedName = 'Local\ABSORB-Round1-US-' + [Guid]::NewGuid().ToString('N')
$twName = 'Local\ABSORB-Round1-TW-' + [Guid]::NewGuid().ToString('N')
$dailyPlan[0].mutex_name = $sharedName
$postClosePlan[0].mutex_name = $sharedName
$postClosePlan[0].wait_milliseconds = 100
$twPlan[0].mutex_name = $twName
$holderReady = Join-Path ([IO.Path]::GetTempPath()) ('absorb-round1-' + [Guid]::NewGuid().ToString('N') + '.ready')
$holderRelease = $holderReady + '.release'
$holder = Start-Job -ScriptBlock {
    param($Name, $Ready, $Release)
    $mutex = [Threading.Mutex]::new($false, $Name)
    try {
        if (-not $mutex.WaitOne(5000)) { throw 'holder could not acquire US mutex' }
        [IO.File]::WriteAllText($Ready, 'ready')
        while (-not [IO.File]::Exists($Release)) { Start-Sleep -Milliseconds 10 }
    } finally {
        try { $mutex.ReleaseMutex() } catch { }
        $mutex.Dispose()
    }
} -ArgumentList $sharedName, $holderReady, $holderRelease
try {
    for ($index = 0; $index -lt 300 -and -not (Test-Path -LiteralPath $holderReady); $index++) { Start-Sleep -Milliseconds 10 }
    if (-not (Test-Path -LiteralPath $holderReady)) { throw 'US holder did not become ready' }
    $receipts = New-Object 'System.Collections.Generic.List[object]'
    try {
        Enter-AbsorbPipelineMutexPlan -Plan $postClosePlan -Receipts $receipts | Out-Null
        throw 'US PostClose did not serialize behind US-Daily'
    } catch [TimeoutException] { }
    $twReceipts = New-Object 'System.Collections.Generic.List[object]'
    $twLeases = @(Enter-AbsorbPipelineMutexPlan -Plan $twPlan -Receipts $twReceipts)
    try {
        if ($twLeases.Count -ne 1) { throw 'TW premarket was blocked by US work' }
    } finally {
        Exit-AbsorbPipelineMutexPlan -Leases $twLeases
    }
    if ($receipts.Count -ne 1 -or $receipts[0].failure_reason -ne 'timeout') { throw 'US contention receipt is incomplete' }
} finally {
    [IO.File]::WriteAllText($holderRelease, 'release')
    Wait-Job -Job $holder -Timeout 5 | Out-Null
    Remove-Job -Job $holder -Force
    Remove-Item -LiteralPath $holderReady, $holderRelease -Force -ErrorAction SilentlyContinue
}
[Console]::WriteLine('serialized')
"""
        completed = subprocess.run(
            [
                "powershell.exe",
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
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("serialized", completed.stdout)

    def test_mutex_abandonment_and_partial_timeout_release_exactly_once(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "invoke_pipeline_task.ps1"
        ).resolve()
        harness = self._powershell_function_import(script) + r"""
$ErrorActionPreference = 'Stop'
$abandonedName = 'Local\ABSORB-Round1-Abandoned-' + [Guid]::NewGuid().ToString('N')
Add-Type -TypeDefinition @'
using System;
using System.Threading;
public static class AbsorbAbandonedMutexProbe {
    public static void Create(string name) {
        var thread = new Thread(() => {
            var mutex = new Mutex(false, name);
            if (!mutex.WaitOne(5000)) throw new Exception("unable to establish abandoned mutex");
        });
        thread.Start();
        thread.Join();
    }
}
'@
[AbsorbAbandonedMutexProbe]::Create($abandonedName)
$abandonedPlan = @([pscustomobject]@{ scope='market'; market='US'; mutex_name=$abandonedName; wait_milliseconds=500 })
$abandonedReceipts = New-Object 'System.Collections.Generic.List[object]'
$abandonedLeases = @(Enter-AbsorbPipelineMutexPlan -Plan $abandonedPlan -Receipts $abandonedReceipts)
Exit-AbsorbPipelineMutexPlan -Leases $abandonedLeases
Exit-AbsorbPipelineMutexPlan -Leases $abandonedLeases
$abandoned = $abandonedReceipts[0]
if (-not $abandoned.ever_acquired -or $abandoned.acquired -or -not $abandoned.released -or $abandoned.acquisition_reason -ne 'abandoned_mutex_acquired') { throw 'abandoned mutex ownership was not safely released' }

$marketName = 'Local\ABSORB-Round1-Market-' + [Guid]::NewGuid().ToString('N')
$publicationName = 'Local\ABSORB-Round1-Publication-' + [Guid]::NewGuid().ToString('N')
$ready = Join-Path ([IO.Path]::GetTempPath()) ('absorb-round1-' + [Guid]::NewGuid().ToString('N') + '.ready')
$release = $ready + '.release'
$holder = Start-Job -ScriptBlock {
    param($Name, $Ready, $Release)
    $mutex = [Threading.Mutex]::new($false, $Name)
    try {
        if (-not $mutex.WaitOne(5000)) { throw 'unable to hold publication mutex' }
        [IO.File]::WriteAllText($Ready, 'ready')
        while (-not [IO.File]::Exists($Release)) { Start-Sleep -Milliseconds 10 }
    } finally {
        try { $mutex.ReleaseMutex() } catch { }
        $mutex.Dispose()
    }
} -ArgumentList $publicationName, $ready, $release
try {
    for ($index = 0; $index -lt 300 -and -not (Test-Path -LiteralPath $ready); $index++) { Start-Sleep -Milliseconds 10 }
    if (-not (Test-Path -LiteralPath $ready)) { throw 'publication holder did not become ready' }
    $plan = @(
        [pscustomobject]@{ scope='market'; market='US'; mutex_name=$marketName; wait_milliseconds=500 },
        [pscustomobject]@{ scope='publication'; market=$null; mutex_name=$publicationName; wait_milliseconds=100 }
    )
    $receipts = New-Object 'System.Collections.Generic.List[object]'
    try {
        Enter-AbsorbPipelineMutexPlan -Plan $plan -Receipts $receipts | Out-Null
        throw 'partial acquisition unexpectedly succeeded'
    } catch [TimeoutException] { }
    if ($receipts.Count -ne 2) { throw 'partial acquisition receipts are incomplete' }
    $market = $receipts[0]
    $publication = $receipts[1]
    if ($market.acquired -or -not $market.ever_acquired -or -not $market.released) { throw 'market lock was not cleaned up after publication timeout' }
    if ($publication.acquired -or $publication.ever_acquired -or $publication.released -or $publication.failure_reason -ne 'timeout' -or $publication.waited_milliseconds -lt 70) { throw 'publication timeout receipt is misleading' }
    $probe = [Threading.Mutex]::new($false, $marketName)
    try {
        if (-not $probe.WaitOne(0)) { throw 'market mutex remains locked after partial failure' }
        $probe.ReleaseMutex()
    } finally { $probe.Dispose() }
} finally {
    [IO.File]::WriteAllText($release, 'release')
    Wait-Job -Job $holder -Timeout 5 | Out-Null
    Remove-Job -Job $holder -Force
    Remove-Item -LiteralPath $ready, $release -Force -ErrorAction SilentlyContinue
}
[Console]::WriteLine('cleanup-ok')
"""
        completed = subprocess.run(
            [
                "powershell.exe",
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
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("cleanup-ok", completed.stdout)

    def test_hidden_task_wrapper_preserves_distinctive_native_exit_code_and_receipt(self):
        scripts = Path(__file__).parents[1] / "scripts"
        invoke_script = (scripts / "invoke_pipeline_task.ps1").resolve()
        native_helper = (scripts / "native_process.ps1").resolve()
        hidden_launcher = (scripts / "run_hidden.vbs").resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_helper_ps = str(native_helper).replace("'", "''")
            root_ps = str(root).replace("'", "''")
            probe = root / "exit_probe.ps1"
            probe.write_text("param([string]$DataRoot)\nexit 73\n", encoding="utf-8-sig")
            probe_ps = str(probe).replace("'", "''")
            logs_ps = str(root / "logs").replace("'", "''")
            harness = root / "invoke_probe.ps1"
            harness.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                + self._powershell_function_import(invoke_script)
                + f". '{native_helper_ps}'\n"
                + (
                    "$code = Invoke-AbsorbPipelineTask "
                    "-Job 'US-Daily' "
                    f"-DataRoot '{root_ps}' "
                    f"-ScriptPath '{probe_ps}' "
                    "-ScriptArguments @() "
                    f"-LogDirectory '{logs_ps}' "
                    "-PowerShellExe (Get-Command powershell.exe -ErrorAction Stop).Source\n"
                    "exit $code\n"
                ),
                encoding="utf-8-sig",
            )
            completed = subprocess.run(
                [
                    "cscript.exe",
                    "//B",
                    "//NoLogo",
                    str(hidden_launcher),
                    r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(harness),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 73, completed.stdout + completed.stderr)
            receipt = json.loads(
                (root / "logs" / "current-US-Daily.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            self.assertFalse(receipt["success"])
            self.assertEqual(receipt["exit_code"], 73)
            self.assertTrue(receipt["mutexes"])
            self.assertTrue(all(item["released"] for item in receipt["mutexes"] if item["ever_acquired"]))

    def test_full_backtest_disable_is_idempotent_and_failure_is_task_failure(self):
        scripts = Path(__file__).parents[1] / "scripts"
        invoke_script = (scripts / "invoke_pipeline_task.ps1").resolve()
        native_helper = (scripts / "native_process.ps1").resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_helper_ps = str(native_helper).replace("'", "''")
            root_ps = str(root).replace("'", "''")
            checkpoint = root / "checkpoints" / "jobs" / "full_backtest" / "current.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            probe = root / "success_probe.ps1"
            probe.write_text("param([string]$DataRoot, [int]$MaxItems)\nexit 0\n", encoding="utf-8-sig")
            probe_ps = str(probe).replace("'", "''")
            logs_ps = str(root / "logs").replace("'", "''")
            harness = (
                "$ErrorActionPreference = 'Stop'\n"
                + self._powershell_function_import(invoke_script)
                + f". '{native_helper_ps}'\n"
                + r"""
$Global:TaskState = 'Disabled'
$Global:DisableCalls = 0
$Global:DisableMode = 'ok'
function Get-ScheduledTask { param([string]$TaskName, $ErrorAction) [pscustomobject]@{ TaskName=$TaskName; State=$Global:TaskState } }
function Disable-ScheduledTask {
    param([string]$TaskName, $ErrorAction)
    $Global:DisableCalls++
    if ($Global:DisableMode -eq 'permission') { throw [UnauthorizedAccessException]::new('sensitive permission detail') }
    $Global:TaskState = 'Disabled'
}
$already = Disable-AbsorbCompletedFullBacktestTask
if (-not $already.already_disabled -or $Global:DisableCalls -ne 0) { throw 'already-disabled task was not idempotent' }
$Global:TaskState = 'Ready'
$disabled = Disable-AbsorbCompletedFullBacktestTask
if ($disabled.already_disabled -or -not $disabled.disabled -or $Global:DisableCalls -ne 1) { throw 'enabled task was not disabled and verified' }
$Global:TaskState = 'Ready'
$Global:DisableMode = 'permission'
"""
                + (
                    "$code = Invoke-AbsorbPipelineTask "
                    "-Job 'FullBacktest' "
                    f"-DataRoot '{root_ps}' "
                    f"-ScriptPath '{probe_ps}' "
                    "-ScriptArguments @('-MaxItems', '500') "
                    f"-LogDirectory '{logs_ps}' "
                    "-PowerShellExe (Get-Command powershell.exe -ErrorAction Stop).Source\n"
                    "$receipt = Get-Content -LiteralPath '"
                    + str(root / "logs" / "current-FullBacktest.json").replace("'", "''")
                    + "' -Raw -Encoding utf8 | ConvertFrom-Json\n"
                    "if ($code -eq 0 -or $receipt.success -or $receipt.exit_code -eq 0) { throw 'disable failure was reported as success' }\n"
                    "if ([string]$receipt.error -match 'sensitive permission detail') { throw 'unsafe disable error leaked' }\n"
                    "[Console]::WriteLine('disable-contract-ok')\n"
                )
            )
            completed = subprocess.run(
                [
                    "powershell.exe",
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
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("disable-contract-ok", completed.stdout)

    def test_installer_captures_final_registration_without_machine_mutation(self):
        installer = (
            Path(__file__).parents[1] / "scripts" / "install_pipeline_tasks.ps1"
        ).resolve()
        installer_ps = str(installer).replace("'", "''")
        harness = r"""
$ErrorActionPreference = 'Stop'
$Global:Registrations = New-Object 'System.Collections.Generic.List[object]'
function Get-Command { param([string]$Name, $ErrorAction) [pscustomobject]@{ Source=('C:\Windows\System32\' + $Name) } }
function New-ScheduledTaskPrincipal { param($UserId, $LogonType, $RunLevel) [pscustomobject]@{ UserId=$UserId; LogonType=$LogonType; RunLevel=$RunLevel } }
function New-ScheduledTaskAction { param($Execute, $Argument, $WorkingDirectory) [pscustomobject]@{ Execute=$Execute; Argument=$Argument; WorkingDirectory=$WorkingDirectory } }
function New-ScheduledTaskTrigger {
    param([switch]$Daily, [switch]$Weekly, $DaysOfWeek, [datetime]$At)
    [pscustomobject]@{ Kind=$(if ($Daily) {'Daily'} else {'Weekly'}); DaysOfWeek=$DaysOfWeek; At=$At }
}
function New-ScheduledTaskSettingsSet {
    param([timespan]$ExecutionTimeLimit, $MultipleInstances, $RestartCount, [timespan]$RestartInterval)
    [pscustomobject]@{ ExecutionTimeLimit=$ExecutionTimeLimit; MultipleInstances=$MultipleInstances; RestartCount=$RestartCount; RestartInterval=$RestartInterval; StartWhenAvailable=$false; WakeToRun=$false }
}
function Register-ScheduledTask {
    param($TaskName, $Action, $Trigger, $Settings, $Principal, [string]$Xml, [switch]$Force)
    $Global:Registrations.Add([pscustomobject]@{ TaskName=$TaskName; Action=$Action; Trigger=$Trigger; Settings=$Settings; Principal=$Principal; Xml=$Xml })
}
function schtasks {
    return '<?xml version="1.0" encoding="UTF-16"?><Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Triggers><CalendarTrigger><StartBoundary>2026-08-24T17:10:00</StartBoundary><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger></Triggers></Task>'
}
""" + f". '{installer_ps}' -DataRoot 'D:\\AbsorbData' -WeeklyDay Saturday\n" + r"""
$full = @($Global:Registrations | Where-Object { $_.TaskName -eq 'ABSORB-FullBacktest' })
if ($full.Count -ne 1) { throw 'FullBacktest was not registered exactly once' }
$registration = $full[0]
if ($registration.Trigger.Kind -ne 'Daily' -or $registration.Trigger.At.ToString('HH:mm') -ne '22:30') { throw 'FullBacktest trigger is not daily at 22:30' }
if ($registration.Settings.ExecutionTimeLimit -ne [TimeSpan]::FromMinutes(225)) { throw 'FullBacktest limit is not PT3H45M' }
if ($registration.Xml -or $registration.Trigger.PSObject.Properties.Name -contains 'Repetition') { throw 'FullBacktest gained repetition' }
if ([IO.Path]::GetFileName($registration.Action.Execute) -ne 'wscript.exe') { throw 'FullBacktest does not use wscript' }
foreach ($required in @('//B','//NoLogo','run_hidden.vbs','-WindowStyle','Hidden','invoke_pipeline_task.ps1')) {
    if ($registration.Action.Argument -notmatch [regex]::Escape($required)) { throw "hidden action is missing $required" }
}
$postXml = @($Global:Registrations | Where-Object { $_.TaskName -eq 'ABSORB-TW-PostClose' -and $_.Xml })
if ($postXml.Count -ne 1 -or $postXml[0].Xml -notmatch '<Interval>PT20M</Interval>' -or $postXml[0].Xml -notmatch '<Duration>PT4H50M</Duration>') { throw 'final post-close XML repetition was not captured' }
[Console]::WriteLine('installer-capture-ok')
"""
        completed = subprocess.run(
            [
                "powershell.exe",
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
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("installer-capture-ok", completed.stdout)

    def test_full_backtest_completed_checkpoint_exits_before_yfinance_import(self):
        cli_root = Path(__file__).parents[1].resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "jobs" / "full_backtest" / "current.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            probe = root / "completed_checkpoint_probe.py"
            probe.write_text(
                "import builtins\n"
                "import datetime\n"
                "import sys\n"
                "import types\n"
                "from types import SimpleNamespace\n"
                "_real_import = builtins.__import__\n"
                "def _blocked(name, *args, **kwargs):\n"
                "    if name == 'yfinance' or name.startswith('yfinance.'):\n"
                "        raise ModuleNotFoundError('deliberately unavailable yfinance')\n"
                "    return _real_import(name, *args, **kwargs)\n"
                "builtins.__import__ = _blocked\n"
                "from stock_papi.batch import full_backtest_cli\n"
                "local_quant = types.ModuleType('local_quant')\n"
                "def load_stock_pipeline(_root):\n"
                "    import yfinance\n"
                "local_quant.load_stock_pipeline = load_stock_pipeline\n"
                "sys.modules['local_quant'] = local_quant\n"
                "source_loader = types.ModuleType('reporting.source_loader')\n"
                "source_loader.load_report_source = lambda _root, market: SimpleNamespace(\n"
                "    manifest=SimpleNamespace(manifest_path='manifests/TW-20260821T000000Z-aaaaaaaaaaaa.json', manifest_sha256='a' * 64, market_as_of=datetime.date(2026, 8, 21)),\n"
                "    stocks=(SimpleNamespace(model_version='model-v1', symbol='2330'),))\n"
                "sys.modules['reporting.source_loader'] = source_loader\n"
                "raise SystemExit(full_backtest_cli.main(['--root', sys.argv[1]]))\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(cli_root)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(probe),
                    str(root),
                ],
                cwd=cli_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("already completed", output)
            self.assertNotIn("deliberately unavailable yfinance", output)

    def test_full_backtest_wrapper_checkpoint_guard_is_python_free(self):
        wrapper = (
            Path(__file__).parents[1] / "scripts" / "run_full_backtest.ps1"
        ).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "jobs" / "full_backtest" / "current.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            harness = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('{str(wrapper).replace("'", "''")}', [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ throw 'run_full_backtest.ps1 did not parse' }}
$definition = @($ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-AbsorbFullBacktestCompletedCheckpoint' }}, $true))[0]
if ($null -eq $definition) {{ throw 'full backtest checkpoint helper was not found' }}
Invoke-Expression $definition.Extent.Text
$env:ABSORB_PYTHON_EXE = 'C:\\missing-yfinance-runtime.exe'
if (-not (Test-AbsorbFullBacktestCompletedCheckpoint -DataRoot '{str(root).replace("'", "''")}')) {{ throw 'completed checkpoint was not accepted' }}
[Console]::WriteLine('completed-without-python')
"""
            completed = subprocess.run(
                [
                    "powershell.exe",
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
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("completed-without-python", completed.stdout)

    def test_pipeline_installer_definition_is_daily_bounded_and_hidden(self):
        installer = (
            Path(__file__).parents[1] / "scripts" / "install_pipeline_tasks.ps1"
        ).resolve()
        harness = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile('{str(installer).replace("'", "''")}', [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {{ throw 'install_pipeline_tasks.ps1 did not parse' }}
$definition = @($ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-AbsorbPipelineTaskDefinitions' }}, $true))[0]
if ($null -eq $definition) {{ throw 'task definition helper was not found' }}
Invoke-Expression $definition.Extent.Text
$fullBacktest = @(Get-AbsorbPipelineTaskDefinitions -WeeklyDay Saturday | Where-Object {{ $_.Name -eq 'ABSORB-FullBacktest' }})
if ($fullBacktest.Count -ne 1) {{ throw 'full backtest definition was not unique' }}
@{{ time = $fullBacktest[0].Time; repeat_minutes = $fullBacktest[0].RepeatMinutes; execution_minutes = [int]$fullBacktest[0].ExecutionTimeLimit.TotalMinutes }} | ConvertTo-Json -Compress
"""
        completed = subprocess.run(
            [
                "powershell.exe",
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
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        definition = json.loads(completed.stdout)
        self.assertEqual(definition["time"], "22:30")
        self.assertIsNone(definition["repeat_minutes"])
        self.assertEqual(definition["execution_minutes"], 225)
        source = installer.read_text(encoding="utf-8")
        for required in ("run_hidden.vbs", "wscript.exe", "//B", "//NoLogo", "-WindowStyle", "Hidden"):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_task_wrapper_records_success_or_failure_without_secrets(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "invoke_pipeline_task.ps1"
        ).read_text(encoding="utf-8")
        for required in (
            "logs\\tasks",
            "current-",
            "Get-Command powershell.exe",
            "Invoke-NativeProcessStreaming",
            ".exit_code",
            "-LogPath $LogPath",
            "$TaskSucceeded = $false",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertIn(
            "Disable-ScheduledTask -TaskName 'ABSORB-FullBacktest'",
            source,
        )
        self.assertNotIn("Invoke-NativeProcessCaptured", source)
        self.assertIn("$Checkpoint.status -eq 'completed'", source)
        self.assertIn("mutexes = @($MutexReceipts.ToArray())", source)
        self.assertIn("wait_milliseconds", source)
        self.assertIn("waited_milliseconds", source)
        for forbidden in (
            "LINE_CHANNEL_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "Bearer",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_gcloud_wrapper_uses_native_exit_code_helper(self):
        scripts = Path(__file__).parents[1] / "scripts"
        helper = scripts / "native_process.ps1"
        release_common = (scripts / "observation_release_common.ps1").read_text(
            encoding="utf-8"
        )
        upload = (scripts / "upload_local_quant.ps1").read_text(encoding="utf-8")

        self.assertTrue(helper.is_file())
        self.assertIn("Invoke-NativeProcessCaptured", release_common)
        self.assertNotIn("$Output = & $Gcloud @Arguments 2>&1", release_common)
        self.assertNotIn("& $Gcloud @Arguments", upload)

    def test_post_close_scheduler_bounded_repetition_and_idempotency(self):
        scripts = Path(__file__).parents[1] / "scripts"
        install_source = (scripts / "install_pipeline_tasks.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("RepetitionInterval='PT20M'", install_source)
        self.assertIn("RepetitionDuration='PT4H50M'", install_source)
        self.assertIn("RepetitionNode", install_source)
        self.assertIn("StopAtDurationEnd", install_source)

        post_close_source = (scripts / "run_tw_post_close_pipeline.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("post_close_pipeline_guard.ps1", post_close_source)
        self.assertIn("Test-PostCloseCompletion", post_close_source)
        self.assertIn("skipping duplicate execution", post_close_source)
        self.assertLess(
            post_close_source.index("post_close_pipeline_guard.ps1"),
            post_close_source.index("python_runtime.ps1"),
        )

    @staticmethod
    def _invoke_completion_guard(guard, root, target_date):
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    f". '{guard}'; "
                    f"if (Test-PostCloseCompletion -DataRoot '{root}' "
                    f"-TargetDate '{target_date}') {{ exit 0 }} else {{ exit 1 }}"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"

    def test_post_close_completion_guard_requires_end_to_end_evidence(self):
        import json
        import tempfile

        guard = (
            Path(__file__).parents[1]
            / "scripts"
            / "post_close_pipeline_guard.ps1"
        )
        self.assertTrue(guard.is_file())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pointer_dir = root / "publish" / "reports" / "v2"
            pointer_path = pointer_dir / "latest-TW-post_close.json"
            status_dir = root / "logs" / "tasks"
            status_path = status_dir / "current-TW-PostClose.json"
            pointer_dir.mkdir(parents=True, exist_ok=True)
            status_dir.mkdir(parents=True, exist_ok=True)

            # 1. Nothing published yet -> must NOT skip (re-run required)
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

            # 2. Local pointer exists but no wrapper receipt (promote done,
            #    upload failed) -> must NOT skip; retry must re-run upload
            pointer_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "absorb-report",
                        "market": "TW",
                        "report_type": "post_close",
                        "source_market_date": "2026-08-17",
                        "metadata": "metadata/a.json",
                        "metadata_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

            # 3. Pointer + failed receipt -> must NOT skip
            status_path.write_text(
                json.dumps(
                    {
                        "job": "TW-PostClose",
                        "started_at": "2026-08-17T17:10:00+08:00",
                        "success": False,
                        "exit_code": 1,
                    }
                ),
                encoding="utf-8",
            )
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

            # 4. Pointer + success receipt for a different day -> must NOT skip
            status_path.write_text(
                json.dumps(
                    {
                        "job": "TW-PostClose",
                        "started_at": "2026-08-16T17:10:00+08:00",
                        "success": True,
                        "exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

            # 5. Pointer + success receipt for the same day -> skip
            status_path.write_text(
                json.dumps(
                    {
                        "job": "TW-PostClose",
                        "started_at": "2026-08-17T21:50:00+08:00",
                        "success": True,
                        "exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertTrue(completed)

            # 6. Corrupt pointer JSON -> must NOT skip (fail-open into re-run)
            pointer_path.write_text("{ not valid json", encoding="utf-8")
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

            # 7. Pointer for another source date -> must NOT skip
            pointer_path.write_text(
                json.dumps({"source_market_date": "2026-08-14"}),
                encoding="utf-8",
            )
            completed, _output = self._invoke_completion_guard(
                guard, root, "2026-08-17"
            )
            self.assertFalse(completed)

    def test_availability_aware_retry_contract_and_date_preservation(self):
        import json
        import tempfile
        from stock_papi.batch.post_close import PostClosePipeline, PostClosePipelineError

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_date = datetime.date(2026, 7, 14)
            publish_dir = root / "publish" / "reports" / "v2"
            publish_dir.mkdir(parents=True, exist_ok=True)
            latest_file = publish_dir / "latest-TW-post_close.json"

            # 1. Unavailable official dataset -> fail-closed (no latest created)
            calls = []
            def failing_source():
                raise RuntimeError("official margin data unavailable")

            wired = {
                "load_source": failing_source,
                "infer": lambda v: calls.append("infer"),
                "settle": lambda v: calls.append("settle"),
                "aggregate": lambda v, i, s: calls.append("aggregate"),
                "render": lambda r: calls.append("render"),
                "publish": lambda r, ren: calls.append("publish"),
                "upload": lambda rec: calls.append("upload"),
                "remote_verify": lambda rec: calls.append("verify"),
                "notify": lambda rec: calls.append("notify"),
            }
            pipeline = PostClosePipeline(
                root,
                target_market_date=target_date,
                source_manifest="quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                source_manifest_sha256="a" * 64,
                model_version="lgbm-5d-v1",
                callbacks=wired,
            )
            with self.assertRaises(RuntimeError):
                pipeline.run(now=datetime.datetime(2026, 7, 14, 17, 10, tzinfo=UTC))
            self.assertEqual(calls, [])
            self.assertFalse(latest_file.exists())

            # 2. Retry succeeds later (e.g. 21:20) without date drift
            def success_source():
                return {
                    "market": "TW",
                    "market_as_of": "2026-07-14",
                    "manifest_path": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                    "manifest_sha256": "a" * 64,
                    "model_version": "lgbm-5d-v1",
                    "failure_rate": 0.0,
                    "sample_data": False,
                }

            def publish_cb(report, rendered):
                calls.append("publish")
                latest_file.write_text(
                    json.dumps({
                        "source_market_date": "2026-07-14",
                        "applicable_trading_date": "2026-07-15",
                        "report_type": "post_close",
                    }),
                    encoding="utf-8",
                )
                return {"content_sha256": "b" * 64}

            wired["load_source"] = success_source
            wired["publish"] = publish_cb
            res = pipeline.run(now=datetime.datetime(2026, 7, 14, 21, 20, tzinfo=UTC))
            self.assertEqual(res["status"], "completed")
            self.assertIn("publish", calls)
            self.assertTrue(latest_file.exists())

            latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
            self.assertEqual(latest_data["source_market_date"], "2026-07-14")
            self.assertEqual(latest_data["applicable_trading_date"], "2026-07-15")


if __name__ == "__main__":
    unittest.main()
