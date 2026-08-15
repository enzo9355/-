import os
import subprocess
import sys
import unittest
from pathlib import Path


class PipelineSchedulerTests(unittest.TestCase):
    def test_historical_target_with_publish_fails_before_data_access(self):
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "run_tw_post_close_pipeline.ps1"
        )
        environment = os.environ.copy()
        environment["ABSORB_PYTHON_EXE"] = sys.executable
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
        self.assertIn("RepeatMinutes=1", script)
        self.assertIn("-RepetitionInterval", script)
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

    def test_observation_only_upload_scopes_pointers_to_tw_post_close(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "upload_local_quant.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("$Markets = if ($ObservationOnly)", script)
        self.assertIn("@('TW')", script)
        self.assertIn("if (-not $ObservationOnly -and", script)
        self.assertIn("$ReportV2Types = if ($ObservationOnly)", script)
        self.assertIn("@('post_close')", script)
        self.assertIn("foreach ($Type in $ReportV2Types)", script)
        self.assertNotIn("foreach ($Market in @('TW', 'US'))", script)

    def test_full_backtest_logs_nonfatal_python_warnings_but_keeps_exit_code(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "run_full_backtest.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$env:ComSpec", source)
        self.assertIn("2>&1", source)
        self.assertIn("$ExitCode = $LASTEXITCODE", source)

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
            "success = $false",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertIn(
            "Disable-ScheduledTask -TaskName 'ABSORB-FullBacktest'",
            source,
        )
        self.assertNotIn("Invoke-NativeProcessCaptured", source)
        self.assertIn("$Checkpoint.status -eq 'completed'", source)
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


if __name__ == "__main__": unittest.main()
