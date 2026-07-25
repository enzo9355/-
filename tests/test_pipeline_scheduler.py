import unittest
from pathlib import Path


class PipelineSchedulerTests(unittest.TestCase):
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
        self.assertIn("if (-not $PublishObservation) { exit 0 }", post_close)
        self.assertNotIn("local_quant.py", post_close)
        self.assertNotIn("'--observation-only'", post_close)
        self.assertNotIn("AllowDegradedBootstrap", post_close)
        self.assertIn("-RequireDashboard", post_close)
        self.assertIn("$ParsedTargetDate.Year", post_close)
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
