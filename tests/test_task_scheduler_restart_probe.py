import json
import os
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

TEMP_ROOT = Path(os.environ.get("TEMP", r"C:\Users\enzo\AppData\Local\Temp"))
PROBE_PREFIX = "ABSORB-Probe-RestartSemantics"


class TaskSchedulerRestartProbeTests(unittest.TestCase):
    """Isolated synthetic probe: document Task Scheduler restart-on-failure semantics.

    Registers a throwaway task (unique name, synthetic probe script only) whose
    action exits with a non-zero code, then observes how many times Task
    Scheduler actually runs it. Never touches production tasks or pipelines.

    Production evidence (2026-08-24) and this probe both show that
    RestartOnFailure does NOT restart a task whose process ran and returned a
    non-zero exit code. Explicit bounded repetition therefore remains the
    deterministic retry mechanism (see PreMarket PT10M/PT1H20M repetition).
    """

    def test_restart_on_failure_does_not_restart_nonzero_exit_runs(self):
        token = uuid.uuid4().hex[:12]
        task_name = f"{PROBE_PREFIX}-{token}"
        temporary = tempfile.TemporaryDirectory(
            dir=str(TEMP_ROOT / "opencode")
        )
        self.addCleanup(temporary.cleanup)
        probe_dir = Path(temporary.name)
        counter_path = probe_dir / "counter.txt"
        log_path = probe_dir / "attempts.jsonl"
        probe = probe_dir / "probe.ps1"
        probe.write_text(
            "$path = $args[0]\n"
            "$count = 0\n"
            "if ([IO.File]::Exists($path)) {\n"
            "  $count = [int]([IO.File]::ReadAllText($path))\n"
            "}\n"
            "$count += 1\n"
            "[IO.File]::WriteAllText($path, [string]$count)\n"
            "Add-Content -LiteralPath $args[1] -Value (@{ attempt = $count; at = [DateTimeOffset]::Now.ToString('o') } | ConvertTo-Json -Compress)\n"
            "exit 7\n",
            encoding="utf-8",
        )
        power_shell = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"

        registration = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "$action = New-ScheduledTaskAction -Execute "
                    f"'{power_shell}' -Argument \"-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `\"{probe}`\" `\"{counter_path}`\" `\"{log_path}`\"\" "
                    "-WorkingDirectory " f"'{probe_dir}'; "
                    f"$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(10); "
                    "$settings = New-ScheduledTaskSettingsSet -RestartCount 2 "
                    "-RestartInterval (New-TimeSpan -Minutes 1) "
                    "-MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10); "
                    "$settings.StartWhenAvailable = $true; "
                    "$settings.DisallowStartIfOnBatteries = $false; "
                    "$settings.StopIfGoingOnBatteries = $false; "
                    "$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited; "
                    f"Register-ScheduledTask -TaskName '{task_name}' -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null; "
                    f"(Get-ScheduledTask -TaskName '{task_name}').State"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        def cleanup():
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"Unregister-ScheduledTask -TaskName '{task_name}' -Confirm:$false -ErrorAction SilentlyContinue",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

        self.addCleanup(cleanup)
        if registration.returncode != 0:
            diagnostic = f"{registration.stdout}\n{registration.stderr}".lower()
            if "access is denied" in diagnostic:
                self.skipTest("Task Scheduler probe registration is denied in this environment")
            self.fail(f"probe task registration failed: {registration.stdout} {registration.stderr}")

        observed_lines = []
        last_result = None
        deadline = time.time() + 180
        while time.time() < deadline:
            if log_path.exists():
                observed_lines = log_path.read_text(
                    encoding="utf-8"
                ).strip().splitlines()
            if observed_lines:
                info = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        f"(Get-ScheduledTaskInfo -TaskName '{task_name}').LastTaskResult",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if info.returncode == 0:
                    last_result = info.stdout.strip()
                break
            time.sleep(5)
        cleanup()

        self.assertGreaterEqual(
            len(observed_lines), 1,
            "probe task did not run at all",
        )
        self.assertEqual(last_result, "7")
        attempts = [json.loads(line) for line in observed_lines]
        self.assertEqual(
            [item["attempt"] for item in attempts],
            [1],
            "RestartOnFailure unexpectedly restarted a non-zero-exit run; "
            "scheduler semantics changed and this contract must be revisited",
        )


if __name__ == "__main__":
    unittest.main()
