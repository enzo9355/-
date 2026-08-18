import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class HiddenSchedulerTests(unittest.TestCase):
    def test_hidden_launcher_executes_and_propagates_exit_code(self):
        launcher = SCRIPTS / "run_hidden.vbs"
        self.assertTrue(launcher.is_file())
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "hidden-launcher-result.txt"
            child_script = Path(temporary) / "hidden-launcher-child.ps1"
            child_script.write_text(
                f"[IO.File]::WriteAllText('{marker}', 'ok')\nexit 37\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "cscript.exe",
                    "//B",
                    "//NoLogo",
                    str(launcher),
                    r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-WindowStyle",
                    "Hidden",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(child_script),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            diagnostic = f"{completed.stdout}\n{completed.stderr}".lower()
            if (
                completed.returncode == 1
                and "settings failed" in diagnostic
                and "access is denied" in diagnostic
            ):
                self.skipTest("Windows Script Host settings are sandbox-blocked")
            self.assertEqual(completed.returncode, 37)
            self.assertEqual(marker.read_text(encoding="utf-8"), "ok")

    def test_installer_uses_gui_hidden_launcher(self):
        source = (SCRIPTS / "install_pipeline_tasks.ps1").read_text(
            encoding="utf-8"
        )
        for required in (
            "run_hidden.vbs",
            "wscript.exe",
            "//B",
            "//NoLogo",
            "-WindowStyle",
            "Hidden",
            "-WorkingDirectory $RepoRoot",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
