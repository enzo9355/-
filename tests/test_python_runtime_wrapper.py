import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


POWERSHELL = shutil.which("powershell.exe")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell 5.1 is required")
class PythonRuntimeWrapperTests(unittest.TestCase):
    @staticmethod
    def helper_path():
        return (
            Path(__file__).parents[1] / "scripts" / "python_runtime.ps1"
        ).resolve()

    @staticmethod
    def ps_quote(value):
        return str(value).replace("'", "''")

    def run_powershell(self, lines, *, env=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = root / "harness.ps1"
            harness.write_text(
                "\n".join(
                    (
                        "$ErrorActionPreference = 'Stop'",
                        "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)",
                        f". '{self.ps_quote(self.helper_path())}'",
                        *lines,
                    )
                ),
                encoding="utf-8-sig",
            )
            return subprocess.run(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(harness),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=root,
                env=env,
            )

    @staticmethod
    def create_leaf(path, content=b""):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path.resolve()

    @staticmethod
    def assert_resolved_file(result, expected):
        actual = result.stdout.strip().splitlines()[-1]
        if not os.path.samefile(actual, expected):
            raise AssertionError(f"resolved {actual!r}, expected {str(expected)!r}")

    def test_explicit_absolute_override_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            override = self.create_leaf(root / "override-python.cmd")
            self.create_leaf(repo / ".venv" / "Scripts" / "python.exe")
            env = os.environ.copy()
            env["ABSORB_PYTHON_EXE"] = str(override)

            result = self.run_powershell(
                (
                    "$resolved = Resolve-AbsorbPythonExecutable "
                    f"-RepoRoot '{self.ps_quote(repo)}'",
                    "[Console]::WriteLine($resolved)",
                ),
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_resolved_file(result, override)

    def test_runner_local_venv_precedes_system_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            venv_python = self.create_leaf(
                repo / ".venv" / "Scripts" / "python.exe"
            )
            system_dir = root / "system"
            self.create_leaf(system_dir / "python.cmd")
            env = os.environ.copy()
            env.pop("ABSORB_PYTHON_EXE", None)
            env["PATH"] = str(system_dir)

            result = self.run_powershell(
                (
                    "$resolved = Resolve-AbsorbPythonExecutable "
                    f"-RepoRoot '{self.ps_quote(repo)}'",
                    "[Console]::WriteLine($resolved)",
                ),
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_resolved_file(result, venv_python)

    def test_system_python_is_used_only_when_override_and_venv_are_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            system_dir = root / "system"
            system_python = self.create_leaf(system_dir / "python.cmd")
            env = os.environ.copy()
            env.pop("ABSORB_PYTHON_EXE", None)
            env["PATH"] = str(system_dir)

            result = self.run_powershell(
                (
                    "$resolved = Resolve-AbsorbPythonExecutable "
                    f"-RepoRoot '{self.ps_quote(repo)}'",
                    "[Console]::WriteLine($resolved)",
                ),
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_resolved_file(result, system_python)

    def test_invalid_present_override_fails_closed_without_falling_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            self.create_leaf(repo / ".venv" / "Scripts" / "python.exe")
            env = os.environ.copy()
            env["ABSORB_PYTHON_EXE"] = "relative-python.exe"

            result = self.run_powershell(
                (
                    "Resolve-AbsorbPythonExecutable "
                    f"-RepoRoot '{self.ps_quote(repo)}' | Out-Null",
                ),
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "ABSORB_PYTHON_EXE must be an existing absolute file path",
                result.stdout + result.stderr,
            )

    def test_missing_runtime_fails_with_safe_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            empty_path = root / "empty-path"
            empty_path.mkdir()
            env = os.environ.copy()
            env.pop("ABSORB_PYTHON_EXE", None)
            env["PATH"] = str(empty_path)

            result = self.run_powershell(
                (
                    "Resolve-AbsorbPythonExecutable "
                    f"-RepoRoot '{self.ps_quote(repo)}' | Out-Null",
                ),
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Python executable was not found",
                result.stdout + result.stderr,
            )

    def test_runtime_smoke_check_imports_from_repo_root_outside_working_directory(self):
        repo = Path(__file__).parents[1].resolve()
        result = self.run_powershell(
            (
                "Assert-AbsorbPythonRuntime "
                f"-PythonExe '{self.ps_quote(Path(sys.executable).resolve())}' "
                f"-RepoRoot '{self.ps_quote(repo)}'",
                "[Console]::WriteLine('ok')",
            )
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_runtime_smoke_check_is_fail_closed_and_does_not_echo_child_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            success = self.create_leaf(
                root / "success-python.cmd",
                b"@echo off\r\nexit /b 0\r\n",
            )
            failure = self.create_leaf(
                root / "failure-python.cmd",
                b"@echo off\r\n>&2 echo sensitive-child-output\r\nexit /b 7\r\n",
            )

            succeeded = self.run_powershell(
                (
                    "Assert-AbsorbPythonRuntime "
                    f"-PythonExe '{self.ps_quote(success)}' "
                    f"-RepoRoot '{self.ps_quote(repo)}'",
                    "[Console]::WriteLine('ok')",
                )
            )
            self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
            self.assertIn("ok", succeeded.stdout)

            failed = self.run_powershell(
                (
                    "Assert-AbsorbPythonRuntime "
                    f"-PythonExe '{self.ps_quote(failure)}' "
                    f"-RepoRoot '{self.ps_quote(repo)}'",
                )
            )
            self.assertNotEqual(failed.returncode, 0)
            combined = failed.stdout + failed.stderr
            self.assertIn("Selected Python runtime cannot import stock_papi", combined)
            self.assertNotIn("sensitive-child-output", combined)

    def test_runtime_smoke_check_can_require_yfinance_without_child_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            runtime = self.create_leaf(
                root / "runtime.cmd",
                (
                    b"@echo off\r\n"
                    b"echo %* | findstr /c:\"yfinance\" >nul && (>&2 echo sensitive-child-output & exit /b 7)\r\n"
                    b"exit /b 0\r\n"
                ),
            )
            result = self.run_powershell(
                (
                    "Assert-AbsorbPythonRuntime "
                    f"-PythonExe '{self.ps_quote(runtime)}' "
                    f"-RepoRoot '{self.ps_quote(repo)}' "
                    "-RequiredImports @('stock_papi', 'yfinance')",
                )
            )
            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("required ABSORB modules", combined)
            self.assertNotIn("sensitive-child-output", combined)


if __name__ == "__main__":
    unittest.main()
