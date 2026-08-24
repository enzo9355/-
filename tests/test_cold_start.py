import os
import subprocess
import sys
import unittest
from pathlib import Path


class ColdStartTests(unittest.TestCase):
    def test_tw_lifecycle_pdf_dependency_is_bounded_and_probed_by_cold_wrappers(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("pypdf>=5,<7", requirements.splitlines())
        for wrapper in (
            "run_tw_post_close_pipeline.ps1",
            "run_tw_pre_market_pipeline.ps1",
        ):
            with self.subTest(wrapper=wrapper):
                source = (root / "scripts" / wrapper).read_text(encoding="utf-8")
                self.assertIn(
                    "-RequiredImports @('stock_papi', 'pypdf')", source
                )

    def test_import_does_not_load_analysis_stack(self):
        env = os.environ.copy()
        env.update({
            "LINE_CHANNEL_ACCESS_TOKEN": "test",
            "LINE_CHANNEL_SECRET": "test",
            "GEMINI_API_KEY": "test",
            "GCP_PROJECT_ID": "",
            "PYTHONWARNINGS": "ignore",
        })
        script = """
import sys
import app

heavy = (
    "pandas", "numpy", "sklearn", "lightgbm", "google.generativeai",
    "matplotlib", "reportlab", "pypdf",
)
loaded = [name for name in heavy if name in sys.modules]
if loaded:
    raise SystemExit("loaded during startup: " + ", ".join(loaded))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_does_not_make_http_requests(self):
        env = os.environ.copy()
        env.update({
            "LINE_CHANNEL_ACCESS_TOKEN": "test",
            "LINE_CHANNEL_SECRET": "test",
            "GCP_PROJECT_ID": "test-project-123",
            "SUPABASE_URL": "",
            "SUPABASE_KEY": "",
            "PYTHONWARNINGS": "ignore",
        })
        script = """
import requests.sessions

def fail(*args, **kwargs):
    raise RuntimeError("HTTP during import")

requests.sessions.Session.request = fail
import app
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
