import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class ObservationReleaseScriptTests(unittest.TestCase):
    def test_common_path_guard_fails_closed_before_parent_becomes_null(self):
        source = (
            SCRIPTS / "observation_release_common.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("$null -ne $Current", source)
        self.assertIn("$null -eq $Current", source)
        self.assertIn("escaped allowlisted root", source)
        self.assertIn("contains a reparse point", source)
        self.assertNotIn("ContainsKey($Current.FullName)", source)

    def test_common_path_guard_accepts_inside_and_rejects_sibling_tree(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "allowlisted"
            sibling = parent / "allowlisted-sibling"
            root.mkdir()
            sibling.mkdir()
            inside = root / "inside.json"
            outside = sibling / "outside.json"
            inside.write_text("{}", encoding="utf-8")
            outside.write_text("{}", encoding="utf-8")

            def quoted(path):
                return str(path).replace("'", "''")

            accepted = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    (
                        "$ErrorActionPreference='Stop'; "
                        f". '{quoted(common)}'; $cache=@{{}}; "
                        "Assert-PathWithinRoot "
                        f"-Path '{quoted(inside)}' "
                        f"-Root '{quoted(root)}' "
                        "-VerifiedDirs $cache"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            rejected = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    (
                        "$ErrorActionPreference='Stop'; "
                        f". '{quoted(common)}'; $cache=@{{}}; "
                        "Assert-PathWithinRoot "
                        f"-Path '{quoted(outside)}' "
                        f"-Root '{quoted(root)}' "
                        "-VerifiedDirs $cache"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "escaped allowlisted root",
                rejected.stdout + rejected.stderr,
            )

    def test_uploader_uses_generation_preconditions_and_remote_readback(self):
        uploader = (SCRIPTS / "upload_local_quant.ps1").read_text(
            encoding="utf-8"
        )
        common = (
            SCRIPTS / "observation_release_common.ps1"
        ).read_text(encoding="utf-8")
        source = uploader + "\n" + common

        for required in (
            "observation_release_common.ps1",
            "Invoke-GcloudConditionalCopy",
            "Assert-GcloudFileMatches",
            "--if-generation-match=",
            "before_generation",
            "after_generation",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for destination in (
            "quant/v1/latest-insights.json",
            "quant/v1/latest-$Market.json",
            "reports/v1/index-TW.json",
            "reports/v1/latest-TW.json",
            "reports/v2/index-TW.json",
            "reports/v2/$LatestName",
            "dashboard/v1/latest-TW.json",
        ):
            with self.subTest(destination=destination):
                self.assertIn(destination, uploader)
        self.assertNotIn(
            'Invoke-GcloudCopy $LatestPath "gs://$Bucket/dashboard/v1/latest-TW.json"',
            uploader,
        )
        self.assertIn("$Latest.schema_version -ne 2", uploader)
        self.assertIn(
            "$Latest.kind -ne 'absorb-observation-dashboard'",
            uploader,
        )
        self.assertIn("$Latest.product_mode -ne 'observation'", uploader)

    def test_lkg_capture_records_absent_or_hash_verified_previous_pointers(self):
        source = (SCRIPTS / "capture_observation_lkg.ps1").read_text(
            encoding="utf-8"
        )

        for required in (
            "quant/v1/latest-TW.json",
            "quant/v1/latest-insights.json",
            "dashboard/v1/latest-TW.json",
            "reports/v1/index-TW.json",
            "reports/v1/latest-TW.json",
            "reports/v2/index-TW.json",
            "reports/v2/latest-TW-post_close.json",
            "reports/v2/latest-TW-pre_market.json",
            "reports/v2/latest-TW-weekly_model.json",
            "exists = $false",
            "generation",
            "sha256",
            "observation-lkg",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn("storage', 'rm'", source)
        self.assertNotIn("--recursive", source)

    def test_rollback_only_restores_or_deletes_exact_applied_generations(self):
        rollback = (SCRIPTS / "rollback_observation.ps1").read_text(
            encoding="utf-8"
        )
        common = (
            SCRIPTS / "observation_release_common.ps1"
        ).read_text(encoding="utf-8")
        source = rollback + "\n" + common

        for required in (
            "SupportsShouldProcess",
            "applied_generation",
            "--if-generation-match=",
            "Invoke-GcloudConditionalDelete",
            "previous_sha256",
            "rollback verification failed",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn("--recursive", source)
        self.assertNotIn("objects/", rollback)

        object_state = common[
            common.index("function Get-GcloudObjectState"):
            common.index("function Invoke-GcloudConditionalCopy")
        ]
        self.assertIn("$PreviousWhatIfPreference = $WhatIfPreference", object_state)
        self.assertIn("$WhatIfPreference = $false", object_state)
        self.assertIn(
            "$WhatIfPreference = $PreviousWhatIfPreference",
            object_state,
        )


if __name__ == "__main__":
    unittest.main()
