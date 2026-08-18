import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class ObservationReleaseScriptTests(unittest.TestCase):
    @staticmethod
    def _windows_powershell(command, *, environment=None):
        return subprocess.run(
            [
                r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )

    @staticmethod
    def _ps_quote(path):
        return str(path).replace("'", "''")

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

    def test_pointer_snapshot_binds_upload_bytes_to_validation_hash(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "latest.json"
            staging = root / "staging"
            staging.mkdir()
            source.write_bytes(b"validated-bytes")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            powershell = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"

            def quoted(path):
                return str(path).replace("'", "''")

            success = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    (
                        "$ErrorActionPreference='Stop'; "
                        f". '{quoted(common)}'; "
                        "$snapshot=New-VerifiedPointerSnapshot "
                        f"-Source '{quoted(source)}' "
                        f"-StagingRunPath '{quoted(staging)}' "
                        f"-ExpectedSha256 '{expected}'; "
                        "Write-Output $snapshot"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            snapshot = Path(success.stdout.strip().splitlines()[-1])
            self.assertEqual(snapshot.read_bytes(), b"validated-bytes")
            for path in staging.rglob("*"):
                if path.is_file():
                    path.chmod(0o600)

            source.write_bytes(b"replaced-after-validation")
            failure = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    (
                        "$ErrorActionPreference='Stop'; "
                        f". '{quoted(common)}'; "
                        "New-VerifiedPointerSnapshot "
                        f"-Source '{quoted(source)}' "
                        f"-StagingRunPath '{quoted(staging)}' "
                        f"-ExpectedSha256 '{expected}'"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn(
                "changed during snapshot",
                failure.stdout + failure.stderr,
            )

    def test_pending_pointer_journal_remains_flat_after_three_appends(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "receipt.json.pending.json"
            command = (
                "$ErrorActionPreference='Stop'; "
                f". '{self._ps_quote(common)}'; "
                f"$path='{self._ps_quote(journal)}'; "
                "Write-GcloudPendingPointerJournal -Path $path -Entry "
                "([ordered]@{uri='gs://bucket/u1';source='s1';expected_generation='1';"
                "source_sha256=('a'*64);created_at='2026-08-17T01:00:00Z'}); "
                "Write-GcloudPendingPointerJournal -Path $path -Entry "
                "([ordered]@{uri='gs://bucket/u2';source='s2';expected_generation='2';"
                "source_sha256=('b'*64);created_at='2026-08-17T01:01:00Z'}); "
                "Write-GcloudPendingPointerJournal -Path $path -Entry "
                "([ordered]@{uri='gs://bucket/u3';source='s3';expected_generation='3';"
                "source_sha256=('c'*64);created_at='2026-08-17T01:02:00Z'})"
            )
            completed = self._windows_powershell(command)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            entries = json.loads(journal.read_text(encoding="utf-8-sig"))
            self.assertEqual(
                [entry["uri"] for entry in entries],
                ["gs://bucket/u1", "gs://bucket/u2", "gs://bucket/u3"],
            )

    def test_pending_pointer_journal_reader_flattens_legacy_ps51_shape(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "receipt.json.pending.json"
            journal.write_text(
                json.dumps(
                    [
                        {
                            "value": [
                                {
                                    "uri": "gs://bucket/u1",
                                    "source": "s1",
                                    "expected_generation": "1",
                                    "source_sha256": "a" * 64,
                                    "created_at": "2026-08-17T01:00:00Z",
                                },
                                {
                                    "uri": "gs://bucket/u2",
                                    "source": "s2",
                                    "expected_generation": "2",
                                    "source_sha256": "b" * 64,
                                    "created_at": "2026-08-17T01:01:00Z",
                                },
                            ],
                            "Count": 2,
                        },
                        {
                            "uri": "gs://bucket/u3",
                            "source": "s3",
                            "expected_generation": "3",
                            "source_sha256": "c" * 64,
                            "created_at": "2026-08-17T01:02:00Z",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            command = (
                "$ErrorActionPreference='Stop'; "
                f". '{self._ps_quote(common)}'; "
                f"$entries=@(Read-GcloudPendingPointerJournal -Path "
                f"'{self._ps_quote(journal)}'); "
                "if (($entries.uri -join '|') -ne "
                "'gs://bucket/u1|gs://bucket/u2|gs://bucket/u3') { exit 41 }"
            )
            completed = self._windows_powershell(command)
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )

    def test_gcloud_json_readback_uses_strict_utf8_file_transport(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "remote.json"
            source.write_text(
                json.dumps(
                    {
                        "title": "ABSORB 盤前風險更新",
                        "summary": ["資料不足，維持盤後觀察"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            fake_gcloud = root / "gcloud.cmd"
            fake_gcloud.write_text(
                "@echo off\n"
                "if /I not \"%1\"==\"storage\" exit /b 21\n"
                "if /I not \"%2\"==\"cp\" exit /b 22\n"
                "copy /b \"%ABSORB_FAKE_GCLOUD_SOURCE%\" \"%5\" >nul\n"
                "exit /b %errorlevel%\n",
                encoding="ascii",
            )
            environment = os.environ.copy()
            environment["ABSORB_FAKE_GCLOUD_SOURCE"] = str(source)
            command = (
                "$ErrorActionPreference='Stop'; "
                f". '{self._ps_quote(common)}'; "
                f"$document=Get-GcloudJson -Gcloud "
                f"'{self._ps_quote(fake_gcloud)}' -Uri 'gs://test/remote.json'; "
                "if ($document.title -ne 'ABSORB 盤前風險更新') { exit 42 }; "
                "if ($document.summary[0] -ne '資料不足，維持盤後觀察') { exit 43 }"
            )
            completed = self._windows_powershell(
                command,
                environment=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )

    def test_successor_receipt_evidence_binds_generation_and_sha(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            receipt_root = Path(temporary) / "observation-lkg"
            capture_root = receipt_root / "successor"
            capture_root.mkdir(parents=True)
            previous = capture_root / "reports-v2-index.json"
            previous.write_bytes(b'{"generation":"successor"}')
            digest = hashlib.sha256(previous.read_bytes()).hexdigest()
            successor = capture_root / "receipt.json"
            successor.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "absorb-observation-lkg",
                        "bucket": "line-stock-bot-498908-quant-snapshots",
                        "capture_id": "successor",
                        "captured_at": "2026-08-18T00:00:00Z",
                        "pointers": [
                            {
                                "name": "reports-v2-index",
                                "uri": "gs://bucket/index.json",
                                "exists": True,
                                "generation": "20",
                                "previous_file": previous.name,
                                "previous_sha256": digest,
                                "previous_size": previous.stat().st_size,
                                "applied_generation": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            command = (
                "$ErrorActionPreference='Stop'; "
                f". '{self._ps_quote(common)}'; "
                "$entry=[pscustomobject]@{uri='gs://bucket/index.json';"
                f"source='{self._ps_quote(previous)}';"
                f"source_sha256='{digest}';expected_generation='10';"
                "created_at='2026-08-17T23:00:00Z'}; "
                "$evidence=Get-VerifiedSuccessorPointerEvidence "
                "-Entry $entry "
                f"-SuccessorReceiptPaths @('{self._ps_quote(successor)}') "
                f"-ReceiptRoot '{self._ps_quote(receipt_root)}' "
                "-Bucket 'line-stock-bot-498908-quant-snapshots'; "
                "if ($evidence.generation -ne '20') { exit 44 }; "
                f"if ($evidence.sha256 -ne '{digest}') {{ exit 45 }}"
            )
            completed = self._windows_powershell(command)
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )

    def test_receipt_pointer_capture_validation_rejects_tampering(self):
        common = SCRIPTS / "observation_release_common.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            capture_root = Path(temporary)
            previous = capture_root / "reports-v2-index.json"
            previous.write_bytes(b'{"captured":true}')
            digest = hashlib.sha256(previous.read_bytes()).hexdigest()
            command = (
                "$ErrorActionPreference='Stop'; "
                f". '{self._ps_quote(common)}'; "
                "$pointer=[pscustomobject]@{"
                "uri='gs://bucket/index.json';exists=$true;generation='10';"
                f"previous_file='{previous.name}';previous_sha256='{digest}';"
                f"previous_size={previous.stat().st_size};applied_generation=$null}}; "
                "Assert-ObservationLkgPointerCapture -Pointer $pointer "
                f"-CaptureRoot '{self._ps_quote(capture_root)}' "
                "-Bucket 'bucket'"
            )
            valid = self._windows_powershell(command)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)

            previous.write_bytes(b'{"captured":false}')
            tampered = self._windows_powershell(command)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn(
                "capture hash mismatch",
                (tampered.stdout + tampered.stderr).lower(),
            )

        reconciler = (SCRIPTS / "reconcile_observation_lkg.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Assert-ObservationLkgPointerCapture", reconciler)

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
