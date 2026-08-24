import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.report_fixtures import write_quant_publish_v3


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_local_quant_task.ps1"
WRAPPER = ROOT / "scripts" / "run_local_quant_task.ps1"
UPLOADER = ROOT / "scripts" / "upload_local_quant.ps1"
COMMON = ROOT / "scripts" / "observation_release_common.ps1"
LIFECYCLE = ROOT / "config" / "quant-snapshot-lifecycle.json"
DOCKERIGNORE = ROOT / ".dockerignore"
GCLOUDIGNORE = ROOT / ".gcloudignore"


class LocalQuantTaskTests(unittest.TestCase):
    def test_uploader_is_allowlisted_atomic_and_non_destructive(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            r"D:\StockPapiData",
            "line-stock-bot-498908-quant-snapshots",
            "Assert-AllowlistedPath",
            "Get-FileHash",
            "objects/[0-9a-f]{64}",
            "manifests/",
            '"latest-$Market.json"',
            'latest-insights.json',
            'market-insights',
            "gcloud",
            "storage",
            "cp",
            "--no-clobber",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "Remove-Item",
            "storage rsync",
            "--recursive",
            "service-account.json",
            "FINMIND_PASSWORD",
            "U72f8c70881c4107fd03e506e97d3b75d",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_uploader_batches_content_addressed_objects(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertIn("$ObjectBatchSize = 100", source)
        self.assertIn("Invoke-GcloudCopyBatch", source)
        self.assertIn("$ValidatedObjectPaths", source)
        self.assertNotIn(
            'Invoke-GcloudCopy $ObjectPath "gs://$Bucket/quant/v1/$ObjectRelative"',
            source,
        )
        self.assertLess(source.index("# Upload objects"), source.index("# Upload manifest"))
        self.assertLess(
            source.index("# Upload manifest"), source.index("# Upload latest pointer")
        )

    def test_uploader_has_fail_closed_manifest_v3_preflight(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            "expected_non_price_symbols",
            "operational_failed_symbols",
            "regular_price_denominator",
            "regular_price_coverage",
            "observation_coverage",
            "trading_status_evidence",
            "evidence_sha256",
            "artifact_sha256",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertLess(
            source.index("expected_non_price_symbols"),
            source.index("# Upload objects"),
        )

    def _run_uploader_preflight(self, root: Path):
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        call_log = root / "gcloud-called.txt"
        (fake_bin / "gcloud.cmd").write_text(
            f'@echo called>>"{call_log}"\r\n@exit /b 99\r\n',
            encoding="ascii",
        )
        quoted_data_root = str(root / "AbsorbData").replace("'", "''")
        quoted_bin = str(fake_bin).replace("'", "''")
        quoted_script = str(UPLOADER).replace("'", "''")
        command = (
            "$ErrorActionPreference='Stop';"
            "Import-Module Microsoft.PowerShell.Utility;"
            f"$env:Path='{quoted_bin};'+$env:Path;"
            "try {"
            f"& '{quoted_script}' -PreflightDataRoot '{quoted_data_root}'"
            "} catch {"
            "Write-Error ($_.Exception.Message + [Environment]::NewLine + $_.ScriptStackTrace);"
            "exit 1}"
        )
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return result, call_log

    def test_uploader_rejects_unknown_or_mixed_schema_without_gcloud_copy(self):
        cases = ((9, 3, "Invalid latest pointer"), (2, 3, "Invalid manifest"))
        for pointer_schema, manifest_schema, expected_error in cases:
            with self.subTest(pointer_schema=pointer_schema), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / "AbsorbData"
                publish = write_quant_publish_v3(data_root)
                latest_path = publish / "latest-TW.json"
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                latest["schema_version"] = pointer_schema
                if manifest_schema != 3:
                    raise AssertionError("test fixture schema is invalid")
                latest_path.write_text(json.dumps(latest), encoding="utf-8")

                result, call_log = self._run_uploader_preflight(root)

                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(call_log.exists(), result.stdout + result.stderr)
                self.assertIn(expected_error, result.stdout + result.stderr)

    def test_uploader_validates_v3_partition_and_status_hashes_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "AbsorbData"
            publish = write_quant_publish_v3(data_root)
            latest_path = publish / "latest-TW.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            manifest["expected_non_price_symbols"]["2303"][
                "evidence_sha256"
            ] = "0" * 64
            encoded = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            relative = latest["manifest"].rsplit("-", 1)[0] + f"-{digest[:12]}.json"
            (publish / relative).write_bytes(encoded)
            latest.update(manifest=relative, manifest_sha256=digest)
            latest_path.write_text(json.dumps(latest), encoding="utf-8")

            result, call_log = self._run_uploader_preflight(root)

            self.assertNotEqual(result.returncode, 0)
            import re
            cleaned_output = re.sub(r"\s+", "", (result.stdout + result.stderr).lower())
            self.assertIn("statusobjectevidencemismatch", cleaned_output)

    def test_uploader_rejects_object_path_not_bound_to_declared_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "AbsorbData"
            publish = write_quant_publish_v3(data_root)
            latest_path = publish / "latest-TW.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            entry = manifest["symbols"]["2330"]
            mismatched = f"objects/{'f' * 64}.json.gz"
            (publish / mismatched).write_bytes((publish / entry["path"]).read_bytes())
            entry["path"] = mismatched
            encoded = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            relative = latest["manifest"].rsplit("-", 1)[0] + f"-{digest[:12]}.json"
            (publish / relative).write_bytes(encoded)
            latest.update(manifest=relative, manifest_sha256=digest)
            latest_path.write_text(json.dumps(latest), encoding="utf-8")

            result, call_log = self._run_uploader_preflight(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(call_log.exists(), result.stdout + result.stderr)
            cleaned_output = re.sub(
                r"\s+", "", (result.stdout + result.stderr).lower()
            )
            self.assertIn("objectpathhashmismatch", cleaned_output)

    def test_uploader_valid_v3_passes_without_gcloud_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_quant_publish_v3(root / "AbsorbData")

            result, call_log = self._run_uploader_preflight(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(call_log.exists(), result.stdout + result.stderr)

    def test_uploader_binds_pointer_updates_to_captured_generations(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            "$ExpectedPointerGenerations",
            "-ExpectedGeneration $ExpectedGeneration",
            "Observation-only pointer destination is not allowlisted",
            "if ($Matches.Count -eq 0) { throw",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_uploader_verifies_quant_objects_after_no_clobber_upload(self):
        source = UPLOADER.read_text(encoding="utf-8")
        object_upload = source.index("# Upload objects")
        manifest_upload = source.index("# Upload manifest")
        section = source[object_upload:manifest_upload]

        self.assertIn("Assert-GcloudFileMatches", section)
        self.assertIn('"gs://$Bucket/quant/v1/$ObjectRelative"', section)

    def test_uploader_verifies_report_research_objects_before_mutable_pointers(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            "Publish-VerifiedReportObject",
            "objects/canonical/",
            "objects/regression/",
            "Report object schema or lineage mismatch",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertLess(
            source.index("Publish-VerifiedReportObject"),
            source.index("# Upload manifest"),
        )

    def test_observation_upload_preserves_captured_report_index_entries(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertIn("Assert-ObservationReportIndexPreservesLkg", source)
        self.assertIn("would clobber captured entry", source)
        self.assertIn("previous_file", source)

    def test_mutable_pointer_readback_is_generation_specific_and_journaled(self):
        common = COMMON.read_text(encoding="utf-8")
        uploader = UPLOADER.read_text(encoding="utf-8")

        for required in (
            "Write-GcloudPendingPointerJournal",
            "Conditional GCS pointer update could not be verified",
            '"$Destination#$([string]$After.generation)"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, common)
        for required in (
            "$PendingPointerJournalPath",
            "pending.json",
            "Observation LKG pending pointer journal exists",
        ):
            with self.subTest(required=required):
                self.assertIn(required, uploader)

    def test_mutable_pointer_upload_uses_verified_staging_snapshot(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            "New-VerifiedPointerSnapshot",
            "$Global:PointerStagingRunPath",
            "-ExpectedSha256 $ExpectedSha256",
            "-Source $Snapshot",
            "pointer-staging",
            "pointer-update.lock",
            "[IO.FileShare]::None",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_uploader_uploads_v3_objects_and_manifest_before_pointer(self):
        source = UPLOADER.read_text(encoding="utf-8")
        self.assertLess(source.index("Read-VerifiedGzipJson"), source.index("# Upload objects"))
        self.assertLess(source.index("# Upload objects"), source.index("# Upload manifest"))
        self.assertLess(source.index("# Upload manifest"), source.index("# Upload latest pointer"))

    def test_uploader_sends_market_insights_before_large_market_snapshots(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertLess(source.index("$InsightsUploaded"), source.index("$UploadedMarkets"))

    def test_uploader_validates_and_uploads_report_latest_last_without_blocking_quant(self):
        source = UPLOADER.read_text(encoding="utf-8")

        for required in (
            r"publish\reports\v1",
            "ReportUploadError",
            "metadata/[0-9a-f]{64}",
            "objects/[0-9a-f]{64}",
            "index-TW.json",
            "reports/v1/",
            "日報上傳失敗",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertLess(
            source.index('"gs://$Bucket/reports/v1/$ReportPdfRelative"'),
            source.index('"gs://$Bucket/reports/v1/index-TW.json"'),
        )
        self.assertLess(
            source.index('"gs://$Bucket/reports/v1/index-TW.json"'),
            source.index('"gs://$Bucket/reports/v1/latest-TW.json"'),
        )

    def test_uploader_validates_v2_and_uploads_immutable_content_before_pointers(self):
        source = UPLOADER.read_text(encoding="utf-8")
        start = source.index("function Publish-ReportsV2")
        section = source[start : source.index("\ntry {", start)]

        for required in (
            r"publish\reports\v2",
            "stock-papi-report-index",
            "stock-papi-report",
            "source_manifest_sha256",
            "Report v2 content hash mismatch",
            "Pre-market report v2 must not contain PDF",
            "--no-clobber",
            "Invoke-GcloudConditionalCopy",
            "Assert-GcloudFileMatches",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertLess(
            section.index('"gs://$Bucket/reports/v2/$MetadataRelative"'),
            section.index('"gs://$Bucket/reports/v2/index-TW.json"'),
        )
        self.assertLess(
            section.index('"gs://$Bucket/reports/v2/index-TW.json"'),
            section.index('"gs://$Bucket/reports/v2/$LatestName"'),
        )

    def test_observation_report_latest_allows_omitted_model_versions(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertIn(
            "$null -ne $Latest.model_versions -and",
            source,
        )

    def test_uploader_serializes_generic_pointer_updates_via_to_array(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertIn(
            "pointer_updates = $Global:PointerUpdates.ToArray()",
            source,
        )
        self.assertNotIn(
            "pointer_updates = @($Global:PointerUpdates)",
            source,
        )

    def test_observation_uploader_auto_captures_rollback_receipt(self):
        source = UPLOADER.read_text(encoding="utf-8")

        self.assertIn(
            "if ($ObservationOnly -and -not $LkgReceiptPath)",
            source,
        )
        self.assertIn("capture_observation_lkg.ps1", source)
        self.assertIn("lkg_receipt = $LkgReceiptPath", source)
        self.assertLess(
            source.index("capture_observation_lkg.ps1"),
            source.index("# Upload objects"),
        )

    def test_lifecycle_deletes_cloud_objects_after_thirty_days(self):
        source = LIFECYCLE.read_text(encoding="utf-8")

        self.assertIn('"type": "Delete"', source)
        self.assertIn('"age": 30', source)

    def test_installer_registers_separate_0935_upload_task(self):
        source = INSTALLER.read_text(encoding="utf-8")

        for required in (
            "StockPapi-QuantUpload",
            "upload_local_quant.ps1",
            "09:35",
            "New-TimeSpan -Hours 1",
            "RunLevel Limited",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_cloud_build_excludes_local_and_untracked_artifacts(self):
        for path in (DOCKERIGNORE, GCLOUDIGNORE):
            source = path.read_text(encoding="utf-8")
            for required in (
                "0.26.0",
                "deliverables/",
                "scripts/build_competition_doc.py",
                ".deps/",
                ".env",
            ):
                with self.subTest(path=path.name, required=required):
                    self.assertIn(required, source)

    def test_installer_enforces_d_drive_schedule_and_resource_limits(self):
        source = INSTALLER.read_text(encoding="utf-8")

        for required in (
            r"D:\StockPapiData",
            "NTFS",
            "$MinimumFreeGB = 100",
            "$MinimumFreeGB * 1GB",
            "02:30",
            "ExecutionTimeLimit",
            "New-TimeSpan -Hours 7",
            "MultipleInstances IgnoreNew",
            "Priority 7",
            "StartWhenAvailable",
            "LogonType Interactive",
            "--init",
            "--dry-run",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_installer_contains_no_market_secret_or_service_account_file(self):
        source = INSTALLER.read_text(encoding="utf-8").lower()

        for forbidden in (
            "alpaca_api_secret_key",
            "finmind_password",
            "service-account.json",
            "0.26.0",
            "interactivetoken",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_installer_prefers_existing_bundled_python_over_windows_alias(self):
        source = INSTALLER.read_text(encoding="utf-8")

        self.assertIn(
            "$PythonExe = if (Test-Path $BundledPython) { $BundledPython } "
            "elseif ($PythonCommand)",
            source,
        )

    def test_installer_does_not_rewrite_an_already_private_acl(self):
        source = INSTALLER.read_text(encoding="utf-8")

        self.assertIn("$AclIsPrivate", source)
        self.assertIn("if (-not $AclIsPrivate)", source)

    def test_wrapper_moves_runtime_caches_to_d_drive_and_runs_market_batch(self):
        source = WRAPPER.read_text(encoding="utf-8")

        for required in (
            r"D:\StockPapiData",
            "$env:TEMP",
            "$env:TMP",
            "$env:XDG_CACHE_HOME",
            "$env:HF_HOME",
            "$env:PYTHONPYCACHEPREFIX",
            "$env:PYTHONPATH",
            "--run",
            "--insights",
            "--market",
            "--limit",
            "5000",
            "--delay",
            "0.5",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

        self.assertNotIn("GEMINI_API_KEY", source)
        self.assertNotIn("FINMIND_PASSWORD", source)
        self.assertNotIn("--limit 200", source)
        self.assertNotIn("--market ALL", source)
        self.assertEqual(source.count("--limit 5000"), 2)
        self.assertLess(source.index("--insights"), source.index("--market TW"))
        self.assertLess(source.index("--market TW"), source.index("Start-Sleep"))
        self.assertLess(source.index("Start-Sleep"), source.index("--market US"))

    def test_wrapper_generates_report_only_after_a_new_tw_manifest_and_continues_on_failure(self):
        source = WRAPPER.read_text(encoding="utf-8")

        self.assertIn("$TwLatestBefore", source)
        self.assertIn("$TwLatestAfter", source)
        self.assertIn("-m reporting.cli", source)
        self.assertIn("日報生成失敗", source)
        self.assertLess(source.index("--market TW"), source.index("-m reporting.cli"))
        self.assertLess(source.index("-m reporting.cli"), source.index("--market US"))
        self.assertNotIn("exit $ReportExitCode", source)

    def test_installer_schedules_wrapper_instead_of_embedding_market_arguments(self):
        source = INSTALLER.read_text(encoding="utf-8")

        self.assertIn("$Wrapper", source)
        self.assertIn("-File", source)
        self.assertIn("run_local_quant_task.ps1", source)


if __name__ == "__main__":
    unittest.main()
