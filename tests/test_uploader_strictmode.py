import datetime
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
UPLOADER = SCRIPTS / "upload_local_quant.ps1"
PRE_MARKET_GUARD = SCRIPTS / "pre_market_pipeline_guard.ps1"
POST_CLOSE_GUARD = SCRIPTS / "post_close_pipeline_guard.ps1"

TARGET = "2026-08-24"
REGULAR_COUNT = 19
STATUS_SYMBOL = "2303"
UNAVAILABLE_SYMBOL = "6001"


def _regular_document(symbol):
    return {
        "schema_version": 2,
        "market": "TW",
        "symbol": symbol,
        "name": f"測試股票 {symbol}",
        "as_of": TARGET,
        "target_market_date": TARGET,
        "observation_as_of": TARGET,
        "latest_regular_price_date": TARGET,
        "observation_kind": "regular_price",
        "model_version": "observation-source-v1",
        "latest": {"Date": TARGET + "T00:00:00.000", "Close": 100.0},
        "backtest": {},
        "daily": [{"Date": TARGET + "T00:00:00.000", "Close": 100.0}],
    }


def _status_document(symbol, evidence=None):
    document = {
        "schema_version": 2,
        "market": "TW",
        "symbol": symbol,
        "name": f"測試股票 {symbol}",
        "as_of": "2026-08-19",
        "target_market_date": TARGET,
        "observation_as_of": TARGET,
        "latest_regular_price_date": "2026-08-19",
        "observation_kind": "official_no_regular_trade",
        "model_version": "observation-source-v1",
        "latest": {"Date": "2026-08-19T00:00:00.000", "Close": 9.7},
        "backtest": {},
        "daily": [{"Date": "2026-08-19T00:00:00.000", "Close": 9.7}],
    }
    if evidence is not None:
        document["trading_status_evidence"] = evidence
    return document


def _valid_evidence(symbol):
    return {
        "schema_version": 1,
        "status": "official_no_regular_trade",
        "market": "TW",
        "symbol": symbol,
        "target_market_date": TARGET,
        "evidence_sha256": "b" * 64,
        "raw_row_sha256": "c" * 64,
        "parser_version": "tw-lifecycle-parser-v2",
    }


def _write_object(publish, document):
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    compressed = gzip.compress(encoded, mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    (publish / "objects" / f"{digest}.json.gz").write_bytes(compressed)
    entry = {
        "path": f"objects/{digest}.json.gz",
        "sha256": digest,
        "size": len(compressed),
        "uncompressed_size": len(encoded),
        "as_of": document["as_of"],
        "observation_as_of": document["observation_as_of"],
        "latest_regular_price_date": document["latest_regular_price_date"],
        "observation_kind": document["observation_kind"],
        "model_version": document["model_version"],
    }
    if document["observation_kind"] != "regular_price":
        entry["evidence_sha256"] = document["trading_status_evidence"][
            "evidence_sha256"
        ]
    return digest, entry


def write_production_shaped_v4_publish(data_root, *, regular_without_evidence=True):
    publish = Path(data_root) / "publish" / "quant" / "v1"
    (publish / "objects").mkdir(parents=True, exist_ok=True)
    (publish / "manifests").mkdir(parents=True, exist_ok=True)
    regular_symbols = [f"{3000 + index:04d}" for index in range(REGULAR_COUNT)]
    entries = {}
    expected = {}
    for symbol in regular_symbols:
        document = _regular_document(symbol)
        if not regular_without_evidence:
            document["trading_status_evidence"] = None
        digest, entry = _write_object(publish, document)
        entries[symbol] = entry
    status_document = _status_document(STATUS_SYMBOL, _valid_evidence(STATUS_SYMBOL))
    status_digest, status_entry = _write_object(publish, status_document)
    entries[STATUS_SYMBOL] = status_entry
    expected[STATUS_SYMBOL] = {
        "status": "official_no_regular_trade",
        "evidence_sha256": "b" * 64,
        "artifact_sha256": status_digest,
        "latest_regular_price_date": "2026-08-19",
    }
    observation = REGULAR_COUNT + 1
    active = observation + 1
    manifest = {
        "schema_version": 4,
        "market": "TW",
        "generated_at": "2026-08-24T18:03:40Z",
        "target_market_date": TARGET,
        "observation_as_of": TARGET,
        "active_universe_count": active,
        "observation_count": observation,
        "regular_price_symbol_count": REGULAR_COUNT,
        "verified_non_price_symbol_count": 1,
        "unavailable_count": 1,
        "unavailable_symbols": [UNAVAILABLE_SYMBOL],
        "operational_failure_count": 0,
        "operational_failed_symbols": [],
        "operational_failure_rate": 0.0,
        "observation_coverage": observation / active,
        "regular_price_denominator": REGULAR_COUNT,
        "regular_price_coverage": 1.0,
        "expected_non_price_symbols": expected,
        "symbols": entries,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    relative = f"manifests/TW-20260824T180340Z-{digest[:12]}.json"
    (publish / relative).write_bytes(manifest_bytes)
    (publish / "latest-TW.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "market": "TW",
                "generated_at": "2026-08-24T18:03:40Z",
                "manifest": relative,
                "manifest_sha256": digest,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return publish


def run_uploader_preflight(root, *, dot_source_guard=None, extra_args=()):
    fake_bin = Path(root) / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    call_log = Path(root) / "gcloud-called.txt"
    (fake_bin / "gcloud.cmd").write_text(
        f'@echo called>>"{call_log}"\r\n@exit /b 99\r\n', encoding="ascii"
    )
    data_root = Path(root) / "AbsorbData"
    quoted_data_root = str(data_root).replace("'", "''")
    quoted_bin = str(fake_bin).replace("'", "''")
    quoted_script = str(UPLOADER).replace("'", "''")
    parts = [
        "$ErrorActionPreference='Stop';",
        "Import-Module Microsoft.PowerShell.Utility;",
        f"$env:Path='{quoted_bin};'+$env:Path;",
    ]
    if dot_source_guard is not None:
        quoted_guard = str(dot_source_guard).replace("'", "''")
        parts.append(f". '{quoted_guard}';")
    argument_text = " ".join(extra_args)
    parts.append("try {")
    parts.append(
        f"& '{quoted_script}' -PreflightDataRoot '{quoted_data_root}' "
        f"{argument_text}"
    )
    parts.append("} catch {")
    parts.append(
        "Write-Error ($_.Exception.Message + [Environment]::NewLine + $_.ScriptStackTrace);"
    )
    parts.append("exit 1}")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "".join(parts),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return completed, call_log


class StrictModeLeakTests(unittest.TestCase):
    def test_dot_sourcing_premarket_guard_preserves_caller_strictness(self):
        for guard in (PRE_MARKET_GUARD, POST_CLOSE_GUARD):
            with self.subTest(guard=guard.name):
                quoted_guard = str(guard).replace("'", "''")
                harness = (
                    "$ErrorActionPreference='Stop';"
                    f". '{quoted_guard}';"
                    "$document = '{\"a\":1}' | ConvertFrom-Json;"
                    "try { $value = $document.missing; "
                    "'no-throw' } catch { 'throw' }"
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
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                self.assertEqual(
                    completed.returncode, 0,
                    completed.stdout + completed.stderr,
                )
                self.assertIn(
                    "no-throw", completed.stdout,
                    "dot-sourcing the guard leaked StrictMode into the caller",
                )

    def test_premarket_completion_helper_still_operates(self):
        quoted_guard = str(PRE_MARKET_GUARD).replace("'", "''")
        with tempfile.TemporaryDirectory() as temporary:
            quoted_root = temporary.replace("'", "''")
            harness = (
                "$ErrorActionPreference='Stop';"
                f". '{quoted_guard}';"
                f"if (Test-PreMarketCompletion -DataRoot '{quoted_root}' "
                "-TargetDate '2026-08-25') { 'completed' } else { 'pending' }"
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
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("pending", completed.stdout)


class UploaderStrictModeTests(unittest.TestCase):
    def test_premarket_guard_dot_source_reproduces_uploader_strictmode_failure(self):
        """Reproduces the 07:30-08:50 production failure before the fix."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            completed, call_log = run_uploader_preflight(
                root, dot_source_guard=PRE_MARKET_GUARD
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(call_log.exists(), "preflight must never call gcloud")
        self.assertIn("Validated quant snapshots: TW", completed.stdout)

    def test_uploader_regular_price_without_evidence_passes_without_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            completed, call_log = run_uploader_preflight(root)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(call_log.exists())
        self.assertIn("Validated quant snapshots: TW", completed.stdout)

    def test_status_document_without_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            entry = manifest["symbols"][STATUS_SYMBOL]
            document = {
                key: value
                for key, value in _status_document(STATUS_SYMBOL).items()
                if key != "trading_status_evidence"
            }
            encoded = json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            compressed = gzip.compress(encoded, mtime=0)
            digest = hashlib.sha256(compressed).hexdigest()
            (publish / "objects" / f"{digest}.json.gz").write_bytes(compressed)
            entry.update(
                path=f"objects/{digest}.json.gz",
                sha256=digest,
                size=len(compressed),
                uncompressed_size=len(encoded),
            )
            manifest["expected_non_price_symbols"][STATUS_SYMBOL][
                "artifact_sha256"
            ] = digest
            self._rewrite_manifest(publish, latest, manifest)

            completed, call_log = run_uploader_preflight(root)

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(call_log.exists())
        self.assertIn(
            "statusobjectevidencemismatch",
            re.sub(r"\s+", "", (completed.stdout + completed.stderr).lower()),
        )

    def test_regular_document_with_unexpected_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            symbol = f"{3000:04d}"
            entry = manifest["symbols"][symbol]
            document = _regular_document(symbol)
            document["trading_status_evidence"] = _valid_evidence(symbol)
            encoded = json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            compressed = gzip.compress(encoded, mtime=0)
            digest = hashlib.sha256(compressed).hexdigest()
            (publish / "objects" / f"{digest}.json.gz").write_bytes(compressed)
            entry.update(
                path=f"objects/{digest}.json.gz",
                sha256=digest,
                size=len(compressed),
                uncompressed_size=len(encoded),
            )
            self._rewrite_manifest(publish, latest, manifest)

            completed, call_log = run_uploader_preflight(root)

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(call_log.exists())
        self.assertIn(
            "regularpriceobjectv3mismatch",
            re.sub(r"\s+", "", (completed.stdout + completed.stderr).lower()),
        )

    def test_evidence_sha256_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            manifest["expected_non_price_symbols"][STATUS_SYMBOL][
                "evidence_sha256"
            ] = "d" * 64
            self._rewrite_manifest(publish, latest, manifest)

            completed, call_log = run_uploader_preflight(root)

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(call_log.exists())
        self.assertIn(
            "statusobjectevidencemismatch",
            re.sub(r"\s+", "", (completed.stdout + completed.stderr).lower()),
        )

    def test_artifact_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            symbol = f"{3000:04d}"
            entry = manifest["symbols"][symbol]
            path = publish / "objects" / f"{entry['sha256']}.json.gz"
            path.write_bytes(b"0" * entry["size"])
            self._rewrite_manifest(publish, latest, manifest)

            completed, call_log = run_uploader_preflight(root)

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(call_log.exists())
        self.assertIn(
            "objecthashmismatch",
            re.sub(r"\s+", "", (completed.stdout + completed.stderr).lower()),
        )

    def test_malformed_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_production_shaped_v4_publish(
                root / "AbsorbData", regular_without_evidence=True
            )
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            entry = manifest["symbols"][STATUS_SYMBOL]
            evidence = _valid_evidence(STATUS_SYMBOL)
            evidence.pop("status")
            document = _status_document(STATUS_SYMBOL, evidence)
            encoded = json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            compressed = gzip.compress(encoded, mtime=0)
            digest = hashlib.sha256(compressed).hexdigest()
            (publish / "objects" / f"{digest}.json.gz").write_bytes(compressed)
            entry.update(
                path=f"objects/{digest}.json.gz",
                sha256=digest,
                size=len(compressed),
                uncompressed_size=len(encoded),
            )
            manifest["expected_non_price_symbols"][STATUS_SYMBOL][
                "artifact_sha256"
            ] = digest
            self._rewrite_manifest(publish, latest, manifest)

            completed, call_log = run_uploader_preflight(root)

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(call_log.exists())

    @staticmethod
    def _rewrite_manifest(publish, latest, manifest):
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        relative = (
            latest["manifest"].rsplit("-", 1)[0] + f"-{digest[:12]}.json"
        )
        (publish / relative).write_bytes(manifest_bytes)
        latest.update(manifest=relative, manifest_sha256=digest)
        (publish / "latest-TW.json").write_text(
            json.dumps(latest, separators=(",", ":")), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
