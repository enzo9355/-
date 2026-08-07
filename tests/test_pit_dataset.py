from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.research.pit_dataset import (
    PIT_REQUIREMENTS,
    audit_pit_availability,
    build_price_research_dataset,
    write_pit_audit,
)
from tests.report_fixtures import warmup_stock_document


def canonical(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def gzip_bytes(content):
    import io

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as stream:
        stream.write(content)
    return output.getvalue()


def build_quant_fixture(root):
    publish = root / "publish" / "quant" / "v1"
    symbols = {}
    start = datetime.date(2025, 1, 1)
    for offset, symbol in enumerate(("1111", "2222")):
        daily = []
        for index in range(60):
            close = 50.0 + offset * 10 + index * (0.2 + offset * 0.05)
            daily.append(
                {
                    "Date": (
                        start + datetime.timedelta(days=index)
                    ).isoformat(),
                    "Close": close,
                    "YF_CLOSE": close,
                    "Volume": 1000 + index * 10,
                    "AI_P": 99.9,
                }
            )
        document = {
            "schema_version": 1,
            "market": "TW",
            "symbol": symbol,
            "as_of": daily[-1]["Date"],
            "daily": daily,
        }
        raw = canonical(document)
        compressed = gzip_bytes(raw)
        digest = hashlib.sha256(compressed).hexdigest()
        path = publish / "objects" / f"{digest}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
        symbols[symbol] = {
            "as_of": daily[-1]["Date"],
            "path": f"objects/{digest}.json.gz",
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": len(raw),
            "model_version": "ignored-existing-model",
        }

    manifest = {
        "schema_version": 2,
        "market": "TW",
        "generated_at": "2025-03-01T09:00:00Z",
        "market_as_of": "2025-03-01",
        "universe_count": 2,
        "symbol_count": 2,
        "failure_count": 0,
        "coverage": 1.0,
        "failure_rate": 0.0,
        "failed_symbols": [],
        "symbols": symbols,
    }
    manifest_bytes = canonical(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    relative = "manifests/TW-20250301T090000Z-" + manifest_hash[:12] + ".json"
    manifest_path = publish / relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    latest = {
        "schema_version": 2,
        "market": "TW",
        "generated_at": "2025-03-01T09:00:00Z",
        "manifest": relative,
        "manifest_sha256": manifest_hash,
    }
    (publish / "latest-TW.json").write_bytes(canonical(latest))
    return relative, manifest_hash


class PitDatasetTests(unittest.TestCase):
    def test_pit_history_uses_ohlcv_when_indicators_are_null(self):
        from stock_papi.research.pit_dataset import _history_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = warmup_stock_document("2330")
            raw = canonical(document)
            compressed = gzip_bytes(raw)
            path = root / "object.json.gz"
            path.write_bytes(compressed)
            rows = _history_rows(
                {"quant_root": root, "manifest": {"market": "TW"}}, "2330",
                {"path": path.name, "size": len(compressed), "sha256": hashlib.sha256(compressed).hexdigest(),
                 "uncompressed_size": len(raw), "as_of": document["as_of"]},
            )
        self.assertEqual(len(rows), 70)
        self.assertEqual(rows[0][0], document["daily"][0]["Date"][:10])
    def test_audit_records_every_requirement_with_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative, manifest_hash = build_quant_fixture(root)

            audit = audit_pit_availability(
                root,
                market="TW",
                now=datetime.datetime(
                    2025, 3, 2, tzinfo=datetime.timezone.utc
                ),
                code_sha="a" * 40,
            )

            self.assertEqual(set(audit["requirements"]), set(PIT_REQUIREMENTS))
            for requirement, result in audit["requirements"].items():
                with self.subTest(requirement=requirement):
                    self.assertIn(result["status"], {"available", "unavailable"})
                    self.assertIsInstance(result["evidence"], dict)
                    self.assertTrue(result["evidence"])
            self.assertEqual(
                audit["requirements"]["manifest_path_sha256"]["status"],
                "available",
            )
            self.assertEqual(
                audit["requirements"]["source_timestamp_revision"]["status"],
                "available",
            )
            self.assertEqual(
                audit["requirements"]["adjusted_price_history"]["status"],
                "available",
            )
            self.assertEqual(
                audit["requirements"]["historical_industry_membership"]["status"],
                "unavailable",
            )
            self.assertEqual(audit["source_manifest"]["path"], f"quant/v1/{relative}")
            self.assertEqual(audit["source_manifest"]["sha256"], manifest_hash)
            self.assertEqual(audit["formal_pit_status"], "BLOCKED")

    def test_audit_and_dataset_are_immutable_and_fully_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative, manifest_hash = build_quant_fixture(root)
            audit = audit_pit_availability(
                root,
                market="TW",
                now=datetime.datetime(
                    2025, 3, 2, tzinfo=datetime.timezone.utc
                ),
                code_sha="b" * 40,
            )
            audit_result = write_pit_audit(root, audit)
            result = build_price_research_dataset(
                root,
                audit,
                git_sha="c" * 40,
                now=datetime.datetime(
                    2025, 3, 2, tzinfo=datetime.timezone.utc
                ),
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["kind"], "absorb-pit-price-dataset")
            self.assertEqual(manifest["source_manifests"][0]["path"], f"quant/v1/{relative}")
            self.assertEqual(
                manifest["source_manifests"][0]["sha256"], manifest_hash
            )
            self.assertEqual(manifest["availability_audit"]["sha256"], audit_result["sha256"])
            self.assertEqual(manifest["code_sha"], "b" * 40)
            self.assertEqual(manifest["git_sha"], "c" * 40)
            self.assertEqual(manifest["feature_schema_version"], 1)
            self.assertEqual(manifest["target_definition"]["horizon_sessions"], 5)
            self.assertEqual(manifest["pit_policy"]["formal_pit_status"], "BLOCKED")
            self.assertEqual(manifest["split_policy"]["purge_sessions"], 5)
            self.assertEqual(manifest["split_policy"]["embargo_sessions"], 5)

            compressed = Path(result["dataset_path"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(compressed).hexdigest(),
                manifest["dataset_sha256"],
            )
            decoded = gzip.decompress(compressed).decode("utf-8")
            self.assertNotIn("AI_P", decoded)
            first = json.loads(decoded.splitlines()[0])
            self.assertEqual(
                set(first),
                {
                    "symbol",
                    "source_market_date",
                    "close",
                    "volume",
                    "return_1",
                    "momentum_5",
                    "momentum_20",
                    "volatility_20",
                    "volume_ratio_20",
                    "future_return_5",
                    "direction_5",
                },
            )

            repeated = build_price_research_dataset(
                root,
                audit,
                git_sha="c" * 40,
                now=datetime.datetime(
                    2025, 3, 2, tzinfo=datetime.timezone.utc
                ),
            )
            self.assertEqual(repeated, result)
            Path(result["dataset_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "immutable"):
                build_price_research_dataset(
                    root,
                    audit,
                    git_sha="c" * 40,
                    now=datetime.datetime(
                        2025, 3, 2, tzinfo=datetime.timezone.utc
                    ),
                )

    def test_formal_dataset_fails_closed_when_dependencies_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_quant_fixture(root)
            audit = audit_pit_availability(
                root,
                market="TW",
                now=datetime.datetime(
                    2025, 3, 2, tzinfo=datetime.timezone.utc
                ),
                code_sha="d" * 40,
            )

            with self.assertRaisesRegex(ValueError, "formal PIT"):
                build_price_research_dataset(
                    root,
                    audit,
                    git_sha="e" * 40,
                    require_formal=True,
                )


if __name__ == "__main__":
    unittest.main()
