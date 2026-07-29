import datetime
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.report_fixtures import (
    stock_document,
    write_quant_publish,
    write_quant_publish_v3,
)


class DailyReportSourceTests(unittest.TestCase):
    def test_v3_loader_separates_regular_and_verified_status_artifacts(self):
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_quant_publish_v3(root)
            source = load_report_source(root, market="TW")

        self.assertEqual(source.manifest.schema_version, 3)
        self.assertEqual(source.manifest.observation_as_of.isoformat(), "2026-07-29")
        by_symbol = {stock.symbol: stock for stock in source.stocks}
        self.assertEqual(by_symbol["2330"].observation_kind, "regular_price")
        self.assertEqual(
            by_symbol["2303"].observation_kind, "official_no_regular_trade"
        )
        self.assertEqual(
            by_symbol["2303"].trading_status_evidence["evidence_sha256"],
            source.manifest.expected_non_price_symbols["2303"]["evidence_sha256"],
        )
        self.assertEqual(by_symbol["2303"].as_of.isoformat(), "2026-07-16")

    def test_accepts_only_valid_manifest_listed_tw_objects(self):
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_quant_publish(
                root, [stock_document("2330"), stock_document("2317")]
            )
            unlisted = publish / "objects" / ("f" * 64 + ".json.gz")
            unlisted.write_bytes(gzip.compress(b"{}"))

            source = load_report_source(root, market="TW")

            self.assertEqual(source.manifest.market_as_of.isoformat(), "2026-07-03")
            self.assertEqual([stock.symbol for stock in source.stocks], ["2317", "2330"])
            self.assertNotIn("f" * 64, {stock.sha256 for stock in source.stocks})

    def test_rejects_manifest_hash_mismatch_and_preserves_trust_boundary(self):
        from reporting.exceptions import ReportSourceError
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_quant_publish(root, [stock_document("2330")])
            latest_path = publish / "latest-TW.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            latest["manifest_sha256"] = "0" * 64
            latest_path.write_text(json.dumps(latest), encoding="utf-8")

            with self.assertRaisesRegex(ReportSourceError, "manifest hash"):
                load_report_source(root, market="TW")

    def test_rejects_path_traversal_object_size_and_non_finite_values(self):
        from reporting.exceptions import ReportSourceError
        from reporting.source_loader import load_report_source

        for corruption in ("path", "size", "non_finite"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                publish = write_quant_publish(root, [stock_document("2330")])
                latest = json.loads((publish / "latest-TW.json").read_text(encoding="utf-8"))
                manifest_path = publish / latest["manifest"]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                entry = manifest["symbols"]["2330"]
                if corruption == "path":
                    entry["path"] = "objects/../../secret.json.gz"
                elif corruption == "size":
                    entry["size"] += 1
                else:
                    document = stock_document("2330")
                    document["daily"][-1]["Close"] = float("nan")
                    encoded = json.dumps(document, allow_nan=True).encode("utf-8")
                    compressed = gzip.compress(encoded, mtime=0)
                    digest = hashlib.sha256(compressed).hexdigest()
                    object_path = publish / "objects" / f"{digest}.json.gz"
                    object_path.write_bytes(compressed)
                    entry.update(
                        path=f"objects/{digest}.json.gz",
                        sha256=digest,
                        size=len(compressed),
                        uncompressed_size=len(encoded),
                    )
                manifest_bytes = json.dumps(
                    manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                manifest_path.write_bytes(manifest_bytes)
                latest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
                (publish / "latest-TW.json").write_text(json.dumps(latest), encoding="utf-8")

                with self.assertRaises(ReportSourceError):
                    load_report_source(root, market="TW")

    def test_rejects_mixed_as_of_wrong_market_and_future_date(self):
        from reporting.exceptions import ReportSourceError
        from reporting.source_loader import load_report_source

        cases = {
            "mixed": [
                stock_document("2330", as_of="2026-07-02"),
                stock_document("2317", as_of="2026-07-03"),
            ],
            "market": [{**stock_document("2330"), "market": "US"}],
            "future": [stock_document("2330", as_of="2099-01-01")],
        }
        for name, documents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_quant_publish(root, documents)
                with self.assertRaises(ReportSourceError):
                    load_report_source(root, market="TW")

    def test_rejects_missing_uncompressed_size_metadata(self):
        from reporting.exceptions import ReportSourceError
        from reporting.source_loader import load_report_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish = write_quant_publish(root, [stock_document("2330")])
            latest = json.loads((publish / "latest-TW.json").read_text(encoding="utf-8"))
            manifest_path = publish / latest["manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["symbols"]["2330"]["uncompressed_size"]
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            latest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            (publish / "latest-TW.json").write_text(json.dumps(latest), encoding="utf-8")

            with self.assertRaises(ReportSourceError):
                load_report_source(root)

    def test_loads_latest_valid_previous_manifest_without_zero_fallback(self):
        from reporting.source_loader import (
            load_previous_report_source,
            load_report_source,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_quant_publish(root, [stock_document("2330", as_of="2026-07-02")])
            write_quant_publish(root, [stock_document("2330", as_of="2026-07-03")])
            current = load_report_source(root)

            previous = load_previous_report_source(root, current.manifest.market_as_of)

            self.assertIsNotNone(previous)
            self.assertEqual(previous.manifest.market_as_of.isoformat(), "2026-07-02")
            self.assertIsNone(
                load_previous_report_source(root, datetime.date(2026, 7, 2))
            )


if __name__ == "__main__":
    unittest.main()
