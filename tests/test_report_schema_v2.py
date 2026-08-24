import datetime
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reporting.exceptions import ReportPublishError, ReportWebError
from reporting.publisher import publish_report_v2
from reporting.web import validate_report_index, validate_report_metadata


def metadata(report_type):
    return {
        "schema_version": 2,
        "report_type": report_type,
        "market": "TW",
        "source_market_date": "2026-07-14",
        "applicable_trading_date": "2026-07-15",
        "published_at": "2026-07-14T10:00:00Z",
        "forecast_start_date": "2026-07-15",
        "forecast_end_date": "2026-07-21",
        "backtest_as_of": "2026-07-14",
        "data_as_of": "2026-07-14",
        "source_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
        "source_manifest_sha256": "a" * 64,
        "model_versions": {"lgbm-5d-v1": 2000},
        "title": "ABSORB 台股盤後報告" if report_type == "post_close" else "ABSORB 台股盤前快報",
        "summary": ["市場維持整理"],
        "warnings": [],
        "content": {"market_action": "控制追價", "industries": ["半導體"]},
    }


class ReportSchemaV2Tests(unittest.TestCase):
    def test_publisher_rejects_existing_index_with_wrong_top_level_market(self):
        for published_market, wrong_market in (("TW", "US"), ("US", "TW")):
            with self.subTest(
                published_market=published_market,
                wrong_market=wrong_market,
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                document = metadata("pre_market")
                document["market"] = published_market
                publish_report_v2(root, document)
                index_path = (
                    root
                    / "publish"
                    / "reports"
                    / "v2"
                    / f"index-{published_market}.json"
                )
                index = json.loads(index_path.read_text(encoding="utf-8"))
                index["market"] = wrong_market
                index_path.write_text(
                    json.dumps(index, ensure_ascii=False), encoding="utf-8"
                )

                with self.assertRaises(ReportPublishError):
                    publish_report_v2(root, document)

    def test_v1_index_maps_explicitly_to_post_close(self):
        pdf = b"%PDF v1"
        digest = hashlib.sha256(pdf).hexdigest()
        metadata_digest = "b" * 64
        legacy = {
            "schema_version": 1,
            "kind": "daily-industry-report-index",
            "market": "TW",
            "updated_at": "2026-07-14T10:00:00Z",
            "reports": [
                {
                    "report_date": "2026-07-14",
                    "data_as_of": "2026-07-14",
                    "generated_at": "2026-07-14T10:00:00Z",
                    "model_versions": {"lgbm-5d-v1": 1},
                    "coverage": 1.0,
                    "pdf_path": f"objects/{digest}.pdf",
                    "pdf_sha256": digest,
                    "pdf_size": len(pdf),
                    "page_count": 1,
                    "metadata": f"metadata/{metadata_digest}.json",
                    "metadata_sha256": metadata_digest,
                }
            ],
        }

        result = validate_report_index(json.dumps(legacy).encode("utf-8"))

        self.assertEqual(result[0]["report_type"], "post_close")
        self.assertEqual(result[0]["source_market_date"], "2026-07-14")
        self.assertEqual(result[0]["applicable_trading_date"], "2026-07-14")

        with self.assertRaises(ReportWebError):
            validate_report_index(
                json.dumps(legacy).encode("utf-8"), expected_version=2
            )

        legacy_metadata = json.dumps(
            {"schema_version": 1}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with self.assertRaises(ReportWebError):
            validate_report_metadata(
                legacy_metadata,
                {"metadata_sha256": hashlib.sha256(legacy_metadata).hexdigest()},
                expected_version=2,
            )

    def test_v2_publishes_post_close_and_pre_market_for_same_applicable_day(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "post-close.pdf"
            pdf.write_bytes(b"%PDF-1.4 post close")
            publish_report_v2(root, metadata("post_close"), pdf_path=pdf, page_count=8)
            publish_report_v2(root, metadata("pre_market"))

            publish = root / "publish" / "reports" / "v2"
            index_bytes = (publish / "index-TW.json").read_bytes()
            reports = validate_report_index(index_bytes)
            index_document = json.loads(index_bytes)

            self.assertEqual(len(reports), 2)
            self.assertEqual(index_document["kind"], "absorb-report-index")
            self.assertEqual(
                {item["report_type"] for item in reports}, {"post_close", "pre_market"}
            )
            pre_market = next(item for item in reports if item["report_type"] == "pre_market")
            self.assertNotIn("pdf_path", pre_market)
            content = (publish / pre_market["metadata"]).read_bytes()
            document = validate_report_metadata(content, pre_market)
            self.assertEqual(document["kind"], "absorb-report")
            self.assertEqual(document["applicable_trading_date"], "2026-07-15")

    def test_v2_reader_accepts_legacy_kind_during_migration(self):
        document = metadata("pre_market")
        document["kind"] = "stock-papi-report"
        canonical_content = json.dumps(
            document["content"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        document["content_sha256"] = hashlib.sha256(canonical_content).hexdigest()
        content = json.dumps(document, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        entry = {
            key: document[key]
            for key in (
                "report_type", "market", "source_market_date",
                "applicable_trading_date", "published_at", "data_as_of",
                "model_versions", "title", "summary", "content_sha256",
            )
        }
        entry["metadata"] = f"metadata/{digest}.json"
        entry["metadata_sha256"] = digest

        validated = validate_report_metadata(content, entry)

        self.assertEqual(validated["kind"], "stock-papi-report")

    def test_v2_duplicate_is_idempotent_and_conflicting_content_preserves_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = metadata("pre_market")
            publish_report_v2(root, first)
            publish_report_v2(root, first)
            index_path = root / "publish" / "reports" / "v2" / "index-TW.json"
            before = index_path.read_bytes()
            self.assertEqual(len(validate_report_index(before)), 1)

            changed = metadata("pre_market")
            changed["content"] = {"market_action": "提高防守"}
            with self.assertRaises(ReportPublishError):
                publish_report_v2(root, changed)

            self.assertEqual(index_path.read_bytes(), before)

    def test_rerunning_older_report_preserves_newer_index_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            post_close = metadata("post_close")
            pre_market = metadata("pre_market")
            pre_market["published_at"] = "2026-07-14T11:00:00Z"
            publish_report_v2(root, post_close)
            publish_report_v2(root, pre_market)
            index_path = root / "publish" / "reports" / "v2" / "index-TW.json"
            before = index_path.read_bytes()

            publish_report_v2(root, post_close)

            self.assertEqual(index_path.read_bytes(), before)

    def test_pre_market_rejects_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "not-needed.pdf"
            pdf.write_bytes(b"%PDF")
            with self.assertRaises(ValueError):
                publish_report_v2(root, metadata("pre_market"), pdf_path=pdf, page_count=1)


if __name__ == "__main__":
    unittest.main()
