import hashlib
import json
import unittest

from reporting.exceptions import ReportWebError
from stock_papi.repositories.report_store import (
    load_report_index,
    load_report_metadata,
    load_report_metadata_by_sha,
    load_report_pdf,
)


class ReportRepositoryTests(unittest.TestCase):
    def test_v2_metadata_loaders_require_explicit_expected_market(self):
        with self.assertRaises(ValueError):
            load_report_metadata(
                {"metadata": "metadata/" + "a" * 64 + ".json"},
                load_object=lambda *_: self.fail("must reject before object read"),
                version="v2",
            )
        with self.assertRaises(ValueError):
            load_report_metadata_by_sha(
                "a" * 64,
                load_object=lambda *_: self.fail("must reject before object read"),
            )

    def test_v2_index_market_must_match_requested_object(self):
        for requested_market, document_market in (("US", "TW"), ("TW", "US")):
            with self.subTest(
                requested_market=requested_market,
                document_market=document_market,
            ):
                content = json.dumps({
                    "schema_version": 2,
                    "kind": "absorb-report-index",
                    "market": document_market,
                    "updated_at": "2026-08-24T12:00:00Z",
                    "reports": [],
                }).encode("utf-8")

                with self.assertRaises(ReportWebError):
                    load_report_index(
                        load_object=lambda *_: content,
                        max_bytes=1234,
                        version="v2",
                        market=requested_market,
                    )

    def test_v2_index_uses_fixed_allowlisted_prefix(self):
        calls = []
        self.assertIsNone(
            load_report_index(
                load_object=lambda path, size: calls.append((path, size)) or None,
                max_bytes=1234,
                version="v2",
            )
        )
        self.assertEqual(calls, [("reports/v2/index-TW.json", 1234)])
        with self.assertRaises(ValueError):
            load_report_index(
                load_object=lambda *_: None,
                max_bytes=1234,
                version="../../secret",
            )

    def test_verified_pdf_is_returned_and_bad_hash_fails_closed(self):
        pdf = b"%PDF verified"
        item = {
            "pdf_path": f"objects/{hashlib.sha256(pdf).hexdigest()}.pdf",
            "pdf_size": len(pdf),
            "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        }
        self.assertEqual(load_report_pdf(item, load_object=lambda *_: pdf), pdf)
        self.assertIsNone(load_report_pdf(item, load_object=lambda *_: b"corrupt"))


if __name__ == "__main__":
    unittest.main()
