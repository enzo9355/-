import datetime
import tempfile
import unittest
from pathlib import Path

import app as stock_app
from stock_papi.repositories import market_insights, quant_snapshots
from tests.report_fixtures import write_quant_publish_v3


class QuantSnapshotRepositoryTests(unittest.TestCase):
    def test_repository_accepts_hash_bound_v3_status_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            publish = write_quant_publish_v3(Path(temporary))

            def load_object(name, _limit):
                path = publish / name.removeprefix("quant/v1/")
                return path.read_bytes() if path.is_file() else None

            cache = {}
            manifest = quant_snapshots.published_quant_manifest(
                "TW",
                today=datetime.date(2026, 7, 30),
                load_object=load_object,
                cache=cache,
            )
            document = quant_snapshots.fetch_quant_snapshot(
                "2303",
                today=datetime.date(2026, 7, 30),
                is_us_ticker_fn=lambda _code: False,
                load_manifest=lambda market, today=None: manifest,
                load_object=load_object,
            )

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(document["observation_kind"], "official_no_regular_trade")
        self.assertEqual(
            document["trading_status_evidence"]["evidence_sha256"],
            manifest["expected_non_price_symbols"]["2303"]["evidence_sha256"],
        )

    def test_compatibility_caches_have_one_canonical_owner(self):
        self.assertIs(stock_app._QUANT_MANIFEST_CACHE, quant_snapshots.QUANT_MANIFEST_CACHE)
        self.assertIs(stock_app._MARKET_INSIGHTS_CACHE, market_insights.MARKET_INSIGHTS_CACHE)


if __name__ == "__main__":
    unittest.main()
