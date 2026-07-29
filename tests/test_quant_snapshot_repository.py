import datetime
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import app as stock_app
from stock_papi.repositories import market_insights, quant_snapshots
from tests.report_fixtures import (
    stock_document,
    write_quant_publish,
    write_quant_publish_v3,
)


def rewrite_repository_manifest(publish: Path, manifest: dict) -> None:
    latest_path = publish / "latest-TW.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    relative = latest["manifest"].rsplit("-", 1)[0] + f"-{digest[:12]}.json"
    (publish / relative).write_bytes(encoded)
    latest.update(manifest=relative, manifest_sha256=digest)
    latest_path.write_text(json.dumps(latest, separators=(",", ":")), encoding="utf-8")


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

    def test_repository_returns_none_for_v3_status_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            publish = write_quant_publish_v3(Path(temporary))
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publish / latest["manifest"]).read_text(encoding="utf-8")
            )
            entry = manifest["symbols"]["2303"]
            with gzip.open(publish / entry["path"], "rt", encoding="utf-8") as stream:
                document = json.load(stream)
            document["trading_status_evidence"]["raw_fields"]["volume"] = "1"
            encoded = json.dumps(
                document,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            compressed = gzip.compress(encoded, mtime=0)
            digest = hashlib.sha256(compressed).hexdigest()
            relative = f"objects/{digest}.json.gz"
            (publish / relative).write_bytes(compressed)
            entry.update(
                path=relative,
                sha256=digest,
                size=len(compressed),
                uncompressed_size=len(encoded),
            )
            manifest["expected_non_price_symbols"]["2303"][
                "artifact_sha256"
            ] = digest
            rewrite_repository_manifest(publish, manifest)

            def load_object(name, _limit):
                path = publish / name.removeprefix("quant/v1/")
                return path.read_bytes() if path.is_file() else None

            manifest = quant_snapshots.published_quant_manifest(
                "TW",
                today=datetime.date(2026, 7, 30),
                load_object=load_object,
                cache={},
            )
            document = quant_snapshots.fetch_quant_snapshot(
                "2303",
                today=datetime.date(2026, 7, 30),
                is_us_ticker_fn=lambda _code: False,
                load_manifest=lambda market, today=None: manifest,
                load_object=load_object,
            )

        self.assertIsNotNone(manifest)
        self.assertIsNone(document)

    def test_repository_keeps_v2_cache_key_separate_from_v3_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2 = write_quant_publish(
                root / "v2", [stock_document("2330", as_of="2026-07-29")]
            )
            v3 = write_quant_publish_v3(root / "v3")
            current = {"publish": v2}

            def load_object(name, _limit):
                path = current["publish"] / name.removeprefix("quant/v1/")
                return path.read_bytes() if path.is_file() else None

            cache = {}
            first = quant_snapshots.published_quant_manifest(
                "TW",
                today=datetime.date(2026, 7, 30),
                load_object=load_object,
                cache=cache,
            )
            current["publish"] = v3
            second = quant_snapshots.published_quant_manifest(
                "TW",
                today=datetime.date(2026, 7, 30),
                load_object=load_object,
                cache=cache,
            )

        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(second["schema_version"], 3)
        self.assertEqual(len(cache), 2)

    def test_compatibility_caches_have_one_canonical_owner(self):
        self.assertIs(stock_app._QUANT_MANIFEST_CACHE, quant_snapshots.QUANT_MANIFEST_CACHE)
        self.assertIs(stock_app._MARKET_INSIGHTS_CACHE, market_insights.MARKET_INSIGHTS_CACHE)


if __name__ == "__main__":
    unittest.main()
