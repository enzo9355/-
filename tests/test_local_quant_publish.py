import datetime
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_quant import (
    TAIPEI,
    ensure_layout,
    publish_market_insights,
    publish_market_snapshot,
    write_stock_artifact,
)
from stock_papi.integrations.market_data.tw_trading_status import evidence_sha256


TARGET = datetime.date(2026, 7, 29)


def v3_document(symbol, *, status=None):
    latest_date = "2026-07-16" if status else TARGET.isoformat()
    document = {
        "schema_version": 2,
        "as_of": latest_date,
        "target_market_date": TARGET.isoformat(),
        "observation_as_of": TARGET.isoformat(),
        "latest_regular_price_date": latest_date,
        "observation_kind": status["status"] if status else "regular_price",
        "trading_status_evidence": status,
        "name": symbol,
        "model_version": "observation-source-v1",
        "latest": {"Date": f"{latest_date}T00:00:00.000", "Close": 100.0},
        "backtest": {},
        "daily": [{"Date": f"{latest_date}T00:00:00.000", "Close": 100.0}],
        "source_lineage": {
            "source_schema_version": "tw-official-historical-v3",
            "observation_as_of": TARGET.isoformat(),
            "latest_regular_price_date": latest_date,
            "observation_kind": status["status"] if status else "regular_price",
        },
    }
    if status:
        document["source_lineage"]["trading_status_evidence_sha256"] = status[
            "evidence_sha256"
        ]
    return document


def no_trade_status(symbol="2303"):
    status = {
        "schema_version": 1,
        "status": "official_no_regular_trade",
        "market": "TW",
        "exchange": "TWSE",
        "symbol": symbol,
        "target_market_date": TARGET.isoformat(),
        "source_id": "twse_price",
        "payload_sha256": "a" * 64,
        "raw_row_sha256": "b" * 64,
        "raw_fields": {"symbol": symbol, "name": symbol, "open": "--", "high": "--", "low": "--", "close": "--", "volume": "0"},
        "parser_version": "tw-official-historical-parser-v3",
    }
    status["evidence_sha256"] = evidence_sha256(status)
    return status


class LocalQuantPublishTests(unittest.TestCase):
    def test_v3_manifest_partitions_regular_status_and_operational_symbols(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            symbols = [f"{number:04d}" for number in range(1000, 1021)]
            status_symbol = symbols[-2]
            failed_symbol = symbols[-1]
            status = no_trade_status(status_symbol)
            for symbol in symbols[:-2]:
                write_stock_artifact(root, "TW", symbol, v3_document(symbol))
            write_stock_artifact(
                root,
                "TW",
                status_symbol,
                v3_document(status_symbol, status=status),
            )

            latest_path = publish_market_snapshot(
                root,
                "TW",
                symbols,
                generated_at=datetime.datetime(2026, 7, 30, 6, tzinfo=TAIPEI),
                failed_symbols=[failed_symbol],
                target_market_date=TARGET,
            )
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (latest_path.parent / latest["manifest"]).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["observation_count"], 20)
        self.assertEqual(manifest["regular_price_symbol_count"], 19)
        self.assertEqual(manifest["expected_non_price_symbol_count"], 1)
        self.assertEqual(manifest["operational_failure_count"], 1)
        self.assertEqual(manifest["regular_price_denominator"], 20)
        self.assertEqual(manifest["regular_price_coverage"], 19 / 20)
        self.assertEqual(manifest["operational_failed_symbols"], [failed_symbol])

    def test_v3_manifest_partitions_regular_and_status_symbols(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            status = no_trade_status()
            write_stock_artifact(root, "TW", "2330", v3_document("2330"))
            write_stock_artifact(root, "TW", "2303", v3_document("2303", status=status))

            latest_path = publish_market_snapshot(
                root,
                "TW",
                ["2303", "2330"],
                generated_at=datetime.datetime(2026, 7, 30, 6, tzinfo=TAIPEI),
                target_market_date=TARGET,
            )
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (latest_path.parent / latest["manifest"]).read_text(encoding="utf-8")
            )

        self.assertEqual(latest["schema_version"], 3)
        self.assertEqual(manifest["schema_version"], 3)
        self.assertNotIn("market_as_of", manifest)
        self.assertEqual(manifest["target_market_date"], TARGET.isoformat())
        self.assertEqual(manifest["observation_as_of"], TARGET.isoformat())
        self.assertEqual(manifest["universe_count"], 2)
        self.assertEqual(manifest["observation_count"], 2)
        self.assertEqual(manifest["regular_price_symbol_count"], 1)
        self.assertEqual(manifest["expected_non_price_symbol_count"], 1)
        self.assertEqual(manifest["operational_failure_count"], 0)
        self.assertEqual(manifest["regular_price_denominator"], 1)
        self.assertEqual(manifest["regular_price_coverage"], 1.0)
        self.assertEqual(manifest["observation_coverage"], 1.0)
        self.assertEqual(manifest["operational_failure_rate"], 0.0)
        expected = manifest["expected_non_price_symbols"]["2303"]
        self.assertEqual(expected["status"], "official_no_regular_trade")
        self.assertEqual(expected["evidence_sha256"], status["evidence_sha256"])
        self.assertEqual(expected["artifact_sha256"], manifest["symbols"]["2303"]["sha256"])
        self.assertEqual(expected["latest_regular_price_date"], "2026-07-16")
        self.assertEqual(manifest["operational_failed_symbols"], [])

    def test_v3_publish_rejects_status_date_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            status = no_trade_status()
            status["target_market_date"] = "2026-07-28"
            status["evidence_sha256"] = evidence_sha256(status)
            write_stock_artifact(
                root, "TW", "2303", v3_document("2303", status=status)
            )

            with self.assertRaisesRegex(RuntimeError, "artifact is invalid"):
                publish_market_snapshot(
                    root,
                    "TW",
                    ["2303"],
                    generated_at=datetime.datetime(2026, 7, 30, 6, tzinfo=TAIPEI),
                    target_market_date=TARGET,
                )

    def test_v3_unknown_missing_price_preserves_previous_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-28", "model_version": "lgbm-5d-v1"},
            )
            latest_path = publish_market_snapshot(
                root,
                "TW",
                ["2330"],
                generated_at=datetime.datetime(2026, 7, 29, 6, tzinfo=TAIPEI),
            )
            before = latest_path.read_bytes()
            write_stock_artifact(root, "TW", "2330", v3_document("2330"))

            with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
                publish_market_snapshot(
                    root,
                    "TW",
                    ["2303", "2330"],
                    generated_at=datetime.datetime(2026, 7, 30, 6, tzinfo=TAIPEI),
                    target_market_date=TARGET,
                )

            self.assertEqual(latest_path.read_bytes(), before)

    def test_market_insights_publish_content_addressed_gzip_and_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            document = {
                "schema_version": 1,
                "as_of": "2026-07-06",
                "industries": [], "mops": [], "etfs": [], "supply_chains": [],
                "sources": ["TWSE"],
            }

            latest_path = publish_market_insights(
                root,
                document,
                generated_at=datetime.datetime(2026, 7, 7, 2, 30, tzinfo=TAIPEI),
            )

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            object_path = latest_path.parent / latest["path"]
            self.assertEqual(latest["schema_version"], 1)
            self.assertEqual(latest["kind"], "market-insights")
            self.assertEqual(hashlib.sha256(object_path.read_bytes()).hexdigest(), latest["sha256"])
            with gzip.open(object_path, "rt", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), document)

    def test_four_percent_failures_publish_with_coverage_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            symbols = [f"{number:04d}" for number in range(100)]
            failed = symbols[-4:]
            for symbol in symbols[:-4]:
                write_stock_artifact(
                    root,
                    "TW",
                    symbol,
                    {"as_of": "2026-07-03", "model_version": "lgbm-5d-v1"},
                )

            latest_path = publish_market_snapshot(
                root,
                "TW",
                symbols,
                failed_symbols=failed,
                generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
            )

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (latest_path.parent / latest["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["universe_count"], 100)
            self.assertEqual(manifest["symbol_count"], 96)
            self.assertEqual(manifest["failure_count"], 4)
            self.assertEqual(manifest["failed_symbols"], failed)
            self.assertEqual(manifest["coverage"], 0.96)

    def test_five_percent_failures_preserve_previous_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            symbols = [f"{number:04d}" for number in range(100)]
            for symbol in symbols[:-5]:
                write_stock_artifact(root, "TW", symbol, {"as_of": "2026-07-03"})
            latest_path = root / "publish" / "quant" / "v1" / "latest-TW.json"
            latest_path.parent.mkdir(parents=True)
            latest_path.write_text('{"previous":true}', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "failure rate"):
                publish_market_snapshot(
                    root,
                    "TW",
                    symbols,
                    failed_symbols=symbols[-5:],
                    generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                )

            self.assertEqual(latest_path.read_text(encoding="utf-8"), '{"previous":true}')

    def test_complete_market_publishes_content_addressed_manifest_and_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-03", "model_version": "lgbm-5d-v1"},
            )
            write_stock_artifact(
                root,
                "TW",
                "2317",
                {"as_of": "2026-07-03", "model_version": "lgbm-5d-v1"},
            )

            latest_path = publish_market_snapshot(
                root,
                "TW",
                ["2330", "2317"],
                generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
            )

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            manifest_path = latest_path.parent / latest["manifest"]
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual(
                latest["manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest()
            )
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["symbol_count"], 2)
            self.assertEqual(manifest["market_as_of"], "2026-07-03")
            self.assertEqual(list(manifest["symbols"]), ["2317", "2330"])
            for entry in manifest["symbols"].values():
                object_path = latest_path.parent / entry["path"]
                self.assertTrue(object_path.is_file())
                self.assertEqual(
                    hashlib.sha256(object_path.read_bytes()).hexdigest(), entry["sha256"]
                )
                with gzip.open(object_path, "rb") as stream:
                    decoded = stream.read()
                self.assertEqual(entry["uncompressed_size"], len(decoded))
                with gzip.open(object_path, "rt", encoding="utf-8") as stream:
                    self.assertEqual(json.load(stream)["schema_version"], 1)

    def test_identical_market_rerun_preserves_existing_manifest_and_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-03", "model_version": "observation-source-v1"},
            )

            latest_path = publish_market_snapshot(
                root,
                "TW",
                ["2330"],
                generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
            )
            original_latest = latest_path.read_bytes()
            original_manifests = sorted((latest_path.parent / "manifests").iterdir())

            rerun_path = publish_market_snapshot(
                root,
                "TW",
                ["2330"],
                generated_at=datetime.datetime(2026, 7, 5, 7, tzinfo=TAIPEI),
            )

            self.assertEqual(rerun_path, latest_path)
            self.assertEqual(rerun_path.read_bytes(), original_latest)
            self.assertEqual(
                sorted((latest_path.parent / "manifests").iterdir()),
                original_manifests,
            )

    def test_changed_market_rerun_publishes_new_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-03", "close": 100},
            )
            latest_path = publish_market_snapshot(
                root,
                "TW",
                ["2330"],
                generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
            )
            original_latest = latest_path.read_bytes()

            write_stock_artifact(
                root,
                "TW",
                "2330",
                {"as_of": "2026-07-03", "close": 101},
            )
            publish_market_snapshot(
                root,
                "TW",
                ["2330"],
                generated_at=datetime.datetime(2026, 7, 5, 7, tzinfo=TAIPEI),
            )

            self.assertNotEqual(latest_path.read_bytes(), original_latest)
            self.assertEqual(
                len(list((latest_path.parent / "manifests").iterdir())),
                2,
            )

    def test_missing_artifact_does_not_replace_previous_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            publish_root = root / "publish" / "quant" / "v1"
            publish_root.mkdir(parents=True)
            latest_path = publish_root / "latest-TW.json"
            latest_path.write_text('{"previous":true}', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
                publish_market_snapshot(
                    root,
                    "TW",
                    ["2330"],
                    generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                )

            self.assertEqual(latest_path.read_text(encoding="utf-8"), '{"previous":true}')

    def test_corrupt_artifact_does_not_create_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            artifact = root / "artifacts" / "stocks" / "US" / "AAPL.json.gz"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"not-gzip")

            with self.assertRaisesRegex(RuntimeError, "artifact is invalid"):
                publish_market_snapshot(
                    root,
                    "US",
                    ["AAPL"],
                    generated_at=datetime.datetime(2026, 7, 5, 6, tzinfo=TAIPEI),
                )

            self.assertFalse(
                (root / "publish" / "quant" / "v1" / "latest-US.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
