import datetime
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock

import pandas as pd

from stock_papi.batch.tw_official_post_close_cli import run
from stock_papi.integrations.market_data.tw_official_bulk import OfficialDailySnapshot, OfficialRequestBudget

TARGET = datetime.date(2026, 7, 24)


def snapshot(price_symbols=("2330", "2303")):
    price = {
        symbol: MappingProxyType({
            "date": "2026-07-24", "stock_id": symbol, "open": 1.0,
            "max": 1.0, "min": 1.0, "close": 1.0, "Trading_Volume": 1.0,
        }) for symbol in price_symbols
    }
    return OfficialDailySnapshot(
        target_date=TARGET,
        price_by_symbol=MappingProxyType(price),
        institutional_by_symbol=MappingProxyType({}),
        margin_by_symbol=MappingProxyType({}),
        source_results=MappingProxyType({}),
        manifest_sha256="a" * 64,
        request_count=6,
        request_budget=OfficialRequestBudget(6, 12, 6, 0, True, "capacity_proven"),
    )


class Pipeline:
    pd = pd
    industry_map = {"全市場": ["2330", "2303"]}

    @staticmethod
    def fetch_finmind_dataset(*_args):
        raise AssertionError("original FinMind fetch must not run")


class TWOfficialPostCloseCLITests(unittest.TestCase):
    def test_prefetches_once_enriches_identity_and_restores_patches(self):
        pipeline = Pipeline()
        original_fetch = pipeline.fetch_finmind_dataset
        observed = {}
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2303", "2330"]
        module.load_stock_pipeline = lambda _root: pipeline

        def original_batch(root, market, symbols, analyze, *args, batch_identity=None, **kwargs):
            observed["identity"] = batch_identity
            return {"next_index": 0, "failed": [], "pending": [], "excluded": []}

        module.run_market_batch = original_batch
        module.build_stock_snapshot = lambda _pipeline, market, symbol, *args, **kwargs: {"symbol": symbol}

        def local_main(argv):
            self.assertIn("--observation-only", argv)
            module.run_market_batch(
                Path("x"), "TW", ["2303", "2330"], lambda _symbol: {},
                batch_identity={"target_market_date": TARGET.isoformat(), "product_mode": "observation"},
            )
            payload = module.build_stock_snapshot(pipeline, "TW", "2330")
            observed["lineage"] = payload["source_lineage"]
            self.assertNotEqual(pipeline.fetch_finmind_dataset, original_fetch)
            return 0

        module.main = local_main
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        builder = Mock(return_value=snapshot())
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "artifacts/stocks/TW/2330.json.gz"
                path.parent.mkdir(parents=True)
                with path.open("wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                        stream.write(json.dumps({
                            "schema_version": 1, "market": "TW", "symbol": "2330",
                            "as_of": "2026-07-23", "daily": [{
                                "Date": "2026-07-23T00:00:00.000", "Open": 1, "High": 1,
                                "Low": 1, "Close": 1, "Volume": 1,
                            }],
                        }).encode())
                result = run(
                    root=Path(temporary), target_market_date=TARGET,
                    limit=5000, delay=0.5, snapshot_builder=builder,
                )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old

        self.assertEqual(result, 0)
        builder.assert_called_once()
        self.assertEqual(observed["identity"]["source_mode"], "tw_official_bulk_v1")
        self.assertEqual(observed["identity"]["official_manifest_sha256"], "a" * 64)
        self.assertEqual(observed["lineage"]["source_mode"], "tw_official_bulk_v1")
        self.assertEqual(pipeline.fetch_finmind_dataset, original_fetch)
        self.assertIs(module.run_market_batch, original_batch)

    def test_source_failure_occurs_before_local_main(self):
        pipeline = Pipeline()
        module = types.ModuleType("local_quant")
        module.get_taiwan_symbols = lambda _pipeline: ["2330"]
        module.load_stock_pipeline = lambda _root: pipeline
        module.main = Mock(return_value=0)
        module.run_market_batch = Mock()
        module.build_stock_snapshot = Mock()
        old = sys.modules.get("local_quant")
        sys.modules["local_quant"] = module
        try:
            with self.assertRaises(RuntimeError):
                run(
                    root=Path("x"), target_market_date=TARGET, limit=1, delay=0,
                    snapshot_builder=lambda *_args: (_ for _ in ()).throw(RuntimeError("source unavailable")),
                )
        finally:
            if old is None:
                sys.modules.pop("local_quant", None)
            else:
                sys.modules["local_quant"] = old
        module.main.assert_not_called()
        module.run_market_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
