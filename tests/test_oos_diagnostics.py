import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_papi.batch.oos_diagnostics import _enrich_point_in_time
from tests.report_fixtures import warmup_stock_document


class OOSDiagnosticsCompatibilityTests(unittest.TestCase):
    def test_oos_filters_null_market_factor_but_keeps_liquidity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = warmup_stock_document("2330")
            document["daily"][0]["MARKET_RET_20"] = None
            raw = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            compressed = gzip.compress(raw, mtime=0)
            object_path = root / "publish" / "quant" / "v1" / "objects" / "2330.json.gz"
            object_path.parent.mkdir(parents=True)
            object_path.write_bytes(compressed)
            manifest = {"symbols": {"2330": {"path": "objects/2330.json.gz", "size": len(compressed), "sha256": hashlib.sha256(compressed).hexdigest()}}}
            content = json.dumps(manifest, separators=(",", ":")).encode()
            manifest_path = root / "publish" / "quant" / "v1" / "manifests" / "fixture.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_bytes(content)
            frame, _metadata = _enrich_point_in_time(
                root,
                {"dataset_manifest": "quant/v1/manifests/fixture.json", "dataset_sha256": hashlib.sha256(content).hexdigest()},
                pd.DataFrame([{"symbol": "2330", "source_market_date": document["daily"][0]["Date"][:10]}]),
            )
        self.assertTrue(frame["liquidity"].notna().all())
        self.assertTrue(frame["market_ret_20"].isna().all())


if __name__ == "__main__":
    unittest.main()
