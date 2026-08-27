import datetime
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


UTC = datetime.timezone.utc


def promoted(market="TW"):
    return {
        "market": market,
        "candidate_sha256": "b" * 64,
        "model_version": "lgbm-5d-v1",
        "feature_schema_version": 1,
        "gates": {
            key: True
            for key in (
                "parity", "leakage", "calibration", "schema", "security",
                "quality", "price_quality",
            )
        },
    }


def source(market="TW"):
    return SimpleNamespace(
        manifest=SimpleNamespace(
            schema_version=4,
            market=market,
            market_as_of=datetime.date(2026, 8, 26),
            manifest_path=f"manifests/{market}-20260826T130000Z-aaaaaaaaaaaa.json",
            manifest_sha256="a" * 64,
        ),
        stocks=(),
    )


def snapshots(market="TW"):
    symbol = "TAIEX" if market == "TW" else "^GSPC"
    return {
        symbol: {
            "schema_version": 2,
            "market": market,
            "symbol": symbol,
            "as_of": "2026-08-26",
            "model_version": "lgbm-5d-v1",
            "feature_schema_version": 1,
            "daily": [
                {
                    "Date": "2026-08-25T00:00:00.000",
                    "Open": 99.0, "High": 101.0, "Low": 98.0, "Close": 100.0,
                },
                {
                    "Date": "2026-08-26T00:00:00.000",
                    "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 100.0,
                    "AI_P": 64.5, "AI_PRED_RET_5": 0.04,
                    "AI_PRED_PRICE_5": 104.0,
                },
            ],
        }
    }


class PredictionProductsCliTests(unittest.TestCase):
    def test_builds_immutable_object_and_local_pointer(self):
        from stock_papi.batch.prediction_products_cli import build

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build(
                root,
                "TW",
                now=datetime.datetime(2026, 8, 26, 14, tzinfo=UTC),
                load_source=lambda _root, market: source(market),
                load_backtest=lambda _root, market: promoted(market),
                prepare_snapshots=lambda _source, _backtest: snapshots("TW"),
                fifth_session=lambda _root, _market, _as_of: datetime.date(2026, 9, 2),
            )

            pointer = json.loads(Path(result["pointer_path"]).read_text(encoding="utf-8"))
            body = Path(result["object_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(body).hexdigest(), pointer["sha256"])
            self.assertEqual(pointer["path"], f"objects/{pointer['sha256']}.json")
            self.assertEqual(json.loads(body)["entities"]["TAIEX"]["predicted_price"], 104.0)

    def test_fails_closed_without_valid_source_promotion_or_entities(self):
        from stock_papi.batch.prediction_products_cli import build

        cases = (
            (lambda _root, market: source("US"), lambda _root, market: promoted(market), lambda *_: snapshots("TW")),
            (lambda _root, market: source(market), lambda _root, market: None, lambda *_: snapshots("TW")),
            (lambda _root, market: source(market), lambda _root, market: {**promoted(market), "gates": {**promoted(market)["gates"], "price_quality": False}}, lambda *_: snapshots("TW")),
            (lambda _root, market: source(market), lambda _root, market: promoted(market), lambda *_: {}),
        )
        for load_source, load_backtest, prepare_snapshots in cases:
            with self.subTest(case=cases.index((load_source, load_backtest, prepare_snapshots))):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(ValueError):
                        build(
                            temporary,
                            "TW",
                            now=datetime.datetime(2026, 8, 26, 14, tzinfo=UTC),
                            load_source=load_source,
                            load_backtest=load_backtest,
                            prepare_snapshots=prepare_snapshots,
                            fifth_session=lambda *_: datetime.date(2026, 9, 2),
                        )


if __name__ == "__main__":
    unittest.main()
