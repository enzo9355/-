import datetime
import unittest

from stock_papi.batch.prediction_products import (
    build_prediction_product,
    validate_prediction_product,
)


def promoted_backtest(market="TW"):
    return {
        "market": market,
        "candidate_sha256": "b" * 64,
        "model_version": "lgbm-5d-v1",
        "feature_schema_version": 1,
        "cutoff": "2026-08-20",
        "promoted_at": "2026-08-21T01:00:00Z",
        "gates": {
            "parity": True,
            "leakage": True,
            "calibration": True,
            "schema": True,
            "security": True,
            "quality": True,
            "price_quality": True,
        },
        "metrics": {"price_mae_ratio": 0.82},
    }


def quant_manifest(market="TW", symbols=("2330", "TAIEX")):
    return {
        "schema_version": 4,
        "market": market,
        "observation_as_of": "2026-08-26",
        "source_manifest": (
            f"quant/v1/manifests/{market}-20260826T130000Z-{'a' * 12}.json"
        ),
        "source_manifest_sha256": "a" * 64,
        "symbols": {symbol: {"as_of": "2026-08-26"} for symbol in symbols},
    }


def snapshot(symbol, market="TW"):
    daily = [{
        "Date": "2026-08-26",
        "Open": 99.0,
        "High": 101.0,
        "Low": 98.0,
        "Close": 100.0,
        "AI_P": 64.5,
        "AI_PRED_RET_5": 0.04,
        "AI_PRED_PRICE_5": 104.0,
    }]
    if symbol in {"TAIEX", "^GSPC", "^IXIC", "^DJI"}:
        daily.insert(0, {
            "Date": "2026-08-25", "Open": 97.0, "High": 100.0,
            "Low": 96.0, "Close": 99.0,
        })
    return {
        "schema_version": 2,
        "market": market,
        "symbol": symbol,
        "as_of": "2026-08-26",
        "model_version": "lgbm-5d-v1",
        "feature_schema_version": 1,
        "daily": daily,
    }


class PredictionProductTests(unittest.TestCase):
    def test_builder_emits_stock_and_allowlisted_index_with_one_semantics(self):
        product = build_prediction_product(
            "TW",
            quant_manifest(),
            {"2330": snapshot("2330"), "TAIEX": snapshot("TAIEX")},
            promoted_backtest(),
            next_session=lambda _market, _as_of, _count: datetime.date(2026, 9, 2),
            generated_at=datetime.datetime(
                2026, 8, 26, 14, tzinfo=datetime.timezone.utc
            ),
        )

        self.assertEqual(product["horizon_sessions"], 5)
        self.assertEqual(product["as_of"], "2026-08-26")
        self.assertEqual(product["entities"]["TAIEX"]["entity_type"], "market_index")
        self.assertEqual(product["entities"]["2330"]["entity_type"], "security")
        self.assertEqual(product["entities"]["2330"]["target_session"], "2026-09-02")
        self.assertEqual(product["entities"]["2330"]["up_probability"], 0.645)
        self.assertEqual(product["entities"]["2330"]["predicted_price"], 104.0)
        self.assertEqual(product["entities"]["2330"]["predicted_change_pct"], 4.0)
        self.assertEqual(len(product["entities"]["TAIEX"]["candles"]), 2)
        self.assertEqual(product["entities"]["TAIEX"]["candles"][-1]["close"], 100.0)
        self.assertNotIn("candles", product["entities"]["2330"])
        self.assertIs(validate_prediction_product(product), product)

    def test_builder_requires_price_quality_and_rejects_unknown_index(self):
        with self.assertRaisesRegex(ValueError, "promotion"):
            build_prediction_product(
                "TW", quant_manifest(), {"2330": snapshot("2330")}, None,
                next_session=lambda *_args: datetime.date(2026, 9, 2),
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )

        backtest = promoted_backtest()
        backtest["gates"]["price_quality"] = False
        with self.assertRaisesRegex(ValueError, "promotion"):
            build_prediction_product(
                "TW", quant_manifest(), {"2330": snapshot("2330")}, backtest,
                next_session=lambda *_args: datetime.date(2026, 9, 2),
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )

        with self.assertRaisesRegex(ValueError, "symbol"):
            build_prediction_product(
                "US",
                quant_manifest("US", ("^RUT",)),
                {"^RUT": snapshot("^RUT", "US")},
                promoted_backtest("US"),
                next_session=lambda *_args: datetime.date(2026, 9, 2),
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )

    def test_builder_rejects_non_finite_or_inconsistent_price(self):
        bad = snapshot("2330")
        bad["daily"][0]["AI_PRED_PRICE_5"] = 105.0
        with self.assertRaisesRegex(ValueError, "prediction"):
            build_prediction_product(
                "TW", quant_manifest(), {"2330": bad}, promoted_backtest(),
                next_session=lambda *_args: datetime.date(2026, 9, 2),
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
