import unittest

from stock_papi.services.prediction_view import prediction_for


def product(as_of="2026-08-26"):
    return {
        "schema_version": 1,
        "kind": "absorb-five-session-predictions",
        "market": "TW",
        "as_of": as_of,
        "entities": {
            "2330": {
                "symbol": "2330",
                "entity_type": "security",
                "as_of": as_of,
                "target_session": "2026-09-02",
                "current_price": 100.0,
                "up_probability": 0.645,
                "predicted_return_5d": 0.04,
                "predicted_price": 104.0,
                "predicted_change_pct": 4.0,
            }
        },
        "model_version": "lgbm-5d-v1",
        "backtest_sha256": "b" * 64,
    }


class PredictionViewTests(unittest.TestCase):
    def test_current_prediction_is_merged_with_plain_language_fields(self):
        result = prediction_for(product(), "TW", "2330", "2026-08-26")

        self.assertEqual(result["status"], "current")
        self.assertEqual(result["probability_pct"], 64.5)
        self.assertEqual(result["predicted_price"], 104.0)
        self.assertEqual(result["target_session"], "2026-09-02")
        self.assertEqual(result["model_version"], "lgbm-5d-v1")

    def test_previous_prediction_is_labeled_and_future_or_wrong_identity_is_rejected(self):
        previous = prediction_for(product("2026-08-26"), "TW", "2330", "2026-08-27")
        self.assertEqual(previous["status"], "previous")
        self.assertEqual(previous["as_of"], "2026-08-26")

        self.assertIsNone(prediction_for(product(), "US", "2330", "2026-08-26"))
        self.assertIsNone(prediction_for(product(), "TW", "9999", "2026-08-26"))
        self.assertIsNone(prediction_for(product(), "TW", "2330", "2026-08-25"))

    def test_untrusted_embedded_prediction_shape_is_rejected(self):
        bad = product()
        bad["entities"]["2330"]["predicted_price"] = "104"
        self.assertIsNone(prediction_for(bad, "TW", "2330", "2026-08-26"))

    def test_index_view_preserves_verified_actual_candles(self):
        value = product()
        value["entities"] = {
            "TAIEX": {
                **value["entities"]["2330"],
                "symbol": "TAIEX",
                "entity_type": "market_index",
                "candles": [
                    {"time": "2026-08-25", "open": 98.0, "high": 100.0, "low": 97.0, "close": 99.0},
                    {"time": "2026-08-26", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0},
                ],
            }
        }

        result = prediction_for(value, "TW", "TAIEX", "2026-08-26")

        self.assertEqual(len(result["candles"]), 2)
        self.assertEqual(result["candles"][-1]["close"], 100.0)


if __name__ == "__main__":
    unittest.main()
