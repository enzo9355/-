import datetime
import importlib
import importlib.util
import math
import unittest


def market_rows(count=280):
    rows = []
    day = datetime.date(2025, 7, 1)
    index = 0
    while len(rows) < count:
        if day.weekday() < 5:
            close = 100 + index * 0.06 + math.sin(index / 6) * 3.5
            rows.append(
                {
                    "Date": day.isoformat(),
                    "Open": close - 0.4,
                    "High": close + 1.1,
                    "Low": close - 1.0,
                    "Close": close,
                }
            )
            index += 1
        day += datetime.timedelta(days=1)
    return rows


class FiveSessionForecastTests(unittest.TestCase):
    def _builder(self):
        spec = importlib.util.find_spec("stock_papi.services.forecast")
        self.assertIsNotNone(spec, "five-session forecast module is missing")
        module = importlib.import_module("stock_papi.services.forecast")
        build = getattr(module, "build_five_session_forecast", None)
        self.assertIsNotNone(build, "five-session forecast builder is missing")
        return build

    def test_forecast_publishes_probability_price_and_oos_evidence(self):
        build = self._builder()

        result = build(market_rows(), market="TW")

        self.assertEqual(result["status"], "published")
        self.assertEqual(result["horizon_sessions"], 5)
        self.assertGreaterEqual(result["probability_up_pct"], 1)
        self.assertLessEqual(result["probability_up_pct"], 99)
        self.assertGreater(result["target_price"], 0)
        self.assertTrue(math.isfinite(result["expected_return_pct"]))
        self.assertEqual(len(result["points"]), 6)
        self.assertEqual(result["points"][0]["value"], rows_close := market_rows()[-1]["Close"])
        self.assertAlmostEqual(result["points"][-1]["value"], result["target_price"], places=2)
        self.assertGreater(result["validation"]["oos_samples"], 30)
        self.assertGreaterEqual(result["validation"]["direction_accuracy_pct"], 0)
        self.assertLessEqual(result["validation"]["direction_accuracy_pct"], 100)
        self.assertGreaterEqual(result["validation"]["price_mae_pct"], 0)
        self.assertEqual(result["model_version"], "lgbm-ohlc-5d-v1")
        self.assertGreater(rows_close, 0)

    def test_forecast_fails_closed_for_short_history(self):
        build = self._builder()
        self.assertIsNone(build(market_rows(80), market="TW"))


if __name__ == "__main__":
    unittest.main()
