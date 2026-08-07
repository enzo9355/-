import json
import unittest

from stock_papi.services.observation_view import build_stock_observation
from tests.report_fixtures import warmup_stock_document


def snapshot():
    rows = []
    for index in range(65):
        close = 100 + index
        rows.append(
            {
                "Date": f"2026-05-{index + 1:02d}T00:00:00.000",
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "MA20": close - 1,
                "MA60": close - 2,
                "RSI": 75,
                "MACD_OSC": 0.3,
                "K": 62,
                "D": 54,
                "VOL_RATIO": 2.2,
                "INST_NET_RATIO": 0.03,
                "ForeignNet": 1000,
                "DATA_PRICE_WARNING": 0,
                "AI_P": 99,
            }
        )
    rows[-1]["Date"] = "2026-07-16T00:00:00.000"
    return {
        "schema_version": 1,
        "market": "TW",
        "symbol": "2330",
        "name": "台積電",
        "as_of": "2026-07-16",
        "model_version": "must-not-leak",
        "latest": rows[-1],
        "daily": rows,
        "backtest": {"accuracy": 100},
    }


class ObservationViewTests(unittest.TestCase):
    def test_warmup_rows_remain_candles_and_null_ma20_is_filtered(self):
        document = build_stock_observation(warmup_stock_document("2330"))
        self.assertEqual(len(json.loads(document["candles"])), 70)
        self.assertEqual(len(json.loads(document["ma20_line"])), 50)

    def test_stock_view_contains_only_actual_observations(self):
        document = build_stock_observation(snapshot())

        self.assertEqual(document["code"], "2330")
        self.assertEqual(document["name"], "台積電")
        self.assertEqual(document["price"], 164.0)
        self.assertEqual(document["prediction_status"], "AI 預測研究中")
        self.assertEqual(document["trend_observation"], "above_ma20_ma60")
        self.assertTrue(document["risk_events"])
        self.assertTrue(json.loads(document["candles"]))
        self.assertTrue(json.loads(document["ma20_line"]))
        encoded = json.dumps(document, ensure_ascii=False)
        for forbidden in (
            '"prob"',
            '"probability"',
            '"recommendation"',
            '"backtest"',
            '"model_version"',
            '"prediction"',
            "五日上漲機率",
            "勝率",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_invalid_or_sample_snapshot_fails_closed(self):
        self.assertIsNone(build_stock_observation(None))
        invalid = snapshot()
        invalid["sample_data"] = True
        self.assertIsNone(build_stock_observation(invalid))
        invalid = snapshot()
        invalid["daily"] = []
        self.assertIsNone(build_stock_observation(invalid))


if __name__ == "__main__":
    unittest.main()
