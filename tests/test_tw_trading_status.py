import datetime
import unittest

from stock_papi.integrations.market_data.tw_trading_status import (
    classify_price_row,
    evidence_sha256,
    resolve_lifecycle_status,
)


TARGET = datetime.date(2026, 7, 29)
FIELDS = (
    "代號",
    "名稱",
    "收盤",
    "漲跌",
    "開盤",
    "最高",
    "最低",
    "均價",
    "成交股數",
)
INDICES = {
    "symbol": 0,
    "name": 1,
    "close": 2,
    "open": 4,
    "high": 5,
    "low": 6,
    "volume": 8,
}


def classify(row):
    return classify_price_row(
        TARGET,
        "tpex_price",
        "TPEx",
        FIELDS,
        row,
        INDICES,
        "a" * 64,
    )


def event(event_type, effective_date):
    document = {
        "schema_version": 1,
        "exchange": "TWSE",
        "symbol": "1459",
        "event_type": event_type,
        "effective_date": effective_date,
        "source_id": "twse_reduction",
        "payload_sha256": "b" * 64,
        "raw_row_sha256": "c" * 64,
        "parser_version": "tw-lifecycle-parser-v2",
    }
    document["evidence_sha256"] = evidence_sha256(document)
    return document


class TwTradingStatusTests(unittest.TestCase):
    def test_blank_ohlc_with_positive_official_volume_is_no_regular_trade(self):
        result = classify(
            ["00886", "永豐美國科技", " ---", "--- ", "---", "---", "---", "44.11", "435"]
        )

        self.assertIsNone(result.price)
        self.assertEqual(result.status["status"], "official_no_regular_trade")
        self.assertEqual(result.status["target_market_date"], "2026-07-29")
        self.assertEqual(result.status["raw_fields"]["volume"], "435")
        self.assertRegex(result.status["raw_row_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(result.status["evidence_sha256"], r"^[0-9a-f]{64}$")

    def test_blank_ohlc_with_zero_or_blank_volume_is_no_regular_trade(self):
        for volume in ("0", "---", None):
            with self.subTest(volume=volume):
                result = classify(
                    ["3064", "泰偉", "---", "---", "---", "---", "---", "18.84", volume]
                )
                self.assertIsNone(result.price)
                self.assertEqual(result.status["status"], "official_no_regular_trade")

    def test_partial_blank_or_prose_ohlc_fails_closed(self):
        rows = (
            ["3064", "泰偉", "18", "0", "---", "18", "18", "18", "0"],
            ["3064", "泰偉", "暫停交易", "0", "---", "---", "---", "18", "0"],
        )
        for row in rows:
            with self.subTest(row=row), self.assertRaisesRegex(
                ValueError, "official price row is invalid"
            ):
                classify(row)

    def test_valid_ohlc_with_zero_volume_remains_regular_price(self):
        result = classify(
            ["6488", "環球晶", "405", "0", "400", "410", "395", "402", "0"]
        )

        self.assertIsNone(result.status)
        self.assertEqual(result.price["close"], 405.0)
        self.assertEqual(result.price["Trading_Volume"], 0.0)

    def test_resume_closes_suspension_on_resume_session(self):
        suspended = resolve_lifecycle_status(
            [event("suspend", "2026-07-23"), event("resume", "2026-08-03")],
            TARGET,
            active=True,
        )
        resumed = resolve_lifecycle_status(
            [event("suspend", "2026-07-23"), event("resume", "2026-08-03")],
            datetime.date(2026, 8, 3),
            active=True,
        )

        self.assertEqual(suspended["status"], "officially_suspended")
        self.assertEqual(suspended["valid_from"], "2026-07-23")
        self.assertEqual(suspended["valid_through_exclusive"], "2026-08-03")
        self.assertIsNone(resumed)

    def test_effective_termination_is_a_universe_disposition_not_suspension(self):
        disposition = resolve_lifecycle_status(
            [event("terminate", "2026-07-28")], TARGET, active=True
        )

        self.assertEqual(disposition["status"], "officially_terminated")
        self.assertNotEqual(disposition["status"], "officially_suspended")


if __name__ == "__main__":
    unittest.main()
