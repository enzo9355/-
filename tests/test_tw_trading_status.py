import datetime
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.integrations.market_data.tw_trading_status import (
    LIFECYCLE_SOURCE_DEFINITIONS,
    classify_price_row,
    evidence_sha256,
    load_lifecycle_snapshot,
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


def lifecycle_payloads():
    return {
        "twse_current_stop": {
            "stat": "ok",
            "tables": [{
                "title": "115年07月30日 停止買賣",
                "fields": ["證券代號", "證券名稱", "違反營業細則條款", "停止買賣原因", "停止買賣開始日期"],
                "data": [["1589", "永冠-KY", "第50-3條", "財報", "115年04月07日"]],
            }],
        },
        "twse_intraday_halt": [{
            "Number": "1", "Code": "4169", "Name": "泰宗",
            "TradingHaltDate": "1150723", "TradingHaltTime": "080000",
            "TradingResumptionDate": "1150724", "TradingResumptionTime": "080000",
        }],
        "twse_reduction_resume": {
            "stat": "OK",
            "fields": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價", "漲停價格", "跌停價格", "開盤競價基準", "除權參考價", "減資原因", "詳細資料"],
            "data": [["115/08/03", "1459", "聯發", "11.85", "12.46", "13.70", "11.25", "12.45", "--", "退還股款", "1459,20260722"]],
            "strDate": "20260803", "endDate": "20260803",
        },
        "twse_reduction_detail_1459_20260722": {
            "stat": "OK",
            "fields": ["股票代號：", "股票名稱：", "停止買賣日期："],
            "data": [["1459", "聯發", "115/07/23"]],
        },
        "twse_termination": [
            {"DelistingDate": "115/06/23", "Company": "森崴能源", "Code": "6806"}
        ],
        "tpex_current_mode": [{
            "Date": "1150729", "SecuritiesCompanyCode": "4804", "CompanyName": "大略-KY",
            "AlteredTrading": "", "PeriodicTrading": "", "ManagedStock": "",
            "MatchingFrequency": "", "SuspensionOfTrading": "Ｙ", " FinancialAnnouncements": "Ｙ",
        }],
        "tpex_suspend_history": [],
        "tpex_termination": {
            "stat": "ok", "date": "2026",
            "tables": [{
                "fields": ["股票代號", "公司名稱", "終止上櫃日期", "終止上櫃原因", "公司資料網址"],
                "data": [["3426", "台興電子企業股份有限公司", "115-06-08", "規則", "https://mops.twse.com.tw/"]],
            }],
        },
    }


class LifecycleResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.headers = {"Content-Length": str(len(self.content))}


class LifecycleSession:
    def __init__(self):
        self.calls = []
        self.payloads = lifecycle_payloads()

    def get(self, url, *, params, headers, timeout):
        source_id = next(
            source_id
            for source_id, definition in LIFECYCLE_SOURCE_DEFINITIONS.items()
            if definition.url == url
        )
        if source_id == "twse_reduction_detail":
            source_id = f"{source_id}_{params['STK_NO']}_{params['FILE_DATE']}"
        self.calls.append((source_id, dict(params)))
        return LifecycleResponse(self.payloads[source_id])


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

    def test_lifecycle_loader_resolves_official_intervals_and_terminations(self):
        session = LifecycleSession()
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = load_lifecycle_snapshot(
                Path(temporary),
                TARGET,
                session=session,
                required_symbols_by_exchange={
                    "TWSE": {"1459", "1589", "6806"},
                    "TPEx": {"3426", "4804"},
                },
                now=datetime.datetime(2026, 7, 30, 1, tzinfo=datetime.timezone.utc),
            )

        self.assertEqual(snapshot.status_by_symbol["1459"]["status"], "officially_suspended")
        self.assertEqual(snapshot.status_by_symbol["1589"]["valid_from"], "2026-04-07")
        self.assertEqual(snapshot.status_by_symbol["4804"]["target_market_date"], "2026-07-29")
        self.assertEqual(set(snapshot.terminated_by_symbol), {"3426", "6806"})
        self.assertEqual(snapshot.request_count, 8)
        self.assertEqual(len(snapshot.source_hashes), 8)

    def test_lifecycle_loader_uses_hash_verified_warm_cache_without_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cold = LifecycleSession()
            first = load_lifecycle_snapshot(
                root,
                TARGET,
                session=cold,
                required_symbols_by_exchange={"TWSE": {"1459"}, "TPEx": set()},
                now=datetime.datetime(2026, 7, 30, 1, tzinfo=datetime.timezone.utc),
            )
            warm = LifecycleSession()
            second = load_lifecycle_snapshot(
                root,
                TARGET,
                session=warm,
                required_symbols_by_exchange={"TWSE": {"1459"}, "TPEx": set()},
                now=datetime.datetime(2026, 7, 30, 1, tzinfo=datetime.timezone.utc),
            )

        self.assertGreater(first.request_count, 0)
        self.assertEqual(second.request_count, 0)
        self.assertEqual(warm.calls, [])
        self.assertEqual(second.source_hashes, first.source_hashes)


if __name__ == "__main__":
    unittest.main()
