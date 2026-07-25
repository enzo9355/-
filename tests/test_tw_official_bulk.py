import datetime
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceFailure,
    SOURCE_DEFINITIONS,
    build_official_daily_snapshot,
    normalize_market_date,
    normalize_symbol,
    parse_number,
    parse_tpex_institutional,
    parse_tpex_margin,
    parse_tpex_price,
    parse_twse_institutional,
    parse_twse_margin,
    parse_twse_price,
    plan_official_request_budget,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_source,
    store_cached_source,
)

TARGET = datetime.date(2026, 7, 24)


class Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.headers = {"Content-Length": str(len(self.content))}


class Session:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        source = next(item for item, definition in SOURCE_DEFINITIONS.items() if definition.url == url)
        payload = self.payloads[source]
        if isinstance(payload, Response):
            return payload
        return Response(payload)


def fixtures():
    fields = [
        "證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)",
        "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
        "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
        "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數",
        "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
        "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)",
        "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數",
    ]
    tpex_row = ["6488", "環球晶"] + [str(i) for i in range(2, 24)]
    return {
        "twse_price": [
            {"Code": "2330", "Date": "1150724", "OpeningPrice": "1,100", "HighestPrice": "1,120", "LowestPrice": "1,090", "ClosingPrice": "1,110", "TradeVolume": "1000"},
            {"Code": "2303", "Date": "1150724", "OpeningPrice": "50", "HighestPrice": "51", "LowestPrice": "49", "ClosingPrice": "50", "TradeVolume": "2000"},
        ],
        "twse_institutional": {
            "stat": "OK", "date": "20260724", "fields": fields,
            "data": [["2330", "台積電", "100", "40", "60", "10", "5", "5", "30", "10", "20", "9", "5", "2", "3", "8", "2", "6", "95"]],
        },
        "twse_margin": [{"股票代號": "2330", "股票名稱": "台積電", "融資今日餘額": "5000", "融券今日餘額": "200"}],
        "tpex_price": [
            {"SecuritiesCompanyCode": "6488", "Date": "1150724", "Open": "400", "High": "410", "Low": "395", "Close": "405", "TradingShares": "3000"},
            {"SecuritiesCompanyCode": "8069", "Date": "1150724", "Open": "100", "High": "102", "Low": "99", "Close": "101", "TradingShares": "4000"},
        ],
        "tpex_institutional": {"stat": "OK", "date": "20260724", "tables": [{"date": "115/07/24", "fields": [f"f{i}" for i in range(24)], "data": [tpex_row]}]},
        "tpex_margin": [{"SecuritiesCompanyCode": "6488", "Date": "1150724", "MarginPurchaseBalance": "1000", "ShortSaleBalance": "50"}],
    }


class TWOfficialContractTests(unittest.TestCase):
    def test_normalizers_and_exact_sources(self):
        self.assertEqual(normalize_symbol(" 2330 "), "2330")
        self.assertEqual(normalize_market_date("1150724"), TARGET)
        self.assertEqual(normalize_market_date("115/07/24"), TARGET)
        self.assertEqual(normalize_market_date("20260724"), TARGET)
        self.assertEqual(parse_number("1,234"), 1234.0)
        self.assertEqual(tuple(SOURCE_DEFINITIONS), (
            "twse_price", "twse_institutional", "twse_margin",
            "tpex_price", "tpex_institutional", "tpex_margin",
        ))
        with self.assertRaises(ValueError):
            normalize_symbol("TOTAL")
        with self.assertRaises(ValueError):
            parse_number("--")

    def test_request_budget_is_bounded(self):
        budget = plan_official_request_budget(cold_source_count=6, retry_attempts=2)
        self.assertEqual(budget.planned_minimum_requests, 6)
        self.assertEqual(budget.planned_worst_case_requests, 12)
        self.assertTrue(budget.capacity_proven)
        blocked = plan_official_request_budget(cold_source_count=6, fallback_symbols=7, fallback_enabled=True)
        self.assertFalse(blocked.capacity_proven)


class TWOfficialParserTests(unittest.TestCase):
    def test_all_six_parsers_map_to_canonical_contract(self):
        data = fixtures()
        self.assertEqual(len(parse_twse_price(data["twse_price"], TARGET)), 2)
        institutional = parse_twse_institutional(data["twse_institutional"], TARGET)
        self.assertEqual({row["name"] for row in institutional}, {"Foreign", "InvestmentTrust", "Dealer"})
        foreign = next(row for row in institutional if row["name"] == "Foreign")
        self.assertEqual((foreign["buy"], foreign["sell"]), (110.0, 45.0))
        dealer = next(row for row in institutional if row["name"] == "Dealer")
        self.assertEqual((dealer["buy"], dealer["sell"]), (13.0, 4.0))
        self.assertEqual(parse_twse_margin(data["twse_margin"], TARGET)[0]["MarginPurchaseTodayBalance"], 5000.0)
        self.assertEqual(len(parse_tpex_price(data["tpex_price"], TARGET)), 2)
        tpex_inst = parse_tpex_institutional(data["tpex_institutional"], TARGET)
        self.assertEqual(len(tpex_inst), 3)
        self.assertEqual(next(row for row in tpex_inst if row["name"] == "Foreign")["buy"], 8.0)
        self.assertEqual(parse_tpex_margin(data["tpex_margin"], TARGET)[0]["ShortSaleTodayBalance"], 50.0)

    def test_wrong_date_and_conflicting_duplicate_fail_closed(self):
        rows = fixtures()["twse_price"]
        bad = list(rows)
        bad[0] = dict(bad[0], Date="1150723")
        with self.assertRaises(ValueError):
            parse_twse_price(bad, TARGET)
        duplicate = rows + [dict(rows[0], ClosingPrice="1,109")]
        with self.assertRaises(ValueError):
            parse_twse_price(duplicate, TARGET)


class TWOfficialCacheTests(unittest.TestCase):
    def test_cache_round_trip_and_hash_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = parse_twse_price(fixtures()["twse_price"], TARGET)
            entry = store_cached_source(
                root, source_id="twse_price", target_date=TARGET, rows=rows,
                symbol_count=2, parser_version="v1", source_url="https://example.test/x?token=secret",
                fetched_at=datetime.datetime(2026, 7, 24, tzinfo=datetime.timezone.utc),
            )
            metadata = entry.metadata_path.read_text(encoding="utf-8")
            self.assertNotIn("secret", metadata)
            loaded = load_cached_source(root, source_id="twse_price", target_date=TARGET, parser_version="v1")
            self.assertEqual(loaded.rows, rows)
            entry.payload_path.write_bytes(entry.payload_path.read_bytes() + b"x")
            with self.assertRaises(OfficialCacheError):
                load_cached_source(root, source_id="twse_price", target_date=TARGET, parser_version="v1")


class TWOfficialOrchestratorTests(unittest.TestCase):
    def test_cold_cache_uses_six_requests_and_warm_cache_uses_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Session(fixtures())
            snapshot = build_official_daily_snapshot(
                Path(temporary), TARGET, session=session,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2}, minimum_chip_symbols=1,
            )
            self.assertEqual(len(session.calls), 6)
            self.assertEqual(snapshot.request_count, 6)
            warm = Session(fixtures())
            second = build_official_daily_snapshot(
                Path(temporary), TARGET, session=warm,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2}, minimum_chip_symbols=1,
            )
            self.assertEqual(len(warm.calls), 0)
            self.assertEqual(second.request_count, 0)
            self.assertEqual(second.manifest_sha256, snapshot.manifest_sha256)

    def test_one_source_failure_returns_no_snapshot(self):
        data = fixtures()
        data["tpex_margin"] = Response({}, status=503)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(OfficialSourceFailure):
            build_official_daily_snapshot(
                Path(temporary), TARGET, session=Session(data), retry_attempts=1,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2}, minimum_chip_symbols=1,
            )


if __name__ == "__main__":
    unittest.main()
