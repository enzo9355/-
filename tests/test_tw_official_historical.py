import datetime
import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceFailure,
    TPEX_INSTITUTIONAL_FIELDS,
    parse_tpex_institutional,
)
from stock_papi.integrations.market_data.tw_official_historical import (
    HISTORICAL_SOURCE_DEFINITIONS,
    MAX_CATCHUP_SESSIONS,
    _params,
    build_historical_daily_snapshot,
    build_official_snapshot_series,
    parse_tpex_margin_report,
    parse_tpex_price_report,
    parse_tpex_price_report_with_status,
    parse_twse_margin_report,
    parse_twse_price_report,
    parse_twse_price_report_with_status,
)
from stock_papi.integrations.market_data.tw_trading_status import (
    LifecycleSnapshot,
    evidence_sha256,
)

TARGET = datetime.date(2026, 7, 24)
CONTRACT_TARGET = datetime.date(2026, 7, 16)


TWSE_T86_FIELDS = [
    "證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)",
    "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
    "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)",
    "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數",
]


def ymd(value):
    return value.strftime("%Y%m%d")


def roc(value):
    return f"{value.year - 1911:03d}/{value.month:02d}/{value.day:02d}"


def payloads(value):
    date_text = ymd(value)
    roc_text = roc(value)
    twse_price_fields = [
        "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
        "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)",
        "漲跌價差", "最後揭示買價", "最後揭示買量", "最後揭示賣價",
        "最後揭示賣量", "本益比",
    ]
    twse_margin_fields = [
        "代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額",
        "次一營業日限額", "買進", "賣出", "現券償還", "前日餘額", "今日餘額",
        "次一營業日限額", "資券互抵", "註記",
    ]
    tpex_price_fields = [
        "代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低", "均價",
        "成交股數", "成交金額(元)", "成交筆數", "最後買價", "最後買量(張數)",
        "最後賣價", "最後賣量(張數)", "發行股數", "次日 參考價", "次日 漲停價",
        "次日 跌停價",
    ]
    tpex_margin_fields = [
        "代號", "名稱", "前資餘額(張)", "資買", "資賣", "現償", "資餘額",
        "資屬證金", "資使用率(%)", "資限額", "前券餘額(張)", "券賣", "券買",
        "券償", "券餘額", "券屬證金", "券使用率(%)", "券限額", "資券相抵(張)",
        "備註",
    ]
    tpex_institutional = ["6488", "環球晶"] + [str(index) for index in range(2, 24)]
    return {
        "twse_price": {
            "stat": "OK", "date": date_text,
            "tables": [
                {"fields": ["指數"], "data": [["TAIEX"]], "title": "指數"},
                {
                    "fields": twse_price_fields,
                    "data": [
                        ["2330", "台積電", "1000", "1", "100", "1100", "1120", "1090", "1110", "+", "10", "1109", "1", "1110", "1", "20"],
                        ["2303", "聯電", "2000", "2", "200", "50", "51", "49", "50", "+", "0", "49.9", "1", "50", "1", "10"],
                    ],
                    "title": f"{roc_text} 每日收盤行情(全部)",
                },
            ],
        },
        "twse_institutional": {
            "stat": "OK", "date": date_text, "fields": TWSE_T86_FIELDS,
            "data": [["2330", "台積電", "100", "40", "60", "10", "5", "5", "30", "10", "20", "9", "5", "2", "3", "8", "2", "6", "95"]],
        },
        "twse_margin": {
            "stat": "OK", "date": date_text,
            "tables": [
                {},
                {
                    "fields": twse_margin_fields,
                    "data": [["2330", "台積電", "1", "2", "0", "4999", "5000", "9999", "1", "2", "0", "199", "200", "999", "0", ""]],
                    "title": f"{roc_text} 融資融券彙總 (股票)",
                },
            ],
        },
        "tpex_price": {
            "stat": "ok", "date": date_text,
            "tables": [{
                "date": roc_text, "fields": tpex_price_fields,
                "data": [
                    ["6488", "環球晶", "405", "+5", "400", "410", "395", "402", "3000", "1", "1", "404", "1", "405", "1", "1", "405", "445", "365"],
                    ["8069", "元太", "101", "+1", "100", "102", "99", "100", "4000", "1", "1", "100", "1", "101", "1", "1", "101", "111", "91"],
                ],
                "title": "上櫃股票行情",
            }],
        },
        "tpex_institutional": {
            "stat": "ok", "date": date_text,
            "tables": [{
    "title": "三大法人買賣明細資訊",
    "columnNum": 25,
    "date": roc_text,
    "fields": list(TPEX_INSTITUTIONAL_FIELDS),
    "data": [tpex_institutional],
}],
        },
        "tpex_margin": {
            "stat": "ok", "date": date_text,
            "tables": [{
                "date": roc_text, "fields": tpex_margin_fields,
                "data": [["6488", "環球晶", "999", "2", "1", "0", "1000", "0", "10", "9999", "49", "2", "1", "0", "50", "0", "1", "999", "0", ""]],
                "title": "上櫃股票融資融券餘額",
            }],
        },
    }


class Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.headers = {"Content-Length": str(len(self.content))}


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers, **_kwargs):
        source_id = next(
            key for key, definition in HISTORICAL_SOURCE_DEFINITIONS.items()
            if definition.url == url
        )
        if source_id.startswith("twse"):
            value = datetime.datetime.strptime(params["date"], "%Y%m%d").date()
        elif source_id in {"tpex_price", "tpex_margin"}:
            value = datetime.datetime.strptime(params["date"], "%Y/%m/%d").date()
        else:
            year, month, day = map(int, params["d"].split("/"))
            value = datetime.date(year + 1911, month, day)
        self.calls.append({
            "source_id": source_id,
            "url": url,
            "date": value,
            "params": dict(params),
            "headers": dict(headers),
        })
        return Response(payloads(value)[source_id])


class BlankPriceSession(Session):
    def get(self, url, *, params, headers, **kwargs):
        response = super().get(url, params=params, headers=headers, **kwargs)
        if self.calls[-1]["source_id"] == "tpex_price":
            payload = json.loads(response.content.decode("utf-8"))
            payload["tables"][0]["data"].append(
                ["00886", "永豐美國科技", "---", "---", "---", "---", "---", "44.11", "435", "19188", "17", "43.60", "1", "43.92", "1", "4451000", "43.92", "9999.95", "0.01"]
            )
            return Response(payload)
        return response


def lifecycle_snapshot(*, statuses=None, terminated=None):
    return LifecycleSnapshot(
        target_date=CONTRACT_TARGET,
        status_by_symbol=MappingProxyType(dict(statuses or {})),
        terminated_by_symbol=MappingProxyType(dict(terminated or {})),
        source_hashes=MappingProxyType({"tpex_current_mode": "d" * 64}),
        request_count=0,
    )


def suspended_status(symbol, exchange="TPEx"):
    document = {
        "schema_version": 1,
        "status": "officially_suspended",
        "market": "TW",
        "exchange": exchange,
        "symbol": symbol,
        "target_market_date": CONTRACT_TARGET.isoformat(),
        "valid_from": CONTRACT_TARGET.isoformat(),
        "valid_through_exclusive": None,
        "evaluated_through": CONTRACT_TARGET.isoformat(),
        "lifecycle_events": [],
        "parser_version": "tw-lifecycle-parser-v2",
    }
    document["evidence_sha256"] = evidence_sha256(document)
    return document


class HistoricalParserTests(unittest.TestCase):
    def test_tpex_blank_row_is_preserved_as_hash_bound_status(self):
        data = payloads(TARGET)["tpex_price"]
        data["tables"][0]["data"].append(
            ["00886", "永豐美國科技", "---", "---", "---", "---", "---", "44.11", "435", "19188", "17", "43.60", "1", "43.92", "1", "4451000", "43.92", "9999.95", "0.01"]
        )

        prices, statuses = parse_tpex_price_report_with_status(
            data, TARGET, "a" * 64
        )

        self.assertNotIn("00886", {row["stock_id"] for row in prices})
        self.assertEqual(statuses["00886"]["status"], "official_no_regular_trade")
        self.assertEqual(statuses["00886"]["raw_fields"]["volume"], "435")

    def test_twse_blank_row_is_preserved_as_hash_bound_status(self):
        data = payloads(TARGET)["twse_price"]
        data["tables"][1]["data"].append(
            ["1213", "大飲", "0", "0", "0", "--", "--", "--", "--", "", "0", "7.10", "3", "7.25", "1", "0"]
        )

        prices, statuses = parse_twse_price_report_with_status(
            data, TARGET, "b" * 64
        )

        self.assertNotIn("1213", {row["stock_id"] for row in prices})
        self.assertEqual(statuses["1213"]["status"], "official_no_regular_trade")

    def test_twse_price_and_margin_tables_map_exact_indices(self):
        data = payloads(TARGET)
        twse_price = parse_twse_price_report(data["twse_price"], TARGET)
        self.assertEqual(len(twse_price), 2)
        self.assertEqual(twse_price[1]["Trading_Volume"], 1000.0)
        twse_margin = parse_twse_margin_report(data["twse_margin"], TARGET)
        self.assertEqual(twse_margin[0]["MarginPurchaseTodayBalance"], 5000.0)
        self.assertEqual(twse_margin[0]["ShortSaleTodayBalance"], 200.0)

    def test_twse_price_and_margin_reject_mismatched_target_date(self):
        data = payloads(TARGET)
        for name, parser in (
            ("twse_price", parse_twse_price_report),
            ("twse_margin", parse_twse_margin_report),
        ):
            with self.subTest(source=name), self.assertRaises(ValueError):
                parser(data[name], TARGET - datetime.timedelta(days=1))

    def test_sanitized_modern_price_and_margin_reports_canonicalize_exact_values(self):
        data = payloads(CONTRACT_TARGET)
        self.assertEqual(parse_tpex_price_report(data["tpex_price"], CONTRACT_TARGET), (
            {"date": "2026-07-16", "stock_id": "6488", "open": 400.0, "max": 410.0,
             "min": 395.0, "close": 405.0, "Trading_Volume": 3000.0},
            {"date": "2026-07-16", "stock_id": "8069", "open": 100.0, "max": 102.0,
             "min": 99.0, "close": 101.0, "Trading_Volume": 4000.0},
        ))
        self.assertEqual(parse_tpex_margin_report(data["tpex_margin"], CONTRACT_TARGET), (
            {"date": "2026-07-16", "stock_id": "6488",
             "MarginPurchaseTodayBalance": 1000.0, "ShortSaleTodayBalance": 50.0},
        ))

    def test_tpex_price_rejects_mismatched_top_level_date(self):
        data = payloads(CONTRACT_TARGET)["tpex_price"]
        data["date"] = "20260724"
        with self.assertRaises(ValueError):
            parse_tpex_price_report(data, CONTRACT_TARGET)

    def test_tpex_price_rejects_mismatched_table_date(self):
        data = payloads(CONTRACT_TARGET)["tpex_price"]
        data["tables"][0]["date"] = "115/07/24"
        with self.assertRaises(ValueError):
            parse_tpex_price_report(data, CONTRACT_TARGET)

    def test_tpex_margin_rejects_mismatched_top_level_date(self):
        data = payloads(CONTRACT_TARGET)["tpex_margin"]
        data["date"] = "20260724"
        with self.assertRaises(ValueError):
            parse_tpex_margin_report(data, CONTRACT_TARGET)

    def test_tpex_margin_rejects_mismatched_table_date(self):
        data = payloads(CONTRACT_TARGET)["tpex_margin"]
        data["tables"][0]["date"] = "115/07/24"
        with self.assertRaises(ValueError):
            parse_tpex_margin_report(data, CONTRACT_TARGET)

    def test_tpex_institutional_report_canonicalizes_exact_rows(self):
        data = payloads(CONTRACT_TARGET)
        self.assertEqual(parse_tpex_institutional(data["tpex_institutional"], CONTRACT_TARGET), (
            {"date": "2026-07-16", "stock_id": "6488", "name": "Dealer", "buy": 20.0, "sell": 21.0},
            {"date": "2026-07-16", "stock_id": "6488", "name": "Foreign", "buy": 8.0, "sell": 9.0},
            {"date": "2026-07-16", "stock_id": "6488", "name": "InvestmentTrust", "buy": 11.0, "sell": 12.0},
        ))


class HistoricalRequestContractTests(unittest.TestCase):
    def test_tpex_price_contract_is_modern(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_price"].url,
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
        )
        self.assertEqual(
            _params("tpex_price", CONTRACT_TARGET),
            {"date": "2026/07/16", "response": "json"},
        )

    def test_tpex_margin_contract_is_modern(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_margin"].url,
            "https://www.tpex.org.tw/www/zh-tw/margin/balance",
        )
        self.assertEqual(
            _params("tpex_margin", CONTRACT_TARGET),
            {"date": "2026/07/16", "response": "json"},
        )

    def test_tpex_institutional_contract_is_unchanged(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_institutional"].url,
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php",
        )
        self.assertEqual(_params("tpex_institutional", CONTRACT_TARGET), {
            "l": "zh-tw", "o": "json", "se": "EW", "t": "D",
            "d": "115/07/16", "s": "0,asc",
        })

    def test_request_headers_are_source_specific(self):
        session = Session()
        with tempfile.TemporaryDirectory() as temporary:
            build_historical_daily_snapshot(
                Path(temporary),
                CONTRACT_TARGET,
                session=session,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
            )

        self.assertEqual(
            {call["source_id"] for call in session.calls},
            set(HISTORICAL_SOURCE_DEFINITIONS),
        )
        for call in session.calls:
            source_id = call["source_id"]
            self.assertEqual(call["url"], HISTORICAL_SOURCE_DEFINITIONS[source_id].url)
            self.assertEqual(call["date"], CONTRACT_TARGET)
            self.assertEqual(call["params"], _params(source_id, CONTRACT_TARGET))
            expected_headers = {"User-Agent": "ABSORB/1.0"}
            if source_id == "tpex_price":
                expected_headers["X-Requested-With"] = "XMLHttpRequest"
            self.assertEqual(call["headers"], expected_headers)


class HistoricalSeriesTests(unittest.TestCase):
    def test_two_dates_use_twelve_cold_requests_then_zero_warm_requests(self):
        dates = (datetime.date(2026, 7, 23), TARGET)
        with tempfile.TemporaryDirectory() as temporary:
            session = Session()
            series = build_official_snapshot_series(
                Path(temporary), dates, session=session,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
            )
            self.assertEqual(len(session.calls), 12)
            self.assertEqual(series.request_count, 12)
            self.assertEqual(series.request_budget.planned_minimum_requests, 12)
            self.assertEqual(series.request_budget.planned_worst_case_requests, 24)
            self.assertEqual(series.dates, dates)
            warm = Session()
            second = build_official_snapshot_series(
                Path(temporary), dates, session=warm,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
            )
            self.assertEqual(warm.calls, [])
            self.assertEqual(second.request_count, 0)
            self.assertEqual(second.manifest_sha256, series.manifest_sha256)

    def test_default_chip_coverage_rejects_truncated_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OfficialSourceFailure) as context:
                build_historical_daily_snapshot(
                    Path(temporary),
                    TARGET,
                    session=Session(),
                    minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                )
        self.assertEqual(context.exception.source_id, "twse_institutional")
        self.assertEqual(context.exception.category, "schema_validation")

    def test_series_rejects_more_than_bounded_catchup(self):
        start = datetime.date(2026, 7, 1)
        dates = [start + datetime.timedelta(days=index) for index in range(MAX_CATCHUP_SESSIONS + 1)]
        with self.assertRaises(ValueError):
            build_official_snapshot_series(Path("x"), dates)


class HistoricalStatusSnapshotTests(unittest.TestCase):
    def test_status_aware_snapshot_skips_lifecycle_when_all_required_prices_exist(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "stock_papi.integrations.market_data.tw_official_historical.load_lifecycle_snapshot"
        ) as lifecycle:
            snapshot = build_historical_daily_snapshot(
                Path(temporary),
                CONTRACT_TARGET,
                session=Session(),
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
                required_symbols_by_exchange={"TWSE": {"2330"}, "TPEx": {"6488"}},
            )

        lifecycle.assert_not_called()
        self.assertEqual(dict(snapshot.trading_status_by_symbol), {})
        self.assertEqual(dict(snapshot.terminated_by_symbol), {})

    def test_blank_raw_row_becomes_status_and_never_a_price(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "stock_papi.integrations.market_data.tw_official_historical.load_lifecycle_snapshot",
            return_value=lifecycle_snapshot(),
        ):
            snapshot = build_historical_daily_snapshot(
                Path(temporary),
                CONTRACT_TARGET,
                session=BlankPriceSession(),
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
                required_symbols_by_exchange={"TWSE": {"2330"}, "TPEx": {"6488", "00886"}},
            )

        self.assertNotIn("00886", snapshot.price_by_symbol)
        self.assertEqual(
            snapshot.trading_status_by_symbol["00886"]["status"],
            "official_no_regular_trade",
        )
        self.assertEqual(
            snapshot.trading_status_by_symbol["00886"]["raw_fields"]["volume"],
            "435",
        )

    def test_absent_price_requires_covering_suspension_or_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "stock_papi.integrations.market_data.tw_official_historical.load_lifecycle_snapshot",
            return_value=lifecycle_snapshot(
                statuses={"4804": suspended_status("4804")}
            ),
        ):
            snapshot = build_historical_daily_snapshot(
                Path(temporary),
                CONTRACT_TARGET,
                session=Session(),
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
                required_symbols_by_exchange={"TWSE": {"2330"}, "TPEx": {"6488", "4804"}},
            )
        self.assertEqual(snapshot.trading_status_by_symbol["4804"]["status"], "officially_suspended")

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "stock_papi.integrations.market_data.tw_official_historical.load_lifecycle_snapshot",
            return_value=lifecycle_snapshot(),
        ), self.assertRaises(OfficialSourceFailure) as context:
            build_historical_daily_snapshot(
                Path(temporary),
                CONTRACT_TARGET,
                session=Session(),
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
                required_symbols_by_exchange={"TWSE": {"2330"}, "TPEx": {"6488", "4804"}},
            )
        self.assertEqual(context.exception.category, "unrecognized_missing_price")


if __name__ == "__main__":
    unittest.main()
