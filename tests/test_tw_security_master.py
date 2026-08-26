import datetime
import types
import unittest

from stock_papi.services.market import sector_signal_item
from stock_papi.services.observation_view import build_stock_observation
from stock_papi.integrations.market_data.tw_security_master import (
    NameChange,
    TaiwanSecurityMasterUnavailable,
    audit_taiwan_universe,
    build_taiwan_security_master,
    normalize_display_name,
)


class TaiwanSecurityMasterTests(unittest.TestCase):
    def setUp(self):
        self.master = build_taiwan_security_master(
            as_of=datetime.date(2026, 8, 26),
            twse_company_rows=[
                {"公司代號": "2330", "公司簡稱": "台積電", "上市日期": "891209"},
            ],
            twse_etf_rows=[
                {"基金代號": "00775B", "基金簡稱": "台新投等債15+", "上市日期": "1080215"},
                {"基金代號": "00904", "基金簡稱": "台新臺灣半導體30", "上市日期": "1110307"},
                {"基金代號": "009803", "基金簡稱": "玉山市值動能50", "上市日期": "1140313"},
                {"基金代號": "009805", "基金簡稱": "台新美國電力基建", "上市日期": "1140513"},
                {"基金代號": "009810", "基金簡稱": "玉山全球藍籌100", "上市日期": "1140716"},
            ],
            tpex_company_rows=[
                {"SecuritiesCompanyCode": "6241", "CompanyAbbreviation": "鑫永洋", "DateOfListing": "920408"},
                {"SecuritiesCompanyCode": "2740", "CompanyAbbreviation": "華軒", "DateOfListing": "1041224"},
                {"SecuritiesCompanyCode": "4530", "CompanyAbbreviation": "天意能創", "DateOfListing": "900430"},
                {"SecuritiesCompanyCode": "4953", "CompanyAbbreviation": "緯致", "DateOfListing": "1030108"},
                {"SecuritiesCompanyCode": "5381", "CompanyAbbreviation": "光譜", "DateOfListing": "880319"},
            ],
            tpex_quote_rows=[],
        )

    def test_all_reported_stale_names_resolve_from_official_master(self):
        expected = {
            "6241": "鑫永洋",
            "2740": "華軒",
            "4530": "天意能創",
            "4953": "緯致",
            "5381": "光譜",
            "00775B": "台新投等債15+",
            "00904": "台新臺灣半導體30",
            "009805": "台新美國電力基建",
            "009803": "玉山市值動能50",
            "009810": "玉山全球藍籌100",
        }
        for symbol, name in expected.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(self.master.resolve_name(symbol), name)

    def test_name_lifecycle_is_point_in_time_and_does_not_activate_future_rename(self):
        master = build_taiwan_security_master(
            as_of=datetime.date(2026, 8, 26),
            twse_company_rows=[
                {"公司代號": "2330", "公司簡稱": "台積電", "上市日期": "891209"},
            ],
            twse_etf_rows=[],
            tpex_company_rows=[
                {"SecuritiesCompanyCode": "6241", "CompanyAbbreviation": "鑫永洋", "DateOfListing": "920408"},
            ],
            tpex_quote_rows=[],
            name_changes=[
                NameChange("6241", "易通展", "鑫永洋", datetime.date(2026, 8, 25), "fixture-current"),
                NameChange("2330", "台積電", "台積電未來名", datetime.date(2026, 9, 1), "fixture-future"),
            ],
        )

        self.assertEqual(master.resolve_name("6241", datetime.date(2026, 8, 24)), "易通展")
        self.assertEqual(master.resolve_name("6241", datetime.date(2026, 8, 25)), "鑫永洋")
        self.assertEqual(master.resolve_name("6241", datetime.date(2026, 8, 26)), "鑫永洋")
        self.assertEqual(master.resolve_name("2330", datetime.date(2026, 8, 31)), "台積電")
        self.assertEqual(master.resolve_name("2330", datetime.date(2026, 9, 1)), "台積電未來名")
        self.assertEqual(master.entries["6241"].listed_date, datetime.date(2003, 4, 8))

    def test_normal_name_and_official_status_marker_are_stable(self):
        self.assertEqual(normalize_display_name("華義*"), "華義")
        self.assertEqual(self.master.resolve_name("2330"), "台積電")

    def test_audit_distinguishes_name_market_listing_and_delisted_findings(self):
        runtime = {
            "6241": types.SimpleNamespace(name="易通展", data_source="tpex", group="半導體業", type="股票"),
            "2330": types.SimpleNamespace(name="台積電", data_source="twse", group="半導體業", type="股票"),
            "1589": types.SimpleNamespace(name="已下市", data_source="twse", group="其他", type="股票"),
        }
        result = audit_taiwan_universe(
            runtime,
            self.master,
            configured_symbols={"6241", "2330", "1589"},
            runtime_snapshot_date=datetime.date(2026, 8, 25),
        )
        self.assertEqual(result["NAME_MISMATCH"]["6241"]["runtime_name"], "易通展")
        self.assertEqual(result["MISSING_NEW_LISTING"], {})
        self.assertIn("1589", result["STALE_DELISTED"])

    def test_unavailable_authoritative_source_is_explicit(self):
        self.assertTrue(issubclass(TaiwanSecurityMasterUnavailable, RuntimeError))

    def test_market_signal_does_not_reuse_an_artifact_name_for_taiwan_symbol(self):
        data = {
            "as_of": "2026-08-26",
            "name": "易通展",
            "prob": 50,
            "bt": {},
            "foreign_flow": {},
        }
        self.assertEqual(
            sector_signal_item(
                "6241",
                data,
                get_stock_name=lambda _symbol: "鑫永洋",
            )["name"],
            "鑫永洋",
        )

    def test_papi_service_receives_the_canonical_registry(self):
        from stock_papi import application

        class StubResolver:
            def __init__(self, entries):
                self.entries = entries

            def get_master(self):
                return types.SimpleNamespace(entries=self.entries)

            def contains(self, _symbol):
                return True

        entries = {"6241": types.SimpleNamespace(name="鑫永洋")}
        original = application.taiwan_security_master
        application.taiwan_security_master = StubResolver(entries)
        try:
            service = application._papi_service()
        finally:
            application.taiwan_security_master = original

        self.assertIs(service.twstock_codes, entries)

    def test_public_observation_does_not_reuse_a_stale_taiwan_artifact_name(self):
        snapshot = {
            "schema_version": 1,
            "market": "TW",
            "symbol": "6241",
            "name": "易通展",
            "as_of": "2026-08-26",
            "daily": [{"Date": "2026-08-26", "Close": 100.0}],
        }

        self.assertEqual(
            build_stock_observation(
                snapshot,
                get_stock_name=lambda symbol, **_kwargs: "鑫永洋"
                if symbol == "6241"
                else symbol,
            )["name"],
            "鑫永洋",
        )


if __name__ == "__main__":
    unittest.main()
