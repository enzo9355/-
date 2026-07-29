import datetime
import json
import tempfile
import unittest
from pathlib import Path

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceFailure,
    TPEX_INSTITUTIONAL_FIELDS,
    normalize_market_date,
    normalize_symbol,
    parse_number,
    parse_tpex_institutional,
    parse_twse_institutional,
    plan_official_request_budget,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_raw_source,
    load_cached_source,
    store_cached_raw_source,
    store_cached_source,
)

TARGET = datetime.date(2026, 7, 24)

TWSE_FIELDS = [
    "證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)",
    "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
    "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)",
    "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數",
]


def twse_payload():
    return {
        "stat": "OK",
        "date": "20260724",
        "fields": TWSE_FIELDS,
        "data": [[
            "2330", "台積電", "100", "40", "60", "10", "5", "5",
            "30", "10", "20", "9", "5", "2", "3", "8", "2", "6", "95",
        ]],
    }


def tpex_payload():
    return {
        "stat": "ok",
        "date": "20260724",
        "tables": [{
            "title": "三大法人買賣明細資訊",
            "columnNum": 25,
            "date": "115/07/24",
            "fields": list(TPEX_INSTITUTIONAL_FIELDS),
            "data": [["6488", "環球晶"] + [str(index) for index in range(2, 24)]],
        }],
    }


class TWOfficialContractTests(unittest.TestCase):
    def test_normalizers_are_strict(self):
        self.assertEqual(normalize_symbol(" 2330 "), "2330")
        self.assertEqual(normalize_market_date("1150724"), TARGET)
        self.assertEqual(normalize_market_date("115/07/24"), TARGET)
        self.assertEqual(normalize_market_date("20260724"), TARGET)
        self.assertEqual(parse_number("1,234"), 1234.0)
        self.assertEqual(parse_number("(25)"), -25.0)
        with self.assertRaises(ValueError):
            normalize_symbol("TOTAL")
        with self.assertRaises(ValueError):
            parse_number("--")

    def test_request_budget_keeps_finmind_fallback_disabled_and_bounded(self):
        budget = plan_official_request_budget(
            cold_source_count=6,
            retry_attempts=2,
        )
        self.assertEqual(budget.planned_minimum_requests, 6)
        self.assertEqual(budget.planned_worst_case_requests, 12)
        self.assertEqual(budget.finmind_requests, 0)
        self.assertTrue(budget.capacity_proven)
        disabled = plan_official_request_budget(
            cold_source_count=6,
            fallback_symbols=1,
            fallback_enabled=False,
        )
        self.assertFalse(disabled.capacity_proven)
        self.assertEqual(disabled.reason, "fallback_disabled")
        exceeded = plan_official_request_budget(
            cold_source_count=6,
            fallback_symbols=7,
            fallback_enabled=True,
        )
        self.assertFalse(exceeded.capacity_proven)
        self.assertEqual(exceeded.reason, "fallback_budget_exceeded")

    def test_structured_failure_retains_no_response_or_credentials(self):
        failure = OfficialSourceFailure(
            "twse_price",
            "http_error",
            http_status=503,
            retryable=True,
            safe_message="HTTP 503",
        )
        self.assertEqual(failure.source_id, "twse_price")
        self.assertEqual(failure.http_status, 503)
        self.assertFalse(hasattr(failure, "response_body"))
        self.assertFalse(hasattr(failure, "authorization"))


class TWOfficialInstitutionalParserTests(unittest.TestCase):
    def test_twse_aggregates_foreign_and_dealer_components(self):
        rows = parse_twse_institutional(twse_payload(), TARGET)
        self.assertEqual({row["name"] for row in rows}, {
            "Foreign", "InvestmentTrust", "Dealer",
        })
        foreign = next(row for row in rows if row["name"] == "Foreign")
        dealer = next(row for row in rows if row["name"] == "Dealer")
        self.assertEqual((foreign["buy"], foreign["sell"]), (110.0, 45.0))
        self.assertEqual((dealer["buy"], dealer["sell"]), (13.0, 4.0))

    def test_tpex_uses_verified_24_column_contract(self):
        rows = parse_tpex_institutional(tpex_payload(), TARGET)
        self.assertEqual(len(rows), 3)
        foreign = next(row for row in rows if row["name"] == "Foreign")
        dealer = next(row for row in rows if row["name"] == "Dealer")
        self.assertEqual((foreign["buy"], foreign["sell"]), (8.0, 9.0))
        self.assertEqual((dealer["buy"], dealer["sell"]), (20.0, 21.0))

    def test_tpex_reordered_or_unlabelled_schema_fails_closed(self):
        payload = tpex_payload()
        payload["tables"][0]["fields"][2], payload["tables"][0]["fields"][3] = (
            payload["tables"][0]["fields"][3],
            payload["tables"][0]["fields"][2],
        )
        with self.assertRaisesRegex(ValueError, "schema fingerprint"):
            parse_tpex_institutional(payload, TARGET)
        payload = tpex_payload()
        payload["tables"][0]["title"] = "unknown"
        with self.assertRaisesRegex(ValueError, "schema fingerprint"):
            parse_tpex_institutional(payload, TARGET)

    def test_date_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_twse_institutional(
                twse_payload(), TARGET - datetime.timedelta(days=1)
            )
        with self.assertRaises(ValueError):
            parse_tpex_institutional(
                tpex_payload(), TARGET - datetime.timedelta(days=1)
            )


class TWOfficialCacheTests(unittest.TestCase):
    def test_cache_round_trip_is_hash_verified_and_secret_free(self):
        rows = parse_twse_institutional(twse_payload(), TARGET)
        with tempfile.TemporaryDirectory() as temporary:
            entry = store_cached_source(
                Path(temporary),
                source_id="twse_institutional",
                target_date=TARGET,
                rows=rows,
                symbol_count=1,
                parser_version="v1",
                source_url="https://example.test/report?token=secret",
                fetched_at=datetime.datetime(
                    2026, 7, 24, tzinfo=datetime.timezone.utc
                ),
            )
            metadata = entry.metadata_path.read_text(encoding="utf-8")
            self.assertNotIn("secret", metadata)
            loaded = load_cached_source(
                Path(temporary),
                source_id="twse_institutional",
                target_date=TARGET,
                parser_version="v1",
            )
            self.assertEqual(loaded.rows, rows)
            entry.payload_path.write_bytes(entry.payload_path.read_bytes() + b"x")
            with self.assertRaises(OfficialCacheError):
                load_cached_source(
                    Path(temporary),
                    source_id="twse_institutional",
                    target_date=TARGET,
                    parser_version="v1",
                )

    def test_raw_price_cache_is_content_addressed_and_hash_verified(self):
        payload = json.dumps(
            {"stat": "OK", "date": "20260724", "data": [["2330"]]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            entry = store_cached_raw_source(
                Path(temporary),
                source_id="twse_price",
                target_date=TARGET,
                payload=payload,
                parser_version="raw-v1",
                source_url="https://example.test/report?token=secret",
                fetched_at=datetime.datetime(
                    2026, 7, 24, tzinfo=datetime.timezone.utc
                ),
            )

            self.assertIn(entry.payload_sha256, entry.payload_path.name)
            self.assertEqual(entry.payload_path.parent.name, "objects")
            self.assertNotIn(
                "secret", entry.metadata_path.read_text(encoding="utf-8")
            )
            loaded = load_cached_raw_source(
                Path(temporary),
                source_id="twse_price",
                target_date=TARGET,
                parser_version="raw-v1",
            )
            self.assertEqual(loaded.payload, payload)
            entry.payload_path.write_bytes(entry.payload_path.read_bytes() + b"x")
            with self.assertRaises(OfficialCacheError):
                load_cached_raw_source(
                    Path(temporary),
                    source_id="twse_price",
                    target_date=TARGET,
                    parser_version="raw-v1",
                )


if __name__ == "__main__":
    unittest.main()
