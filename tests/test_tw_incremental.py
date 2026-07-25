import datetime
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialDailySnapshot,
    OfficialRequestBudget,
)
from stock_papi.integrations.market_data.tw_official_historical import (
    OfficialSnapshotSeries,
)
from stock_papi.quant.tw_artifact_audit import audit_artifact_dates
from stock_papi.quant.tw_incremental import (
    IncrementalHistoryError,
    OfficialCompatFetcher,
    load_incremental_artifact,
)

TARGET = datetime.date(2026, 7, 24)


def snapshot_for(value, close, manifest_char):
    return OfficialDailySnapshot(
        target_date=value,
        price_by_symbol=MappingProxyType({"2330": MappingProxyType({
            "date": value.isoformat(), "stock_id": "2330",
            "open": close - 10.0, "max": close + 10.0,
            "min": close - 20.0, "close": close,
            "Trading_Volume": 1000.0,
        })}),
        institutional_by_symbol=MappingProxyType({"2330": (
            MappingProxyType({"date": value.isoformat(), "stock_id": "2330", "name": "Foreign", "buy": 100.0, "sell": 20.0}),
            MappingProxyType({"date": value.isoformat(), "stock_id": "2330", "name": "InvestmentTrust", "buy": 20.0, "sell": 10.0}),
            MappingProxyType({"date": value.isoformat(), "stock_id": "2330", "name": "Dealer", "buy": 5.0, "sell": 5.0}),
        )}),
        margin_by_symbol=MappingProxyType({"2330": MappingProxyType({
            "date": value.isoformat(), "stock_id": "2330",
            "MarginPurchaseTodayBalance": 5000.0,
            "ShortSaleTodayBalance": 200.0,
        })}),
        source_results=MappingProxyType({}),
        manifest_sha256=manifest_char * 64,
        request_count=6,
        request_budget=OfficialRequestBudget(6, 12, 6, 0, True, "capacity_proven"),
        source_mode="tw_official_bulk_v2",
        source_schema_version="tw-official-historical-v1",
    )


def snapshot():
    return snapshot_for(TARGET, 1110.0, "a")


def series():
    first_date = datetime.date(2026, 7, 23)
    first = snapshot_for(first_date, 1090.0, "b")
    second = snapshot()
    document = {
        "source_mode": "tw_official_bulk_v2",
        "target_date": TARGET.isoformat(),
        "snapshots": [
            {"date": first_date.isoformat(), "manifest_sha256": first.manifest_sha256},
            {"date": TARGET.isoformat(), "manifest_sha256": second.manifest_sha256},
        ],
    }
    digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    return OfficialSnapshotSeries(
        target_date=TARGET,
        snapshots=MappingProxyType({first_date: first, TARGET: second}),
        manifest_sha256=digest,
        request_count=12,
        request_budget=OfficialRequestBudget(12, 24, 12, 0, True, "capacity_proven"),
    )


def write_artifact(root, *, daily, as_of="2026-07-22", symbol="2330"):
    path = Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "market": "TW",
        "symbol": symbol,
        "as_of": as_of,
        "daily": daily,
    }
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(document).encode())
    return path


def history(date="2026-07-22T00:00:00.000"):
    return [{
        "Date": date,
        "Open": 1000.0,
        "High": 1020.0,
        "Low": 990.0,
        "Close": 1010.0,
        "Volume": 900.0,
        "InstitutionalNet": 60.0,
        "ForeignNet": 50.0,
        "MarginBalance": 4900.0,
        "ShortBalance": 180.0,
    }]


class TWOfficialIncrementalTests(unittest.TestCase):
    def test_serves_history_plus_target_without_finmind(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=history())
            fetcher = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            price = fetcher("TaiwanStockPrice", "2330", "2026-07-01", "2026-07-24")
            self.assertEqual(list(price["date"]), ["2026-07-22", "2026-07-24"])
            institutional = fetcher(
                "TaiwanStockInstitutionalInvestorsBuySell",
                "2330", "2026-07-01", "2026-07-24",
            )
            self.assertEqual(len(institutional), 6)
            old_foreign = institutional[
                (institutional.date == "2026-07-22")
                & (institutional.name == "Foreign")
            ].iloc[0]
            self.assertEqual(old_foreign.buy - old_foreign.sell, 50)
            margin = fetcher(
                "TaiwanStockMarginPurchaseShortSale",
                "2330", "2026-07-01", "2026-07-24",
            )
            self.assertEqual(list(margin["MarginPurchaseTodayBalance"]), [4900.0, 5000.0])
            lineage = fetcher.lineage_for("2330")
            self.assertEqual(lineage["official_series_manifest_sha256"], "a" * 64)
            self.assertNotIn("token", json.dumps(lineage).lower())

    def test_snapshot_series_fills_each_missing_trading_session_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=history())
            fetcher = OfficialCompatFetcher(Path(temporary), series(), pd=pd)
            price = fetcher("TaiwanStockPrice", "2330", "2026-07-01", "2026-07-24")
            self.assertEqual(
                list(price["date"]),
                ["2026-07-22", "2026-07-23", "2026-07-24"],
            )
            self.assertEqual(list(price["close"]), [1010.0, 1090.0, 1110.0])
            lineage = fetcher.lineage_for("2330")
            self.assertEqual(
                lineage["official_snapshot_dates"],
                ["2026-07-23", "2026-07-24"],
            )
            self.assertEqual(len(lineage["official_snapshot_manifests"]), 2)

    def test_existing_exact_intermediate_date_is_verified_not_duplicated(self):
        existing = history() + [{
            "Date": "2026-07-23T00:00:00.000",
            "Open": 1080.0, "High": 1100.0, "Low": 1070.0,
            "Close": 1090.0, "Volume": 1000.0,
            "InstitutionalNet": 90.0, "ForeignNet": 80.0,
            "MarginBalance": 5000.0, "ShortBalance": 200.0,
        }]
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=existing, as_of="2026-07-23")
            fetcher = OfficialCompatFetcher(Path(temporary), series(), pd=pd)
            price = fetcher("TaiwanStockPrice", "2330", "2026-07-01", "2026-07-24")
            self.assertEqual(
                list(price["date"]),
                ["2026-07-22", "2026-07-23", "2026-07-24"],
            )

    def test_missing_or_future_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fetcher = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            with self.assertRaises(IncrementalHistoryError):
                fetcher("TaiwanStockPrice", "2330", "2026-07-01", "2026-07-24")
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=history("2026-07-25T00:00:00.000"),
                as_of="2026-07-25",
            )
            with self.assertRaises(IncrementalHistoryError):
                audit_artifact_dates(Path(temporary), ["2330"], target_date=TARGET)

    def test_artifact_declared_as_of_must_match_latest_daily_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=history(), as_of="2026-07-23")
            with self.assertRaises(IncrementalHistoryError):
                load_incremental_artifact(Path(temporary), "2330")

    def test_audit_records_latest_dates_and_unavailable_symbols(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=history(), symbol="2330")
            audit = audit_artifact_dates(
                Path(temporary), ["2330", "2303"], target_date=TARGET
            )
            self.assertEqual(
                audit.latest_by_symbol["2330"],
                datetime.date(2026, 7, 22),
            )
            self.assertEqual(audit.unavailable_symbols, ("2303",))
            self.assertEqual(audit.latest_date_counts, {"2026-07-22": 1})

    def test_same_date_mismatch_fails_and_match_is_not_duplicated(self):
        matching = [{
            "Date": "2026-07-24T00:00:00.000",
            "Open": 1100.0, "High": 1120.0,
            "Low": 1090.0, "Close": 1110.0, "Volume": 1000.0,
            "InstitutionalNet": 90.0, "ForeignNet": 80.0,
            "MarginBalance": 5000.0, "ShortBalance": 200.0,
        }]
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(temporary, daily=matching, as_of="2026-07-24")
            fetcher = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            price = fetcher("TaiwanStockPrice", "2330", "2026-07-24", "2026-07-24")
            self.assertEqual(len(price), 1)
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=[dict(matching[0], Close=1109.0)],
                as_of="2026-07-24",
            )
            fetcher = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            with self.assertRaises(IncrementalHistoryError):
                fetcher("TaiwanStockPrice", "2330", "2026-07-24", "2026-07-24")


if __name__ == "__main__":
    unittest.main()
