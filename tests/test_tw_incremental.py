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
_MISSING = object()


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
        source_schema_version="tw-official-historical-v2",
    )


def snapshot():
    return snapshot_for(TARGET, 1110.0, "a")


def series():
    first_date = datetime.date(2026, 7, 23)
    first = snapshot_for(first_date, 1090.0, "b")
    second = snapshot()
    document = {
        "source_mode": "tw_official_bulk_v2",
        "source_schema_version": "tw-official-historical-v2",
        "target_date": TARGET.isoformat(),
        "snapshots": [
            {"date": first_date.isoformat(), "manifest_sha256": first.manifest_sha256},
            {"date": TARGET.isoformat(), "manifest_sha256": second.manifest_sha256},
        ],
    }
    digest = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OfficialSnapshotSeries(
        target_date=TARGET,
        snapshots=MappingProxyType({first_date: first, TARGET: second}),
        manifest_sha256=digest,
        request_count=12,
        request_budget=OfficialRequestBudget(12, 24, 12, 0, True, "capacity_proven"),
    )


def write_artifact(
    root,
    *,
    daily,
    as_of="2026-07-22",
    symbol="2330",
    source_lineage=_MISSING,
):
    path = Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "market": "TW",
        "symbol": symbol,
        "as_of": as_of,
        "daily": daily,
    }
    if source_lineage is not _MISSING:
        document["source_lineage"] = source_lineage
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


def target_history(**changes):
    row = {
        "Date": "2026-07-24T00:00:00.000",
        "Open": 1.0,
        "High": 2.0,
        "Low": 0.5,
        "Close": 1.5,
        "Volume": 7.0,
        "InstitutionalNet": 11.0,
        "ForeignNet": 9.0,
        "MarginBalance": 22.0,
        "ShortBalance": 3.0,
    }
    row.update(changes)
    return [row]


def official_lineage(*, schema_version="tw-official-historical-v2", reconciliation=None):
    manifest_document = {
        "source_mode": "tw_official_bulk_v2",
        "source_schema_version": schema_version,
        "target_date": TARGET.isoformat(),
        "snapshots": [
            {"date": TARGET.isoformat(), "manifest_sha256": "a" * 64}
        ],
    }
    result = {
        "source_mode": "tw_official_bulk_v2",
        "source_schema_version": schema_version,
        "target_market_date": TARGET.isoformat(),
        "official_series_manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest_document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "official_snapshot_dates": [TARGET.isoformat()],
        "official_snapshot_manifests": [
            {"date": TARGET.isoformat(), "manifest_sha256": "a" * 64}
        ],
        "historical_artifact_sha256": "c" * 64,
        "historical_as_of": "2026-07-23",
        "symbol": "2330",
        "official_target_price_available": True,
    }
    if reconciliation is not None:
        result["legacy_reconciliation"] = reconciliation
    return result


def reconciliation_record(
    *,
    replaced_date=None,
    legacy_as_of=None,
    snapshot_dates=None,
):
    replaced_date = replaced_date or TARGET.isoformat()
    legacy_as_of = legacy_as_of or TARGET.isoformat()
    snapshot_dates = snapshot_dates or (TARGET.isoformat(),)
    snapshot_manifests = [
        {
            "date": value,
            "manifest_sha256": ("a" if value == TARGET.isoformat() else "b") * 64,
        }
        for value in snapshot_dates
    ]
    manifest_document = {
        "source_mode": "tw_official_bulk_v2",
        "source_schema_version": "tw-official-historical-v2",
        "target_date": snapshot_dates[-1],
        "snapshots": snapshot_manifests,
    }
    return {
        "schema_version": 1,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": "d" * 64,
        "legacy_artifact_as_of": legacy_as_of,
        "official_source_mode": "tw_official_bulk_v2",
        "official_source_schema_version": "tw-official-historical-v2",
        "official_series_manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest_document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "official_snapshot_dates": list(snapshot_dates),
        "official_snapshot_manifests": snapshot_manifests,
        "replaced_dates": [replaced_date],
        "price_replaced_dates": [replaced_date],
        "institutional_replaced_dates": [replaced_date],
        "margin_replaced_dates": [replaced_date],
        "date_evidence": [{
            "date": replaced_date,
            "price_replaced": True,
            "institutional_replaced": True,
            "margin_replaced": True,
        }],
    }


def snapshot_without_optional(*, price=True, row_date=None, row_symbol="2330"):
    value = TARGET
    price_rows = {}
    if price:
        price_rows["2330"] = MappingProxyType({
            "date": row_date or value.isoformat(),
            "stock_id": row_symbol,
            "open": 1100.0,
            "max": 1120.0,
            "min": 1090.0,
            "close": 1110.0,
            "Trading_Volume": 1000.0,
        })
    return OfficialDailySnapshot(
        target_date=value,
        price_by_symbol=MappingProxyType(price_rows),
        institutional_by_symbol=MappingProxyType({}),
        margin_by_symbol=MappingProxyType({}),
        source_results=MappingProxyType({}),
        manifest_sha256="a" * 64,
        request_count=6,
        request_budget=OfficialRequestBudget(
            6, 12, 6, 0, True, "capacity_proven"
        ),
        source_mode="tw_official_bulk_v2",
        source_schema_version="tw-official-historical-v2",
    )


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
            self.assertEqual(
                lineage["official_series_manifest_sha256"],
                "465df97b3dce102d52133e047071a045efae70e1e89a130bce29431a60a20104",
            )
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


class TWLegacyOverlapReconciliationTests(unittest.TestCase):
    def _fetcher(self, root, source=None):
        return OfficialCompatFetcher(
            Path(root),
            source or snapshot(),
            pd=pd,
            legacy_overlap_policy="replace_verified_legacy",
        )

    def test_replace_verified_legacy_overrides_overlapping_price(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            price = self._fetcher(temporary)(
                "TaiwanStockPrice", "2330", TARGET.isoformat(), TARGET.isoformat()
            )
            self.assertEqual(len(price), 1)
            self.assertEqual(
                price.iloc[0][
                    ["open", "max", "min", "close", "Trading_Volume"]
                ].tolist(),
                [1100.0, 1120.0, 1090.0, 1110.0, 1000.0],
            )

    def test_replace_verified_legacy_overrides_institutional_and_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            fetcher = self._fetcher(temporary)
            institutional = fetcher(
                "TaiwanStockInstitutionalInvestorsBuySell",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            margin = fetcher(
                "TaiwanStockMarginPurchaseShortSale",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            self.assertEqual(
                institutional[["name", "buy", "sell"]].values.tolist(),
                [
                    ["Foreign", 100.0, 20.0],
                    ["InvestmentTrust", 20.0, 10.0],
                    ["Dealer", 5.0, 5.0],
                ],
            )
            self.assertEqual(
                margin.iloc[0][
                    ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]
                ].tolist(),
                [5000.0, 200.0],
            )

    def test_replace_verified_legacy_preserves_missing_optional_official_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            fetcher = self._fetcher(temporary, snapshot_without_optional())
            institutional = fetcher(
                "TaiwanStockInstitutionalInvestorsBuySell",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            margin = fetcher(
                "TaiwanStockMarginPurchaseShortSale",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            evidence = fetcher.reconciliation_for("2330")
            foreign = institutional[institutional.name == "Foreign"].iloc[0]
            self.assertEqual(foreign.buy - foreign.sell, 9.0)
            self.assertEqual(
                margin.iloc[0][
                    ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]
                ].tolist(),
                [22.0, 3.0],
            )
            self.assertEqual(evidence["institutional_replaced_dates"], [])
            self.assertEqual(evidence["margin_replaced_dates"], [])
            self.assertEqual(
                evidence["date_evidence"],
                [{
                    "date": TARGET.isoformat(),
                    "price_replaced": True,
                    "institutional_replaced": False,
                    "margin_replaced": False,
                }],
            )

    def test_replace_verified_legacy_recalculates_without_duplicate_dates(self):
        overlap = target_history()[0]
        overlap["Date"] = "2026-07-23T00:00:00.000"
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=history() + [overlap],
                as_of="2026-07-23",
            )
            price = self._fetcher(temporary, series())(
                "TaiwanStockPrice", "2330", "2026-07-01", TARGET.isoformat()
            )
            self.assertEqual(
                price[["date", "close"]].values.tolist(),
                [
                    ["2026-07-22", 1010.0],
                    ["2026-07-23", 1090.0],
                    ["2026-07-24", 1110.0],
                ],
            )

    def test_reconciliation_requires_official_price_for_every_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "official price is unavailable for legacy overlap",
            ):
                self._fetcher(
                    temporary, snapshot_without_optional(price=False)
                )(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_reconciliation_rejects_official_row_identity_mismatch(self):
        for source in (
            snapshot_without_optional(row_date="2026-07-23"),
            snapshot_without_optional(row_symbol="2303"),
        ):
            with self.subTest(row=source.price_by_symbol["2330"]):
                with tempfile.TemporaryDirectory() as temporary:
                    write_artifact(
                        temporary,
                        daily=target_history(),
                        as_of=TARGET.isoformat(),
                    )
                    with self.assertRaisesRegex(
                        IncrementalHistoryError,
                        "official row identity is invalid",
                    ):
                        self._fetcher(temporary, source)(
                            "TaiwanStockPrice",
                            "2330",
                            TARGET.isoformat(),
                            TARGET.isoformat(),
                        )

    def test_reconciliation_rejects_optional_official_row_identity_mismatch(self):
        base = snapshot()
        institutional = MappingProxyType({"2330": (
            MappingProxyType({
                "date": "2026-07-23",
                "stock_id": "2330",
                "name": "Foreign",
                "buy": 1.0,
                "sell": 0.0,
            }),
        )})
        source = OfficialDailySnapshot(
            target_date=base.target_date,
            price_by_symbol=base.price_by_symbol,
            institutional_by_symbol=institutional,
            margin_by_symbol=base.margin_by_symbol,
            source_results=base.source_results,
            manifest_sha256=base.manifest_sha256,
            request_count=base.request_count,
            request_budget=base.request_budget,
            source_mode=base.source_mode,
            source_schema_version=base.source_schema_version,
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "official row identity is invalid",
            ):
                self._fetcher(temporary, source)(
                    "TaiwanStockInstitutionalInvestorsBuySell",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_strict_policy_still_rejects_overlap_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            fetcher = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            with self.assertRaises(IncrementalHistoryError):
                fetcher(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_official_lineage_v1_still_rejects_overlap_mismatch(self):
        self._assert_official_lineage_remains_strict("tw-official-historical-v1")

    def test_official_lineage_v2_still_rejects_overlap_mismatch(self):
        self._assert_official_lineage_remains_strict("tw-official-historical-v2")

    def _assert_official_lineage_remains_strict(self, schema_version):
        with tempfile.TemporaryDirectory() as temporary:
            prior = write_artifact(
                temporary,
                daily=history("2026-07-23T00:00:00.000"),
                as_of="2026-07-23",
            )
            lineage = official_lineage(schema_version=schema_version)
            lineage["historical_artifact_sha256"] = hashlib.sha256(
                prior.read_bytes()
            ).hexdigest()
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            with self.assertRaises(IncrementalHistoryError):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_unknown_lineage_is_not_treated_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage={"source_mode": "finmind"},
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_malformed_official_lineage_is_not_treated_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage={"source_mode": "tw_official_bulk_v2"},
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_non_dict_lineage_is_not_treated_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage="legacy",
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_unknown_official_schema_is_not_treated_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage=official_lineage(schema_version="unknown-v99"),
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_reconciliation_rejects_invalid_official_source_identity(self):
        base = snapshot()
        cases = (
            {"source_mode": "finmind"},
            {"source_schema_version": "unknown-v99"},
            {"manifest_sha256": "g" * 64},
        )
        for changes in cases:
            values = {
                "target_date": base.target_date,
                "price_by_symbol": base.price_by_symbol,
                "institutional_by_symbol": base.institutional_by_symbol,
                "margin_by_symbol": base.margin_by_symbol,
                "source_results": base.source_results,
                "manifest_sha256": base.manifest_sha256,
                "request_count": base.request_count,
                "request_budget": base.request_budget,
                "source_mode": base.source_mode,
                "source_schema_version": base.source_schema_version,
            }
            values.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._fetcher(
                        tempfile.gettempdir(),
                        OfficialDailySnapshot(**values),
                    )

    def test_reconciliation_rejects_invalid_numeric_or_duplicate_category(self):
        base = snapshot()
        invalid_price = MappingProxyType({"2330": MappingProxyType({
            **dict(base.price_by_symbol["2330"]),
            "close": float("nan"),
        })})
        duplicate_institutional = MappingProxyType({"2330": (
            base.institutional_by_symbol["2330"][0],
            base.institutional_by_symbol["2330"][0],
        )})
        cases = (
            (invalid_price, base.institutional_by_symbol, "TaiwanStockPrice"),
            (
                base.price_by_symbol,
                duplicate_institutional,
                "TaiwanStockInstitutionalInvestorsBuySell",
            ),
        )
        for price_rows, institutional_rows, dataset in cases:
            source = OfficialDailySnapshot(
                target_date=base.target_date,
                price_by_symbol=price_rows,
                institutional_by_symbol=institutional_rows,
                margin_by_symbol=base.margin_by_symbol,
                source_results=base.source_results,
                manifest_sha256=base.manifest_sha256,
                request_count=base.request_count,
                request_budget=base.request_budget,
                source_mode=base.source_mode,
                source_schema_version=base.source_schema_version,
            )
            with self.subTest(dataset=dataset):
                with tempfile.TemporaryDirectory() as temporary:
                    write_artifact(
                        temporary,
                        daily=target_history(),
                        as_of=TARGET.isoformat(),
                    )
                    with self.assertRaises(IncrementalHistoryError):
                        self._fetcher(temporary, source)(
                            dataset,
                            "2330",
                            TARGET.isoformat(),
                            TARGET.isoformat(),
                        )

    def test_official_lineage_rejects_tampered_canonical_series_identity(self):
        lineage = official_lineage()
        lineage["official_series_manifest_sha256"] = "f" * 64
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=matching,
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_reconciliation_evidence_is_independent_of_query_range(self):
        overlap = target_history()[0]
        overlap["Date"] = "2026-07-23T00:00:00.000"
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=history() + [overlap],
                as_of="2026-07-23",
            )
            fetcher = self._fetcher(temporary, series())
            fetcher(
                "TaiwanStockPrice",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            self.assertEqual(
                fetcher.reconciliation_for("2330")["replaced_dates"],
                ["2026-07-23"],
            )

    def test_preserved_reconciliation_evidence_cannot_be_mutated_by_caller(self):
        prior = reconciliation_record()
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            lineage = official_lineage(reconciliation=prior)
            lineage["historical_as_of"] = TARGET.isoformat()
            write_artifact(
                temporary,
                daily=matching,
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            fetcher = self._fetcher(temporary)
            first = fetcher.lineage_for("2330")
            first["legacy_reconciliation"]["date_evidence"][0][
                "margin_replaced"
            ] = False
            self.assertTrue(
                fetcher.lineage_for("2330")["legacy_reconciliation"]
                ["date_evidence"][0]["margin_replaced"]
            )

    def test_official_lineage_rejects_impossible_reconciliation_dates(self):
        impossible = reconciliation_record(
            replaced_date="2026-07-23",
            legacy_as_of="2026-07-22",
        )
        lineage = official_lineage(reconciliation=impossible)
        lineage["historical_as_of"] = "2026-07-22"
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(
                    Open=1100.0,
                    High=1120.0,
                    Low=1090.0,
                    Close=1110.0,
                    Volume=1000.0,
                    InstitutionalNet=90.0,
                    ForeignNet=80.0,
                    MarginBalance=5000.0,
                    ShortBalance=200.0,
                ),
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact lineage is not eligible for reconciliation: TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_official_lineage_rejects_impossible_preserved_time_identity(self):
        impossible_records = (
            reconciliation_record(
                snapshot_dates=(TARGET.isoformat(), "2026-07-25"),
            ),
            reconciliation_record(
                replaced_date="2026-07-23",
                legacy_as_of=TARGET.isoformat(),
                snapshot_dates=("2026-07-23", TARGET.isoformat()),
            ),
        )
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        for reconciliation in impossible_records:
            with self.subTest(reconciliation=reconciliation):
                lineage = official_lineage(reconciliation=reconciliation)
                lineage["historical_as_of"] = TARGET.isoformat()
                with tempfile.TemporaryDirectory() as temporary:
                    write_artifact(
                        temporary,
                        daily=matching,
                        as_of=TARGET.isoformat(),
                        source_lineage=lineage,
                    )
                    with self.assertRaisesRegex(
                        IncrementalHistoryError,
                        "historical artifact lineage is not eligible for reconciliation: TW:2330",
                    ):
                        self._fetcher(temporary)(
                            "TaiwanStockPrice",
                            "2330",
                            TARGET.isoformat(),
                            TARGET.isoformat(),
                        )

    def test_single_snapshot_reconciliation_lineage_round_trips_as_official(self):
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
            )
            reconciler = self._fetcher(temporary)
            for dataset in OfficialCompatFetcher.SUPPORTED_DATASETS:
                reconciler(
                    dataset,
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )
            write_artifact(
                temporary,
                daily=matching,
                as_of=TARGET.isoformat(),
                source_lineage=reconciler.lineage_for("2330"),
            )
            strict = OfficialCompatFetcher(Path(temporary), snapshot(), pd=pd)
            price = strict(
                "TaiwanStockPrice",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            self.assertEqual(price.iloc[0]["close"], 1110.0)

    def test_official_lineage_allows_symbol_history_after_series_start(self):
        lineage = official_lineage()
        snapshot_dates = ["2026-07-22", TARGET.isoformat()]
        snapshot_manifests = [
            {"date": "2026-07-22", "manifest_sha256": "b" * 64},
            {"date": TARGET.isoformat(), "manifest_sha256": "a" * 64},
        ]
        manifest_document = {
            "source_mode": "tw_official_bulk_v2",
            "source_schema_version": "tw-official-historical-v2",
            "target_date": TARGET.isoformat(),
            "snapshots": snapshot_manifests,
        }
        lineage.update(
            historical_as_of="2026-07-23",
            official_snapshot_dates=snapshot_dates,
            official_snapshot_manifests=snapshot_manifests,
            official_series_manifest_sha256=hashlib.sha256(
                json.dumps(
                    manifest_document,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_artifact(
                temporary,
                daily=matching,
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            price = self._fetcher(temporary)(
                "TaiwanStockPrice",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            self.assertEqual(price.iloc[0]["close"], 1110.0)

    def test_reconciliation_evidence_records_original_hash_and_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_artifact(
                temporary,
                daily=target_history(),
                as_of=TARGET.isoformat(),
                source_lineage=None,
            )
            original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            fetcher = self._fetcher(temporary)
            for dataset in OfficialCompatFetcher.SUPPORTED_DATASETS:
                fetcher(
                    dataset,
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )
            evidence = fetcher.reconciliation_for("2330")
            self.assertEqual(evidence["schema_version"], 1)
            self.assertEqual(evidence["mode"], "replace_verified_legacy")
            self.assertEqual(evidence["legacy_artifact_sha256"], original_sha)
            self.assertEqual(evidence["replaced_dates"], [TARGET.isoformat()])
            self.assertEqual(evidence["price_replaced_dates"], [TARGET.isoformat()])
            self.assertEqual(
                evidence["institutional_replaced_dates"], [TARGET.isoformat()]
            )
            self.assertEqual(evidence["margin_replaced_dates"], [TARGET.isoformat()])
            self.assertEqual(
                fetcher.lineage_for("2330")["legacy_reconciliation"], evidence
            )

    def test_later_official_run_preserves_legacy_reconciliation_evidence(self):
        prior = reconciliation_record()
        matching = target_history(
            Open=1100.0,
            High=1120.0,
            Low=1090.0,
            Close=1110.0,
            Volume=1000.0,
            InstitutionalNet=90.0,
            ForeignNet=80.0,
            MarginBalance=5000.0,
            ShortBalance=200.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            lineage = official_lineage(reconciliation=prior)
            lineage["historical_as_of"] = TARGET.isoformat()
            write_artifact(
                temporary,
                daily=matching,
                as_of=TARGET.isoformat(),
                source_lineage=lineage,
            )
            fetcher = self._fetcher(temporary)
            fetcher(
                "TaiwanStockPrice",
                "2330",
                TARGET.isoformat(),
                TARGET.isoformat(),
            )
            self.assertEqual(
                fetcher.lineage_for("2330")["legacy_reconciliation"], prior
            )

    def test_reconciliation_does_not_bootstrap_missing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                IncrementalHistoryError,
                "historical artifact is unavailable for TW:2330",
            ):
                self._fetcher(temporary)(
                    "TaiwanStockPrice",
                    "2330",
                    TARGET.isoformat(),
                    TARGET.isoformat(),
                )

    def test_unknown_legacy_overlap_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unknown legacy overlap policy"):
                OfficialCompatFetcher(
                    Path(temporary),
                    snapshot(),
                    pd=pd,
                    legacy_overlap_policy="relaxed",
                )


if __name__ == "__main__":
    unittest.main()
