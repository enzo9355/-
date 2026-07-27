"""Serve validated TW history plus a bounded series of official trading-day rows."""

from __future__ import annotations

import copy
import datetime as _datetime
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from stock_papi.integrations.market_data.tw_official_bulk import OfficialDailySnapshot

MAX_COMPRESSED_BYTES = 5 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
SOURCE_MODE = "tw_official_bulk_v2"
LEGACY_OVERLAP_POLICIES = frozenset({"strict", "replace_verified_legacy"})
KNOWN_OFFICIAL_SCHEMA_VERSIONS = frozenset({
    "tw-official-historical-v1",
    "tw-official-historical-v2",
})
_MISSING = object()


class IncrementalHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class IncrementalArtifact:
    symbol: str
    document: dict[str, Any]
    compressed_sha256: str
    latest_date: _datetime.date


@dataclass(frozen=True)
class ArtifactDateAudit:
    latest_by_symbol: Mapping[str, _datetime.date]
    unavailable_symbols: tuple[str, ...]
    earliest_latest_date: _datetime.date | None
    latest_date_counts: Mapping[str, int]

    @property
    def available_count(self) -> int:
        return len(self.latest_by_symbol)


def _parse_date(value: Any) -> _datetime.date:
    try:
        return _datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return _datetime.date.fromisoformat(str(value)[:10])
        except ValueError as exc:
            raise IncrementalHistoryError("historical row date is invalid") from exc


def _artifact_path(root: Path, symbol: str) -> Path:
    return Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"


def load_incremental_artifact(root: Path, symbol: str) -> IncrementalArtifact:
    path = _artifact_path(Path(root), symbol)
    try:
        compressed_size = path.stat().st_size
        if not 0 < compressed_size <= MAX_COMPRESSED_BYTES:
            raise ValueError("compressed artifact size")
        compressed = path.read_bytes()
        if len(compressed) != compressed_size:
            raise ValueError("compressed artifact changed")
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            decoded = stream.read(MAX_UNCOMPRESSED_BYTES + 1)
        if not decoded or len(decoded) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("artifact expansion")
        document = json.loads(decoded.decode("utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("market") != "TW"
            or document.get("symbol") != symbol
            or not isinstance(document.get("daily"), list)
            or not document["daily"]
        ):
            raise ValueError("artifact schema")
        declared_as_of = _datetime.date.fromisoformat(str(document["as_of"]))
        dates = []
        for row in document["daily"]:
            if not isinstance(row, dict):
                raise ValueError("artifact row")
            dates.append(_parse_date(row.get("Date")))
        if len(dates) != len(set(dates)):
            raise ValueError("artifact duplicate dates")
        latest_date = max(dates)
        if declared_as_of != latest_date:
            raise ValueError("artifact as_of mismatch")
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, gzip.BadGzipFile) as exc:
        raise IncrementalHistoryError(
            f"historical artifact is unavailable for TW:{symbol}"
        ) from exc
    return IncrementalArtifact(
        symbol=symbol,
        document=document,
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        latest_date=latest_date,
    )


def audit_artifact_dates(
    root: Path,
    symbols: Iterable[str],
    *,
    target_date: _datetime.date,
) -> ArtifactDateAudit:
    if not isinstance(target_date, _datetime.date) or isinstance(target_date, _datetime.datetime):
        raise TypeError("target_date must be a date")
    latest_by_symbol: dict[str, _datetime.date] = {}
    unavailable = []
    counts: dict[str, int] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol)
        try:
            artifact = load_incremental_artifact(root, symbol)
            if artifact.latest_date > target_date:
                raise IncrementalHistoryError(
                    f"historical artifact is newer than target for TW:{symbol}"
                )
        except IncrementalHistoryError:
            unavailable.append(symbol)
            continue
        latest_by_symbol[symbol] = artifact.latest_date
        key = artifact.latest_date.isoformat()
        counts[key] = counts.get(key, 0) + 1
    earliest = min(latest_by_symbol.values()) if latest_by_symbol else None
    return ArtifactDateAudit(
        latest_by_symbol=MappingProxyType(latest_by_symbol),
        unavailable_symbols=tuple(sorted(set(unavailable))),
        earliest_latest_date=earliest,
        latest_date_counts=MappingProxyType(dict(sorted(counts.items()))),
    )


class OfficialCompatFetcher:
    """FinMind-compatible callable backed only by local history and official snapshots."""

    SUPPORTED_DATASETS = {
        "TaiwanStockPrice",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
    }

    def __init__(
        self,
        root: Path,
        source: Any,
        *,
        pd: Any,
        legacy_overlap_policy: str = "strict",
    ):
        if legacy_overlap_policy not in LEGACY_OVERLAP_POLICIES:
            raise ValueError("unknown legacy overlap policy")
        self.root = Path(root)
        is_daily_snapshot = isinstance(source, OfficialDailySnapshot)
        if is_daily_snapshot:
            self.snapshots = MappingProxyType({source.target_date: source})
            self.target_date = source.target_date
            self.series_manifest_sha256 = source.manifest_sha256
            self.source_schema_version = source.source_schema_version
            self.source_mode = getattr(source, "source_mode", SOURCE_MODE)
        else:
            snapshots = getattr(source, "snapshots", None)
            if not isinstance(snapshots, Mapping) or not snapshots:
                raise TypeError("official snapshot series is invalid")
            normalized = dict(sorted(snapshots.items()))
            if any(
                not isinstance(value, _datetime.date)
                or isinstance(value, _datetime.datetime)
                or not isinstance(snapshot, OfficialDailySnapshot)
                or snapshot.target_date != value
                for value, snapshot in normalized.items()
            ):
                raise TypeError("official snapshot series is invalid")
            self.snapshots = MappingProxyType(normalized)
            self.target_date = max(normalized)
            if getattr(source, "target_date", None) != self.target_date:
                raise TypeError("official snapshot series target is invalid")
            self.series_manifest_sha256 = str(getattr(source, "manifest_sha256", ""))
            self.source_schema_version = str(getattr(source, "source_schema_version", ""))
            self.source_mode = str(getattr(source, "source_mode", SOURCE_MODE))
        canonical_series_sha256 = self._canonical_series_sha256(
            self.source_mode,
            self.source_schema_version,
            self.target_date,
            (
                (value, snapshot.manifest_sha256)
                for value, snapshot in self.snapshots.items()
            ),
        )
        if is_daily_snapshot:
            self.series_manifest_sha256 = canonical_series_sha256
        if (
            self.source_mode != SOURCE_MODE
            or self.source_schema_version not in KNOWN_OFFICIAL_SCHEMA_VERSIONS
            or not self._is_sha256(self.series_manifest_sha256)
            or any(
                snapshot.source_mode != SOURCE_MODE
                or snapshot.source_schema_version != self.source_schema_version
                or not self._is_sha256(snapshot.manifest_sha256)
                for snapshot in self.snapshots.values()
            )
            or self.series_manifest_sha256 != canonical_series_sha256
        ):
            raise ValueError("official series manifest is invalid")
        self.pd = pd
        self.legacy_overlap_policy = legacy_overlap_policy
        self._artifacts: dict[str, IncrementalArtifact] = {}
        self._lineage_kinds: dict[str, str] = {}
        self._existing_reconciliations: dict[str, dict[str, Any]] = {}
        self._reconciliation_dates: dict[
            str, dict[_datetime.date, dict[str, bool]]
        ] = {}

    def _load_artifact(self, symbol: str) -> IncrementalArtifact:
        if symbol not in self._artifacts:
            self._artifacts[symbol] = load_incremental_artifact(self.root, symbol)
        artifact = self._artifacts[symbol]
        if artifact.latest_date > self.target_date:
            raise IncrementalHistoryError(
                f"historical artifact is newer than target for TW:{symbol}"
            )
        return artifact

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _canonical_series_sha256(
        source_mode: str,
        source_schema_version: str,
        target_date: _datetime.date,
        snapshot_manifests: Iterable[tuple[_datetime.date, str]],
    ) -> str:
        document = {
            "source_mode": source_mode,
            "source_schema_version": source_schema_version,
            "target_date": target_date.isoformat(),
            "snapshots": [
                {
                    "date": value.isoformat(),
                    "manifest_sha256": manifest_sha256,
                }
                for value, manifest_sha256 in snapshot_manifests
            ],
        }
        return hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _date_list(value: Any) -> tuple[_datetime.date, ...] | None:
        if not isinstance(value, list) or not value:
            return None
        try:
            dates = tuple(_datetime.date.fromisoformat(item) for item in value)
        except (TypeError, ValueError):
            return None
        if dates != tuple(sorted(set(dates))):
            return None
        return dates

    @classmethod
    def _valid_reconciliation(
        cls,
        value: Any,
        *,
        target_date: _datetime.date,
    ) -> bool:
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("mode") != "replace_verified_legacy"
            or not cls._is_sha256(value.get("legacy_artifact_sha256"))
            or value.get("official_source_mode") != SOURCE_MODE
            or value.get("official_source_schema_version")
            not in KNOWN_OFFICIAL_SCHEMA_VERSIONS
            or not cls._is_sha256(
                value.get("official_series_manifest_sha256")
            )
        ):
            return False
        try:
            legacy_as_of = _datetime.date.fromisoformat(
                value["legacy_artifact_as_of"]
            )
        except (KeyError, TypeError, ValueError):
            return False
        snapshot_dates = cls._date_list(value.get("official_snapshot_dates"))
        manifests = value.get("official_snapshot_manifests")
        if (
            snapshot_dates is None
            or not isinstance(manifests, list)
            or len(manifests) != len(snapshot_dates)
            or legacy_as_of > target_date
            or snapshot_dates[-1] > target_date
        ):
            return False
        for item, snapshot_date in zip(manifests, snapshot_dates):
            if (
                not isinstance(item, dict)
                or item.get("date") != snapshot_date.isoformat()
                or not cls._is_sha256(item.get("manifest_sha256"))
            ):
                return False
        if value["official_series_manifest_sha256"] != cls._canonical_series_sha256(
            value["official_source_mode"],
            value["official_source_schema_version"],
            snapshot_dates[-1],
            (
                (snapshot_date, item["manifest_sha256"])
                for snapshot_date, item in zip(snapshot_dates, manifests)
            ),
        ):
            return False
        replaced = cls._date_list(value.get("replaced_dates"))
        price = cls._date_list(value.get("price_replaced_dates"))
        if (
            replaced is None
            or price != replaced
            or replaced[-1] > target_date
            or replaced[-1] != legacy_as_of
            or not set(replaced).issubset(snapshot_dates)
        ):
            return False
        optional_dates = {}
        for name in ("institutional_replaced_dates", "margin_replaced_dates"):
            raw = value.get(name)
            if raw == []:
                optional_dates[name] = ()
                continue
            parsed = cls._date_list(raw)
            if parsed is None or not set(parsed).issubset(replaced):
                return False
            optional_dates[name] = parsed
        evidence = value.get("date_evidence")
        if not isinstance(evidence, list) or len(evidence) != len(replaced):
            return False
        for row, row_date in zip(evidence, replaced):
            if (
                not isinstance(row, dict)
                or row.get("date") != row_date.isoformat()
                or row.get("price_replaced") is not True
                or not isinstance(row.get("institutional_replaced"), bool)
                or not isinstance(row.get("margin_replaced"), bool)
                or row["institutional_replaced"]
                != (row_date in optional_dates["institutional_replaced_dates"])
                or row["margin_replaced"]
                != (row_date in optional_dates["margin_replaced_dates"])
            ):
                return False
        return True

    @classmethod
    def _valid_official_lineage(
        cls,
        lineage: Any,
        artifact: IncrementalArtifact,
    ) -> bool:
        if (
            not isinstance(lineage, dict)
            or lineage.get("source_mode") != SOURCE_MODE
            or lineage.get("source_schema_version")
            not in KNOWN_OFFICIAL_SCHEMA_VERSIONS
            or lineage.get("symbol") != artifact.symbol
            or not cls._is_sha256(
                lineage.get("official_series_manifest_sha256")
            )
            or not cls._is_sha256(lineage.get("historical_artifact_sha256"))
            or not isinstance(lineage.get("official_target_price_available"), bool)
        ):
            return False
        try:
            target_date = _datetime.date.fromisoformat(
                lineage["target_market_date"]
            )
            historical_as_of = _datetime.date.fromisoformat(
                lineage["historical_as_of"]
            )
        except (KeyError, TypeError, ValueError):
            return False
        snapshot_dates = cls._date_list(lineage.get("official_snapshot_dates"))
        if (
            snapshot_dates is None
            or target_date != artifact.latest_date
            or historical_as_of > target_date
            or snapshot_dates[-1] != target_date
        ):
            return False
        manifests = lineage.get("official_snapshot_manifests")
        if not isinstance(manifests, list) or len(manifests) != len(snapshot_dates):
            return False
        for item, value in zip(manifests, snapshot_dates):
            if (
                not isinstance(item, dict)
                or item.get("date") != value.isoformat()
                or not cls._is_sha256(item.get("manifest_sha256"))
            ):
                return False
        if lineage["official_series_manifest_sha256"] != cls._canonical_series_sha256(
            lineage["source_mode"],
            lineage["source_schema_version"],
            target_date,
            (
                (value, item["manifest_sha256"])
                for value, item in zip(snapshot_dates, manifests)
            ),
        ):
            return False
        reconciliation = lineage.get("legacy_reconciliation", _MISSING)
        return (
            reconciliation is _MISSING
            or cls._valid_reconciliation(
                reconciliation,
                target_date=target_date,
            )
        )

    def _lineage_kind(self, symbol: str) -> str:
        if symbol in self._lineage_kinds:
            return self._lineage_kinds[symbol]
        artifact = self._load_artifact(symbol)
        lineage = artifact.document.get("source_lineage", _MISSING)
        if lineage is _MISSING or lineage is None:
            kind = "legacy"
        elif self._valid_official_lineage(lineage, artifact):
            kind = "official"
            reconciliation = lineage.get("legacy_reconciliation")
            if reconciliation is not None:
                self._existing_reconciliations[symbol] = copy.deepcopy(
                    reconciliation
                )
        else:
            raise IncrementalHistoryError(
                "historical artifact lineage is not eligible for reconciliation: "
                f"TW:{symbol}"
            )
        self._lineage_kinds[symbol] = kind
        return kind

    @staticmethod
    def _number(row: dict[str, Any], name: str, default: float | None = None) -> float:
        value = row.get(name, default)
        if value is None or isinstance(value, bool):
            raise IncrementalHistoryError(f"historical field is invalid: {name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise IncrementalHistoryError(f"historical field is invalid: {name}") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise IncrementalHistoryError(f"historical field is invalid: {name}")
        return number

    def _daily_rows(
        self,
        symbol: str,
        start: _datetime.date,
        end: _datetime.date,
    ) -> list[dict[str, Any]]:
        artifact = self._load_artifact(symbol)
        rows = []
        for item in artifact.document["daily"]:
            row_date = _parse_date(item.get("Date"))
            if start <= row_date <= end:
                rows.append(dict(item, _date=row_date))
        return sorted(rows, key=lambda row: row["_date"])

    @staticmethod
    def _net_rows(
        date_text: str,
        symbol: str,
        total: float,
        foreign: float,
    ) -> list[dict[str, Any]]:
        remainder = total - foreign
        return [
            {
                "date": date_text,
                "stock_id": symbol,
                "name": name,
                "buy": max(net, 0.0),
                "sell": max(-net, 0.0),
            }
            for name, net in (
                ("Foreign", foreign),
                ("InvestmentTrust", remainder),
                ("Dealer", 0.0),
            )
        ]

    @staticmethod
    def _validate_official_row(
        snapshot: OfficialDailySnapshot,
        symbol: str,
        row: Mapping[str, Any],
    ) -> None:
        if (
            row.get("date") != snapshot.target_date.isoformat()
            or row.get("stock_id") != symbol
        ):
            raise IncrementalHistoryError(
                f"official row identity is invalid for TW:{symbol}"
            )

    @staticmethod
    def _validate_official_numbers(
        row: Mapping[str, Any],
        names: Iterable[str],
        symbol: str,
    ) -> None:
        for name in names:
            value = row.get(name)
            if value is None or isinstance(value, bool):
                raise IncrementalHistoryError(
                    f"official row is invalid for TW:{symbol}"
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise IncrementalHistoryError(
                    f"official row is invalid for TW:{symbol}"
                ) from exc
            if number != number or number in (float("inf"), float("-inf")):
                raise IncrementalHistoryError(
                    f"official row is invalid for TW:{symbol}"
                )

    @classmethod
    def _official_price(
        cls,
        snapshot: OfficialDailySnapshot,
        symbol: str,
    ) -> dict[str, Any] | None:
        row = snapshot.price_by_symbol.get(symbol)
        if row is None:
            return None
        cls._validate_official_row(snapshot, symbol, row)
        cls._validate_official_numbers(
            row,
            ("open", "max", "min", "close", "Trading_Volume"),
            symbol,
        )
        return dict(row)

    @classmethod
    def _official_institutional(
        cls,
        snapshot: OfficialDailySnapshot,
        symbol: str,
    ) -> list[dict[str, Any]]:
        rows = snapshot.institutional_by_symbol.get(symbol, ())
        names = []
        for row in rows:
            cls._validate_official_row(snapshot, symbol, row)
            name = row.get("name")
            if name not in {"Foreign", "InvestmentTrust", "Dealer"}:
                raise IncrementalHistoryError(
                    f"official row is invalid for TW:{symbol}"
                )
            names.append(name)
            cls._validate_official_numbers(row, ("buy", "sell"), symbol)
        if len(names) != len(set(names)):
            raise IncrementalHistoryError(
                f"official row is invalid for TW:{symbol}"
            )
        return [dict(row) for row in rows]

    @classmethod
    def _official_margin(
        cls,
        snapshot: OfficialDailySnapshot,
        symbol: str,
    ) -> dict[str, Any] | None:
        row = snapshot.margin_by_symbol.get(symbol)
        if row is None:
            return None
        cls._validate_official_row(snapshot, symbol, row)
        cls._validate_official_numbers(
            row,
            ("MarginPurchaseTodayBalance", "ShortSaleTodayBalance"),
            symbol,
        )
        return dict(row)

    def _record_reconciliation(
        self,
        symbol: str,
        value: _datetime.date,
        snapshot: OfficialDailySnapshot,
    ) -> None:
        price = self._official_price(snapshot, symbol)
        if price is None:
            raise IncrementalHistoryError(
                f"official price is unavailable for legacy overlap: TW:{symbol}"
            )
        institutional = bool(self._official_institutional(snapshot, symbol))
        margin = self._official_margin(snapshot, symbol) is not None
        current = {
            "price_replaced": True,
            "institutional_replaced": institutional,
            "margin_replaced": margin,
        }
        recorded = self._reconciliation_dates.setdefault(symbol, {}).get(value)
        if recorded is not None and recorded != current:
            raise IncrementalHistoryError(
                f"official reconciliation evidence changed for TW:{symbol}"
            )
        self._reconciliation_dates[symbol][value] = current

    def _reconciliation_plan(
        self,
        symbol: str,
    ) -> dict[_datetime.date, dict[str, bool]]:
        if symbol in self._reconciliation_dates:
            return self._reconciliation_dates[symbol]
        self._reconciliation_dates[symbol] = {}
        artifact_dates = {
            _parse_date(row.get("Date"))
            for row in self._load_artifact(symbol).document["daily"]
        }
        for value, snapshot in self.snapshots.items():
            if value in artifact_dates:
                self._record_reconciliation(symbol, value, snapshot)
        return self._reconciliation_dates[symbol]

    def _verify_existing(
        self,
        symbol: str,
        historical: dict[str, Any],
        snapshot: OfficialDailySnapshot,
    ) -> None:
        official_price = self._official_price(snapshot, symbol)
        if official_price is None:
            return
        for historical_name, official_name in (
            ("Open", "open"),
            ("High", "max"),
            ("Low", "min"),
            ("Close", "close"),
            ("Volume", "Trading_Volume"),
        ):
            if self._number(historical, historical_name) != float(official_price[official_name]):
                raise IncrementalHistoryError(
                    f"existing row conflicts with official source for TW:{symbol}"
                )
        institutional = self._official_institutional(snapshot, symbol)
        if institutional:
            expected_total = sum(
                float(item["buy"]) - float(item["sell"])
                for item in institutional
            )
            expected_foreign = sum(
                float(item["buy"]) - float(item["sell"])
                for item in institutional
                if item["name"] == "Foreign"
            )
            if (
                self._number(historical, "InstitutionalNet", 0.0) != expected_total
                or self._number(historical, "ForeignNet", 0.0) != expected_foreign
            ):
                raise IncrementalHistoryError(
                    f"existing chip row conflicts with official source for TW:{symbol}"
                )
        margin = self._official_margin(snapshot, symbol)
        if margin is not None and (
            self._number(historical, "MarginBalance", 0.0)
            != float(margin["MarginPurchaseTodayBalance"])
            or self._number(historical, "ShortBalance", 0.0)
            != float(margin["ShortSaleTodayBalance"])
        ):
            raise IncrementalHistoryError(
                f"existing margin row conflicts with official source for TW:{symbol}"
            )

    def __call__(self, dataset: str, code: str, start_date: str, end_date: str):
        if dataset not in self.SUPPORTED_DATASETS:
            raise IncrementalHistoryError(
                f"unsupported official compatibility dataset: {dataset}"
            )
        symbol = str(code)
        start = _datetime.date.fromisoformat(start_date)
        end = _datetime.date.fromisoformat(end_date)
        historical = self._daily_rows(symbol, start, end)
        historical_by_date = {row["_date"]: row for row in historical}
        lineage_kind = self._lineage_kind(symbol)
        replace_legacy = (
            self.legacy_overlap_policy == "replace_verified_legacy"
            and lineage_kind == "legacy"
        )
        replacement_dates = (
            set(self._reconciliation_plan(symbol)).intersection(
                historical_by_date
            )
            if replace_legacy
            else set()
        )
        for value, snapshot in self.snapshots.items():
            existing = historical_by_date.get(value)
            if existing is not None and not replace_legacy:
                self._verify_existing(symbol, existing, snapshot)

        rows: list[dict[str, Any]] = []
        if dataset == "TaiwanStockPrice":
            for row in historical:
                if row["_date"] in replacement_dates:
                    rows.append(
                        self._official_price(
                            self.snapshots[row["_date"]], symbol
                        )
                    )
                else:
                    rows.append({
                        "date": row["_date"].isoformat(),
                        "stock_id": symbol,
                        "open": self._number(row, "Open"),
                        "max": self._number(row, "High"),
                        "min": self._number(row, "Low"),
                        "close": self._number(row, "Close"),
                        "Trading_Volume": self._number(row, "Volume", 0.0),
                    })
            for value, snapshot in self.snapshots.items():
                if start <= value <= end and value not in historical_by_date:
                    official = self._official_price(snapshot, symbol)
                    if official is not None:
                        rows.append(official)
        elif dataset == "TaiwanStockInstitutionalInvestorsBuySell":
            for row in historical:
                official = (
                    self._official_institutional(
                        self.snapshots[row["_date"]], symbol
                    )
                    if row["_date"] in replacement_dates
                    else []
                )
                if official:
                    rows.extend(official)
                else:
                    rows.extend(
                        self._net_rows(
                            row["_date"].isoformat(),
                            symbol,
                            self._number(row, "InstitutionalNet", 0.0),
                            self._number(row, "ForeignNet", 0.0),
                        )
                    )
            for value, snapshot in self.snapshots.items():
                if (
                    start <= value <= end
                    and value not in historical_by_date
                    and self._official_price(snapshot, symbol) is not None
                ):
                    rows.extend(self._official_institutional(snapshot, symbol))
        else:
            for row in historical:
                official = (
                    self._official_margin(self.snapshots[row["_date"]], symbol)
                    if row["_date"] in replacement_dates
                    else None
                )
                if official is not None:
                    rows.append(official)
                else:
                    rows.append({
                        "date": row["_date"].isoformat(),
                        "stock_id": symbol,
                        "MarginPurchaseTodayBalance": self._number(
                            row, "MarginBalance", 0.0
                        ),
                        "ShortSaleTodayBalance": self._number(
                            row, "ShortBalance", 0.0
                        ),
                    })
            for value, snapshot in self.snapshots.items():
                if (
                    start <= value <= end
                    and value not in historical_by_date
                    and self._official_price(snapshot, symbol) is not None
                ):
                    official = self._official_margin(snapshot, symbol)
                    if official is not None:
                        rows.append(official)
        if not rows:
            return self.pd.DataFrame()
        return (
            self.pd.DataFrame(rows)
            .sort_values("date")
            .drop_duplicates(
                subset=[
                    column
                    for column in ("date", "stock_id", "name")
                    if column in rows[0]
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )

    def reconciliation_for(self, symbol: str) -> dict[str, Any] | None:
        if (
            self.legacy_overlap_policy == "replace_verified_legacy"
            and self._lineage_kind(symbol) == "legacy"
        ):
            self._reconciliation_plan(symbol)
        recorded = self._reconciliation_dates.get(symbol)
        if not recorded:
            return None
        artifact = self._load_artifact(symbol)
        dates = sorted(recorded)
        return {
            "schema_version": 1,
            "mode": "replace_verified_legacy",
            "legacy_artifact_sha256": artifact.compressed_sha256,
            "legacy_artifact_as_of": artifact.latest_date.isoformat(),
            "official_source_mode": self.source_mode,
            "official_source_schema_version": self.source_schema_version,
            "official_series_manifest_sha256": self.series_manifest_sha256,
            "official_snapshot_dates": [
                value.isoformat() for value in self.snapshots
            ],
            "official_snapshot_manifests": [
                {
                    "date": value.isoformat(),
                    "manifest_sha256": snapshot.manifest_sha256,
                }
                for value, snapshot in self.snapshots.items()
            ],
            "replaced_dates": [value.isoformat() for value in dates],
            "price_replaced_dates": [value.isoformat() for value in dates],
            "institutional_replaced_dates": [
                value.isoformat()
                for value in dates
                if recorded[value]["institutional_replaced"]
            ],
            "margin_replaced_dates": [
                value.isoformat()
                for value in dates
                if recorded[value]["margin_replaced"]
            ],
            "date_evidence": [
                {"date": value.isoformat(), **recorded[value]}
                for value in dates
            ],
        }

    def lineage_for(self, symbol: str) -> dict[str, Any]:
        artifact = self._load_artifact(symbol)
        self._lineage_kind(symbol)
        lineage = {
            "source_mode": self.source_mode,
            "source_schema_version": self.source_schema_version,
            "target_market_date": self.target_date.isoformat(),
            "official_series_manifest_sha256": self.series_manifest_sha256,
            "official_snapshot_dates": [value.isoformat() for value in self.snapshots],
            "official_snapshot_manifests": [
                {
                    "date": value.isoformat(),
                    "manifest_sha256": snapshot.manifest_sha256,
                }
                for value, snapshot in self.snapshots.items()
            ],
            "historical_artifact_sha256": artifact.compressed_sha256,
            "historical_as_of": artifact.latest_date.isoformat(),
            "symbol": symbol,
            "official_target_price_available": (
                symbol in self.snapshots[self.target_date].price_by_symbol
            ),
        }
        reconciliation = (
            self.reconciliation_for(symbol)
            or self._existing_reconciliations.get(symbol)
        )
        if reconciliation is not None:
            lineage["legacy_reconciliation"] = copy.deepcopy(reconciliation)
        return lineage
