"""Serve validated TW history plus a bounded series of official trading-day rows."""

from __future__ import annotations

import copy
import datetime as _datetime
import gzip
import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable

from stock_papi.integrations.market_data.tw_official_bulk import OfficialDailySnapshot
from stock_papi.integrations.market_data.tw_trading_status import evidence_sha256
from stock_papi.quant.features import CALCULATED_COLUMNS

MAX_COMPRESSED_BYTES = 5 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
SOURCE_MODE = "tw_official_bulk_v2"
LEGACY_OVERLAP_POLICIES = frozenset({"strict", "replace_verified_legacy"})
KNOWN_OFFICIAL_SCHEMA_VERSIONS = frozenset({
    "tw-official-historical-v1",
    "tw-official-historical-v2",
    "tw-official-historical-v3",
})
_MISSING = object()
_RECONCILIATION_HISTORY_FIELDS = frozenset({
    "schema_version",
    "symbol",
    "reconciled_artifact_sha256",
    "history_sha256",
    "reconciliation",
})
RECOVERY_DERIVED_FIELDS: frozenset[str] = frozenset((
    *CALCULATED_COLUMNS,
    "AI_P",
    "FUTURE_RET_5",
    "T",
))
_RECOVERY_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "mode",
    "symbol",
    "recovery_target_market_date",
    "input_artifact_sha256",
    "original_artifact_sha256",
    "backup_target_market_date",
    "backup_series_manifest_sha256",
    "backup_manifest_entry_sha256",
    "backup_object_size",
    "backup_object_uncompressed_size",
    "restored_start_date",
    "restored_end_date",
    "restored_row_count",
    "restored_daily_sha256",
    "receipt_sha256",
})
_RECOVERY_OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


class IncrementalHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class IncrementalArtifact:
    symbol: str
    document: dict[str, Any]
    compressed_sha256: str
    latest_date: _datetime.date
    observation_date: _datetime.date
    trading_status_evidence: Mapping[str, Any] | None


@dataclass(frozen=True)
class HistoryRecoveryResult:
    merged_daily: tuple[Mapping[str, Any], ...]
    restored_candidates: tuple[Mapping[str, Any], ...]
    backup_daily: tuple[Mapping[str, Any], ...]
    input_artifact_sha256: str
    original_artifact_sha256: str
    expected_result_sha256: str
    backup_target_market_date: _datetime.date
    backup_series_manifest_sha256: str
    backup_manifest_entry: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    existing_receipt: Mapping[str, Any] | None


HistoryRecoveryResolver = Callable[
    [str, IncrementalArtifact],
    HistoryRecoveryResult | None,
]


@dataclass(frozen=True)
class ArtifactDateAudit:
    latest_by_symbol: Mapping[str, _datetime.date]
    observation_by_symbol: Mapping[str, _datetime.date]
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


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            name = str(key)
            if name in normalized:
                raise IncrementalHistoryError("canonical JSON mapping keys collide")
            normalized[name] = _canonical_json_value(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
    raise IncrementalHistoryError("canonical JSON value is invalid")


def _canonical_recovery_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise IncrementalHistoryError("daily history row is invalid")
    value = {
        name: item
        for name, item in row.items()
        if name not in RECOVERY_DERIVED_FIELDS and not str(name).startswith("_")
    }
    value["Date"] = _parse_date(value.get("Date")).isoformat()
    return value


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
            or document.get("schema_version") not in {1, 2}
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
        if document["schema_version"] == 1:
            observation_date = latest_date
            trading_status_evidence = None
        else:
            target_date = _datetime.date.fromisoformat(
                str(document["target_market_date"])
            )
            observation_date = _datetime.date.fromisoformat(
                str(document["observation_as_of"])
            )
            latest_regular_price_date = _datetime.date.fromisoformat(
                str(document["latest_regular_price_date"])
            )
            observation_kind = document.get("observation_kind")
            status_value = document.get("trading_status_evidence")
            if (
                target_date != observation_date
                or latest_regular_price_date != latest_date
                or latest_date > observation_date
                or observation_kind not in {
                    "regular_price",
                    "official_no_regular_trade",
                    "officially_suspended",
                }
            ):
                raise ValueError("artifact observation dates mismatch")
            if observation_kind == "regular_price":
                if latest_date != observation_date or status_value is not None:
                    raise ValueError("regular artifact observation is invalid")
                trading_status_evidence = None
            else:
                if (
                    not isinstance(status_value, dict)
                    or status_value.get("status") != observation_kind
                    or status_value.get("market") != "TW"
                    or status_value.get("symbol") != symbol
                    or status_value.get("target_market_date")
                    != observation_date.isoformat()
                    or status_value.get("evidence_sha256")
                    != evidence_sha256(status_value)
                ):
                    raise ValueError("artifact trading status is invalid")
                trading_status_evidence = status_value
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, gzip.BadGzipFile) as exc:
        raise IncrementalHistoryError(
            f"historical artifact is unavailable for TW:{symbol}"
        ) from exc
    return IncrementalArtifact(
        symbol=symbol,
        document=document,
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        latest_date=latest_date,
        observation_date=observation_date,
        trading_status_evidence=trading_status_evidence,
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
    observation_by_symbol: dict[str, _datetime.date] = {}
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
        observation_by_symbol[symbol] = artifact.observation_date
        key = artifact.latest_date.isoformat()
        counts[key] = counts.get(key, 0) + 1
    earliest = min(latest_by_symbol.values()) if latest_by_symbol else None
    return ArtifactDateAudit(
        latest_by_symbol=MappingProxyType(latest_by_symbol),
        observation_by_symbol=MappingProxyType(observation_by_symbol),
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
        recovery_resolver: HistoryRecoveryResolver | None = None,
    ):
        if legacy_overlap_policy not in LEGACY_OVERLAP_POLICIES:
            raise ValueError("unknown legacy overlap policy")
        if recovery_resolver is not None and not callable(recovery_resolver):
            raise TypeError("history recovery resolver is invalid")
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
        self.recovery_resolver = recovery_resolver
        self._artifacts: dict[str, IncrementalArtifact] = {}
        self._history_recovery: dict[str, HistoryRecoveryResult | None] = {}
        self._lineage_kinds: dict[str, str] = {}
        self._existing_reconciliations: dict[str, dict[str, Any]] = {}
        self._existing_reconciliation_history: dict[
            str, list[dict[str, Any]]
        ] = {}
        self._reconciliation_dates: dict[
            str, dict[_datetime.date, dict[str, str]]
        ] = {}
        self._existing_recovery_receipts: dict[str, dict[str, Any]] = {}

    def _load_artifact(self, symbol: str) -> IncrementalArtifact:
        if symbol not in self._artifacts:
            self._artifacts[symbol] = load_incremental_artifact(self.root, symbol)
        artifact = self._artifacts[symbol]
        if artifact.latest_date > self.target_date:
            raise IncrementalHistoryError(
                f"historical artifact is newer than target for TW:{symbol}"
            )
        return artifact

    def _ensure_history_recovery(
        self,
        symbol: str,
    ) -> HistoryRecoveryResult | None:
        if symbol not in self._history_recovery:
            artifact = self._load_artifact(symbol)
            self._history_recovery[symbol] = (
                None
                if self.recovery_resolver is None
                else self.recovery_resolver(symbol, artifact)
            )
        return self._history_recovery[symbol]

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _canonical_json_sha256(value: Any) -> str:
        payload = json.dumps(
            _canonical_json_value(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _canonical_recovery_source_row(
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _canonical_recovery_source_row(row)

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
        if not isinstance(value, (list, tuple)) or not value:
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
            not isinstance(value, Mapping)
            or value.get("schema_version") != 2
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
            or not isinstance(manifests, (list, tuple))
            or len(manifests) != len(snapshot_dates)
            or legacy_as_of > target_date
            or snapshot_dates[-1] > target_date
        ):
            return False
        for item, snapshot_date in zip(manifests, snapshot_dates):
            if (
                not isinstance(item, Mapping)
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
        overlap = cls._date_list(value.get("overlap_dates"))
        if (
            overlap is None
            or overlap[-1] > target_date
            or overlap[-1] != legacy_as_of
            or not set(overlap).issubset(snapshot_dates)
        ):
            return False
        partitions: dict[str, tuple[_datetime.date, ...]] = {}
        for dataset in ("price", "institutional", "margin"):
            replaced_name = f"{dataset}_replaced_dates"
            preserved_name = f"{dataset}_preserved_no_official_row_dates"
            for name in (replaced_name, preserved_name):
                raw = value.get(name)
                parsed = () if raw in ([], ()) else cls._date_list(raw)
                if parsed is None:
                    return False
                partitions[name] = parsed
            replaced = set(partitions[replaced_name])
            preserved = set(partitions[preserved_name])
            if replaced & preserved or replaced | preserved != set(overlap):
                return False
        evidence = value.get("date_evidence")
        if not isinstance(evidence, (list, tuple)) or len(evidence) != len(overlap):
            return False
        for row, row_date in zip(evidence, overlap):
            if (
                not isinstance(row, Mapping)
                or row.get("date") != row_date.isoformat()
            ):
                return False
            for dataset in ("price", "institutional", "margin"):
                expected = (
                    "replaced_official"
                    if row_date in partitions[f"{dataset}_replaced_dates"]
                    else "preserved_legacy_no_official_row"
                )
                if row.get(f"{dataset}_action") != expected:
                    return False
        return True

    @staticmethod
    def _positive_int(value: Any) -> bool:
        return type(value) is int and value > 0

    @classmethod
    def _valid_recovery_receipt(
        cls,
        value: Any,
        *,
        symbol: str,
        recovery_bindings: Iterable[tuple[Mapping[str, Any], str]],
    ) -> bool:
        if (
            not isinstance(value, Mapping)
            or set(value) != _RECOVERY_RECEIPT_FIELDS
            or value.get("schema_version") != 1
            or value.get("mode") != "restore_verified_reconciliation_backup"
            or value.get("symbol") != symbol
            or not all(
                cls._is_sha256(value.get(name))
                for name in (
                    "input_artifact_sha256",
                    "original_artifact_sha256",
                    "backup_series_manifest_sha256",
                    "backup_manifest_entry_sha256",
                    "restored_daily_sha256",
                    "receipt_sha256",
                )
            )
            or not all(
                cls._positive_int(value.get(name))
                for name in (
                    "backup_object_size",
                    "backup_object_uncompressed_size",
                    "restored_row_count",
                )
            )
        ):
            return False
        try:
            recovery_target = _datetime.date.fromisoformat(
                value["recovery_target_market_date"]
            )
            backup_target = _datetime.date.fromisoformat(
                value["backup_target_market_date"]
            )
            start = _datetime.date.fromisoformat(value["restored_start_date"])
            end = _datetime.date.fromisoformat(value["restored_end_date"])
            unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
            if (
                start > end
                or recovery_target < backup_target
                or cls._canonical_json_sha256(unsigned) != value["receipt_sha256"]
            ):
                return False
        except (IncrementalHistoryError, KeyError, TypeError, ValueError):
            return False
        for reconciliation, artifact_sha256 in recovery_bindings:
            dates = cls._date_list(reconciliation.get("official_snapshot_dates"))
            if (
                dates is not None
                and artifact_sha256 == value["input_artifact_sha256"]
                and reconciliation.get("legacy_artifact_sha256")
                == value["original_artifact_sha256"]
                and dates[-1] == backup_target
                and reconciliation.get("official_series_manifest_sha256")
                == value["backup_series_manifest_sha256"]
            ):
                return True
        return False

    @staticmethod
    def _normalized_persisted_daily(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)) or not value:
            raise IncrementalHistoryError("persisted daily history is invalid")
        rows = []
        previous: _datetime.date | None = None
        for row in value:
            if not isinstance(row, Mapping):
                raise IncrementalHistoryError("persisted daily history is invalid")
            day = _parse_date(row.get("Date"))
            if previous is not None and day <= previous:
                raise IncrementalHistoryError("persisted daily history is invalid")
            normalized = dict(row)
            normalized["Date"] = day.isoformat()
            rows.append(normalized)
            previous = day
        return rows

    def _validated_recovery_backup(
        self,
        result: HistoryRecoveryResult,
        *,
        symbol: str,
    ) -> tuple[
        dict[_datetime.date, dict[str, Any]],
        tuple[_datetime.date, ...],
        dict[_datetime.date, dict[str, str]],
        str,
        dict[str, Any],
    ]:
        if (
            not isinstance(result, HistoryRecoveryResult)
            or not self._is_sha256(result.input_artifact_sha256)
            or not self._is_sha256(result.original_artifact_sha256)
            or not self._is_sha256(result.expected_result_sha256)
            or not isinstance(result.backup_target_market_date, _datetime.date)
            or isinstance(result.backup_target_market_date, _datetime.datetime)
            or not self._is_sha256(result.backup_series_manifest_sha256)
            or not isinstance(result.backup_manifest_entry, Mapping)
            or not self._valid_reconciliation(
                result.reconciliation,
                target_date=result.backup_target_market_date,
            )
        ):
            raise IncrementalHistoryError("daily history recovery binding is invalid")
        entry = result.backup_manifest_entry
        entry_sha256 = self._canonical_json_sha256(entry)
        if (
            entry.get("original_sha256") != result.original_artifact_sha256
            or entry.get("new_sha256") != result.expected_result_sha256
            or not self._positive_int(entry.get("original_size"))
            or not self._positive_int(entry.get("original_uncompressed_size"))
            or result.reconciliation.get("legacy_artifact_sha256")
            != result.original_artifact_sha256
            or result.reconciliation.get("official_series_manifest_sha256")
            != result.backup_series_manifest_sha256
            or _datetime.date.fromisoformat(
                result.reconciliation["official_snapshot_dates"][-1]
            ) != result.backup_target_market_date
        ):
            raise IncrementalHistoryError("daily history recovery binding is invalid")
        backup: dict[_datetime.date, dict[str, Any]] = {}
        previous: _datetime.date | None = None
        for row in result.backup_daily:
            if not isinstance(row, Mapping):
                raise IncrementalHistoryError("daily history recovery backup is invalid")
            day = _parse_date(row.get("Date"))
            if previous is not None and day <= previous:
                raise IncrementalHistoryError("daily history recovery backup is invalid")
            backup[day] = dict(row, Date=day.isoformat())
            previous = day
        candidates = []
        for row in result.restored_candidates:
            if not isinstance(row, Mapping):
                raise IncrementalHistoryError("daily history recovery backup is invalid")
            day = _parse_date(row.get("Date"))
            if day not in backup or day in candidates:
                raise IncrementalHistoryError("daily history recovery backup is invalid")
            candidates.append(day)
        if candidates != sorted(candidates):
            raise IncrementalHistoryError("daily history recovery backup is invalid")
        evidence = {
            _datetime.date.fromisoformat(row["date"]): dict(row)
            for row in result.reconciliation["date_evidence"]
        }
        return backup, tuple(candidates), evidence, entry_sha256, dict(entry)

    def _validate_recovered_final_row(
        self,
        *,
        symbol: str,
        final: Mapping[str, Any],
        backup: Mapping[str, Any],
        day: _datetime.date,
        evidence: Mapping[_datetime.date, Mapping[str, str]],
    ) -> None:
        if final.get("Date") != backup.get("Date"):
            raise IncrementalHistoryError("daily history recovery final row is invalid")
        actions = evidence.get(day, {})
        snapshot = self.snapshots.get(day)
        if actions.get("price_action") == "replaced_official":
            if snapshot is None:
                raise IncrementalHistoryError("daily history recovery official row is unavailable")
            official = self._official_price(snapshot, symbol)
            if official is None or any(
                self._number(final, historical) != float(official[official_name])
                for historical, official_name in (
                    ("Open", "open"), ("High", "max"), ("Low", "min"),
                    ("Close", "close"), ("Volume", "Trading_Volume"),
                )
            ):
                raise IncrementalHistoryError("daily history recovery final row is invalid")
        else:
            for name in _RECOVERY_OHLCV_FIELDS:
                if self._number(final, name) != self._number(backup, name):
                    raise IncrementalHistoryError("daily history recovery final row is invalid")
        if actions.get("institutional_action") == "replaced_official":
            if snapshot is None:
                raise IncrementalHistoryError("daily history recovery official row is unavailable")
            institutional = self._official_institutional(snapshot, symbol)
            total = sum(float(row["buy"]) - float(row["sell"]) for row in institutional)
            foreign = sum(
                float(row["buy"]) - float(row["sell"])
                for row in institutional if row["name"] == "Foreign"
            )
            if (
                self._number(final, "InstitutionalNet", 0.0) != total
                or self._number(final, "ForeignNet", 0.0) != foreign
            ):
                raise IncrementalHistoryError("daily history recovery final row is invalid")
        else:
            for name in ("InstitutionalNet", "ForeignNet"):
                if self._number(final, name, 0.0) != self._number(
                    backup, name, 0.0
                ):
                    raise IncrementalHistoryError("daily history recovery final row is invalid")
        if actions.get("margin_action") == "replaced_official":
            if snapshot is None:
                raise IncrementalHistoryError("daily history recovery official row is unavailable")
            margin = self._official_margin(snapshot, symbol)
            if margin is None or (
                self._number(final, "MarginBalance", 0.0)
                != float(margin["MarginPurchaseTodayBalance"])
                or self._number(final, "ShortBalance", 0.0)
                != float(margin["ShortSaleTodayBalance"])
            ):
                raise IncrementalHistoryError("daily history recovery final row is invalid")
        else:
            for name in ("MarginBalance", "ShortBalance"):
                if self._number(final, name, 0.0) != self._number(
                    backup, name, 0.0
                ):
                    raise IncrementalHistoryError("daily history recovery final row is invalid")

    def _finalize_daily_history_recovery(
        self,
        result: HistoryRecoveryResult,
        *,
        symbol: str,
        recovery_target_market_date: _datetime.date,
        persisted_daily: Any,
    ) -> dict[str, Any] | None:
        if (
            not isinstance(recovery_target_market_date, _datetime.date)
            or isinstance(recovery_target_market_date, _datetime.datetime)
        ):
            raise IncrementalHistoryError("daily history recovery target is invalid")
        final = self._normalized_persisted_daily(persisted_daily)
        final_by_date = {
            _datetime.date.fromisoformat(row["Date"]): row for row in final
        }
        backup, candidates, evidence, entry_sha256, entry = self._validated_recovery_backup(
            result, symbol=symbol
        )
        existing = result.existing_receipt
        if existing is not None:
            if not self._valid_recovery_receipt(
                dict(existing),
                symbol=symbol,
                recovery_bindings=((result.reconciliation, result.expected_result_sha256),),
            ):
                raise IncrementalHistoryError("daily history recovery receipt is invalid")
            receipt = dict(existing)
            try:
                receipt_recovery_target = _datetime.date.fromisoformat(
                    receipt["recovery_target_market_date"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise IncrementalHistoryError(
                    "daily history recovery receipt is invalid"
            ) from exc
            if (
                not (
                    result.backup_target_market_date
                    <= receipt_recovery_target
                    <= recovery_target_market_date
                )
                or receipt["original_artifact_sha256"]
                != result.original_artifact_sha256
                or receipt["backup_target_market_date"]
                != result.backup_target_market_date.isoformat()
                or receipt["backup_series_manifest_sha256"]
                != result.backup_series_manifest_sha256
                or receipt["backup_manifest_entry_sha256"] != entry_sha256
                or receipt["backup_object_size"] != entry["original_size"]
                or receipt["backup_object_uncompressed_size"]
                != entry["original_uncompressed_size"]
                or entry.get("new_sha256") != result.expected_result_sha256
            ):
                raise IncrementalHistoryError("daily history recovery receipt is invalid")
            start = _datetime.date.fromisoformat(receipt["restored_start_date"])
            end = _datetime.date.fromisoformat(receipt["restored_end_date"])
            authorized = [day for day in sorted(backup) if start <= day <= end]
            if (
                not authorized
                or len(authorized) != receipt["restored_row_count"]
                or authorized[0] != start
                or authorized[-1] != end
            ):
                raise IncrementalHistoryError("daily history recovery receipt is invalid")
        else:
            authorized = [day for day in candidates if day in final_by_date]
            if not authorized:
                return None
            start, end = authorized[0], authorized[-1]
        retained = []
        missing = []
        for day in authorized:
            row = final_by_date.get(day)
            if row is None:
                missing.append(day)
                continue
            self._validate_recovered_final_row(
                symbol=symbol,
                final=row,
                backup=backup[day],
                day=day,
                evidence=evidence,
            )
            retained.append(self._canonical_recovery_source_row(row))
        if existing is not None:
            if not missing:
                retained_dates = [
                    _datetime.date.fromisoformat(row["Date"])
                    for row in retained
                ]
                if (
                    receipt["restored_daily_sha256"]
                    != self._canonical_json_sha256(retained)
                    or receipt["restored_row_count"] != len(retained)
                    or retained_dates[0].isoformat()
                    != receipt["restored_start_date"]
                    or retained_dates[-1].isoformat()
                    != receipt["restored_end_date"]
                ):
                    raise IncrementalHistoryError("daily history recovery receipt is invalid")
            elif len(missing) == len(authorized):
                floor = _datetime.date.fromisoformat(final[0]["Date"])
                if any(day >= floor for day in missing):
                    raise IncrementalHistoryError("daily history recovery receipt is invalid")
            else:
                raise IncrementalHistoryError("daily history recovery receipt is invalid")
            return receipt
        unsigned = {
            "schema_version": 1,
            "mode": "restore_verified_reconciliation_backup",
            "symbol": symbol,
            "recovery_target_market_date": recovery_target_market_date.isoformat(),
            "input_artifact_sha256": result.input_artifact_sha256,
            "original_artifact_sha256": result.original_artifact_sha256,
            "backup_target_market_date": result.backup_target_market_date.isoformat(),
            "backup_series_manifest_sha256": result.backup_series_manifest_sha256,
            "backup_manifest_entry_sha256": entry_sha256,
            "backup_object_size": entry["original_size"],
            "backup_object_uncompressed_size": entry["original_uncompressed_size"],
            "restored_start_date": start.isoformat(),
            "restored_end_date": end.isoformat(),
            "restored_row_count": len(retained),
            "restored_daily_sha256": self._canonical_json_sha256(retained),
        }
        return {**unsigned, "receipt_sha256": self._canonical_json_sha256(unsigned)}

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
        status_aware = (
            lineage.get("source_schema_version")
            == "tw-official-historical-v3"
        )
        if (
            snapshot_dates is None
            or (
                target_date
                != (
                    artifact.observation_date
                    if status_aware
                    else artifact.latest_date
                )
            )
            or historical_as_of > target_date
            or snapshot_dates[-1] != target_date
        ):
            return False
        if status_aware:
            status = artifact.trading_status_evidence
            observation_kind = (
                status.get("status") if status is not None else "regular_price"
            )
            if (
                artifact.document.get("schema_version") != 2
                or lineage.get("observation_as_of") != target_date.isoformat()
                or lineage.get("latest_regular_price_date")
                != artifact.latest_date.isoformat()
                or lineage.get("observation_kind") != observation_kind
                or lineage.get("official_target_price_available")
                != (status is None)
                or (
                    status is None
                    and "trading_status_evidence_sha256" in lineage
                )
                or (
                    status is not None
                    and lineage.get("trading_status_evidence_sha256")
                    != status.get("evidence_sha256")
                )
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
        history = lineage.get("legacy_reconciliation_history", _MISSING)
        if history is not _MISSING:
            if (
                reconciliation is not _MISSING
                or not isinstance(history, list)
                or not history
            ):
                return False
            history_dates = []
            artifact_hashes = []
            for item in history:
                reconciliation_value = (
                    item.get("reconciliation")
                    if isinstance(item, dict)
                    else None
                )
                dates = (
                    cls._date_list(
                        reconciliation_value.get("official_snapshot_dates")
                    )
                    if isinstance(reconciliation_value, dict)
                    else None
                )
                if (
                    not isinstance(item, dict)
                    or set(item) != _RECONCILIATION_HISTORY_FIELDS
                    or item.get("schema_version") != 2
                    or item.get("symbol") != artifact.symbol
                    or not cls._is_sha256(
                        item.get("reconciled_artifact_sha256")
                    )
                    or not cls._is_sha256(item.get("history_sha256"))
                    or dates is None
                    or not cls._valid_reconciliation(
                        reconciliation_value,
                        target_date=target_date,
                    )
                ):
                    return False
                history_sha256 = cls._canonical_json_sha256(
                    {
                        name: value
                        for name, value in item.items()
                        if name != "history_sha256"
                    }
                )
                if item["history_sha256"] != history_sha256:
                    return False
                history_dates.append(dates[-1])
                artifact_hashes.append(item["reconciled_artifact_sha256"])
            if (
                history_dates != sorted(set(history_dates))
                or len(artifact_hashes) != len(set(artifact_hashes))
            ):
                return False
            receipt = lineage.get("daily_history_recovery", _MISSING)
            return (
                receipt is _MISSING
                or cls._valid_recovery_receipt(
                    receipt,
                    symbol=artifact.symbol,
                    recovery_bindings=(
                        (
                            item["reconciliation"],
                            item["reconciled_artifact_sha256"],
                        )
                        for item in history
                    ),
                )
            )
        if reconciliation is _MISSING:
            return "daily_history_recovery" not in lineage
        valid_direct = (
            cls._valid_reconciliation(
                reconciliation,
                target_date=target_date,
            )
            and lineage["historical_artifact_sha256"]
            == reconciliation["legacy_artifact_sha256"]
            and lineage["historical_as_of"]
            == reconciliation["legacy_artifact_as_of"]
            and lineage["source_mode"] == reconciliation["official_source_mode"]
            and lineage["source_schema_version"]
            == reconciliation["official_source_schema_version"]
            and lineage["official_series_manifest_sha256"]
            == reconciliation["official_series_manifest_sha256"]
            and lineage["official_snapshot_dates"]
            == reconciliation["official_snapshot_dates"]
            and lineage["official_snapshot_manifests"]
            == reconciliation["official_snapshot_manifests"]
        )
        return valid_direct and "daily_history_recovery" not in lineage

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
            history = lineage.get("legacy_reconciliation_history")
            if history is not None:
                self._existing_reconciliation_history[symbol] = copy.deepcopy(
                    history
                )
            receipt = lineage.get("daily_history_recovery")
            if receipt is not None:
                self._existing_recovery_receipts[symbol] = copy.deepcopy(receipt)
        else:
            if isinstance(lineage, dict) and lineage.get("source_mode") == SOURCE_MODE:
                kind = "legacy"
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
        recovery = self._ensure_history_recovery(symbol)
        daily = (
            recovery.merged_daily
            if recovery is not None
            else artifact.document["daily"]
        )
        rows: list[dict[str, Any]] = []
        for item in daily:
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

    def status_for(self, symbol: str) -> dict[str, Any] | None:
        symbol = str(symbol)
        snapshot = self.snapshots[self.target_date]
        status = snapshot.trading_status_by_symbol.get(symbol)
        if status is None:
            return None
        document = json.loads(
            json.dumps(dict(status), ensure_ascii=False, allow_nan=False)
        )
        if (
            document.get("status") not in {
                "official_no_regular_trade", "officially_suspended"
            }
            or document.get("market") != "TW"
            or document.get("symbol") != symbol
            or document.get("target_market_date") != self.target_date.isoformat()
            or document.get("evidence_sha256") != evidence_sha256(document)
            or symbol in snapshot.price_by_symbol
            or symbol in snapshot.terminated_by_symbol
        ):
            raise IncrementalHistoryError(
                f"official status is invalid for TW:{symbol}"
            )
        return document

    def _record_reconciliation(
        self,
        symbol: str,
        value: _datetime.date,
        snapshot: OfficialDailySnapshot,
    ) -> None:
        price = self._official_price(snapshot, symbol)
        institutional = self._official_institutional(snapshot, symbol)
        margin = self._official_margin(snapshot, symbol)
        current = {
            "price_action": (
                "replaced_official"
                if price is not None
                else "preserved_legacy_no_official_row"
            ),
            "institutional_action": (
                "replaced_official"
                if institutional
                else "preserved_legacy_no_official_row"
            ),
            "margin_action": (
                "replaced_official"
                if margin is not None
                else "preserved_legacy_no_official_row"
            ),
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
    ) -> dict[_datetime.date, dict[str, str]]:
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
        reconciliation = (
            {
                value: actions
                for value, actions in self._reconciliation_plan(symbol).items()
                if value in historical_by_date
            }
            if replace_legacy
            else {}
        )
        for value, snapshot in self.snapshots.items():
            existing = historical_by_date.get(value)
            if existing is not None and not replace_legacy:
                self._verify_existing(symbol, existing, snapshot)

        rows: list[dict[str, Any]] = []
        if dataset == "TaiwanStockPrice":
            for row in historical:
                if (
                    reconciliation.get(row["_date"], {}).get("price_action")
                    == "replaced_official"
                ):
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
                    if reconciliation.get(row["_date"], {}).get(
                        "institutional_action"
                    ) == "replaced_official"
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
                    if reconciliation.get(row["_date"], {}).get(
                        "margin_action"
                    ) == "replaced_official"
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
            "schema_version": 2,
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
            "overlap_dates": [value.isoformat() for value in dates],
            **{
                f"{dataset}_{suffix}_dates": [
                    value.isoformat()
                    for value in dates
                    if recorded[value][f"{dataset}_action"] == action
                ]
                for dataset in ("price", "institutional", "margin")
                for suffix, action in (
                    ("replaced", "replaced_official"),
                    (
                        "preserved_no_official_row",
                        "preserved_legacy_no_official_row",
                    ),
                )
            },
            "date_evidence": [
                {"date": value.isoformat(), **recorded[value]}
                for value in dates
            ],
        }

    def lineage_for(
        self,
        symbol: str,
        *,
        persisted_daily: Any | None = None,
    ) -> dict[str, Any]:
        artifact = self._load_artifact(symbol)
        recovery = self._ensure_history_recovery(symbol)
        self._lineage_kind(symbol)
        status = self.status_for(symbol)
        latest_regular_price_date = artifact.latest_date
        if status is not None and persisted_daily is not None:
            normalized_persisted_daily = self._normalized_persisted_daily(
                persisted_daily
            )
            latest_regular_price_date = _parse_date(
                normalized_persisted_daily[-1]["Date"]
            )
            if not (
                artifact.latest_date
                <= latest_regular_price_date
                < self.target_date
            ):
                raise IncrementalHistoryError(
                    "status lineage latest regular price date is invalid"
                )
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
            "observation_as_of": self.target_date.isoformat(),
            "latest_regular_price_date": (
                self.target_date.isoformat()
                if status is None
                else latest_regular_price_date.isoformat()
            ),
            "observation_kind": (
                status["status"] if status is not None else "regular_price"
            ),
        }
        if status is not None:
            lineage["trading_status_evidence_sha256"] = status[
                "evidence_sha256"
            ]
        reconciliation = self.reconciliation_for(symbol)
        existing_reconciliation = self._existing_reconciliations.get(symbol)
        if reconciliation is None and existing_reconciliation is not None:
            same_series = (
                existing_reconciliation["official_source_mode"]
                == self.source_mode
                and existing_reconciliation["official_source_schema_version"]
                == self.source_schema_version
                and existing_reconciliation["official_series_manifest_sha256"]
                == self.series_manifest_sha256
                and existing_reconciliation["official_snapshot_dates"]
                == lineage["official_snapshot_dates"]
                and existing_reconciliation["official_snapshot_manifests"]
                == lineage["official_snapshot_manifests"]
            )
            if same_series:
                reconciliation = existing_reconciliation
                lineage.update(
                    historical_artifact_sha256=reconciliation[
                        "legacy_artifact_sha256"
                    ],
                    historical_as_of=reconciliation["legacy_artifact_as_of"],
                )
            else:
                reconciliation_copy = copy.deepcopy(existing_reconciliation)
                history_item = {
                    "schema_version": 2,
                    "symbol": symbol,
                    "reconciled_artifact_sha256": artifact.compressed_sha256,
                    "reconciliation": reconciliation_copy,
                }
                history_item["history_sha256"] = self._canonical_json_sha256(
                    history_item
                )
                lineage["legacy_reconciliation_history"] = [
                    *copy.deepcopy(
                        self._existing_reconciliation_history.get(symbol, [])
                    ),
                    history_item,
                ]
        elif symbol in self._existing_reconciliation_history:
            lineage["legacy_reconciliation_history"] = copy.deepcopy(
                self._existing_reconciliation_history[symbol]
            )
        if reconciliation is not None:
            lineage["legacy_reconciliation"] = copy.deepcopy(reconciliation)
        if recovery is not None:
            if persisted_daily is None:
                raise IncrementalHistoryError(
                    "final persisted daily history is required for recovery"
                )
            receipt = self._finalize_daily_history_recovery(
                recovery,
                symbol=symbol,
                recovery_target_market_date=self.target_date,
                persisted_daily=persisted_daily,
            )
            if receipt is not None:
                direct = self._existing_reconciliations.get(symbol)
                if direct is not None:
                    if (
                        self._existing_reconciliation_history.get(symbol)
                        or recovery.expected_result_sha256
                        != recovery.input_artifact_sha256
                        or direct["legacy_artifact_sha256"]
                        != recovery.original_artifact_sha256
                        or _datetime.date.fromisoformat(
                            direct["official_snapshot_dates"][-1]
                        ) != recovery.backup_target_market_date
                        or direct["official_series_manifest_sha256"]
                        != recovery.backup_series_manifest_sha256
                    ):
                        raise IncrementalHistoryError(
                            "daily history recovery direct binding is invalid"
                        )
                    history_item = {
                        "schema_version": 2,
                        "symbol": symbol,
                        "reconciled_artifact_sha256": recovery.input_artifact_sha256,
                        "reconciliation": copy.deepcopy(direct),
}
                    history_item["history_sha256"] = self._canonical_json_sha256(
                        history_item
                    )
                    lineage.pop("legacy_reconciliation", None)
                    lineage["legacy_reconciliation_history"] = [history_item]
                elif not self._existing_reconciliation_history.get(symbol):
                    if recovery.reconciliation is None:
                        raise IncrementalHistoryError(
                            "daily history recovery binding is invalid"
                        )
                    history_item = {
                        "schema_version": 2,
                        "symbol": symbol,
                        "reconciled_artifact_sha256": recovery.input_artifact_sha256,
                        "reconciliation": _canonical_json_value(
                            recovery.reconciliation
                        ),
                    }
                    history_item["history_sha256"] = self._canonical_json_sha256(
                        history_item
                    )
                    lineage["legacy_reconciliation_history"] = [history_item]
                lineage["daily_history_recovery"] = dict(receipt)
        elif symbol in self._existing_recovery_receipts:
            lineage["daily_history_recovery"] = copy.deepcopy(
                self._existing_recovery_receipts[symbol]
            )
        return lineage
