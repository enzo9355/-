"""Immutable backup state for explicit TW legacy artifact reconciliation."""

from __future__ import annotations

import contextlib
import copy
import datetime
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any

from stock_papi.quant.tw_incremental import (
    MAX_COMPRESSED_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    SOURCE_MODE,
    HistoryRecoveryResult,
    IncrementalArtifact,
    IncrementalHistoryError,
    OfficialCompatFetcher,
    load_incremental_artifact,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SYMBOL_RE = re.compile(r"[0-9]{4,5}[0-9A-Z]?")
_MANIFEST_FIELDS = {
    "schema_version",
    "target_market_date",
    "official_series_manifest_sha256",
    "entries",
}
_ENTRY_FIELDS = {
    "symbol",
    "status",
    "original_sha256",
    "original_size",
    "original_uncompressed_size",
    "backup_path",
    "overlap_dates",
    "new_sha256",
}


class LegacyReconciliationError(RuntimeError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _assert_safe_child(root: Path, path: Path) -> Path:
    root = _absolute(root)
    path = _absolute(path)
    if path != root and not path.is_relative_to(root):
        raise LegacyReconciliationError("legacy reconciliation path is invalid")
    current = root
    if _is_reparse(current):
        raise LegacyReconciliationError("legacy reconciliation path is invalid")
    for part in path.relative_to(root).parts:
        current /= part
        if _is_reparse(current):
            raise LegacyReconciliationError("legacy reconciliation path is invalid")
    return path


def _decode_gzip(raw: bytes) -> bytes:
    if not 0 < len(raw) <= MAX_COMPRESSED_BYTES:
        raise LegacyReconciliationError("legacy reconciliation artifact is invalid")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
            decoded = stream.read(MAX_UNCOMPRESSED_BYTES + 1)
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        raise LegacyReconciliationError(
            "legacy reconciliation artifact is invalid"
        ) from exc
    if not decoded or len(decoded) > MAX_UNCOMPRESSED_BYTES:
        raise LegacyReconciliationError("legacy reconciliation artifact is invalid")
    return decoded


def _read_bytes(path: Path, *, max_bytes: int = MAX_COMPRESSED_BYTES) -> bytes:
    try:
        size = path.stat().st_size
        if not 0 < size <= max_bytes:
            raise ValueError("size")
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise LegacyReconciliationError(
            "legacy reconciliation artifact is unavailable"
        ) from exc
    except ValueError as exc:
        raise LegacyReconciliationError(
            "legacy reconciliation artifact is invalid"
        ) from exc
    if len(raw) != size or len(raw) > max_bytes:
        raise LegacyReconciliationError("legacy reconciliation artifact changed")
    return raw


def _read_verified_object(
    root: Path,
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_uncompressed_size: int,
    expected_bytes: bytes | None = None,
) -> tuple[bytes, bytes]:
    _assert_safe_child(root, path)
    raw = _read_bytes(path)
    if (
        len(raw) != expected_size
        or _sha256(raw) != expected_sha256
        or (expected_bytes is not None and raw != expected_bytes)
    ):
        raise LegacyReconciliationError(
            "legacy reconciliation backup object conflicts"
        )
    decoded = _decode_gzip(raw)
    if len(decoded) != expected_uncompressed_size:
        raise LegacyReconciliationError(
            "legacy reconciliation backup object conflicts"
        )
    return raw, decoded


def _dates(value: Any) -> tuple[datetime.date, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        parsed = tuple(datetime.date.fromisoformat(item) for item in value)
    except (TypeError, ValueError):
        return None
    if parsed != tuple(sorted(set(parsed))):
        return None
    return parsed


_RECOVERY_MISSING = object()
_OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def _finite_number(value: Any) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LegacyReconciliationError("daily history OHLCV is invalid")
    try:
        if not math.isfinite(value):
            raise LegacyReconciliationError("daily history OHLCV is invalid")
    except OverflowError as exc:
        raise LegacyReconciliationError("daily history OHLCV is invalid") from exc
    return value


def _validated_daily_by_date(value: Any) -> dict[datetime.date, dict[str, Any]]:
    if not isinstance(value, list):
        raise LegacyReconciliationError("daily history rows are invalid")
    rows: dict[datetime.date, dict[str, Any]] = {}
    previous: datetime.date | None = None
    for row in value:
        if not isinstance(row, dict):
            raise LegacyReconciliationError("daily history rows are invalid")
        try:
            day = datetime.datetime.fromisoformat(row["Date"]).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise LegacyReconciliationError("daily history date is invalid") from exc
        if previous is not None and day <= previous:
            raise LegacyReconciliationError("daily history dates are invalid")
        for name in _OHLCV_FIELDS:
            _finite_number(row.get(name))
        rows[day] = dict(row)
        previous = day
    return rows


def _merge_recovery_daily(
    active_daily: Any,
    backup_daily: Any,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    active = _validated_daily_by_date(active_daily)
    backup = _validated_daily_by_date(backup_daily)
    for day in active.keys() & backup.keys():
        for name in _OHLCV_FIELDS:
            if _finite_number(active[day].get(name)) != _finite_number(
                backup[day].get(name)
            ):
                raise LegacyReconciliationError("daily history OHLCV conflict")
    merged = {**backup, **active}
    restored_dates = sorted(backup.keys() - active.keys())
    if restored_dates and restored_dates[-1] >= min(active):
        raise LegacyReconciliationError("daily history recovery is not a missing prefix")
    return (
        tuple(dict(merged[day]) for day in sorted(merged)),
        tuple(dict(backup[day]) for day in restored_dates),
    )


def _immutable_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable_copy(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_copy(item) for item in value)
    return copy.deepcopy(value)


_FALLBACK_RECOVERY_KIND = "historical_artifact_sha256"


def _recovery_fallback_snapshot_manifest(
    root: Path,
    as_of: datetime.date,
) -> tuple[str, str]:
    from stock_papi.integrations.market_data.tw_official_historical import (
        build_historical_daily_snapshot,
    )

    snapshot = build_historical_daily_snapshot(root, as_of)
    return snapshot.manifest_sha256, snapshot.source_schema_version


def _recover_via_historical_artifact_sha256(
    root: Path,
    symbol: str,
    artifact: IncrementalArtifact,
) -> HistoryRecoveryResult | None:
    lineage = artifact.document.get("source_lineage") or {}
    historical_sha = lineage.get("historical_artifact_sha256")
    historical_as_of = lineage.get("historical_as_of")
    if not _is_sha256(historical_sha) or not isinstance(historical_as_of, str):
        raise LegacyReconciliationError(
            "daily history recovery fallback binding is unavailable"
        )
    try:
        historical_as_of_date = datetime.date.fromisoformat(historical_as_of)
    except ValueError as exc:
        raise LegacyReconciliationError(
            "daily history recovery fallback binding is unavailable"
        ) from exc

    base = (
        Path(root)
        / "quarantine"
        / "tw-recovery"
        / "legacy-reconciliation"
        / "v2"
    )
    if not base.is_dir():
        raise LegacyReconciliationError(
            "daily history recovery fallback backup is unavailable"
        )
    _assert_safe_child(Path(root), base)

    candidates: list[
        tuple[datetime.date, str, dict[str, Any], dict[str, Any]]
    ] = []
    for target_dir in sorted(base.iterdir()):
        if not target_dir.is_dir():
            continue
        try:
            target_date = datetime.date.fromisoformat(target_dir.name)
        except ValueError:
            continue
        for series_dir in sorted(target_dir.iterdir()):
            if not series_dir.is_dir() or not _is_sha256(series_dir.name):
                continue
            store = LegacyArtifactBackupStore(
                Path(root),
                target_date=target_date,
                series_manifest_sha256=series_dir.name,
            )
            try:
                manifest = store._load_manifest(required=True)
            except LegacyReconciliationError as exc:
                raise LegacyReconciliationError(
                    "daily history recovery fallback backup is invalid"
                ) from exc
            entry = manifest["entries"].get(symbol)
            if (
                entry is None
                or entry.get("original_sha256") != historical_sha
                or entry.get("status") != "applied"
                or not _is_sha256(entry.get("new_sha256"))
            ):
                continue
            try:
                backup_document, verified_entry = store.read_original_document(
                    symbol=symbol,
                    original_sha256=historical_sha,
                    expected_result_sha256=entry["new_sha256"],
                )
            except LegacyReconciliationError as exc:
                raise LegacyReconciliationError(
                    "daily history recovery fallback backup object conflicts"
                ) from exc
            candidates.append(
                (target_date, series_dir.name, backup_document, verified_entry)
            )

    if not candidates:
        raise LegacyReconciliationError(
            "daily history recovery fallback backup is unavailable"
        )

    identities = {
        (entry["original_sha256"], entry["original_size"], entry["original_uncompressed_size"])
        for _target, _series, _document, entry in candidates
    }
    new_shas = {entry["new_sha256"] for _target, _series, _document, entry in candidates}
    as_ofs = {document["as_of"] for _target, _series, document, _entry in candidates}
    if len(identities) != 1 or len(new_shas) != 1 or len(as_ofs) != 1:
        raise LegacyReconciliationError(
            "daily history recovery fallback authorization is ambiguous"
        )

    _target, _series, backup_document, entry = candidates[0]
    backup_as_of = datetime.date.fromisoformat(backup_document["as_of"])
    if backup_as_of != historical_as_of_date:
        raise LegacyReconciliationError(
            "daily history recovery fallback identity is invalid"
        )

    manifest_sha256, source_schema_version = _recovery_fallback_snapshot_manifest(
        Path(root), backup_as_of
    )
    as_of_text = backup_as_of.isoformat()
    series_manifest_sha256 = OfficialCompatFetcher._canonical_series_sha256(
        SOURCE_MODE,
        source_schema_version,
        backup_as_of,
        [(backup_as_of, manifest_sha256)],
    )
    reconciliation: dict[str, Any] = {
        "schema_version": 2,
        "mode": "replace_verified_legacy",
        "legacy_artifact_sha256": historical_sha,
        "legacy_artifact_as_of": as_of_text,
        "official_source_mode": SOURCE_MODE,
        "official_source_schema_version": source_schema_version,
        "official_series_manifest_sha256": series_manifest_sha256,
        "official_snapshot_dates": [as_of_text],
        "official_snapshot_manifests": [
            {"date": as_of_text, "manifest_sha256": manifest_sha256}
        ],
        "overlap_dates": [as_of_text],
        "price_replaced_dates": [],
        "price_preserved_no_official_row_dates": [as_of_text],
        "institutional_replaced_dates": [],
        "institutional_preserved_no_official_row_dates": [as_of_text],
        "margin_replaced_dates": [],
        "margin_preserved_no_official_row_dates": [as_of_text],
        "date_evidence": [
            {
                "date": as_of_text,
                "price_action": "preserved_legacy_no_official_row",
                "institutional_action": "preserved_legacy_no_official_row",
                "margin_action": "preserved_legacy_no_official_row",
            }
        ],
        "recovery_kind": _FALLBACK_RECOVERY_KIND,
    }

    merged_daily, restored_candidates = _merge_recovery_daily(
        artifact.document.get("daily"), backup_document.get("daily")
    )
    existing_receipt = lineage.get("daily_history_recovery")
    if existing_receipt is not None and not isinstance(existing_receipt, dict):
        raise LegacyReconciliationError("daily history recovery receipt is invalid")
    return HistoryRecoveryResult(
        merged_daily=tuple(_immutable_copy(row) for row in merged_daily),
        restored_candidates=tuple(_immutable_copy(row) for row in restored_candidates),
        backup_daily=tuple(
            _immutable_copy(row)
            for row in _validated_daily_by_date(backup_document.get("daily")).values()
        ),
        input_artifact_sha256=artifact.compressed_sha256,
        original_artifact_sha256=historical_sha,
        expected_result_sha256=entry["new_sha256"],
        backup_target_market_date=backup_as_of,
        backup_series_manifest_sha256=series_manifest_sha256,
        backup_manifest_entry=_immutable_copy(entry),
        reconciliation=_immutable_copy(reconciliation),
        existing_receipt=(
            _immutable_copy(existing_receipt) if existing_receipt is not None else None
        ),
    )


def resolve_truncated_daily_history(
    root: Path,
    symbol: str,
    artifact: IncrementalArtifact,
) -> HistoryRecoveryResult | None:
    lineage = artifact.document.get("source_lineage", _RECOVERY_MISSING)
    if lineage is _RECOVERY_MISSING or lineage is None:
        return None
    if not OfficialCompatFetcher._valid_official_lineage(lineage, artifact):
        raise LegacyReconciliationError(
            f"daily history recovery lineage is invalid for TW:{symbol}"
        )
    if artifact.symbol != symbol:
        raise LegacyReconciliationError("daily history recovery symbol is invalid")
    has_direct = "legacy_reconciliation" in lineage
    history = lineage.get("legacy_reconciliation_history", _RECOVERY_MISSING)
    has_history = history is not _RECOVERY_MISSING
    if not has_direct and not has_history:
        return _recover_via_historical_artifact_sha256(root, symbol, artifact)
    if (
        not has_direct
        and has_history
        and isinstance(history, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("reconciliation"), dict)
            and item["reconciliation"].get("recovery_kind") == _FALLBACK_RECOVERY_KIND
            for item in history
        )
    ):
        return None

    candidates: list[tuple[dict[str, Any], str]] = []
    direct = lineage.get("legacy_reconciliation", _RECOVERY_MISSING)
    if direct is not _RECOVERY_MISSING:
        candidates.append((direct, artifact.compressed_sha256))
    history = lineage.get("legacy_reconciliation_history", _RECOVERY_MISSING)
    if history is not _RECOVERY_MISSING:
        if not isinstance(history, list):
            raise LegacyReconciliationError("daily history recovery lineage is invalid")
        for item in history:
            if not isinstance(item, dict):
                raise LegacyReconciliationError("daily history recovery lineage is invalid")
            candidates.append(
                (item.get("reconciliation"), item.get("reconciled_artifact_sha256"))
            )
    if not candidates:
        raise LegacyReconciliationError("daily history recovery lineage is invalid")

    verified: list[
        tuple[
            tuple[Any, ...],
            datetime.date,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            str,
        ]
    ] = []
    for reconciliation, expected_result_sha256 in candidates:
        if not isinstance(reconciliation, dict) or not _is_sha256(
            expected_result_sha256
        ):
            raise LegacyReconciliationError("daily history recovery lineage is invalid")
        try:
            target = datetime.date.fromisoformat(
                reconciliation["official_snapshot_dates"][-1]
            )
            series_manifest_sha256 = reconciliation[
                "official_series_manifest_sha256"
            ]
            original_sha256 = reconciliation["legacy_artifact_sha256"]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise LegacyReconciliationError(
                "daily history recovery lineage is invalid"
            ) from exc
        try:
            store = LegacyArtifactBackupStore(
                Path(root),
                target_date=target,
                series_manifest_sha256=series_manifest_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise LegacyReconciliationError(
                "daily history recovery lineage is invalid"
            ) from exc
        backup_document, entry = store.read_original_document(
            symbol=symbol,
            original_sha256=original_sha256,
            expected_result_sha256=expected_result_sha256,
        )
        entry_sha256 = OfficialCompatFetcher._canonical_json_sha256(entry)
        if entry_sha256 is None:
            raise LegacyReconciliationError("daily history recovery binding is invalid")
        binding = (
            entry["original_sha256"],
            target.isoformat(),
            series_manifest_sha256,
            expected_result_sha256,
            entry["backup_path"],
            entry["original_size"],
            entry["original_uncompressed_size"],
            entry_sha256,
        )
        verified.append(
            (
                binding,
                target,
                reconciliation,
                entry,
                backup_document,
                expected_result_sha256,
            )
        )

    original_hashes = {binding[0] for binding, *_rest in verified}
    bindings = {binding for binding, *_rest in verified}
    if len(original_hashes) != 1 or len(bindings) != 1:
        raise LegacyReconciliationError(
            "daily history recovery authorization is ambiguous"
        )
    _binding, target, selected_reconciliation, entry, backup_document, expected = verified[0]
    merged_daily, restored_candidates = _merge_recovery_daily(
        artifact.document.get("daily"), backup_document.get("daily")
    )
    existing_receipt = lineage.get("daily_history_recovery")
    if existing_receipt is not None and not isinstance(existing_receipt, dict):
        raise LegacyReconciliationError("daily history recovery receipt is invalid")
    return HistoryRecoveryResult(
        merged_daily=tuple(_immutable_copy(row) for row in merged_daily),
        restored_candidates=tuple(_immutable_copy(row) for row in restored_candidates),
        backup_daily=tuple(
            _immutable_copy(row) for row in _validated_daily_by_date(
                backup_document.get("daily")
            ).values()
        ),
        input_artifact_sha256=artifact.compressed_sha256,
        original_artifact_sha256=entry["original_sha256"],
        expected_result_sha256=expected,
        backup_target_market_date=target,
        backup_series_manifest_sha256=selected_reconciliation[
            "official_series_manifest_sha256"
        ],
        backup_manifest_entry=_immutable_copy(entry),
        reconciliation=_immutable_copy(selected_reconciliation),
        existing_receipt=(
            _immutable_copy(existing_receipt) if existing_receipt is not None else None
        ),
    )


class LegacyArtifactBackupStore:
    def __init__(
        self,
        root: Path,
        *,
        target_date: datetime.date,
        series_manifest_sha256: str,
    ):
        if (
            not isinstance(target_date, datetime.date)
            or isinstance(target_date, datetime.datetime)
        ):
            raise TypeError("target_date must be a date")
        if not _is_sha256(series_manifest_sha256):
            raise ValueError("official series manifest is invalid")
        self.root = _absolute(Path(root))
        if not self.root.is_dir():
            raise ValueError("legacy reconciliation root is invalid")
        _assert_safe_child(self.root, self.root)
        self.target_date = target_date
        self.series_manifest_sha256 = series_manifest_sha256
        self.target_parent = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / target_date.isoformat()
        )
        self.backup_root = self.target_parent / series_manifest_sha256
        self.objects_dir = self.backup_root / "objects"
        self.manifest_path = self.backup_root / "manifest.json"

    def _expected_artifact_path(self, symbol: str) -> Path:
        if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
            raise LegacyReconciliationError("legacy reconciliation symbol is invalid")
        return self.root / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"

    def _validate_artifact_path(self, symbol: str, artifact_path: Path) -> Path:
        expected = self._expected_artifact_path(symbol)
        actual = _assert_safe_child(self.root, Path(artifact_path))
        if actual != expected:
            raise LegacyReconciliationError("legacy reconciliation path is invalid")
        return actual

    def _base_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "target_market_date": self.target_date.isoformat(),
            "official_series_manifest_sha256": self.series_manifest_sha256,
            "entries": {},
        }

    def _validate_entry(self, symbol: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
            raise LegacyReconciliationError("legacy reconciliation manifest is invalid")
        original_sha = value.get("original_sha256")
        overlap = _dates(value.get("overlap_dates"))
        status = value.get("status")
        new_sha = value.get("new_sha256")
        if (
            value.get("symbol") != symbol
            or _SYMBOL_RE.fullmatch(symbol) is None
            or status not in {"backup_complete", "applied"}
            or not _is_sha256(original_sha)
            or not isinstance(value.get("original_size"), int)
            or isinstance(value.get("original_size"), bool)
            or value["original_size"] <= 0
            or not isinstance(value.get("original_uncompressed_size"), int)
            or isinstance(value.get("original_uncompressed_size"), bool)
            or value["original_uncompressed_size"] <= 0
            or value.get("backup_path") != f"objects/{original_sha}.json.gz"
            or overlap is None
            or overlap[-1] > self.target_date
            or (status == "backup_complete" and new_sha is not None)
            or (status == "applied" and not _is_sha256(new_sha))
        ):
            raise LegacyReconciliationError("legacy reconciliation manifest is invalid")
        return value

    def _load_manifest(self, *, required: bool = False) -> dict[str, Any]:
        old_target = (
            self.target_parent.parent.parent
            / "v1"
            / self.target_date.isoformat()
        )
        _assert_safe_child(self.root, old_target)
        if old_target.exists():
            raise LegacyReconciliationError(
                "legacy reconciliation manifest schema is unsupported"
            )
        _assert_safe_child(self.root, self.manifest_path)
        if not self.manifest_path.exists():
            if required:
                raise LegacyReconciliationError(
                    "legacy reconciliation manifest is unavailable"
                )
            return self._base_manifest()
        try:
            document = json.loads(
                _read_bytes(
                    self.manifest_path, max_bytes=MAX_UNCOMPRESSED_BYTES
                ).decode("utf-8")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation manifest is invalid"
            ) from exc
        if (
            not isinstance(document, dict)
            or set(document) != _MANIFEST_FIELDS
            or document.get("schema_version") != 2
            or document.get("target_market_date") != self.target_date.isoformat()
            or document.get("official_series_manifest_sha256")
            != self.series_manifest_sha256
            or not isinstance(document.get("entries"), dict)
        ):
            raise LegacyReconciliationError("legacy reconciliation manifest is invalid")
        for symbol, entry in document["entries"].items():
            self._validate_entry(symbol, entry)
        return document

    def _ensure_directories(self) -> None:
        _assert_safe_child(self.root, self.backup_root)
        try:
            self.objects_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation backup directory is unavailable"
            ) from exc
        _assert_safe_child(self.root, self.objects_dir)

    @contextlib.contextmanager
    def _manifest_transaction(self):
        lock_root = (
            self.root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / ".locks"
            / self.target_date.isoformat()
        )
        _assert_safe_child(self.root, lock_root)
        try:
            lock_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation lock is unavailable"
            ) from exc
        _assert_safe_child(self.root, lock_root)
        lock_path = lock_root / f"{self.series_manifest_sha256}.lock"
        _assert_safe_child(self.root, lock_path)
        if _is_reparse(lock_path):
            raise LegacyReconciliationError(
                "legacy reconciliation lock is invalid"
            )
        try:
            with lock_path.open("a+b") as stream:
                if _is_reparse(lock_path):
                    raise LegacyReconciliationError(
                        "legacy reconciliation lock is invalid"
                    )
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                    unlock = msvcrt.locking
                    unlock_args = (stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    unlock = fcntl.flock
                    unlock_args = (stream.fileno(), fcntl.LOCK_UN)
                try:
                    yield
                finally:
                    stream.seek(0)
                    unlock(*unlock_args)
        except LegacyReconciliationError:
            raise
        except OSError as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation lock is unavailable"
            ) from exc

    @staticmethod
    def _write_descriptor(descriptor: int, payload: bytes) -> None:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _write_manifest(self, document: dict[str, Any]) -> None:
        self._ensure_directories()
        payload = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if self.manifest_path.exists():
            try:
                if _read_bytes(
                    self.manifest_path, max_bytes=MAX_UNCOMPRESSED_BYTES
                ) == payload:
                    return
            except OSError as exc:
                raise LegacyReconciliationError(
                    "legacy reconciliation manifest is unavailable"
                ) from exc
        descriptor, temporary = tempfile.mkstemp(
            prefix=".manifest-",
            suffix=".tmp",
            dir=self.backup_root,
        )
        temporary_path = Path(temporary)
        try:
            self._write_descriptor(descriptor, payload)
            _assert_safe_child(self.root, temporary_path)
            os.replace(temporary_path, self.manifest_path)
            _assert_safe_child(self.root, self.manifest_path)
        except (OSError, LegacyReconciliationError) as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, LegacyReconciliationError):
                raise
            raise LegacyReconciliationError(
                "legacy reconciliation manifest write failed"
            ) from exc

    def _verify_object(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int,
        expected_uncompressed_size: int,
        expected_bytes: bytes | None = None,
    ) -> None:
        _read_verified_object(
            self.root,
            path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            expected_uncompressed_size=expected_uncompressed_size,
            expected_bytes=expected_bytes,
        )

    def read_original_document(
        self,
        *,
        symbol: str,
        original_sha256: str,
        expected_result_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(symbol, str)
            or _SYMBOL_RE.fullmatch(symbol) is None
            or not _is_sha256(original_sha256)
            or not _is_sha256(expected_result_sha256)
        ):
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )
        manifest = self._load_manifest(required=True)
        entry = manifest["entries"].get(symbol)
        if entry is None:
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )
        entry = self._validate_entry(symbol, entry)
        if (
            entry["status"] != "applied"
            or entry["original_sha256"] != original_sha256
            or entry["new_sha256"] != expected_result_sha256
            or entry["backup_path"] != f"objects/{original_sha256}.json.gz"
        ):
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )
        _raw, decoded = _read_verified_object(
            self.root,
            self.backup_root / entry["backup_path"],
            expected_sha256=original_sha256,
            expected_size=entry["original_size"],
            expected_uncompressed_size=entry["original_uncompressed_size"],
        )
        try:
            document = json.loads(decoded.decode("utf-8"))
            daily = document["daily"]
            as_of = datetime.date.fromisoformat(document["as_of"])
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            ) from exc
        if (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("market") != "TW"
            or document.get("symbol") != symbol
            or not isinstance(daily, list)
            or not daily
        ):
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )
        dates = []
        for row in daily:
            try:
                date = datetime.datetime.fromisoformat(row["Date"]).date()
            except (KeyError, TypeError, ValueError) as exc:
                raise LegacyReconciliationError(
                    "legacy reconciliation backup object conflicts"
                ) from exc
            if not isinstance(row, dict):
                raise LegacyReconciliationError(
                    "legacy reconciliation backup object conflicts"
                )
            for name in ("Open", "High", "Low", "Close", "Volume"):
                value = row.get(name)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise LegacyReconciliationError(
                        "legacy reconciliation backup object conflicts"
                    )
                try:
                    valid_number = math.isfinite(value)
                except OverflowError:
                    valid_number = False
                if not valid_number:
                    raise LegacyReconciliationError(
                        "legacy reconciliation backup object conflicts"
                    )
            dates.append(date)
        if dates != sorted(set(dates)) or as_of != dates[-1]:
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )
        return document, dict(entry)

    def _publish_object(
        self,
        raw: bytes,
        *,
        original_sha256: str,
        uncompressed_size: int,
    ) -> Path:
        self._ensure_directories()
        target = self.objects_dir / f"{original_sha256}.json.gz"
        if target.exists() or target.is_symlink():
            self._verify_object(
                target,
                expected_sha256=original_sha256,
                expected_size=len(raw),
                expected_uncompressed_size=uncompressed_size,
                expected_bytes=raw,
            )
            return target
        descriptor, temporary = tempfile.mkstemp(
            prefix=".object-",
            suffix=".tmp",
            dir=self.objects_dir,
        )
        temporary_path = Path(temporary)
        try:
            self._write_descriptor(descriptor, raw)
            self._verify_object(
                temporary_path,
                expected_sha256=original_sha256,
                expected_size=len(raw),
                expected_uncompressed_size=uncompressed_size,
                expected_bytes=raw,
            )
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                pass
            self._verify_object(
                target,
                expected_sha256=original_sha256,
                expected_size=len(raw),
                expected_uncompressed_size=uncompressed_size,
                expected_bytes=raw,
            )
        except (OSError, LegacyReconciliationError) as exc:
            if isinstance(exc, LegacyReconciliationError):
                raise
            raise LegacyReconciliationError(
                "legacy reconciliation backup object write failed"
            ) from exc
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return target

    def _validate_evidence(
        self,
        symbol: str,
        value: Any,
        *,
        original_sha256: str,
        artifact_as_of: datetime.date,
    ) -> tuple[datetime.date, ...]:
        if (
            not OfficialCompatFetcher._valid_reconciliation(
                value, target_date=self.target_date
            )
            or value.get("schema_version") != 2
            or value.get("mode") != "replace_verified_legacy"
            or value.get("legacy_artifact_sha256") != original_sha256
            or value.get("legacy_artifact_as_of") != artifact_as_of.isoformat()
            or value.get("official_source_mode") != "tw_official_bulk_v2"
            or value.get("official_series_manifest_sha256")
            != self.series_manifest_sha256
        ):
            raise LegacyReconciliationError(
                f"legacy reconciliation evidence is invalid for TW:{symbol}"
            )
        overlap = _dates(value.get("overlap_dates"))
        if (
            overlap is None
            or overlap[-1] != artifact_as_of
            or overlap[-1] > self.target_date
        ):
            raise LegacyReconciliationError(
                f"legacy reconciliation evidence is invalid for TW:{symbol}"
            )
        return overlap

    def _validate_original(
        self,
        symbol: str,
        path: Path,
        value: Any,
    ) -> tuple[bytes, int, tuple[datetime.date, ...]]:
        try:
            artifact = load_incremental_artifact(self.root, symbol)
        except IncrementalHistoryError as exc:
            raise LegacyReconciliationError(
                f"legacy reconciliation artifact is invalid for TW:{symbol}"
            ) from exc
        raw = _read_bytes(path)
        decoded = _decode_gzip(raw)
        original_sha = _sha256(raw)
        if original_sha != artifact.compressed_sha256:
            raise LegacyReconciliationError(
                f"legacy reconciliation artifact changed for TW:{symbol}"
            )
        replaced = self._validate_evidence(
            symbol,
            value,
            original_sha256=original_sha,
            artifact_as_of=artifact.latest_date,
        )
        return raw, len(decoded), replaced

    def _verify_entry_object(self, entry: dict[str, Any]) -> None:
        self._verify_object(
            self.backup_root / entry["backup_path"],
            expected_sha256=entry["original_sha256"],
            expected_size=entry["original_size"],
            expected_uncompressed_size=entry["original_uncompressed_size"],
        )

    def _validate_expected_result(
        self,
        symbol: str,
        path: Path,
        entry: dict[str, Any],
    ) -> str:
        try:
            artifact = load_incremental_artifact(self.root, symbol)
        except IncrementalHistoryError as exc:
            raise LegacyReconciliationError(
                f"legacy reconciliation result is invalid for TW:{symbol}"
            ) from exc
        raw = _read_bytes(path)
        current_sha = _sha256(raw)
        lineage = artifact.document.get("source_lineage")
        reconciliation = (
            lineage.get("legacy_reconciliation")
            if isinstance(lineage, dict)
            else None
        )
        if (
            current_sha != artifact.compressed_sha256
            or artifact.latest_date != self.target_date
            or not OfficialCompatFetcher._valid_official_lineage(
                lineage, artifact
            )
            or not isinstance(lineage, dict)
            or lineage.get("source_mode") != "tw_official_bulk_v2"
            or lineage.get("target_market_date") != self.target_date.isoformat()
            or lineage.get("official_series_manifest_sha256")
            != self.series_manifest_sha256
            or not isinstance(reconciliation, dict)
            or reconciliation.get("schema_version") != 2
            or reconciliation.get("mode") != "replace_verified_legacy"
            or reconciliation.get("legacy_artifact_sha256")
            != entry["original_sha256"]
            or reconciliation.get("official_series_manifest_sha256")
            != self.series_manifest_sha256
            or reconciliation.get("overlap_dates") != entry["overlap_dates"]
        ):
            raise LegacyReconciliationError(
                f"legacy reconciliation result is invalid for TW:{symbol}"
            )
        return current_sha

    def _current_entry_state(
        self,
        symbol: str,
        entry: dict[str, Any],
    ) -> tuple[str, Path, str]:
        path = self._validate_artifact_path(
            symbol, self._expected_artifact_path(symbol)
        )
        self._verify_entry_object(entry)
        try:
            current_sha = _sha256(_read_bytes(path))
        except LegacyReconciliationError as exc:
            raise LegacyReconciliationError(
                f"legacy reconciliation state conflict for TW:{symbol}"
            ) from exc
        if entry["status"] == "backup_complete" and current_sha == entry[
            "original_sha256"
        ]:
            return "original", path, current_sha
        try:
            result_sha = self._validate_expected_result(symbol, path, entry)
        except LegacyReconciliationError as exc:
            raise LegacyReconciliationError(
                f"legacy reconciliation state conflict for TW:{symbol}"
            ) from exc
        if entry["status"] == "applied" and result_sha != entry["new_sha256"]:
            raise LegacyReconciliationError(
                f"legacy reconciliation state conflict for TW:{symbol}"
            )
        return "result", path, result_sha

    def backup_before_write(
        self,
        *,
        symbol: str,
        artifact_path: Path,
        evidence: dict[str, Any] | None,
    ) -> str:
        with self._manifest_transaction():
            return self._backup_before_write(
                symbol=symbol,
                artifact_path=artifact_path,
                evidence=evidence,
            )

    def _backup_before_write(
        self,
        *,
        symbol: str,
        artifact_path: Path,
        evidence: dict[str, Any] | None,
    ) -> str:
        path = self._validate_artifact_path(symbol, artifact_path)
        manifest = self._load_manifest()
        entry = manifest["entries"].get(symbol)
        if entry is None:
            if evidence is None:
                return "passthrough"
            raw, uncompressed_size, overlap = self._validate_original(
                symbol, path, evidence
            )
            original_sha = _sha256(raw)
            self._publish_object(
                raw,
                original_sha256=original_sha,
                uncompressed_size=uncompressed_size,
            )
            manifest["entries"][symbol] = {
                "symbol": symbol,
                "status": "backup_complete",
                "original_sha256": original_sha,
                "original_size": len(raw),
                "original_uncompressed_size": uncompressed_size,
                "backup_path": f"objects/{original_sha}.json.gz",
                "overlap_dates": [value.isoformat() for value in overlap],
                "new_sha256": None,
            }
            self._write_manifest(manifest)
            return "write"

        entry = self._validate_entry(symbol, entry)
        state, _current_path, _current_sha = self._current_entry_state(symbol, entry)
        if entry["status"] == "applied":
            return "noop"
        if state == "original":
            try:
                _raw, _size, overlap = self._validate_original(
                    symbol, path, evidence
                )
            except LegacyReconciliationError as exc:
                raise LegacyReconciliationError(
                    f"legacy reconciliation state conflict for TW:{symbol}"
                ) from exc
            if [value.isoformat() for value in overlap] != entry["overlap_dates"]:
                raise LegacyReconciliationError(
                    f"legacy reconciliation state conflict for TW:{symbol}"
                )
            return "write"
        entry["status"] = "applied"
        entry["new_sha256"] = _current_sha
        self._write_manifest(manifest)
        return "noop"

    def mark_applied(self, *, symbol: str, artifact_path: Path) -> Path:
        with self._manifest_transaction():
            return self._mark_applied(symbol=symbol, artifact_path=artifact_path)

    def _mark_applied(self, *, symbol: str, artifact_path: Path) -> Path:
        path = self._validate_artifact_path(symbol, artifact_path)
        manifest = self._load_manifest(required=True)
        entry = manifest["entries"].get(symbol)
        if entry is None:
            raise LegacyReconciliationError(
                f"legacy reconciliation state conflict for TW:{symbol}"
            )
        entry = self._validate_entry(symbol, entry)
        state, _current_path, new_sha = self._current_entry_state(symbol, entry)
        if state != "result":
            raise LegacyReconciliationError(
                f"legacy reconciliation state conflict for TW:{symbol}"
            )
        if entry["status"] == "applied":
            return path
        entry["status"] = "applied"
        entry["new_sha256"] = new_sha
        self._write_manifest(manifest)
        return path

    def assert_current_state_complete(self) -> dict[str, str]:
        with self._manifest_transaction():
            return self._assert_current_state_complete()

    def _assert_current_state_complete(self) -> dict[str, str]:
        manifest = self._load_manifest()
        for symbol, entry in manifest["entries"].items():
            if entry["status"] != "applied":
                raise LegacyReconciliationError(
                    "legacy reconciliation state is incomplete"
                )
            self._current_entry_state(symbol, entry)
        return {
            symbol: entry["new_sha256"]
            for symbol, entry in manifest["entries"].items()
        }

    @classmethod
    def discover_resume(
        cls,
        root: Path,
        *,
        target_date: datetime.date,
    ) -> tuple[str, datetime.date] | None:
        if (
            not isinstance(target_date, datetime.date)
            or isinstance(target_date, datetime.datetime)
        ):
            raise TypeError("target_date must be a date")
        root = _absolute(Path(root))
        old_parent = (
            root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v1"
            / target_date.isoformat()
        )
        _assert_safe_child(root, old_parent)
        if old_parent.exists():
            raise LegacyReconciliationError(
                "legacy reconciliation manifest schema is unsupported"
            )
        parent = (
            root
            / "quarantine"
            / "tw-recovery"
            / "legacy-reconciliation"
            / "v2"
            / target_date.isoformat()
        )
        _assert_safe_child(root, parent)
        if not parent.exists():
            return None
        try:
            candidates = []
            with os.scandir(parent) as entries:
                for entry in entries:
                    directory = Path(entry.path)
                    if (
                        not entry.is_dir(follow_symlinks=False)
                        or _is_reparse(directory)
                        or not (directory / "manifest.json").is_file()
                    ):
                        raise LegacyReconciliationError(
                            "legacy reconciliation resume state is invalid"
                        )
                    candidates.append(directory)
        except OSError as exc:
            raise LegacyReconciliationError(
                "legacy reconciliation resume state is unavailable"
            ) from exc
        results = []
        for directory in candidates:
            if not _is_sha256(directory.name):
                raise LegacyReconciliationError(
                    "legacy reconciliation resume state is invalid"
                )
            store = cls(
                root,
                target_date=target_date,
                series_manifest_sha256=directory.name,
            )
            manifest = store._load_manifest(required=True)
            if not manifest["entries"]:
                raise LegacyReconciliationError(
                    "legacy reconciliation resume state is invalid"
                )
            for symbol, entry in manifest["entries"].items():
                store._current_entry_state(symbol, entry)
            baseline = min(
                datetime.date.fromisoformat(value)
                for entry in manifest["entries"].values()
                for value in entry["overlap_dates"]
            )
            results.append((directory.name, baseline))
        if len(results) > 1:
            raise LegacyReconciliationError(
                "multiple legacy reconciliation series found"
            )
        return results[0] if results else None
