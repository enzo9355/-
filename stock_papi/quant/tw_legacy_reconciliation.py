"""Immutable backup state for explicit TW legacy artifact reconciliation."""

from __future__ import annotations

import contextlib
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from stock_papi.quant.tw_incremental import (
    MAX_COMPRESSED_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    IncrementalHistoryError,
    OfficialCompatFetcher,
    load_incremental_artifact,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SYMBOL_RE = re.compile(r"[0-9]{4,6}")
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
        _assert_safe_child(self.root, path)
        raw = _read_bytes(path)
        decoded = _decode_gzip(raw)
        if (
            _sha256(raw) != expected_sha256
            or len(raw) != expected_size
            or len(decoded) != expected_uncompressed_size
            or (expected_bytes is not None and raw != expected_bytes)
        ):
            raise LegacyReconciliationError(
                "legacy reconciliation backup object conflicts"
            )

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
