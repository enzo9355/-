"""Fail-closed local cache for verified TWSE/TPEx canonical snapshots."""

from __future__ import annotations

import datetime as _datetime
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

CACHE_SCHEMA_VERSION = 1
MAX_CANONICAL_BYTES = 20 * 1024 * 1024
MAX_COMPRESSED_BYTES = 10 * 1024 * 1024


class OfficialCacheError(RuntimeError):
    """Raised when an existing cache entry cannot be trusted."""


@dataclass(frozen=True)
class OfficialCacheEntry:
    source_id: str
    target_date: _datetime.date
    rows: tuple[dict[str, Any], ...]
    content_sha256: str
    compressed_sha256: str
    compressed_size: int
    row_count: int
    symbol_count: int
    parser_version: str
    payload_path: Path
    metadata_path: Path


def _canonical_json_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return json.dumps(
        list(rows),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    content = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write_bytes(path, content)


def _cache_dir(root: Path, target_date: _datetime.date) -> Path:
    return Path(root) / "source-cache" / "tw-official" / "v1" / target_date.isoformat()


def _metadata_path(root: Path, target_date: _datetime.date, source_id: str) -> Path:
    return _cache_dir(root, target_date) / f"{source_id}.metadata.json"


def _safe_source_identifier(source_url: str) -> str:
    return str(source_url).split("?", 1)[0]


def store_cached_source(
    root: Path,
    *,
    source_id: str,
    target_date: _datetime.date,
    rows: Iterable[Mapping[str, Any]],
    symbol_count: int,
    parser_version: str,
    source_url: str,
    fetched_at: _datetime.datetime,
    date_verification: str = "explicit",
) -> OfficialCacheEntry:
    if not source_id or not parser_version:
        raise ValueError("source_id and parser_version are required")
    if not isinstance(target_date, _datetime.date):
        raise TypeError("target_date must be a date")
    canonical_rows = tuple(dict(row) for row in rows)
    content = _canonical_json_bytes(canonical_rows)
    if not content or len(content) > MAX_CANONICAL_BYTES:
        raise OfficialCacheError("canonical source payload size is invalid")
    content_sha256 = hashlib.sha256(content).hexdigest()

    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(content)
    compressed = buffer.getvalue()
    if not compressed or len(compressed) > MAX_COMPRESSED_BYTES:
        raise OfficialCacheError("compressed source payload size is invalid")
    compressed_sha256 = hashlib.sha256(compressed).hexdigest()

    directory = _cache_dir(Path(root), target_date)
    payload_path = directory / f"{source_id}-{content_sha256}.json.gz"
    metadata_path = _metadata_path(Path(root), target_date, source_id)
    if not payload_path.exists():
        _atomic_write_bytes(payload_path, compressed)
    elif hashlib.sha256(payload_path.read_bytes()).hexdigest() != compressed_sha256:
        raise OfficialCacheError("existing source payload hash mismatch")

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_id": source_id,
        "target_date": target_date.isoformat(),
        "row_count": len(canonical_rows),
        "symbol_count": int(symbol_count),
        "content_sha256": content_sha256,
        "compressed_sha256": compressed_sha256,
        "compressed_size": len(compressed),
        "parser_version": parser_version,
        "source_url_identifier": _safe_source_identifier(source_url),
        "fetched_at": fetched_at.isoformat(),
        "date_verification": str(date_verification),
        "payload": payload_path.name,
    }
    _atomic_write_json(metadata_path, metadata)
    return OfficialCacheEntry(
        source_id=source_id,
        target_date=target_date,
        rows=canonical_rows,
        content_sha256=content_sha256,
        compressed_sha256=compressed_sha256,
        compressed_size=len(compressed),
        row_count=len(canonical_rows),
        symbol_count=int(symbol_count),
        parser_version=parser_version,
        payload_path=payload_path,
        metadata_path=metadata_path,
    )


def load_cached_source(
    root: Path,
    *,
    source_id: str,
    target_date: _datetime.date,
    parser_version: str,
) -> OfficialCacheEntry | None:
    metadata_path = _metadata_path(Path(root), target_date, source_id)
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != CACHE_SCHEMA_VERSION
            or metadata.get("source_id") != source_id
            or metadata.get("target_date") != target_date.isoformat()
        ):
            raise OfficialCacheError("source cache metadata schema mismatch")
        if metadata.get("parser_version") != parser_version:
            return None
        payload_name = metadata["payload"]
        if not isinstance(payload_name, str) or "/" in payload_name or "\\" in payload_name:
            raise OfficialCacheError("source cache payload path is invalid")
        payload_path = metadata_path.parent / payload_name
        compressed = payload_path.read_bytes()
        if (
            not compressed
            or len(compressed) > MAX_COMPRESSED_BYTES
            or len(compressed) != int(metadata["compressed_size"])
            or hashlib.sha256(compressed).hexdigest() != metadata["compressed_sha256"]
        ):
            raise OfficialCacheError("source cache compressed hash mismatch")
        import io

        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            content = stream.read(MAX_CANONICAL_BYTES + 1)
        if not content or len(content) > MAX_CANONICAL_BYTES:
            raise OfficialCacheError("source cache expansion is invalid")
        if hashlib.sha256(content).hexdigest() != metadata["content_sha256"]:
            raise OfficialCacheError("source cache content hash mismatch")
        rows = json.loads(content.decode("utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise OfficialCacheError("source cache rows are invalid")
        if len(rows) != int(metadata["row_count"]):
            raise OfficialCacheError("source cache row count mismatch")
        symbol_count = len({str(row.get("stock_id")) for row in rows if row.get("stock_id")})
        if symbol_count != int(metadata["symbol_count"]):
            raise OfficialCacheError("source cache symbol count mismatch")
    except OfficialCacheError:
        raise
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, gzip.BadGzipFile) as exc:
        raise OfficialCacheError("source cache is invalid") from exc

    return OfficialCacheEntry(
        source_id=source_id,
        target_date=target_date,
        rows=tuple(dict(row) for row in rows),
        content_sha256=str(metadata["content_sha256"]),
        compressed_sha256=str(metadata["compressed_sha256"]),
        compressed_size=int(metadata["compressed_size"]),
        row_count=len(rows),
        symbol_count=symbol_count,
        parser_version=parser_version,
        payload_path=payload_path,
        metadata_path=metadata_path,
    )
