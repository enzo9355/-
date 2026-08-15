"""Private, disabled-by-default TEJ research integration.

TEJ is deliberately kept outside the official TW production data path.  This
module owns entitlement discovery, immutable private caching, explicit entity
and field mappings, point-in-time selection, research factors, and advisory
comparisons with official values.  It never writes public artifacts and never
uses a TEJ value as market truth.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TEJ_PROVIDER = "TEJ"
TEJ_ENABLED_ENV = "TEJ_ENABLED"
TEJ_API_KEY_ENV = "TEJ_API_KEY"
TEJ_SCHEMA_VERSION = 1
TEJ_API_DOCUMENTATION = "https://api.tej.com.tw/document_python.html"
TEJ_MAX_RECORDS = 250_000
TEJ_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
TEJ_MAX_METADATA_BYTES = 1 * 1024 * 1024
TEJ_PRIVATE_ARTIFACT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_TABLE = re.compile(r"^[A-Za-z0-9_./-]{1,128}$")
_SENSITIVE_NAMES = ("api_key", "apikey", "token", "password", "secret")


class TejError(RuntimeError):
    """Operational or schema error with a secret-free machine status."""

    def __init__(self, status: str, detail: str = "", *, retry_after=None):
        self.status = str(status)
        self.retry_after = (
            int(retry_after)
            if type(retry_after) is int and 0 < retry_after <= 86400
            else None
        )
        self.detail = str(detail)[:200]
        self.safe_message = (
            f"TEJ status={self.status}"
            + (f" detail={self.detail}" if self.detail else "")
        )
        super().__init__(self.safe_message)

    def to_dict(self, **extra):
        result = {
            "provider": TEJ_PROVIDER,
            "status": self.status,
            "safe_message": self.safe_message,
        }
        if self.retry_after is not None:
            result["retry_after_seconds"] = self.retry_after
        result.update(
            {
                key: value
                for key, value in extra.items()
                if key not in _SENSITIVE_NAMES
            }
        )
        return result


class TejSchemaError(TejError):
    def __init__(self, detail: str):
        super().__init__("schema_mismatch", detail)


def _enabled(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _timestamp(value=None) -> str:
    value = value or datetime.datetime.now(datetime.timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_timestamp(value, label="timestamp") -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.datetime.fromisoformat(text)
        except ValueError as exc:
            raise TejSchemaError(f"{label} is invalid") from exc
    else:
        raise TejSchemaError(f"{label} is invalid")
    if result.tzinfo is None or result.utcoffset() is None:
        raise TejSchemaError(f"{label} must include timezone")
    return result.astimezone(datetime.timezone.utc)


def _parse_date(value, label="effective_date") -> datetime.date:
    if isinstance(value, datetime.datetime):
        value = value.date().isoformat()
    elif isinstance(value, datetime.date):
        value = value.isoformat()
    else:
        value = str(value or "").split("T", 1)[0]
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise TejSchemaError(f"{label} is invalid") from exc


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TejSchemaError("payload is not finite JSON") from exc


def _safe_child(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped allowlisted root") from exc
    return candidate


def validate_tej_data_root(root) -> Path:
    """Allow CLI operations only on the two approved local data roots."""

    candidate = Path(root).resolve()
    allowed = {
        Path(r"D:\AbsorbData").resolve(),
        Path(r"D:\StockPapiData").resolve(),
    }
    if os.path.normcase(str(candidate)) not in {
        os.path.normcase(str(path)) for path in allowed
    }:
        raise ValueError("TEJ data root is not allowlisted")
    return candidate


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_immutable(path: Path, content: bytes):
    """Publish one immutable local object without exposing a partial final file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ValueError("immutable TEJ artifact is unreadable") from exc
        if existing != content:
            raise ValueError("immutable TEJ artifact conflict")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.chmod(temporary_name, 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # os.link is an atomic create-if-absent operation on the NTFS volume
        # used by the Windows runner. It never overwrites a prior LKG object.
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != content:
                raise ValueError("immutable TEJ artifact conflict")
        except OSError as exc:
            raise ValueError("atomic TEJ artifact claim is unavailable") from exc
        if path.read_bytes() != content:
            raise ValueError("immutable TEJ artifact readback mismatch")
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise
    try:
        Path(temporary_name).unlink()
    except OSError:
        pass


def _credential_metadata_rejected(metadata: Mapping[str, Any]) -> bool:
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            name = str(key).lower()
            if any(token in name for token in _SENSITIVE_NAMES):
                return True
            if _credential_metadata_rejected(value):
                return True
    elif isinstance(metadata, (list, tuple)):
        return any(_credential_metadata_rejected(value) for value in metadata)
    return False


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return _timestamp(value) if isinstance(value, datetime.datetime) else value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _records_from_response(
    response, *, max_records: int = TEJ_MAX_RECORDS
) -> list[dict[str, Any]]:
    if hasattr(response, "to_dict"):
        try:
            response = response.to_dict(orient="records")
        except TypeError:
            response = response.to_dict()
    if isinstance(response, Mapping):
        response = response.get("data", response.get("records"))
    if not isinstance(response, list):
        raise TejSchemaError("TEJ response records are not a list")
    records = []
    for record in response:
        if len(records) >= max_records:
            raise TejError("payload_too_large", "TEJ response exceeds row limit")
        if not isinstance(record, Mapping):
            raise TejSchemaError("TEJ response record is not an object")
        records.append(_json_safe(record))
    if not records:
        raise TejError("empty_dataset", "TEJ returned no records")
    return records


def _exception_status(exc: Exception, *, dataset=False) -> tuple[str, int | None]:
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(exc, "http_status", None)
    text = f"{type(exc).__name__} {exc}".lower()
    if status_code in (401,):
        return "authentication_failed", None
    if status_code in (402, 429) or "rate" in text or "quota" in text:
        retry_after = getattr(exc, "retry_after", None)
        return "rate_limited", retry_after if isinstance(retry_after, int) else None
    if status_code in (403,) or "entitl" in text or "permission" in text:
        return ("dataset_not_entitled" if dataset else "authentication_failed"), None
    if isinstance(status_code, int) and status_code >= 500:
        return "server_network_error", None
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        token in text for token in ("timeout", "connection", "network")
    ):
        return "server_network_error", None
    if "schema" in text or "column" in text or "field" in text:
        return "schema_mismatch", None
    return "server_network_error", None


def _table_codes(info) -> list[str] | None:
    if not isinstance(info, Mapping):
        return None
    tables = info.get("tables")
    if isinstance(tables, Mapping):
        tables = list(tables)
    if not isinstance(tables, list):
        return None
    result = []
    for table in tables:
        if isinstance(table, str):
            code = table
        elif isinstance(table, Mapping):
            code = table.get("code") or table.get("table") or table.get("id")
        else:
            code = None
        if isinstance(code, str) and _SAFE_TABLE.fullmatch(code):
            result.append(code)
    return sorted(set(result))


class TejClient:
    """Small adapter around the documented ``tejapi`` package.

    The package is imported only when TEJ is explicitly enabled.  Callers may
    inject a compatible API object in tests; the object must provide
    ``ApiConfig.info()``, ``ApiConfig.api_key`` and ``get()``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str | None = None,
        api=None,
        logger=None,
        sleep_fn=time.sleep,
        max_retries=2,
        max_records=TEJ_MAX_RECORDS,
        max_payload_bytes=TEJ_MAX_PAYLOAD_BYTES,
    ):
        self.enabled = bool(enabled)
        self.api_key = str(api_key or "").strip() or None
        self.api = api
        self.logger = logger or logging.getLogger(__name__)
        self.sleep_fn = sleep_fn
        self.max_retries = min(max(int(max_retries), 0), 3)
        self.max_records = min(max(int(max_records), 1), TEJ_MAX_RECORDS)
        self.max_payload_bytes = min(
            max(int(max_payload_bytes), 1), TEJ_MAX_PAYLOAD_BYTES
        )
        self._entitled_datasets = None

    @classmethod
    def from_env(cls, *, api=None, logger=None, sleep_fn=time.sleep):
        return cls(
            enabled=_enabled(os.getenv(TEJ_ENABLED_ENV, "false")),
            api_key=os.getenv(TEJ_API_KEY_ENV),
            api=api,
            logger=logger,
            sleep_fn=sleep_fn,
        )

    def _api_module(self):
        if self.api is not None:
            return self.api
        try:
            import tejapi
        except ImportError as exc:
            raise TejError(
                "server_network_error",
                "documented tejapi client is unavailable",
            ) from exc
        self.api = tejapi
        return tejapi

    def discover(self):
        if not self.enabled:
            return {"provider": TEJ_PROVIDER, "status": "disabled"}
        if not self.api_key:
            return {
                "provider": TEJ_PROVIDER,
                "status": "authentication_unavailable",
                "reason": "TEJ_API_KEY is absent",
            }
        try:
            api = self._api_module()
            api.ApiConfig.api_key = self.api_key
            info = api.ApiConfig.info()
            tables = _table_codes(info)
            if tables is None:
                raise TejSchemaError("TEJ account info tables are invalid")
            self._entitled_datasets = set(tables)
            safe_limits = {}
            for key in (
                "reqDayLimit",
                "rowsDayLimit",
                "rowsMonthLimit",
                "todayReqCount",
                "todayRows",
                "monthRows",
            ):
                if isinstance(info, Mapping) and key in info:
                    safe_limits[key] = info[key]
            return {
                "provider": TEJ_PROVIDER,
                "status": "authentication_valid",
                "entitled_datasets": tables,
                "dataset_count": len(tables),
                "limits": safe_limits,
            }
        except TejError as exc:
            self.logger.warning("%s", exc.safe_message)
            return exc.to_dict()
        except Exception as exc:
            status, retry_after = _exception_status(exc)
            error = TejError(status, "account discovery failed", retry_after=retry_after)
            self.logger.warning("%s", error.safe_message)
            return error.to_dict()

    def check_dataset(self, table: str):
        if not isinstance(table, str) or _SAFE_TABLE.fullmatch(table) is None:
            return TejError("schema_mismatch", "dataset code is invalid").to_dict(
                dataset=table if isinstance(table, str) else ""
            )
        result = self.discover()
        if result.get("status") != "authentication_valid":
            return {**result, "dataset": table}
        if table not in set(result.get("entitled_datasets") or []):
            return {
                "provider": TEJ_PROVIDER,
                "status": "dataset_not_entitled",
                "dataset": table,
            }
        return {
            "provider": TEJ_PROVIDER,
            "status": "dataset_entitled",
            "dataset": table,
        }

    def fetch_dataset(self, table: str, *, filters=None, columns=None):
        gate = self.check_dataset(table)
        if gate.get("status") != "dataset_entitled":
            return gate
        try:
            api = self._api_module()
            api.ApiConfig.api_key = self.api_key
            kwargs = dict(filters or {})
            if columns:
                kwargs["opts"] = {"columns": columns}
            kwargs["paginate"] = True
            for attempt in range(self.max_retries + 1):
                try:
                    response = api.get(table, **kwargs)
                    records = _records_from_response(
                        response, max_records=self.max_records
                    )
                    if len(_canonical(records)) > self.max_payload_bytes:
                        raise TejError(
                            "payload_too_large",
                            "TEJ response exceeds byte limit",
                        )
                    return {
                        "provider": TEJ_PROVIDER,
                        "status": "dataset_entitled",
                        "dataset": table,
                        "row_count": len(records),
                        "records": records,
                        "query": _json_safe(kwargs),
                    }
                except TejError as exc:
                    if exc.status != "rate_limited" or attempt >= self.max_retries:
                        return exc.to_dict(dataset=table)
                    delay = min(exc.retry_after or 2**attempt, 60)
                except Exception as exc:
                    status, retry_after = _exception_status(exc, dataset=True)
                    if status != "rate_limited" or attempt >= self.max_retries:
                        error = TejError(status, "dataset request failed", retry_after=retry_after)
                        self.logger.warning("%s", error.safe_message)
                        return error.to_dict(dataset=table)
                    delay = min(retry_after or 2**attempt, 60)
                self.sleep_fn(delay)
        except Exception as exc:
            if isinstance(exc, TejError):
                return exc.to_dict(dataset=table)
            error = TejError("server_network_error", "dataset request setup failed")
            self.logger.warning("%s", error.safe_message)
            return error.to_dict(dataset=table)


def write_tej_raw_cache(root, payload, metadata: Mapping[str, Any]):
    """Write exact private raw evidence and content-addressed metadata."""

    if not isinstance(metadata, Mapping) or _credential_metadata_rejected(metadata):
        raise ValueError("TEJ credential fields are not allowed in cache metadata")
    if metadata.get("provider") != TEJ_PROVIDER:
        raise ValueError("TEJ cache provider is invalid")
    if type(metadata.get("row_count")) is not int or metadata["row_count"] < 0:
        raise ValueError("TEJ cache row_count is invalid")
    if isinstance(payload, list) and metadata["row_count"] != len(payload):
        raise ValueError("TEJ cache row_count mismatch")
    content = _canonical(payload)
    if len(content) > TEJ_MAX_PAYLOAD_BYTES:
        raise ValueError("TEJ raw payload exceeds the configured size limit")
    payload_sha = hashlib.sha256(content).hexdigest()
    root = Path(root).resolve()
    raw_root = root / "raw" / "tej" / "v1"
    raw_path = raw_root / "objects" / f"{payload_sha}.json"
    _write_immutable(raw_path, content)
    bound_metadata = {
        "schema_version": TEJ_SCHEMA_VERSION,
        "kind": "absorb-tej-raw-metadata",
        **_json_safe(dict(metadata)),
        "payload_sha256": payload_sha,
        "payload_size": len(content),
        "payload_path": f"objects/{payload_sha}.json",
    }
    metadata_content = _canonical(bound_metadata)
    if len(metadata_content) > TEJ_MAX_METADATA_BYTES:
        raise ValueError("TEJ cache metadata exceeds the configured size limit")
    metadata_sha = hashlib.sha256(metadata_content).hexdigest()
    metadata_path = raw_root / "metadata" / f"{metadata_sha}.json"
    _write_immutable(metadata_path, metadata_content)
    return {
        "raw_path": str(raw_path),
        "metadata_path": str(metadata_path),
        "payload_sha256": payload_sha,
        "metadata_sha256": metadata_sha,
        "row_count": bound_metadata["row_count"],
    }


def load_tej_raw_cache(root, metadata_path):
    root = Path(root).resolve()
    raw_root = root / "raw" / "tej" / "v1"
    metadata_path = Path(metadata_path).resolve()
    try:
        metadata_path.relative_to(raw_root.resolve())
    except ValueError as exc:
        raise ValueError("TEJ metadata path escaped private raw root") from exc
    if metadata_path.parent != (raw_root / "metadata").resolve():
        raise ValueError("TEJ metadata path is outside the metadata directory")
    metadata_content = metadata_path.read_bytes()
    if len(metadata_content) > TEJ_MAX_METADATA_BYTES:
        raise ValueError("TEJ metadata exceeds the configured size limit")
    metadata = json.loads(metadata_content)
    if (
        metadata.get("schema_version") != TEJ_SCHEMA_VERSION
        or metadata.get("kind") != "absorb-tej-raw-metadata"
        or metadata_path.stem != hashlib.sha256(metadata_content).hexdigest()
        or _SHA256.fullmatch(str(metadata.get("payload_sha256") or "")) is None
        or metadata.get("payload_path")
        != f"objects/{metadata.get('payload_sha256')}.json"
        or type(metadata.get("payload_size")) is not int
        or metadata.get("payload_size") < 0
    ):
        raise ValueError("TEJ metadata identity is invalid")
    payload_path = _safe_child(raw_root, metadata["payload_path"], "TEJ raw payload")
    content = payload_path.read_bytes()
    if (
        len(content) > TEJ_MAX_PAYLOAD_BYTES
        or len(content) != metadata["payload_size"]
        or hashlib.sha256(content).hexdigest() != metadata["payload_sha256"]
    ):
        raise ValueError("TEJ raw payload hash mismatch")
    return {
        "payload": json.loads(content),
        "metadata": metadata,
        "metadata_sha256": hashlib.sha256(metadata_content).hexdigest(),
    }


def _numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _safe_identity(identity):
    if not isinstance(identity, Mapping):
        return {"source": "unknown", "sha256": None}
    digest = str(identity.get("sha256") or "").lower()
    return {
        "source": str(identity.get("source") or "unknown")[:80],
        "sha256": digest if _SHA256.fullmatch(digest) else None,
    }


def normalize_pit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    table: str,
    payload_sha256: str,
    field_map: Mapping[str, Any],
    entity_map: Mapping[str, str],
):
    """Normalize only records covered by an explicit source field contract."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TejSchemaError("TEJ records must be a sequence")
    if _SAFE_TABLE.fullmatch(str(table)) is None:
        raise TejSchemaError("TEJ dataset code is invalid")
    if _SHA256.fullmatch(str(payload_sha256)) is None:
        raise TejSchemaError("TEJ payload identity is invalid")
    required = {"entity", "effective_date", "available_at", "fields"}
    if not isinstance(field_map, Mapping) or not required.issubset(field_map):
        raise TejSchemaError("explicit TEJ field map is incomplete")
    if not isinstance(entity_map, Mapping) or not entity_map:
        raise TejSchemaError("explicit TEJ entity map is incomplete")
    entity_pairs = {}
    for source_entity, symbol in entity_map.items():
        if not isinstance(source_entity, str) or not source_entity.strip():
            raise TejSchemaError("TEJ entity map keys are invalid")
        if not isinstance(symbol, str) or not symbol.strip():
            raise TejSchemaError("TEJ entity map values are invalid")
        entity_pairs[source_entity] = symbol
    if len(set(entity_pairs.values())) != len(entity_pairs):
        raise TejSchemaError("TEJ entity map is not one-to-one")
    fields = field_map.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise TejSchemaError("explicit TEJ field map has no fields")
    entity_field = str(field_map["entity"])
    effective_field = str(field_map["effective_date"])
    available_field = str(field_map["available_at"])
    if not all((entity_field, effective_field, available_field)):
        raise TejSchemaError("TEJ field map control fields are invalid")
    normalized_field_names = set()
    for normalized_name, source_name in fields.items():
        if (
            not isinstance(normalized_name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", normalized_name)
            or not isinstance(source_name, str)
            or not source_name.strip()
        ):
            raise TejSchemaError("TEJ field map names are invalid")
        if normalized_name in normalized_field_names:
            raise TejSchemaError("TEJ field map contains duplicate fields")
        normalized_field_names.add(normalized_name)
    normalized = []
    present_fields = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TejSchemaError(f"TEJ record {index} is not an object")
        entity_id = record.get(entity_field)
        if entity_id is None or str(entity_id) not in entity_pairs:
            raise TejSchemaError(f"TEJ entity mapping is missing: record {index}")
        if effective_field not in record:
            raise TejSchemaError(f"TEJ effective_date field is missing: {effective_field}")
        if available_field not in record:
            raise TejSchemaError(f"TEJ available_at field is missing: {available_field}")
        effective_date = _parse_date(record[effective_field])
        available_at = _parse_timestamp(record[available_field], "available_at")
        values = {}
        for normalized_name, source_name in fields.items():
            if source_name not in record or record[source_name] is None:
                continue
            value = _json_safe(record[source_name])
            values[normalized_name] = value
            present_fields.add(normalized_name)
        normalized.append(
            {
                "schema_version": TEJ_SCHEMA_VERSION,
                "kind": "absorb-tej-pit-row",
                "provider": TEJ_PROVIDER,
                "dataset": table,
                "symbol": entity_pairs[str(entity_id)],
                "tej_entity_id": str(entity_id),
                "effective_date": effective_date.isoformat(),
                "available_at": _timestamp(available_at),
                "source_payload_sha256": str(payload_sha256),
                "values": values,
            }
        )
    if not normalized:
        raise TejError("empty_dataset", "TEJ normalization produced no rows")
    if not present_fields:
        raise TejSchemaError("explicit TEJ fields are absent from every record")
    return sorted(
        normalized,
        key=lambda row: (
            row["symbol"],
            row["effective_date"],
            row["available_at"],
            row["source_payload_sha256"],
        ),
    )


def pit_asof_join(rows, *, prediction_time, effective_date=None):
    """Select the latest visible revision without future/restatement leakage."""

    cutoff = _parse_timestamp(prediction_time, "prediction_time")
    effective_cutoff = (
        _parse_date(effective_date, "effective_date")
        if effective_date is not None
        else None
    )
    revisions = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TejSchemaError("normalized PIT row is not an object")
        available_at = _parse_timestamp(row.get("available_at"), "available_at")
        row_date = _parse_date(row.get("effective_date"), "effective_date")
        if available_at > cutoff or (
            effective_cutoff is not None and row_date > effective_cutoff
        ):
            continue
        key = (str(row.get("symbol")), row_date.isoformat())
        current = revisions.get(key)
        identity = (
            _timestamp(available_at),
            str(row.get("source_payload_sha256") or ""),
        )
        if current is not None:
            current_identity = (
                current["available_at"],
                str(current.get("source_payload_sha256") or ""),
            )
            if identity[0] == current_identity[0] and identity[1] != current_identity[1]:
                raise TejSchemaError("conflicting TEJ PIT revisions at the same announcement")
            if identity == current_identity:
                if _canonical(row) != _canonical(current):
                    raise TejSchemaError(
                        "conflicting TEJ PIT rows share the same revision identity"
                    )
                continue
            if identity <= current_identity:
                continue
        revisions[key] = dict(row)
    latest = {}
    for row in revisions.values():
        dataset = str(row.get("dataset") or "")
        symbol = str(row["symbol"])
        key = (dataset, symbol)
        current = latest.get(key)
        identity = (
            str(row["effective_date"]),
            str(row["available_at"]),
            str(row.get("source_payload_sha256") or ""),
        )
        if current is None or identity > current[0]:
            latest[key] = (identity, row)
    return [
        row
        for _, row in sorted(
            latest.values(),
            key=lambda item: (
                str(item[1].get("dataset") or ""),
                str(item[1]["symbol"]),
            ),
        )
    ]


_FACTOR_FIELD_FAMILY = {
    "pe": "VALUE",
    "pb": "VALUE",
    "dividend_yield": "VALUE",
    "revenue_yoy": "GROWTH",
    "revenue_mom": "GROWTH",
    "revenue_acceleration": "MOMENTUM",
    "eps": "GROWTH",
    "eps_ttm": "GROWTH",
    "roe": "QUALITY",
    "roa": "QUALITY",
    "gross_margin": "QUALITY",
    "operating_margin": "QUALITY",
    "operating_cash_flow": "QUALITY",
    "free_cash_flow": "QUALITY",
    "debt_ratio": "RISK",
    "current_ratio": "LIQUIDITY",
    "institutional_flow": "SENTIMENT",
    "foreign_net_flow": "SENTIMENT",
}
_FACTOR_FEATURE_NAMES = frozenset(
    f"tej_{family.lower()}_{field}_{suffix}"
    for field, family in _FACTOR_FIELD_FAMILY.items()
    for suffix in ("percentile", "zscore")
)
_FACTOR_ROW_KEYS = {
    "symbol",
    "effective_date",
    "available_at",
    "factor_as_of",
    "source_payload_sha256",
    "values",
    "factors",
}


def _percentile(values, value):
    ordered = sorted(values)
    if len(ordered) == 1:
        return 0.5
    rank = sum(item < value for item in ordered)
    ties = sum(item == value for item in ordered)
    return (rank + max(ties - 1, 0) / 2) / (len(ordered) - 1)


def build_factor_snapshot(
    rows,
    *,
    as_of,
    effective_date=None,
    source_normalized_sha256=None,
    field_map_sha256=None,
    entity_map_sha256=None,
):
    rows = list(rows)
    source_normalized_sha256 = source_normalized_sha256 or _canonical_sha(rows)
    field_map_sha256 = field_map_sha256 or _canonical_sha(
        {"fields": sorted({key for row in rows for key in (row.get("values") or {})})}
    )
    entity_map_sha256 = entity_map_sha256 or _canonical_sha(
        {"symbols": sorted({str(row.get("symbol")) for row in rows})}
    )
    if not all(
        _SHA256.fullmatch(str(value))
        for value in (
            source_normalized_sha256,
            field_map_sha256,
            entity_map_sha256,
        )
    ):
        raise TejSchemaError("TEJ factor lineage identity is invalid")
    visible = pit_asof_join(
        rows,
        prediction_time=as_of,
        effective_date=effective_date,
    )
    if not visible:
        return {
            "schema_version": TEJ_SCHEMA_VERSION,
            "kind": "absorb-tej-factor-snapshot",
            "status": "unavailable",
            "reason": "no TEJ observations were visible at the requested as-of time",
            "as_of": _timestamp(_parse_timestamp(as_of, "as_of")),
            "rows": [],
            "feature_manifest": [],
            "factor_families": [],
            "source_normalized_sha256": source_normalized_sha256,
            "field_map_sha256": field_map_sha256,
            "entity_map_sha256": entity_map_sha256,
            "production_model_changed": False,
        }
    field_values = defaultdict(list)
    for row in visible:
        for field, value in (row.get("values") or {}).items():
            numeric = _numeric(value)
            if numeric is not None and field in _FACTOR_FIELD_FAMILY:
                field_values[field].append(numeric)
    feature_manifest = []
    for field in sorted(field_values):
        family = _FACTOR_FIELD_FAMILY[field]
        feature_manifest.extend(
            [
                f"tej_{family.lower()}_{field}_percentile",
                f"tej_{family.lower()}_{field}_zscore",
            ]
        )
    result_rows = []
    factor_as_of = _timestamp(_parse_timestamp(as_of, "as_of"))
    for row in visible:
        factors = {}
        for field, values in field_values.items():
            numeric = _numeric((row.get("values") or {}).get(field))
            if numeric is None:
                continue
            mean = sum(values) / len(values)
            variance = sum((item - mean) ** 2 for item in values) / len(values)
            scale = math.sqrt(variance)
            family = _FACTOR_FIELD_FAMILY[field].lower()
            factors[f"tej_{family}_{field}_percentile"] = _percentile(values, numeric)
            factors[f"tej_{family}_{field}_zscore"] = (
                (numeric - mean) / scale if scale > 1e-12 else 0.0
            )
        result_rows.append(
            {
                "symbol": row["symbol"],
                "effective_date": row["effective_date"],
                "available_at": row["available_at"],
                "factor_as_of": factor_as_of,
                "source_payload_sha256": row["source_payload_sha256"],
                "values": {
                    field: value
                    for field, value in (row.get("values") or {}).items()
                    if field in _FACTOR_FIELD_FAMILY
                },
                "factors": factors,
            }
        )
    return {
        "schema_version": TEJ_SCHEMA_VERSION,
        "kind": "absorb-tej-factor-snapshot",
        "status": "available" if feature_manifest else "unavailable",
        "reason": None if feature_manifest else "no supported numeric TEJ fields were mapped",
        "as_of": factor_as_of,
        "rows": result_rows,
        "feature_manifest": sorted(feature_manifest),
        "factor_families": sorted(
            {_FACTOR_FIELD_FAMILY[field] for field in field_values}
        ),
        "source_normalized_sha256": source_normalized_sha256,
        "field_map_sha256": field_map_sha256,
        "entity_map_sha256": entity_map_sha256,
        "production_model_changed": False,
    }


def validate_factor_snapshot(document):
    """Validate a content-addressed factor envelope before model use."""

    if not isinstance(document, Mapping):
        raise TejSchemaError("TEJ factor snapshot is not an object")
    allowed = {
        "schema_version",
        "kind",
        "status",
        "reason",
        "as_of",
        "rows",
        "feature_manifest",
        "factor_families",
        "source_normalized_sha256",
        "field_map_sha256",
        "entity_map_sha256",
        "production_model_changed",
    }
    if set(document) != allowed:
        raise TejSchemaError("TEJ factor snapshot schema keys are invalid")
    if (
        document.get("schema_version") != TEJ_SCHEMA_VERSION
        or document.get("kind") != "absorb-tej-factor-snapshot"
        or document.get("status") != "available"
        or document.get("production_model_changed") is not False
    ):
        raise TejSchemaError("TEJ factor snapshot envelope is invalid")
    as_of = _parse_timestamp(document.get("as_of"), "factor as_of")
    for field in (
        "source_normalized_sha256",
        "field_map_sha256",
        "entity_map_sha256",
    ):
        if _SHA256.fullmatch(str(document.get(field) or "")) is None:
            raise TejSchemaError(f"TEJ factor lineage is invalid: {field}")
    manifest = document.get("feature_manifest")
    if (
        not isinstance(manifest, list)
        or manifest != sorted(set(manifest))
        or not manifest
        or any(feature not in _FACTOR_FEATURE_NAMES for feature in manifest)
    ):
        raise TejSchemaError("TEJ factor feature manifest is invalid")
    families = document.get("factor_families")
    if (
        not isinstance(families, list)
        or families != sorted(set(families))
        or not set(families).issubset(set(_FACTOR_FIELD_FAMILY.values()))
    ):
        raise TejSchemaError("TEJ factor family manifest is invalid")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TejSchemaError("TEJ factor snapshot rows are unavailable")
    actual_features = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _FACTOR_ROW_KEYS:
            raise TejSchemaError(f"TEJ factor row {index} schema is invalid")
        if not str(row.get("symbol") or "").strip():
            raise TejSchemaError(f"TEJ factor row {index} symbol is invalid")
        _parse_date(row.get("effective_date"), "factor effective_date")
        available_at = _parse_timestamp(row.get("available_at"), "factor available_at")
        factor_as_of = _parse_timestamp(row.get("factor_as_of"), "factor factor_as_of")
        if available_at > as_of or factor_as_of != as_of:
            raise TejSchemaError(f"TEJ factor row {index} is not as-of safe")
        if _SHA256.fullmatch(str(row.get("source_payload_sha256") or "")) is None:
            raise TejSchemaError(f"TEJ factor row {index} source identity is invalid")
        values = row.get("values")
        if not isinstance(values, Mapping) or any(
            field not in _FACTOR_FIELD_FAMILY for field in values
        ):
            raise TejSchemaError(f"TEJ factor row {index} values are invalid")
        factors = row.get("factors")
        if not isinstance(factors, Mapping):
            raise TejSchemaError(f"TEJ factor row {index} factors are invalid")
        for feature, value in factors.items():
            if feature not in _FACTOR_FEATURE_NAMES or feature not in manifest:
                raise TejSchemaError(f"TEJ factor feature is not declared: {feature}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TejSchemaError(f"TEJ factor value is invalid: {feature}")
            if not math.isfinite(float(value)):
                raise TejSchemaError(f"TEJ factor value is non-finite: {feature}")
            actual_features.add(feature)
    if actual_features != set(manifest):
        raise TejSchemaError("TEJ factor manifest does not match row features")
    return document


def _validate_normalized_artifact(document):
    allowed = {
        "schema_version",
        "kind",
        "source_metadata_sha256",
        "field_map_sha256",
        "entity_map_sha256",
        "row_count",
        "rows",
    }
    if not isinstance(document, Mapping) or set(document) != allowed:
        raise TejSchemaError("TEJ normalized artifact schema keys are invalid")
    if (
        document.get("schema_version") != TEJ_SCHEMA_VERSION
        or document.get("kind") != "absorb-tej-pit-normalized"
        or any(
            _SHA256.fullmatch(str(document.get(field) or "")) is None
            for field in (
                "source_metadata_sha256",
                "field_map_sha256",
                "entity_map_sha256",
            )
        )
        or type(document.get("row_count")) is not int
        or document.get("row_count") < 1
        or not isinstance(document.get("rows"), list)
        or len(document["rows"]) != document["row_count"]
    ):
        raise TejSchemaError("TEJ normalized artifact envelope is invalid")
    row_keys = {
        "schema_version",
        "kind",
        "provider",
        "dataset",
        "symbol",
        "tej_entity_id",
        "effective_date",
        "available_at",
        "source_payload_sha256",
        "values",
    }
    for index, row in enumerate(document["rows"]):
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise TejSchemaError(f"TEJ normalized row {index} schema is invalid")
        if (
            row.get("schema_version") != TEJ_SCHEMA_VERSION
            or row.get("kind") != "absorb-tej-pit-row"
            or row.get("provider") != TEJ_PROVIDER
            or _SAFE_TABLE.fullmatch(str(row.get("dataset") or "")) is None
            or not str(row.get("symbol") or "").strip()
            or not str(row.get("tej_entity_id") or "").strip()
            or _SHA256.fullmatch(str(row.get("source_payload_sha256") or "")) is None
            or not isinstance(row.get("values"), Mapping)
        ):
            raise TejSchemaError(f"TEJ normalized row {index} is invalid")
        _parse_date(row.get("effective_date"), "normalized effective_date")
        _parse_timestamp(row.get("available_at"), "normalized available_at")
    return document


def load_tej_private_artifact(root, artifact_path, *, category, expected_kind):
    """Load a hash-verified research artifact from the private TEJ tree."""

    root = Path(root).resolve()
    private_root = (root / "research" / "tej" / "v1").resolve()
    path = Path(artifact_path).resolve()
    expected_parent = (private_root / category).resolve()
    if path.parent != expected_parent or path.suffix.lower() != ".json":
        raise ValueError("TEJ artifact path is outside its private category")
    if _SHA256.fullmatch(path.stem) is None:
        raise ValueError("TEJ artifact filename identity is invalid")
    try:
        path.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("TEJ artifact path escaped private root") from exc
    content = path.read_bytes()
    if len(content) > TEJ_PRIVATE_ARTIFACT_BYTES:
        raise ValueError("TEJ private artifact exceeds the configured size limit")
    if hashlib.sha256(content).hexdigest() != path.stem:
        raise ValueError("TEJ private artifact hash mismatch")
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("TEJ private artifact is not valid JSON") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != TEJ_SCHEMA_VERSION
        or document.get("kind") != expected_kind
    ):
        raise ValueError("TEJ private artifact kind or schema is invalid")
    if expected_kind == "absorb-tej-pit-normalized":
        _validate_normalized_artifact(document)
    elif expected_kind == "absorb-tej-factor-snapshot":
        validate_factor_snapshot(document)
    else:
        raise ValueError("TEJ private artifact kind is not supported")
    return {
        "document": document,
        "path": str(path),
        "sha256": path.stem,
        "size": len(content),
    }


def compare_official_truth(
    official_rows,
    tej_rows,
    *,
    official_identity: Mapping[str, Any],
    tej_identity: Mapping[str, Any],
    checked_at,
):
    """Return advisory discrepancies; official rows remain authoritative."""

    def _index(rows, label):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TejSchemaError(f"{label} rows are not a sequence")
        indexed = {}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TejSchemaError(f"{label} row {index} is not an object")
            symbol = str(row.get("symbol") or "").strip()
            market_date = str(row.get("market_date") or "").strip()
            if not symbol or not market_date:
                raise TejSchemaError(f"{label} row {index} identity is missing")
            key = (symbol, market_date)
            if key in indexed:
                raise TejSchemaError(f"{label} has duplicate row identity")
            indexed[key] = row
        return indexed

    official = _index(official_rows, "official")
    tej = _index(tej_rows, "TEJ")
    mismatches = []
    comparable_row_count = 0
    comparable_field_count = 0
    for key in sorted(set(official) & set(tej)):
        comparable_row_count += 1
        official_row = official[key]
        tej_row = tej[key]
        for field in ("close", "volume"):
            if field not in official_row or field not in tej_row:
                continue
            comparable_field_count += 1
            official_value = official_row[field]
            tej_value = tej_row[field]
            try:
                equal = math.isclose(float(official_value), float(tej_value), rel_tol=1e-9, abs_tol=1e-9)
            except (TypeError, ValueError):
                equal = official_value == tej_value
            if not equal:
                mismatches.append(
                    {
                        "symbol": key[0],
                        "market_date": key[1],
                        "field": field,
                        "official_value": official_value,
                        "tej_value": tej_value,
                        "official_identity": _safe_identity(official_identity),
                        "tej_identity": _safe_identity(tej_identity),
                        "checked_at": _timestamp(_parse_timestamp(checked_at, "checked_at")),
                    }
                )
    return {
        "schema_version": TEJ_SCHEMA_VERSION,
        "kind": "absorb-tej-shadow-comparison",
        "status": (
            "mismatch"
            if mismatches
            else "match"
            if comparable_field_count
            else "unavailable"
        ),
        "reason": (
            None
            if comparable_field_count
            else "no_comparable_rows"
            if not comparable_row_count
            else "no_comparable_fields"
        ),
        "comparable_row_count": comparable_row_count,
        "comparable_field_count": comparable_field_count,
        "mismatches": mismatches,
        "override_official": False,
        "production_blocking": False,
        "checked_at": _timestamp(_parse_timestamp(checked_at, "checked_at")),
    }


__all__ = [
    "TEJ_API_DOCUMENTATION",
    "TEJ_API_KEY_ENV",
    "TEJ_ENABLED_ENV",
    "TEJ_MAX_METADATA_BYTES",
    "TEJ_MAX_PAYLOAD_BYTES",
    "TEJ_MAX_RECORDS",
    "TejClient",
    "TejError",
    "TejSchemaError",
    "build_factor_snapshot",
    "compare_official_truth",
    "load_tej_raw_cache",
    "load_tej_private_artifact",
    "normalize_pit_records",
    "pit_asof_join",
    "validate_factor_snapshot",
    "validate_tej_data_root",
    "write_tej_raw_cache",
]
