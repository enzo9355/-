"""Hash-bound TW non-price row and lifecycle contracts."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceDefinition,
    OfficialSourceFailure,
    normalize_symbol,
    normalize_market_date,
    parse_number,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_raw_source,
    store_cached_raw_source,
)


STATUS_SCHEMA_VERSION = 1
STATUS_PARSER_VERSION = "tw-official-historical-parser-v3"
LIFECYCLE_PARSER_VERSION = "tw-lifecycle-parser-v2"
_EMPTY_PRICE_MARKERS = {"", "-", "--", "---", "----"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class HistoricalLifecycleUnavailable(ValueError):
    pass


LIFECYCLE_SOURCE_DEFINITIONS = MappingProxyType({
    "twse_current_stop": OfficialSourceDefinition(
        "twse_current_stop", "TWSE", "lifecycle",
        "https://www.twse.com.tw/rwd/zh/violation/stop", "json", 5 * 1024 * 1024,
    ),
    "twse_intraday_halt": OfficialSourceDefinition(
        "twse_intraday_halt", "TWSE", "lifecycle",
        "https://openapi.twse.com.tw/v1/exchangeReport/TWTAWU", "json", 5 * 1024 * 1024,
    ),
    "twse_reduction_resume": OfficialSourceDefinition(
        "twse_reduction_resume", "TWSE", "lifecycle",
        "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU", "json", 5 * 1024 * 1024,
    ),
    "twse_reduction_detail": OfficialSourceDefinition(
        "twse_reduction_detail", "TWSE", "lifecycle",
        "https://www.twse.com.tw/rwd/zh/reducation/TWTAVUDetail", "json", 5 * 1024 * 1024,
    ),
    "twse_termination": OfficialSourceDefinition(
        "twse_termination", "TWSE", "lifecycle",
        "https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml", "json", 5 * 1024 * 1024,
    ),
    "twse_listing_change_20260728": OfficialSourceDefinition(
        "twse_listing_change_20260728", "TWSE", "lifecycle",
        "https://investoredu.twse.com.tw/FileSystem/FileUpload/88ff18ef-5726-4b33-b207-f92310023328.pdf",
        "pdf",
        1024 * 1024,
    ),
    "tpex_current_mode": OfficialSourceDefinition(
        "tpex_current_mode", "TPEx", "lifecycle",
        "https://www.tpex.org.tw/openapi/v1/tpex_cmode", "json", 5 * 1024 * 1024,
    ),
    "tpex_suspend_history": OfficialSourceDefinition(
        "tpex_suspend_history", "TPEx", "lifecycle",
        "https://www.tpex.org.tw/openapi/v1/tpex_spendi_history", "json", 5 * 1024 * 1024,
    ),
    "tpex_termination": OfficialSourceDefinition(
        "tpex_termination", "TPEx", "lifecycle",
        "https://www.tpex.org.tw/www/zh-tw/company/deListed", "json", 5 * 1024 * 1024,
    ),
})

TWSE_LISTING_CHANGE_SOURCE_BINDINGS = MappingProxyType({
    "twse_listing_change_20260728": MappingProxyType({
        "announcement_date": "2026-07-28",
        "expected_records": (
            MappingProxyType({
                "symbol": "2867",
                "event_type": "suspend",
                "effective_date": "2026-08-20",
            }),
            MappingProxyType({
                "symbol": "2867",
                "event_type": "terminate",
                "effective_date": "2026-09-01",
            }),
        ),
        "payload_size_bytes": 139878,
        "payload_sha256": "3ff4455c1435b5d0dc62803953241d184c13775662eb46f2feaf25d3d300c768",
    }),
})


@dataclass(frozen=True)
class LifecycleSnapshot:
    target_date: _datetime.date
    status_by_symbol: Mapping[str, Mapping[str, Any]]
    terminated_by_symbol: Mapping[str, Mapping[str, Any]]
    source_hashes: Mapping[str, str]
    request_count: int


@dataclass(frozen=True)
class PriceRowClassification:
    price: dict[str, Any] | None
    status: dict[str, Any] | None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def evidence_sha256(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("evidence_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"<[^>]*>", "", str(value)).replace("&nbsp;", " ").strip()


def _is_empty_price(value: Any) -> bool:
    return value is None or _text(value) in _EMPTY_PRICE_MARKERS


def classify_price_row(
    target_date: _datetime.date,
    source_id: str,
    exchange: str,
    fields: Sequence[str],
    raw_row: Sequence[Any],
    indices: Mapping[str, int],
    payload_sha256: str,
) -> PriceRowClassification:
    required = {"symbol", "name", "open", "high", "low", "close", "volume"}
    if (
        not isinstance(target_date, _datetime.date)
        or isinstance(target_date, _datetime.datetime)
        or exchange not in {"TWSE", "TPEx"}
        or not _SHA256.fullmatch(str(payload_sha256))
        or set(indices) != required
        or isinstance(raw_row, (str, bytes))
    ):
        raise ValueError("official price row is invalid")
    try:
        positions = {name: int(indices[name]) for name in required}
        if any(index < 0 for index in positions.values()):
            raise ValueError
        if len(raw_row) != len(fields) or max(positions.values()) >= len(raw_row):
            raise ValueError
        symbol = normalize_symbol(_text(raw_row[positions["symbol"]]))
    except (IndexError, TypeError, ValueError):
        raise ValueError("official price row is invalid") from None

    raw_fields = {
        name: raw_row[positions[name]]
        for name in ("symbol", "name", "open", "high", "low", "close", "volume")
    }
    blank = {
        name: _is_empty_price(raw_fields[name])
        for name in ("open", "high", "low", "close")
    }
    if all(blank.values()):
        volume = raw_fields["volume"]
        if not _is_empty_price(volume):
            try:
                if float(parse_number(volume)) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("official price row is invalid") from None
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "official_no_regular_trade",
            "market": "TW",
            "exchange": exchange,
            "symbol": symbol,
            "target_market_date": target_date.isoformat(),
            "source_id": str(source_id),
            "payload_sha256": str(payload_sha256),
            "raw_row_sha256": hashlib.sha256(_canonical_bytes(list(raw_row))).hexdigest(),
            "raw_fields": raw_fields,
            "parser_version": STATUS_PARSER_VERSION,
        }
        status["evidence_sha256"] = evidence_sha256(status)
        return PriceRowClassification(price=None, status=status)
    if any(blank.values()) or _is_empty_price(raw_fields["volume"]):
        raise ValueError("official price row is invalid")

    try:
        open_value = float(parse_number(raw_fields["open"]))
        high = float(parse_number(raw_fields["high"]))
        low = float(parse_number(raw_fields["low"]))
        close = float(parse_number(raw_fields["close"]))
        volume = float(parse_number(raw_fields["volume"]))
    except (TypeError, ValueError):
        raise ValueError("official price row is invalid") from None
    if (
        not all(math.isfinite(value) for value in (open_value, high, low, close, volume))
        or min(open_value, high, low, close) <= 0
        or volume < 0
        or high < max(open_value, close, low)
        or low > min(open_value, close, high)
    ):
        raise ValueError("official price row is invalid")
    return PriceRowClassification(
        price={
            "date": target_date.isoformat(),
            "stock_id": symbol,
            "open": open_value,
            "max": high,
            "min": low,
            "close": close,
            "Trading_Volume": volume,
        },
        status=None,
    )


def _event_date(event: Mapping[str, Any]) -> _datetime.date:
    try:
        value = _datetime.date.fromisoformat(str(event["effective_date"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("official lifecycle event is invalid") from None
    return value


def resolve_lifecycle_status(
    events: Sequence[Mapping[str, Any]],
    target_date: _datetime.date,
    *,
    active: bool,
) -> dict[str, Any] | None:
    if not isinstance(target_date, _datetime.date) or isinstance(
        target_date, _datetime.datetime
    ):
        raise TypeError("target_date must be a date")
    if not events:
        return None
    normalized = []
    identity = None
    seen = set()
    for source in events:
        event = dict(source)
        if (
            event.get("schema_version") != 1
            or event.get("event_type") not in {"suspend", "resume", "terminate"}
            or not _SHA256.fullmatch(str(event.get("payload_sha256") or ""))
            or not _SHA256.fullmatch(str(event.get("raw_row_sha256") or ""))
            or event.get("parser_version") != LIFECYCLE_PARSER_VERSION
            or event.get("evidence_sha256") != evidence_sha256(event)
        ):
            raise ValueError("official lifecycle event is invalid")
        current_identity = (str(event.get("exchange")), normalize_symbol(event.get("symbol")))
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("official lifecycle events mix symbols")
        key = (event["event_type"], _event_date(event), event["evidence_sha256"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(event)

    precedence = {"suspend": 0, "resume": 1, "terminate": 2}
    normalized.sort(
        key=lambda event: (_event_date(event), precedence[event["event_type"]])
    )
    terminated = None
    open_suspend = None
    closed_at = None
    chain = []
    for event in normalized:
        event_date = _event_date(event)
        event_type = event["event_type"]
        if event_type == "terminate":
            if terminated is not None:
                raise ValueError("official lifecycle events conflict")
            terminated = event
            if open_suspend is not None and event_date >= _event_date(open_suspend):
                closed_at = event_date
            chain.append(event)
            continue
        if event_type == "suspend":
            if terminated is not None and event_date > _event_date(terminated):
                terminated = None
                open_suspend = None
                closed_at = None
                chain = []
            if open_suspend is not None and closed_at is None:
                raise ValueError("official lifecycle events conflict")
            open_suspend = event
            closed_at = None
            chain = [event]
            continue
        if open_suspend is None or closed_at is not None or event_date < _event_date(open_suspend):
            raise ValueError("official lifecycle events conflict")
        closed_at = event_date
        chain.append(event)

    if terminated is not None and _event_date(terminated) <= target_date:
        result = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "officially_terminated",
            "market": "TW",
            "exchange": identity[0],
            "symbol": identity[1],
            "target_market_date": target_date.isoformat(),
            "effective_date": _event_date(terminated).isoformat(),
            "lifecycle_events": normalized,
            "parser_version": LIFECYCLE_PARSER_VERSION,
        }
        result["evidence_sha256"] = evidence_sha256(result)
        return result
    if open_suspend is None or _event_date(open_suspend) > target_date:
        return None
    if closed_at is not None and closed_at <= target_date:
        return None
    result = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "officially_suspended",
        "market": "TW",
        "exchange": identity[0],
        "symbol": identity[1],
        "target_market_date": target_date.isoformat(),
        "valid_from": _event_date(open_suspend).isoformat(),
        "valid_through_exclusive": closed_at.isoformat() if closed_at else None,
        "evaluated_through": target_date.isoformat(),
        "lifecycle_events": chain,
        "parser_version": LIFECYCLE_PARSER_VERSION,
    }
    result["evidence_sha256"] = evidence_sha256(result)
    return result


def validate_status_evidence(
    document: Mapping[str, Any],
    *,
    symbol: str | None = None,
    target_date: _datetime.date | None = None,
) -> dict[str, Any]:
    value = dict(document)
    try:
        evidence_target = _datetime.date.fromisoformat(
            str(value["target_market_date"])
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("trading status evidence is invalid") from None
    if (
        value.get("schema_version") != STATUS_SCHEMA_VERSION
        or value.get("status")
        not in {"official_no_regular_trade", "officially_suspended"}
        or value.get("market") != "TW"
        or value.get("exchange") not in {"TWSE", "TPEx"}
        or normalize_symbol(value.get("symbol")) != str(value.get("symbol"))
        or (symbol is not None and value.get("symbol") != normalize_symbol(symbol))
        or (target_date is not None and evidence_target != target_date)
        or value.get("evidence_sha256") != evidence_sha256(value)
    ):
        raise ValueError("trading status evidence is invalid")
    if value["status"] == "official_no_regular_trade":
        raw_fields = value.get("raw_fields")
        expected_source = {"TWSE": "twse_price", "TPEx": "tpex_price"}[
            value["exchange"]
        ]
        if (
            value.get("source_id") != expected_source
            or not _SHA256.fullmatch(str(value.get("payload_sha256") or ""))
            or not _SHA256.fullmatch(str(value.get("raw_row_sha256") or ""))
            or value.get("parser_version") != STATUS_PARSER_VERSION
            or not isinstance(raw_fields, dict)
            or set(raw_fields)
            != {"symbol", "name", "open", "high", "low", "close", "volume"}
            or normalize_symbol(raw_fields.get("symbol")) != value["symbol"]
            or not all(
                _is_empty_price(raw_fields[name])
                for name in ("open", "high", "low", "close")
            )
        ):
            raise ValueError("trading status evidence is invalid")
        if not _is_empty_price(raw_fields["volume"]):
            try:
                if float(parse_number(raw_fields["volume"])) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("trading status evidence is invalid") from None
        return value
    events = value.get("lifecycle_events")
    if (
        value.get("parser_version") != LIFECYCLE_PARSER_VERSION
        or not isinstance(events, list)
        or not events
        or resolve_lifecycle_status(events, evidence_target, active=True) != value
    ):
        raise ValueError("trading status evidence is invalid")
    return value


def _lifecycle_params(
    source_id: str,
    target_date: _datetime.date,
    *,
    symbol: str | None = None,
    file_date: str | None = None,
) -> dict[str, str]:
    if source_id == "twse_current_stop":
        return {"response": "json"}
    if source_id == "twse_reduction_resume":
        return {"date": target_date.strftime("%Y%m%d"), "response": "json"}
    if source_id == "twse_reduction_detail":
        if symbol is None or not re.fullmatch(r"\d{8}", str(file_date or "")):
            raise ValueError("TWSE reduction detail identity is invalid")
        return {"STK_NO": normalize_symbol(symbol), "FILE_DATE": str(file_date), "response": "json"}
    if source_id == "tpex_termination":
        return {"code": "", "date": str(target_date.year), "reason": "-1"}
    return {}


def _roc_date(value: Any) -> _datetime.date:
    text = _text(value)
    text = text.replace("年", "/").replace("月", "/").replace("日", "")
    return normalize_market_date(text)


def _raw_row_sha256(row: Any) -> str:
    return hashlib.sha256(_canonical_bytes(row)).hexdigest()


def _lifecycle_event(
    *,
    exchange: str,
    symbol: Any,
    event_type: str,
    effective_date: Any,
    source_id: str,
    payload_sha256: str,
    raw_row: Any,
) -> dict[str, Any]:
    event = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "exchange": exchange,
        "symbol": normalize_symbol(symbol),
        "event_type": event_type,
        "effective_date": _roc_date(effective_date).isoformat(),
        "source_id": source_id,
        "payload_sha256": payload_sha256,
        "raw_row_sha256": _raw_row_sha256(raw_row),
        "raw_fields": raw_row,
        "parser_version": LIFECYCLE_PARSER_VERSION,
    }
    event["evidence_sha256"] = evidence_sha256(event)
    return event


def _as_rows(payload: Any, label: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{label} schema is invalid")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} schema is invalid")
    return rows


def _extract_current_mode_effective_date(
    source_id: str,
    payload: bytes,
) -> _datetime.date | None:
    if source_id != "tpex_current_mode":
        return None
    try:
        document = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        return None
    rows = _as_rows(document, source_id)
    if not rows or "Date" not in rows[0]:
        return None
    return _roc_date(rows[0]["Date"])


def _extract_pdf_text(payload: bytes) -> str:
    if not isinstance(payload, bytes) or not payload.startswith(b"%PDF-"):
        raise ValueError("official lifecycle PDF is invalid")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload), strict=True)
        if reader.is_encrypted or not reader.pages:
            raise ValueError
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise ValueError("official lifecycle PDF is invalid") from exc
    if not text.strip():
        raise ValueError("official lifecycle PDF text is empty")
    return text


def _listing_change_expected_records(
    binding: Mapping[str, Any],
) -> tuple[tuple[str, str, _datetime.date], ...]:
    expected_binding_fields = {
        "announcement_date",
        "expected_records",
        "payload_size_bytes",
        "payload_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_binding_fields:
        raise ValueError("TWSE listing change binding is invalid")
    announcement_text = binding.get("announcement_date")
    payload_size = binding.get("payload_size_bytes")
    payload_sha256 = binding.get("payload_sha256")
    records = binding.get("expected_records")
    try:
        announcement_date = _datetime.date.fromisoformat(announcement_text)
    except (TypeError, ValueError):
        raise ValueError("TWSE listing change binding is invalid") from None
    if (
        announcement_date.isoformat() != announcement_text
        or not isinstance(payload_size, int)
        or isinstance(payload_size, bool)
        or payload_size <= 0
        or not isinstance(payload_sha256, str)
        or _SHA256.fullmatch(payload_sha256) is None
        or not isinstance(records, tuple)
        or not records
    ):
        raise ValueError("TWSE listing change binding is invalid")

    normalized = []
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"symbol", "event_type", "effective_date"}
        ):
            raise ValueError("TWSE listing change expected record is invalid")
        symbol = record.get("symbol")
        event_type = record.get("event_type")
        effective_text = record.get("effective_date")
        try:
            normalized_symbol = normalize_symbol(symbol)
            effective_date = _datetime.date.fromisoformat(effective_text)
        except (TypeError, ValueError):
            raise ValueError(
                "TWSE listing change expected record is invalid"
            ) from None
        if (
            symbol != normalized_symbol
            or event_type not in {"suspend", "terminate"}
            or effective_date.isoformat() != effective_text
            or effective_date < announcement_date
        ):
            raise ValueError("TWSE listing change expected record is invalid")
        normalized.append((normalized_symbol, event_type, effective_date))
    if len(normalized) != len(set(normalized)):
        raise ValueError("TWSE listing change expected records are duplicated")
    return tuple(normalized)


def _validate_listing_change_raw_payload(
    source_id: str,
    payload: bytes,
) -> Mapping[str, Any]:
    binding = TWSE_LISTING_CHANGE_SOURCE_BINDINGS.get(source_id)
    try:
        if binding is None:
            raise ValueError("TWSE listing change binding is missing")
        _listing_change_expected_records(binding)
    except ValueError as exc:
        raise OfficialSourceFailure(
            source_id, "cache_invalid", safe_message=str(exc)
        ) from None
    if (
        len(payload) != binding["payload_size_bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["payload_sha256"]
    ):
        raise OfficialSourceFailure(
            source_id,
            "cache_invalid",
            safe_message="official PDF binding mismatch",
        )
    return binding


def _load_lifecycle_payload(
    root: Path,
    target_date: _datetime.date,
    *,
    definition: OfficialSourceDefinition,
    cache_source_id: str,
    params: Mapping[str, str],
    session: Any,
    timeout: int,
    fetched_at: _datetime.datetime,
) -> tuple[Any, str, int]:
    try:
        cached = load_cached_raw_source(
            root,
            source_id=cache_source_id,
            target_date=target_date,
            parser_version=LIFECYCLE_PARSER_VERSION,
        )
    except OfficialCacheError as exc:
        raise OfficialSourceFailure(
            cache_source_id, "cache_invalid", safe_message=str(exc)
        ) from None
    request_count = 0
    if cached is not None:
        if (
            cached.effective_date is not None
            and cached.effective_date != target_date
        ):
            raise HistoricalLifecycleUnavailable(
                f"cached {cache_source_id} effective date "
                f"{cached.effective_date.isoformat()} "
                f"does not match requested target date "
                f"{target_date.isoformat()}"
            )
    if cached is None:
        try:
            response = session.get(
                definition.url,
                params=dict(params),
                headers={"User-Agent": "ABSORB/1.0"},
                timeout=timeout,
            )
        except Exception as exc:
            raise OfficialSourceFailure(
                cache_source_id,
                "transport_error",
                retryable=True,
                safe_message=type(exc).__name__,
            ) from None
        request_count = 1
        status = int(getattr(response, "status_code", 0))
        if status != 200:
            raise OfficialSourceFailure(
                cache_source_id,
                "http_error",
                http_status=status,
                retryable=status >= 500,
                safe_message=f"HTTP {status}",
            )
        content = bytes(getattr(response, "content", b""))
        if not content or len(content) > definition.max_bytes:
            raise OfficialSourceFailure(
                cache_source_id,
                "response_too_large",
                safe_message="response size is invalid",
            )
        if definition.response_kind == "pdf":
            _validate_listing_change_raw_payload(cache_source_id, content)
        effective_date = _extract_current_mode_effective_date(
            cache_source_id, content
        )
        if (
            effective_date is not None
            and effective_date != target_date
        ):
            raise HistoricalLifecycleUnavailable(
                f"fetched {cache_source_id} effective date "
                f"{effective_date.isoformat()} "
                f"does not match requested target date "
                f"{target_date.isoformat()}"
            )
        try:
            cached = store_cached_raw_source(
                root,
                source_id=cache_source_id,
                target_date=target_date,
                payload=content,
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url=definition.url,
                fetched_at=fetched_at,
                date_verification="lifecycle_contract",
                effective_date=effective_date,
            )
        except (OfficialCacheError, OSError, ValueError) as exc:
            raise OfficialSourceFailure(
                cache_source_id, "cache_invalid", safe_message=str(exc)
            ) from None
    if definition.response_kind == "pdf":
        binding = _validate_listing_change_raw_payload(
            cache_source_id, cached.payload
        )
        try:
            extracted_text = _extract_pdf_text(cached.payload)
        except ValueError as exc:
            raise OfficialSourceFailure(
                cache_source_id,
                "schema_validation",
                safe_message=str(exc),
            ) from None
        payload = {
            "schema_version": 1,
            "source_id": cache_source_id,
            "source_url": definition.url,
            "announcement_date": binding["announcement_date"],
            "payload_size_bytes": len(cached.payload),
            "payload_sha256": cached.payload_sha256,
            "extracted_text": extracted_text,
        }
    else:
        try:
            payload = json.loads(cached.payload.decode("utf-8-sig"))
        except (UnicodeError, ValueError):
            raise OfficialSourceFailure(
                cache_source_id,
                "schema_validation",
                safe_message="response JSON is invalid",
            ) from None
    return payload, cached.payload_sha256, request_count


def _twse_current_stop_events(
    payload: Any,
    payload_sha256: str,
    target_date: _datetime.date,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or str(payload.get("stat", "")).lower() != "ok":
        raise ValueError("TWSE current stop schema is invalid")
    expected = ["證券代號", "證券名稱", "違反營業細則條款", "停止買賣原因", "停止買賣開始日期"]
    matches = []
    for table in payload.get("tables") or []:
        if isinstance(table, Mapping) and table.get("fields") == expected and isinstance(table.get("data"), list):
            matches.append(table)
    if len(matches) != 1:
        raise ValueError("TWSE current stop table is ambiguous")
    title_match = re.search(r"(\d{3})年(\d{2})月(\d{2})日", str(matches[0].get("title") or ""))
    if not title_match:
        raise ValueError("TWSE current stop observation date is invalid")
    observed = _datetime.date(int(title_match.group(1)) + 1911, int(title_match.group(2)), int(title_match.group(3)))
    if observed < target_date:
        raise ValueError("TWSE current stop evidence is stale")
    events = []
    for row in matches[0]["data"]:
        if not isinstance(row, list) or len(row) != len(expected):
            raise ValueError("TWSE current stop row is invalid")
        events.append(_lifecycle_event(
            exchange="TWSE", symbol=row[0], event_type="suspend",
            effective_date=row[4], source_id="twse_current_stop",
            payload_sha256=payload_sha256, raw_row=row,
        ))
    return events


def _twse_intraday_events(payload: Any, payload_sha256: str) -> list[dict[str, Any]]:
    events = []
    for row in _as_rows(payload, "TWSE intraday halt"):
        required = {"Code", "TradingHaltDate", "TradingResumptionDate"}
        if not required <= set(row):
            raise ValueError("TWSE intraday halt schema is invalid")
        if _text(row.get("TradingHaltDate")):
            events.append(_lifecycle_event(
                exchange="TWSE", symbol=row["Code"], event_type="suspend",
                effective_date=row["TradingHaltDate"], source_id="twse_intraday_halt",
                payload_sha256=payload_sha256, raw_row=dict(row),
            ))
        if _text(row.get("TradingResumptionDate")):
            events.append(_lifecycle_event(
                exchange="TWSE", symbol=row["Code"], event_type="resume",
                effective_date=row["TradingResumptionDate"], source_id="twse_intraday_halt",
                payload_sha256=payload_sha256, raw_row=dict(row),
            ))
    return events


def _twse_reduction_rows(payload: Any, payload_sha256: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    if not isinstance(payload, Mapping) or str(payload.get("stat", "")).lower() != "ok":
        raise ValueError("TWSE reduction resume schema is invalid")
    fields = payload.get("fields")
    rows = payload.get("data")
    if not isinstance(fields, list) or fields[:3] != ["恢復買賣日期", "股票代號", "名稱"] or "詳細資料" not in fields or not isinstance(rows, list):
        raise ValueError("TWSE reduction resume schema is invalid")
    detail_index = fields.index("詳細資料")
    events = []
    details = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise ValueError("TWSE reduction resume row is invalid")
        symbol = normalize_symbol(row[1])
        events.append(_lifecycle_event(
            exchange="TWSE", symbol=symbol, event_type="resume",
            effective_date=row[0], source_id="twse_reduction_resume",
            payload_sha256=payload_sha256, raw_row=row,
        ))
        match = re.fullmatch(r"\s*(\d{4,6})\s*,\s*(\d{8})\s*", str(row[detail_index]))
        if not match or normalize_symbol(match.group(1)) != symbol:
            raise ValueError("TWSE reduction detail reference is invalid")
        details.append((symbol, match.group(2)))
    return events, details


def _twse_reduction_detail_event(payload: Any, payload_sha256: str, source_id: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or str(payload.get("stat", "")).lower() != "ok":
        raise ValueError("TWSE reduction detail schema is invalid")
    fields = payload.get("fields")
    rows = payload.get("data")
    required = ["股票代號：", "股票名稱：", "停止買賣日期："]
    if not isinstance(fields, list) or fields[:3] != required or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("TWSE reduction detail schema is invalid")
    row = rows[0]
    if not isinstance(row, list) or len(row) != len(fields):
        raise ValueError("TWSE reduction detail row is invalid")
    return _lifecycle_event(
        exchange="TWSE", symbol=row[0], event_type="suspend",
        effective_date=row[2], source_id=source_id,
        payload_sha256=payload_sha256, raw_row=row,
    )


def _twse_termination_events(payload: Any, payload_sha256: str) -> list[dict[str, Any]]:
    events = []
    for row in _as_rows(payload, "TWSE termination"):
        if not {"Code", "DelistingDate"} <= set(row):
            raise ValueError("TWSE termination schema is invalid")
        events.append(_lifecycle_event(
            exchange="TWSE", symbol=row["Code"], event_type="terminate",
            effective_date=row["DelistingDate"], source_id="twse_termination",
            payload_sha256=payload_sha256, raw_row=dict(row),
        ))
    return events


def _twse_listing_change_events(
    payload: Any,
    payload_sha256: str,
    source_id: str,
) -> list[dict[str, Any]]:
    binding = TWSE_LISTING_CHANGE_SOURCE_BINDINGS.get(source_id)
    definition = LIFECYCLE_SOURCE_DEFINITIONS.get(source_id)
    expected_fields = {
        "schema_version",
        "source_id",
        "source_url",
        "announcement_date",
        "payload_size_bytes",
        "payload_sha256",
        "extracted_text",
    }
    if (
        binding is None
        or definition is None
        or not isinstance(payload, Mapping)
        or set(payload) != expected_fields
        or payload.get("schema_version") != 1
        or payload.get("source_id") != source_id
        or payload.get("source_url") != definition.url
        or payload.get("announcement_date") != binding["announcement_date"]
        or payload.get("payload_size_bytes") != binding["payload_size_bytes"]
        or payload.get("payload_sha256") != binding["payload_sha256"]
        or payload_sha256 != binding["payload_sha256"]
    ):
        raise ValueError("TWSE listing change schema is invalid")
    expected_records = _listing_change_expected_records(binding)
    text = payload.get("extracted_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TWSE listing change schema is invalid")
    date = (
        r"(?P<{prefix}_year>\d{{2,3}})\s*年\s*"
        r"(?P<{prefix}_month>\d{{1,2}})\s*月\s*"
        r"(?P<{prefix}_day>\d{{1,2}})\s*日"
    )
    announcement_pattern = re.compile(
        r"中華民國\s*" + date.format(prefix="announcement")
    )
    announcement_matches = list(announcement_pattern.finditer(text))
    if len(announcement_matches) != 1:
        raise ValueError("TWSE listing change announcement date is invalid")
    announcement_match = announcement_matches[0]
    announcement_date = _datetime.date(
        int(announcement_match.group("announcement_year")) + 1911,
        int(announcement_match.group("announcement_month")),
        int(announcement_match.group("announcement_day")),
    )
    if announcement_date.isoformat() != binding["announcement_date"]:
        raise ValueError("TWSE listing change announcement date is invalid")
    pattern = re.compile(
        r"公司代號\s*[：:]\s*(?P<symbol>\d{4,6}).*?自\s*"
        + date.format(prefix="suspend")
        + r"\s*起停止買賣.*?並自\s*"
        + date.format(prefix="terminate")
        + r"\s*起終止上市",
        re.DOTALL,
    )
    records: dict[tuple[str, _datetime.date, _datetime.date], dict[str, Any]] = {}
    for match in pattern.finditer(text):
        suspend_date = _datetime.date(
            int(match.group("suspend_year")) + 1911,
            int(match.group("suspend_month")),
            int(match.group("suspend_day")),
        )
        termination_date = _datetime.date(
            int(match.group("terminate_year")) + 1911,
            int(match.group("terminate_month")),
            int(match.group("terminate_day")),
        )
        if not announcement_date <= suspend_date < termination_date:
            raise ValueError("TWSE listing change dates are invalid")
        symbol = normalize_symbol(match.group("symbol"))
        raw_fields = {
            "announcement_date": announcement_date.isoformat(),
            "symbol": symbol,
            "suspension_date": suspend_date.isoformat(),
            "termination_date": termination_date.isoformat(),
            "source_url": definition.url,
        }
        records[(symbol, suspend_date, termination_date)] = raw_fields
    if not records:
        raise ValueError("TWSE listing change row is invalid")
    parsed_records = {
        (symbol, event_type, effective_date)
        for symbol, suspend_date, termination_date in records
        for event_type, effective_date in (
            ("suspend", suspend_date),
            ("terminate", termination_date),
        )
    }
    if parsed_records != set(expected_records):
        raise ValueError("TWSE listing change event set is invalid")

    events = []
    for (symbol, suspend_date, termination_date), raw_fields in records.items():
        events.append(_lifecycle_event(
            exchange="TWSE",
            symbol=symbol,
            event_type="suspend",
            effective_date=suspend_date,
            source_id=source_id,
            payload_sha256=payload_sha256,
            raw_row=raw_fields,
        ))
        events.append(_lifecycle_event(
            exchange="TWSE",
            symbol=symbol,
            event_type="terminate",
            effective_date=termination_date,
            source_id=source_id,
            payload_sha256=payload_sha256,
            raw_row=raw_fields,
        ))
    return events


def _tpex_mode_events(payload: Any, payload_sha256: str, target_date: _datetime.date) -> list[dict[str, Any]]:
    events = []
    for row in _as_rows(payload, "TPEx current mode"):
        if not {"Date", "SecuritiesCompanyCode", "SuspensionOfTrading"} <= set(row):
            raise ValueError("TPEx current mode schema is invalid")
        if _roc_date(row["Date"]) != target_date:
            raise HistoricalLifecycleUnavailable(
                "TPEx current mode effective date "
                f"{_roc_date(row['Date']).isoformat()} "
                f"does not match requested target {target_date.isoformat()}"
            )
        marker = _text(row["SuspensionOfTrading"])
        if marker not in {"", "Ｙ"}:
            raise ValueError("TPEx current mode marker is invalid")
        if marker == "Ｙ":
            events.append(_lifecycle_event(
                exchange="TPEx", symbol=row["SecuritiesCompanyCode"], event_type="suspend",
                effective_date=target_date, source_id="tpex_current_mode",
                payload_sha256=payload_sha256, raw_row=dict(row),
            ))
    return events


def _tpex_history_events(payload: Any, payload_sha256: str) -> list[dict[str, Any]]:
    events = []
    for row in _as_rows(payload, "TPEx suspend history"):
        required = {"SecuritiesCompanyCode", "DateOfSuspendedTrading", "DateOfResumedTrading"}
        if not required <= set(row):
            raise ValueError("TPEx suspend history schema is invalid")
        if not re.fullmatch(r"\d{4,6}", _text(row["SecuritiesCompanyCode"])):
            continue
        if _text(row["DateOfSuspendedTrading"]):
            events.append(_lifecycle_event(
                exchange="TPEx", symbol=row["SecuritiesCompanyCode"], event_type="suspend",
                effective_date=row["DateOfSuspendedTrading"], source_id="tpex_suspend_history",
                payload_sha256=payload_sha256, raw_row=dict(row),
            ))
        if _text(row["DateOfResumedTrading"]):
            events.append(_lifecycle_event(
                exchange="TPEx", symbol=row["SecuritiesCompanyCode"], event_type="resume",
                effective_date=row["DateOfResumedTrading"], source_id="tpex_suspend_history",
                payload_sha256=payload_sha256, raw_row=dict(row),
            ))
    return events


def _tpex_termination_events(payload: Any, payload_sha256: str, target_date: _datetime.date) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or str(payload.get("stat", "")).lower() != "ok" or str(payload.get("date")) != str(target_date.year):
        raise ValueError("TPEx termination schema is invalid")
    matches = []
    expected = ["股票代號", "公司名稱", "終止上櫃日期", "終止上櫃原因", "公司資料網址"]
    for table in payload.get("tables") or []:
        if isinstance(table, Mapping) and table.get("fields") == expected and isinstance(table.get("data"), list):
            matches.append(table)
    if len(matches) != 1:
        raise ValueError("TPEx termination table is ambiguous")
    events = []
    for row in matches[0]["data"]:
        if not isinstance(row, list) or len(row) != len(expected):
            raise ValueError("TPEx termination row is invalid")
        events.append(_lifecycle_event(
            exchange="TPEx", symbol=row[0], event_type="terminate",
            effective_date=row[2], source_id="tpex_termination",
            payload_sha256=payload_sha256, raw_row=row,
        ))
    return events


def load_lifecycle_snapshot(
    root: Path,
    target_date: _datetime.date,
    *,
    session: Any,
    required_symbols_by_exchange: Mapping[str, Sequence[str] | set[str]],
    now: _datetime.datetime | None = None,
    timeout: int = 30,
) -> LifecycleSnapshot:
    if not isinstance(target_date, _datetime.date) or isinstance(target_date, _datetime.datetime):
        raise TypeError("target_date must be a date")
    requested = {
        exchange: {normalize_symbol(symbol) for symbol in symbols}
        for exchange, symbols in required_symbols_by_exchange.items()
    }
    if not set(requested) <= {"TWSE", "TPEx"}:
        raise ValueError("lifecycle exchange is invalid")
    fetched_at = now or _datetime.datetime.now(_datetime.timezone.utc)
    if fetched_at.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    events: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    request_count = 0

    def load(source_id: str, *, cache_source_id: str | None = None, params: Mapping[str, str] | None = None) -> tuple[Any, str]:
        nonlocal request_count
        definition = LIFECYCLE_SOURCE_DEFINITIONS[source_id]
        payload, payload_sha256, requests = _load_lifecycle_payload(
            Path(root), target_date, definition=definition,
            cache_source_id=cache_source_id or source_id,
            params=params if params is not None else _lifecycle_params(source_id, target_date),
            session=session, timeout=timeout, fetched_at=fetched_at,
        )
        request_count += requests
        source_hashes[cache_source_id or source_id] = payload_sha256
        return payload, payload_sha256

    try:
        if requested.get("TWSE"):
            payload, digest = load("twse_current_stop")
            events.extend(_twse_current_stop_events(payload, digest, target_date))
            payload, digest = load("twse_intraday_halt")
            events.extend(_twse_intraday_events(payload, digest))
            payload, digest = load("twse_reduction_resume")
            reduction_events, details = _twse_reduction_rows(payload, digest)
            events.extend(reduction_events)
            for symbol, file_date in details:
                if symbol not in requested["TWSE"]:
                    continue
                cache_source_id = f"twse_reduction_detail_{symbol}_{file_date}"
                try:
                    payload, digest = load(
                        "twse_reduction_detail",
                        cache_source_id=cache_source_id,
                        params=_lifecycle_params(
                            "twse_reduction_detail", target_date,
                            symbol=symbol, file_date=file_date,
                        ),
                    )
                    event = _twse_reduction_detail_event(payload, digest, cache_source_id)
                    if event["symbol"] != symbol:
                        raise ValueError("TWSE reduction detail symbol mismatch")
                    events.append(event)
                except (OfficialSourceFailure, ValueError) as exc:
                    pass
            payload, digest = load("twse_termination")
            events.extend(_twse_termination_events(payload, digest))
            for source_id, binding in TWSE_LISTING_CHANGE_SOURCE_BINDINGS.items():
                covered_symbols = {
                    symbol
                    for symbol, _event_type, _effective_date
                    in _listing_change_expected_records(binding)
                }
                if (
                    target_date < _datetime.date.fromisoformat(
                        str(binding["announcement_date"])
                    )
                    or not requested["TWSE"] & covered_symbols
                ):
                    continue
                payload, digest = load(source_id)
                events.extend(_twse_listing_change_events(
                    payload, digest, source_id
                ))
        if requested.get("TPEx"):
            payload, digest = load("tpex_current_mode")
            events.extend(_tpex_mode_events(payload, digest, target_date))
            payload, digest = load("tpex_suspend_history")
            events.extend(_tpex_history_events(payload, digest))
            payload, digest = load("tpex_termination")
            events.extend(_tpex_termination_events(payload, digest, target_date))
    except OfficialSourceFailure:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OfficialSourceFailure(
            "tw_lifecycle", "schema_validation", safe_message=str(exc)
        ) from None

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((event["exchange"], event["symbol"]), []).append(event)
    statuses: dict[str, Mapping[str, Any]] = {}
    terminated: dict[str, Mapping[str, Any]] = {}
    for exchange, symbols in requested.items():
        for symbol in symbols:
            result = resolve_lifecycle_status(
                grouped.get((exchange, symbol), ()), target_date, active=True
            )
            if result is None:
                continue
            destination = terminated if result["status"] == "officially_terminated" else statuses
            if symbol in destination and destination[symbol] != result:
                raise OfficialSourceFailure(
                    "tw_lifecycle", "cross_source_duplicate",
                    safe_message=f"conflicting lifecycle symbol {symbol}",
                )
            destination[symbol] = MappingProxyType(result)
    if set(statuses) & set(terminated):
        raise OfficialSourceFailure(
            "tw_lifecycle", "schema_validation",
            safe_message="lifecycle status and termination overlap",
        )
    return LifecycleSnapshot(
        target_date=target_date,
        status_by_symbol=MappingProxyType(dict(statuses)),
        terminated_by_symbol=MappingProxyType(dict(terminated)),
        source_hashes=MappingProxyType(dict(sorted(source_hashes.items()))),
        request_count=request_count,
    )
