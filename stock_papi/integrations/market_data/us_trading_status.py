"""Authoritative US trading status and halt evidence contracts."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import html
import json
import re
from typing import Any, Mapping, Sequence
from stock_papi.integrations.market_data.us_universe import (
    US_ACCEPTED_EXCHANGES,
    validate_us_ticker,
)

STATUS_SCHEMA_VERSION = 1
STATUS_PARSER_VERSION = "us-official-status-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGULAR_SESSION_OPEN = _datetime.time(9, 30)
_REGULAR_SESSION_CLOSE = _datetime.time(16, 0)


class USStatusSourceError(Exception):
    """Base class for failures in the required official US status source."""


class USStatusOperationalError(USStatusSourceError):
    """Network, timeout, HTTP, or transport failure from the official source."""


class USStatusSchemaError(USStatusSourceError):
    """Malformed XML or unexpected official source schema."""


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


def create_us_status_evidence(
    *,
    status: str,
    symbol: str,
    target_market_date: _datetime.date,
    exchange: str,
    source_id: str,
    payload_sha256: str,
    raw_fields: dict[str, Any] | None = None,
    lifecycle_events: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if status not in {"official_no_regular_trade", "officially_suspended"}:
        raise ValueError(f"unsupported US status: {status}")
    valid_symbol = validate_us_ticker(symbol)
    exchange_upper = exchange.upper()
    if exchange_upper not in US_ACCEPTED_EXCHANGES:
        raise ValueError(f"unsupported US exchange: {exchange}")
    if not _SHA256.fullmatch(str(payload_sha256)):
        raise ValueError("invalid payload_sha256")
    if not isinstance(raw_fields, dict) or raw_fields.get("effective_on_target_session") is not True:
        raise ValueError("US status evidence must prove target-session effectiveness")

    doc: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "market": "US",
        "exchange": exchange_upper,
        "symbol": valid_symbol,
        "target_market_date": target_market_date.isoformat(),
        "source_id": str(source_id),
        "payload_sha256": str(payload_sha256),
        "parser_version": STATUS_PARSER_VERSION,
    }
    doc["raw_fields"] = dict(raw_fields)
    if lifecycle_events is not None:
        doc["lifecycle_events"] = list(lifecycle_events)

    doc["evidence_sha256"] = evidence_sha256(doc)
    return doc


def validate_us_status_evidence(
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
        raise ValueError("US trading status evidence target date is invalid") from None

    if (
        value.get("schema_version") != STATUS_SCHEMA_VERSION
        or value.get("status") not in {"official_no_regular_trade", "officially_suspended"}
        or value.get("market") != "US"
        or value.get("exchange") not in US_ACCEPTED_EXCHANGES
        or validate_us_ticker(value.get("symbol", "")) != str(value.get("symbol"))
        or (symbol is not None and value.get("symbol") != validate_us_ticker(symbol))
        or (target_date is not None and evidence_target != target_date)
        or not _SHA256.fullmatch(str(value.get("payload_sha256") or ""))
        or value.get("parser_version") != STATUS_PARSER_VERSION
        or value.get("evidence_sha256") != evidence_sha256(value)
        or not isinstance(value.get("raw_fields"), dict)
        or value["raw_fields"].get("effective_on_target_session") is not True
    ):
        raise ValueError("US trading status evidence schema or hash is invalid")

    return value


def _parse_status_date(value: str) -> _datetime.date:
    value = str(value).strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return _datetime.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid US status date: {value!r}")


def _parse_status_time(value: str | None) -> _datetime.time | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().upper()
    for fmt in (
        "%H:%M:%S.%f",
        "%H:%M:%S",
        "%H:%M",
        "%I:%M:%S.%f %p",
        "%I:%M:%S %p",
        "%I:%M %p",
    ):
        try:
            return _datetime.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"invalid US status time: {value!r}")


def is_halt_effective_for_target_session(
    *,
    halt_date: _datetime.date,
    target_market_date: _datetime.date,
    halt_time: str | None = None,
    resumption_date: _datetime.date | None = None,
    resumption_time: str | None = None,
) -> bool:
    """Return true only when the halt interval covers the complete target session.

    An old halt without an explicit resumption/active-through interval is not
    enough evidence for a historical target session. A same-day resumption
    before the regular close means the symbol is not a full-session non-price
    observation.
    """
    if not isinstance(halt_date, _datetime.date) or not isinstance(
        target_market_date, _datetime.date
    ):
        raise ValueError("halt and target dates must be dates")
    if resumption_date is not None and not isinstance(resumption_date, _datetime.date):
        raise ValueError("resumption_date must be a date")

    start_time = _parse_status_time(halt_time) or _datetime.time(0, 0)
    resume_time = _parse_status_time(resumption_time)
    session_start = _datetime.datetime.combine(
        target_market_date, _REGULAR_SESSION_OPEN
    )
    session_close = _datetime.datetime.combine(
        target_market_date, _REGULAR_SESSION_CLOSE
    )
    halt_start = _datetime.datetime.combine(halt_date, start_time)
    if halt_start > session_close:
        return False

    if halt_date < target_market_date and resumption_date is None:
        return False
    if resumption_date is None:
        return halt_start <= session_start
    if resumption_date < target_market_date:
        return False
    if resumption_date > target_market_date:
        return True
    if resume_time is None:
        return False
    resumption = _datetime.datetime.combine(resumption_date, resume_time)
    return halt_start <= session_start and resumption > session_close


def _description_cells(description: str) -> list[str]:
    matches = re.findall(r"<td\b[^>]*>(.*?)</td>", description, flags=re.I | re.S)
    return [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(cell))).strip()
        for cell in matches
    ]


def _description_headers(description: str) -> list[str]:
    matches = re.findall(r"<th\b[^>]*>(.*?)</th>", description, flags=re.I | re.S)
    return [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(cell))).strip()
        for cell in matches
    ]


def _normalise_description_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _parse_halt_fields(
    *,
    symbol: str | None,
    market: str | None,
    reason: str | None,
    halt_date: str | None,
    halt_time: str | None = None,
    issue_name: str | None = None,
    resumption_date: str | None = None,
    resumption_quote_time: str | None = None,
    resumption_trade_time: str | None = None,
) -> dict[str, Any]:
    if not symbol or not market or not halt_date:
        raise USStatusSchemaError("Nasdaq halt item required fields are missing")
    if (resumption_quote_time or resumption_trade_time) and not resumption_date:
        raise USStatusSchemaError(
            "Nasdaq halt item resumption time has no resumption date"
        )
    try:
        parsed_halt_date = _parse_status_date(halt_date)
        parsed_resumption_date = (
            _parse_status_date(resumption_date) if resumption_date else None
        )
        parsed_halt_time = _parse_status_time(halt_time)
        parsed_resumption_quote_time = _parse_status_time(resumption_quote_time)
        parsed_resumption_trade_time = _parse_status_time(resumption_trade_time)
    except ValueError as exc:
        raise USStatusSchemaError(str(exc)) from exc
    return {
        "symbol": str(symbol).strip().upper().replace(".", "-"),
        "market": str(market).strip().upper(),
        "reason": str(reason or "").strip(),
        "reason_code": str(reason or "").strip(),
        "issue_name": str(issue_name or "").strip(),
        "halt_date": parsed_halt_date,
        "halt_time": parsed_halt_time,
        "resumption_date": parsed_resumption_date,
        "resumption_quote_time": parsed_resumption_quote_time,
        "resumption_trade_time": parsed_resumption_trade_time,
        "resumption_time": parsed_resumption_trade_time or parsed_resumption_quote_time,
    }


def _parse_halt_description(description: str) -> dict[str, Any]:
    cells = _description_cells(description)
    if len(cells) < 5:
        raise USStatusSchemaError("Nasdaq halt item table is incomplete")

    headers = _description_headers(description)
    if headers and len(headers) == len(cells):
        mapped = {
            _normalise_description_label(header): cell
            for header, cell in zip(headers, cells)
        }
        return _parse_halt_fields(
            symbol=mapped.get("issue_symbol"),
            market=mapped.get("market"),
            reason=mapped.get("reason_code"),
            halt_date=mapped.get("halt_date"),
            halt_time=mapped.get("halt_time"),
            issue_name=mapped.get("issue_name"),
            resumption_date=mapped.get("resumption_date"),
            resumption_quote_time=mapped.get("resumption_quote_time"),
            resumption_trade_time=mapped.get("resumption_trade_time"),
        )

    # Compatibility with the compact table shape used by older test fixtures.
    symbol = cells[0].upper().replace(".", "-")
    market = cells[2].upper()
    reason = cells[3]
    date_tokens = re.findall(r"(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})", " ".join(cells[4:]))
    if not date_tokens:
        raise USStatusSchemaError("Nasdaq halt item has no halt date")
    time_tokens = re.findall(
        r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b",
        " ".join(cells[4:]),
        flags=re.I,
    )
    return _parse_halt_fields(
        symbol=symbol,
        market=market,
        reason=reason,
        halt_date=date_tokens[0],
        halt_time=time_tokens[0] if time_tokens else None,
        resumption_date=date_tokens[1] if len(date_tokens) > 1 else None,
        resumption_trade_time=time_tokens[1] if len(time_tokens) > 1 else None,
    )


def _item_field(item: Any, field_name: str) -> str | None:
    expected = field_name.lower()
    for node in item.iter():
        if node.tag.rsplit("}", 1)[-1].lower() == expected:
            value = (node.text or "").strip()
            return value or None
    return None


def _parse_halt_item(item: Any, description: str | None) -> dict[str, Any]:
    structured = {
        field: _item_field(item, field)
        for field in (
            "HaltDate",
            "HaltTime",
            "IssueSymbol",
            "IssueName",
            "Market",
            "ReasonCode",
            "ResumptionDate",
            "ResumptionQuoteTime",
            "ResumptionTradeTime",
        )
    }
    structured_required = ("HaltDate", "IssueSymbol", "Market")
    if any(structured.values()):
        if not all(structured.get(field) for field in structured_required):
            raise USStatusSchemaError(
                "Nasdaq halt item structured fields are incomplete"
            )
        return _parse_halt_fields(
            symbol=structured["IssueSymbol"],
            market=structured["Market"],
            reason=structured["ReasonCode"],
            halt_date=structured["HaltDate"],
            halt_time=structured["HaltTime"],
            issue_name=structured["IssueName"],
            resumption_date=structured["ResumptionDate"],
            resumption_quote_time=structured["ResumptionQuoteTime"],
            resumption_trade_time=structured["ResumptionTradeTime"],
        )
    if description:
        return _parse_halt_description(description)
    raise USStatusSchemaError("Nasdaq halt item has no structured fields")


def fetch_nasdaq_trade_halts(
    target_market_date: _datetime.date,
    *,
    timeout: int = 10,
    mock_xml: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch authoritative US trading halts from official Nasdaq TradeHalts feed."""
    import urllib.error
    import urllib.request
    import xml.etree.ElementTree as ET

    if mock_xml is not None:
        raw_bytes = mock_xml.encode("utf-8")
    else:
        req = urllib.request.Request(
            "http://www.nasdaqtrader.com/rss.aspx?feed=tradehalts",
            headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            raise USStatusOperationalError(
                f"Nasdaq trade-halt HTTP error: {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise USStatusOperationalError(
                f"Nasdaq trade-halt transport failure: {exc}"
            ) from exc
        except Exception as exc:
            raise USStatusOperationalError(
                f"Nasdaq trade-halt source failure: {exc}"
            ) from exc

    payload_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        root = ET.fromstring(raw_bytes.decode("utf-8"))
    except (UnicodeError, ET.ParseError) as exc:
        raise USStatusSchemaError("Nasdaq trade-halt XML is malformed") from exc

    root_name = root.tag.rsplit("}", 1)[-1].lower()
    if root_name not in {"rss", "rdf"}:
        raise USStatusSchemaError("Nasdaq trade-halt XML root is unexpected")
    items = [
        item for item in root.iter() if item.tag.rsplit("}", 1)[-1].lower() == "item"
    ]
    if root_name == "rss" and not any(
        node.tag.rsplit("}", 1)[-1].lower() == "channel" for node in root.iter()
    ):
        raise USStatusSchemaError("Nasdaq trade-halt RSS channel is missing")

    status_map: dict[str, dict[str, Any]] = {}
    for item in items:
        descriptions = [
            node for node in item.iter() if node.tag.rsplit("}", 1)[-1].lower() == "description"
        ]
        description = descriptions[0].text if descriptions and descriptions[0].text else None
        parsed = _parse_halt_item(item, description)
        try:
            valid_sym = validate_us_ticker(parsed["symbol"])
            market_name = parsed["market"]
            if market_name not in {"NASDAQ", "NYSE", "AMEX", "CBOE", "BATS"}:
                raise ValueError("unsupported Nasdaq halt market")
        except ValueError as exc:
            raise USStatusSchemaError("Nasdaq halt item identity is invalid") from exc

        if not is_halt_effective_for_target_session(
            halt_date=parsed["halt_date"],
            target_market_date=target_market_date,
            halt_time=(parsed["halt_time"].strftime("%H:%M:%S") if parsed["halt_time"] else None),
            resumption_date=parsed["resumption_date"],
            resumption_time=(
                parsed["resumption_time"].strftime("%H:%M:%S")
                if parsed["resumption_time"]
                else None
            ),
        ):
            continue

        raw_fields = {
            "reason": parsed["reason"],
            "reason_code": parsed["reason_code"],
            "issue_name": parsed["issue_name"],
            "halt_date": parsed["halt_date"].isoformat(),
            "effective_on_target_session": True,
        }
        if parsed["halt_time"]:
            raw_fields["halt_time"] = parsed["halt_time"].strftime("%H:%M:%S")
        if parsed["resumption_date"]:
            raw_fields["resumption_date"] = parsed["resumption_date"].isoformat()
        if parsed["resumption_quote_time"]:
            raw_fields["resumption_quote_time"] = parsed["resumption_quote_time"].strftime(
                "%H:%M:%S"
            )
        if parsed["resumption_trade_time"]:
            raw_fields["resumption_trade_time"] = parsed["resumption_trade_time"].strftime(
                "%H:%M:%S"
            )
        doc = create_us_status_evidence(
            status="officially_suspended",
            symbol=valid_sym,
            target_market_date=target_market_date,
            exchange=market_name,
            source_id="nasdaq_tradehalts_rss",
            payload_sha256=payload_hash,
            raw_fields=raw_fields,
        )
        previous = status_map.get(valid_sym)
        if previous is not None and previous != doc:
            raise USStatusSchemaError(
                f"Nasdaq halt source contains conflicting duplicate rows for {valid_sym}"
            )
        status_map[valid_sym] = doc

    return status_map


def get_us_trading_status_snapshot(
    target_market_date: _datetime.date,
    symbols: Sequence[str] | None = None,
    *,
    mock_xml: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build authoritative target-date trading status evidence map."""
    halts = fetch_nasdaq_trade_halts(target_market_date, mock_xml=mock_xml)
    if symbols is None:
        return halts
    sym_set = set(symbols)
    return {s: doc for s, doc in halts.items() if s in sym_set}
