"""Authoritative US trading status and halt evidence contracts."""

from __future__ import annotations

import datetime as _datetime
import hashlib
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
    if raw_fields is not None:
        doc["raw_fields"] = raw_fields
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
    ):
        raise ValueError("US trading status evidence schema or hash is invalid")

    return value


def fetch_nasdaq_trade_halts(
    target_market_date: _datetime.date,
    *,
    timeout: int = 10,
    mock_xml: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch authoritative US trading halts from official Nasdaq TradeHalts feed."""
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
        except Exception:
            return {}

    payload_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        root = ET.fromstring(raw_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return {}

    status_map = {}
    target_fmt = target_market_date.strftime("%m/%d/%Y")

    for item in root.findall(".//item"):
        desc = item.find("description")
        if desc is None or not desc.text:
            continue
        # Extract fields from description table HTML
        text = desc.text
        # Look for Halt Date, Symbol, Reason Code, Market
        symbol_match = re.search(r"<td>([A-Z0-9\.\-]+)</td>\s*<td>([^<]*)</td>\s*<td>(NASDAQ|NYSE|AMEX|CBOE|BATS)</td>\s*<td>([A-Z0-9]+)</td>", text)
        date_match = re.search(r"<td>(\d{2}/\d{2}/\d{4})</td>", text)
        if not symbol_match or not date_match:
            continue

        raw_sym = symbol_match.group(1).replace(".", "-")
        market_name = symbol_match.group(3)
        reason = symbol_match.group(4)
        halt_date_str = date_match.group(1)

        try:
            halt_date = _datetime.datetime.strptime(halt_date_str, "%m/%d/%Y").date()
        except ValueError:
            continue

        # If halt was active on or before target date
        if halt_date <= target_market_date:
            try:
                valid_sym = validate_us_ticker(raw_sym)
                doc = create_us_status_evidence(
                    status="officially_suspended",
                    symbol=valid_sym,
                    target_market_date=target_market_date,
                    exchange=market_name,
                    source_id="nasdaq_tradehalts_rss",
                    payload_sha256=payload_hash,
                    raw_fields={"reason": reason, "halt_date": halt_date.isoformat()},
                )
                status_map[valid_sym] = doc
            except ValueError:
                continue

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
