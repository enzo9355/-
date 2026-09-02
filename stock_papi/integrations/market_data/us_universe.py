"""Authoritative US stock market universe from official SEC exchange listings."""

import datetime
import base64
import hashlib
import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import zoneinfo

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")

SEC_US_UNIVERSE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NASDAQ_US_UNIVERSE_URLS = (
    "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json",
)
NASDAQ_SYMBOL_DIRECTORY_URLS = (
    (
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "nasdaqtrader:nasdaqlisted",
    ),
    (
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "nasdaqtrader:otherlisted",
    ),
)
NYSE_SYMBOL_MAPPING_INDEX_URL = "https://ftp.nyse.com/NYSESymbolMapping/"
SEC_US_UNIVERSE_MAX_BYTES = 15 * 1024 * 1024
NASDAQ_SYMBOL_DIRECTORY_MAX_BYTES = 8 * 1024 * 1024
NYSE_SYMBOL_MAPPING_INDEX_MAX_BYTES = 2 * 1024 * 1024
NYSE_SYMBOL_MAPPING_MAX_BYTES = 12 * 1024 * 1024
NASDAQ_CORPORATE_ACTION_ALERT_URL = "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-576"
NASDAQ_CORPORATE_ACTION_ALERT_SOURCE_ID = "nasdaqtrader:corporate_action:ECA2026-576"
NASDAQ_CORPORATE_ACTION_ALERT_CLAIMS_SHA256 = "228096681f4d7b2f280f7fc7df4c063336227bc4ef6ee011fcb54d598ed822a2"
NASDAQ_CORPORATE_ACTION_ALERT_EFFECTIVE_DATE = datetime.date(2026, 8, 17)
NASDAQ_CORPORATE_ACTION_ALERT_MAX_BYTES = 2 * 1024 * 1024
US_UNIVERSE_CACHE_CONTRACT_VERSION = 8
US_ACCEPTED_EXCHANGES = frozenset(
    {"NASDAQ", "NYSE", "CBOE", "BATS", "NYS", "ASE", "AMEX"}
)
CRYPTO_SECURITY_TERMS = frozenset(
    {"bitcoin", "ethereum", "crypto", "token", "digital asset", "solana", "xrp"}
)
VALID_US_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?$")


def validate_us_ticker(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not (1 <= len(symbol) <= 10) or not VALID_US_TICKER_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid US ticker: {symbol!r}")
    return symbol


@dataclass(frozen=True)
class USUniverseBreakdown:
    configured_listed_count: int
    eligible_listed_count: int
    active_universe_count: int
    excluded_exchange_count: int
    excluded_crypto_count: int
    excluded_invalid_count: int
    excluded_derivative_count: int
    derivative_breakdown: dict[str, int]
    terminated_delisted_count: int | None
    exchange_counts: dict[str, int]
    symbols: list[str]
    exclusions_by_symbol: dict[str, dict[str, Any]]
    security_metadata_status: str = "unavailable"
    security_metadata_sources: list[dict[str, Any]] = field(default_factory=list)
    security_type_counts: dict[str, int] = field(default_factory=dict)
    security_eligibility_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    lifecycle_evidence_status: str = "lifecycle_evidence_unavailable"
    lifecycle_evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_events_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)


SUPPORTED_SECURITY_TYPES = frozenset(
    {
        "COMMON_EQUITY",
        "ADR",
        "ETF",
        "ETN",
        "WARRANT",
        "RIGHT",
        "UNIT",
        "PREFERRED",
        "UNKNOWN",
    }
)
DERIVATIVE_SECURITY_TYPES = frozenset({"ETN", "WARRANT", "RIGHT", "UNIT", "PREFERRED"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
NYSE_SECURITY_TYPE_CODES = {
    "C": "COMMON_EQUITY",
    "E": "ETF",
    "H": "ADR",
    "I": "UNIT",
    "M": "PREFERRED",
    "O": "COMMON_EQUITY",
    "P": "PREFERRED",
    "R": "RIGHT",
    "W": "WARRANT",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _evidence_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _metadata_source_record(
    *,
    source_id: str,
    source_url: str,
    payload_sha256: str,
    as_of: str | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    clean = dict(record)
    clean.pop("evidence_sha256", None)
    return {
        **clean,
        "source_id": source_id,
        "source_url": source_url,
        "source_identity": f"{source_id}:{payload_sha256}",
        "payload_sha256": payload_sha256,
        "as_of": as_of,
        "evidence_sha256": _evidence_sha256(clean),
    }


def _directory_as_of(text: str) -> str | None:
    match = re.search(
        r"File Creation Time:\s*(?:File Creation Time:\s*)?(\d{8})(\d{2}:?\d{2})",
        text,
        flags=re.I,
    )
    if not match:
        return None
    raw = match.group(1) + re.sub(r"[: ]", "", match.group(2))
    try:
        return datetime.datetime.strptime(raw, "%m%d%Y%H%M").date().isoformat()
    except ValueError:
        return None


def _official_security_type_from_name(name: str, *, etf: str | None = None) -> str:
    """Classify only from fields published by an exchange directory."""
    name_lower = str(name or "").strip().lower()
    if re.search(r"\bunit(?:s)?\b", name_lower):
        return "UNIT"
    if re.search(r"\bwarrant(?:s)?\b", name_lower):
        return "WARRANT"
    if re.search(
        r"\bright(?:s)?(?:\s*,|\s*$|\s+each|\s+entitl|\s+to\b|\s*\()",
        name_lower,
    ):
        return "RIGHT"
    if re.search(r"\betn(?:s)?\b", name_lower):
        return "ETN"
    if str(etf or "").strip().upper() == "Y":
        return "ETF"
    if re.search(r"american depositary|\badr(?:s)?\b|\bads\b", name_lower):
        return "ADR"
    if "preferred" in name_lower or "preference" in name_lower:
        return "PREFERRED"
    if re.search(r"\b(?:common stock|common shares?|ordinary shares?)\b", name_lower):
        return "COMMON_EQUITY"
    return "UNKNOWN"


def _official_symbol_aliases(
    value: str,
    *,
    security_type: str | None = None,
) -> set[str]:
    """Return conservative SEC-ticker aliases for exchange-native symbols."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    upper = raw.upper()
    aliases: set[str] = set()
    p_marker = re.fullmatch(r"^(.+)[p]([A-Z])$", raw) is not None
    right_marker = (
        security_type == "RIGHT"
        and re.fullmatch(r"^(.+?)(?:r|rw)$", raw) is not None
    )

    def add(candidate: str) -> None:
        candidate = re.sub(r"\s+", "-", str(candidate or "").strip().upper())
        if not candidate:
            return
        try:
            aliases.add(validate_us_ticker(candidate))
        except ValueError:
            return

    if not p_marker and not right_marker:
        add(upper)

    if "$" in upper:
        root, suffix = upper.split("$", 1)
        if suffix:
            add(f"{root}-P{suffix}")
        else:
            add(f"{root}-P")

    if raw and re.fullmatch(r"^(.+)[p]([A-Z])$", raw):
        match = re.fullmatch(r"^(.+)[p]([A-Z])$", raw)
        assert match is not None
        add(f"{match.group(1)}-P{match.group(2)}")

    if right_marker:
        match = re.fullmatch(r"^(.+?)(r|rw)$", raw)
        assert match is not None
        add(f"{match.group(1)}-{'RW' if match.group(2) == 'rw' else 'RI'}")

    if upper.endswith("="):
        add(upper[:-1] + "-UN")
    elif upper.endswith("+"):
        add(upper[:-1] + "-WT")
    elif upper.endswith("^"):
        add(upper[:-1] + "-RI")

    if "." in upper:
        root, suffix = upper.rsplit(".", 1)
        if suffix == "U":
            add(f"{root}-UN")
        elif suffix in {"W", "WS", "WT"}:
            add(f"{root}-WT")
        elif suffix in {"R", "RT"}:
            add(f"{root}-RI")
        elif security_type == "WARRANT" and re.fullmatch(r"[A-Z]", suffix):
            add(f"{root}-WT{suffix}")
        elif security_type == "RIGHT" and re.fullmatch(r"[A-Z]", suffix):
            add(f"{root}-RI{suffix}")
        elif security_type == "UNIT" and re.fullmatch(r"[A-Z]", suffix):
            add(f"{root}-UN{suffix}")
        else:
            add(f"{root}-{suffix}")

    if re.fullmatch(r"^(.+)\s+PR[A-Z]$", upper):
        match = re.fullmatch(r"^(.+)\s+PR([A-Z])$", upper)
        assert match is not None
        add(f"{match.group(1)}-P{match.group(2)}")
    elif re.fullmatch(r"^(.+)\s+P[A-Z]$", upper):
        match = re.fullmatch(r"^(.+)\s+P([A-Z])$", upper)
        assert match is not None
        add(f"{match.group(1)}-P{match.group(2)}")
    elif re.fullmatch(r"^(.+)\s+U$", upper):
        add(upper[:-2] + "-UN")
    elif re.fullmatch(r"^(.+)\s+(?:WS|W)$", upper):
        add(upper.rsplit(" ", 1)[0] + "-WT")
    elif re.fullmatch(r"^(.+)\s+(?:RT|R)$", upper):
        add(upper.rsplit(" ", 1)[0] + "-RI")

    return aliases


def parse_nasdaq_security_directory(
    text: str,
    *,
    source_id: str,
    source_url: str = "",
    as_of: str | None = None,
) -> dict[str, Any]:
    """Parse an official Nasdaq Trader symbol directory with provenance."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nasdaq security directory is empty")
    lines = [line.lstrip("\ufeff") for line in text.splitlines() if line.strip()]
    header_index = next(
        (index for index, line in enumerate(lines) if "|" in line and "Security Name" in line),
        None,
    )
    if header_index is None:
        raise ValueError("Nasdaq security directory header is missing")
    headers = [item.strip() for item in lines[header_index].split("|")]
    symbol_field = next(
        (field_name for field_name in ("Symbol", "ACT Symbol", "Nasdaq Symbol") if field_name in headers),
        None,
    )
    if symbol_field is None or "Security Name" not in headers:
        raise ValueError("Nasdaq security directory schema is incomplete")
    positions = {name: index for index, name in enumerate(headers)}
    payload_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    effective_as_of = as_of or _directory_as_of(text)
    records: dict[str, dict[str, Any]] = {}
    for line in lines[header_index + 1 :]:
        if line.lower().startswith("file creation time"):
            continue
        values = line.split("|")
        if len(values) < len(headers):
            raise ValueError("Nasdaq security directory row is incomplete")
        raw_symbol = values[positions[symbol_field]].strip()
        if not raw_symbol:
            continue
        security_name = values[positions["Security Name"]].strip()
        etf = values[positions["ETF"]].strip().upper() if "ETF" in positions else None
        security_type = _official_security_type_from_name(security_name, etf=etf)
        aliases = _official_symbol_aliases(raw_symbol, security_type=security_type)
        for field_name in ("CQS Symbol", "NASDAQ Symbol"):
            if field_name in positions:
                aliases.update(
                    _official_symbol_aliases(
                        values[positions[field_name]].strip(),
                        security_type=security_type,
                    )
                )
        if not aliases:
            raise ValueError("Nasdaq security directory row has no valid symbol alias")
        record = {
            "security_name": security_name,
            "exchange_code": values[positions["Exchange"]].strip() if "Exchange" in positions else None,
            "etf": etf,
            "test_issue": values[positions["Test Issue"]].strip().upper() if "Test Issue" in positions else None,
            "financial_status": values[positions["Financial Status"]].strip().upper() if "Financial Status" in positions else None,
            "security_type": security_type,
            "source_symbol": raw_symbol.upper(),
            "cqs_symbol": values[positions["CQS Symbol"]].strip() if "CQS Symbol" in positions else None,
        }
        for symbol in sorted(aliases):
            enriched = _metadata_source_record(
                source_id=source_id,
                source_url=source_url,
                payload_sha256=payload_sha256,
                as_of=effective_as_of,
                record={"symbol": symbol, **record},
            )
            previous = records.get(symbol)
            if previous is not None and previous != enriched:
                raise ValueError(f"Nasdaq security directory contains conflicting {symbol}")
            records[symbol] = enriched
    if not records:
        raise ValueError("Nasdaq security directory contains no symbols")
    return {
        "source_id": source_id,
        "source_url": source_url,
        "source_identity": f"{source_id}:{payload_sha256}",
        "payload_sha256": payload_sha256,
        "as_of": effective_as_of,
        "records": records,
    }


def _xml_child_text(element: ET.Element, field_name: str) -> str:
    for child in list(element):
        if child.tag.rsplit("}", 1)[-1] == field_name:
            return str(child.text or "").strip()
    return ""


def parse_nyse_security_mapping(
    text: str,
    *,
    source_id: str,
    source_url: str = "",
    as_of: str | None = None,
) -> dict[str, Any]:
    """Parse NYSE's first-party XML symbol mapping with typed security codes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("NYSE security mapping is empty")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("NYSE security mapping XML is invalid") from exc
    rows = [
        element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "SymbolMap"
    ]
    if not rows:
        raise ValueError("NYSE security mapping contains no SymbolMap rows")
    payload_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    effective_as_of = as_of
    if effective_as_of is None:
        match = re.search(r"(\d{8})", source_url)
        if match:
            try:
                effective_as_of = datetime.datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
            except ValueError:
                effective_as_of = None

    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_symbol = _xml_child_text(row, "Symbol")
        cqs_symbol = _xml_child_text(row, "CQS_Symbol")
        listed_market = _xml_child_text(row, "ListedMarket")
        security_type_code = _xml_child_text(row, "Security_Type").upper()
        if not raw_symbol or not listed_market or not security_type_code:
            raise ValueError("NYSE security mapping row is incomplete")
        security_type = NYSE_SECURITY_TYPE_CODES.get(security_type_code, "UNKNOWN")
        aliases = _official_symbol_aliases(raw_symbol, security_type=security_type)
        aliases.update(_official_symbol_aliases(cqs_symbol, security_type=security_type))
        if not aliases:
            raise ValueError("NYSE security mapping row has no valid symbol alias")
        record = {
            "exchange_code": listed_market,
            "etf": "Y" if security_type == "ETF" else None,
            "security_type": security_type,
            "security_type_code": security_type_code,
            "source_symbol": raw_symbol,
            "cqs_symbol": cqs_symbol or None,
        }
        for symbol in sorted(aliases):
            enriched = _metadata_source_record(
                source_id=source_id,
                source_url=source_url,
                payload_sha256=payload_sha256,
                as_of=effective_as_of,
                record={"symbol": symbol, **record},
            )
            previous = records.get(symbol)
            if previous is not None and previous != enriched:
                raise ValueError(f"NYSE security mapping contains conflicting {symbol}")
            records[symbol] = enriched
    if not records:
        raise ValueError("NYSE security mapping contains no symbols")
    return {
        "source_id": source_id,
        "source_url": source_url,
        "source_identity": f"{source_id}:{payload_sha256}",
        "payload_sha256": payload_sha256,
        "as_of": effective_as_of,
        "records": records,
    }


def fetch_nyse_security_mapping(*, timeout: int = 15) -> dict[str, Any]:
    """Fetch the latest dated NYSE first-party XML symbol mapping."""
    index_request = urllib.request.Request(
        NYSE_SYMBOL_MAPPING_INDEX_URL,
        headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
    )
    with urllib.request.urlopen(index_request, timeout=timeout) as resp:
        index_content = resp.read(NYSE_SYMBOL_MAPPING_INDEX_MAX_BYTES + 1)
    if len(index_content) > NYSE_SYMBOL_MAPPING_INDEX_MAX_BYTES:
        raise RuntimeError("NYSE security mapping index response is too large")
    index_text = index_content.decode("utf-8", errors="strict")
    filenames = sorted(
        set(re.findall(r"NYSESymbolMapping_(\d{8})\.xml", index_text, flags=re.I)),
        reverse=True,
    )
    if not filenames:
        raise RuntimeError("NYSE security mapping index contains no dated XML")
    filename = f"NYSESymbolMapping_{filenames[0]}.xml"
    source_url = NYSE_SYMBOL_MAPPING_INDEX_URL + filename
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        content = resp.read(NYSE_SYMBOL_MAPPING_MAX_BYTES + 1)
    if len(content) > NYSE_SYMBOL_MAPPING_MAX_BYTES:
        raise RuntimeError("NYSE security mapping response is too large")
    document = parse_nyse_security_mapping(
        content.decode("utf-8-sig"),
        source_id="nyse:security_mapping",
        source_url=source_url,
        as_of=datetime.datetime.strptime(filenames[0], "%Y%m%d").date().isoformat(),
    )
    return {
        "records": document["records"],
        "sources": [
            {
                "source_id": document["source_id"],
                "source_url": document["source_url"],
                "source_identity": document["source_identity"],
                "payload_sha256": document["payload_sha256"],
                "as_of": document["as_of"],
                "status": "healthy",
            }
        ],
        "status": "healthy",
    }


def fetch_nasdaq_security_directory(*, timeout: int = 15) -> dict[str, Any]:
    """Fetch first-party Nasdaq symbol directories used for security eligibility."""
    documents = []
    errors = []
    for url, source_id in NASDAQ_SYMBOL_DIRECTORY_URLS:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read(NASDAQ_SYMBOL_DIRECTORY_MAX_BYTES + 1)
            if len(content) > NASDAQ_SYMBOL_DIRECTORY_MAX_BYTES:
                raise RuntimeError("Nasdaq security directory response is too large")
            documents.append(
                parse_nasdaq_security_directory(
                    content.decode("utf-8"),
                    source_id=source_id,
                    source_url=url,
                )
            )
        except Exception as exc:
            errors.append({"source_id": source_id, "source_url": url, "error_type": type(exc).__name__})
    if not documents:
        raise RuntimeError("all Nasdaq security directories are unavailable")
    records: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for document in documents:
        for symbol, record in document["records"].items():
            previous = records.get(symbol)
            if previous is not None and (
                previous.get("security_name") != record.get("security_name")
                or previous.get("etf") != record.get("etf")
                or previous.get("security_type") != record.get("security_type")
            ):
                conflicts.add(symbol)
                continue
            records[symbol] = record
    for symbol in conflicts:
        records.pop(symbol, None)
    return {
        "records": records,
        "sources": [
            {
                "source_id": document["source_id"],
                "source_url": document["source_url"],
                "source_identity": document["source_identity"],
                "payload_sha256": document["payload_sha256"],
                "as_of": document["as_of"],
                "status": "healthy",
            }
            for document in documents
        ]
        + errors,
        "status": "healthy" if not errors else "partial",
        "conflicted_symbols": sorted(conflicts),
    }


def fetch_official_us_security_metadata(
    *,
    timeout: int = 15,
    target_market_date: datetime.date | None = None,
) -> dict[str, Any]:
    """Combine first-party Nasdaq and NYSE metadata without weakening conflicts."""
    documents: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    partial_source = False
    fetchers = (
        (fetch_nasdaq_security_directory, "nasdaqtrader:symbol_directory", "https://www.nasdaqtrader.com/"),
        (fetch_nyse_security_mapping, "nyse:security_mapping", NYSE_SYMBOL_MAPPING_INDEX_URL),
    )
    for fetcher, source_id, source_url in fetchers:
        try:
            document = fetcher(timeout=timeout)
            documents.append(document)
            sources.extend(dict(item) for item in (document.get("sources") or []))
            partial_source = partial_source or document.get("status") != "healthy"
        except Exception as exc:
            error = {
                "source_id": source_id,
                "source_url": source_url,
                "status": "unavailable",
                "error_type": type(exc).__name__,
            }
            errors.append(error)
            sources.append(error)
    if not documents:
        raise RuntimeError("all first-party US security metadata sources are unavailable")

    records: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for document in documents:
        for symbol, record in (document.get("records") or {}).items():
            previous = records.get(symbol)
            if previous is None:
                records[symbol] = record
                continue
            previous_type = str(previous.get("security_type") or "UNKNOWN").upper()
            current_type = str(record.get("security_type") or "UNKNOWN").upper()
            if (
                previous_type != "UNKNOWN"
                and current_type != "UNKNOWN"
                and previous_type != current_type
            ):
                if previous_type == "ETN" or current_type == "ETN":
                    etn_record = previous if previous_type == "ETN" else record
                    etn_source = str(etn_record.get("source_id") or "")
                    if etn_source.startswith("nasdaqtrader:"):
                        # Nasdaq's explicit ETN name is stronger than a
                        # generic ETF flag from another exchange mapping.
                        records[symbol] = etn_record
                        continue
                if previous_type in DERIVATIVE_SECURITY_TYPES and current_type in DERIVATIVE_SECURITY_TYPES:
                    # Both official records prove the instrument is outside the
                    # equity-observation scope; retain the later typed exchange
                    # record without inventing a common-equity classification.
                    records[symbol] = record
                    continue
                conflicts.add(symbol)
                continue
            if previous_type == "UNKNOWN" and current_type != "UNKNOWN":
                records[symbol] = record
    for symbol in conflicts:
        records.pop(symbol, None)
    return {
        "records": records,
        "sources": sources,
        "status": "healthy" if not errors and not partial_source else "partial",
        "conflicted_symbols": sorted(conflicts),
    }


def _heuristic_security_type(symbol: str, name: str, raw_ticker: str) -> str:
    """Return a non-authoritative secondary signal for audit only."""
    name_lower = name.lower()
    sym_upper = symbol.upper()
    raw_upper = raw_ticker.upper()
    if re.search(r"\bwarrants?\b", name_lower) or sym_upper.endswith(("-WT", "-WTA", "-WTB", "-WTC", "WS")) or (len(sym_upper) == 5 and sym_upper.endswith("W")):
        return "WARRANT"
    if re.search(r"\brights?\b", name_lower) or sym_upper.endswith("-RI") or (len(sym_upper) == 5 and sym_upper.endswith("R")):
        return "RIGHT"
    if re.search(r"\bunits?\b", name_lower) or sym_upper.endswith("-UN") or (len(sym_upper) == 5 and sym_upper.endswith("U")):
        return "UNIT"
    if "preferred" in name_lower or "pfd" in name_lower or "-P" in sym_upper or "/PR" in raw_upper or "-PR" in sym_upper:
        return "PREFERRED"
    if "etf" in name_lower:
        return "ETF"
    return "UNKNOWN"


def _authoritative_security_classification(
    symbol: str,
    sec_name: str,
    raw_ticker: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    heuristic = _heuristic_security_type(symbol, sec_name, raw_ticker)
    evidence = dict(metadata or {})
    explicit = str(evidence.get("security_type") or "").strip().upper()
    if explicit in SUPPORTED_SECURITY_TYPES and explicit != "UNKNOWN":
        classification = explicit
        method = "authoritative_security_type_field"
    elif re.search(r"\betn(?:s)?\b", str(evidence.get("security_name") or "").lower()):
        classification = "ETN"
        method = "exchange_security_name_explicit"
    elif str(evidence.get("etf") or "").strip().upper() == "Y":
        classification = "ETF"
        method = "exchange_etf_flag"
    else:
        exchange_name = str(evidence.get("security_name") or "").lower()
        if re.search(r"\b(warrants?|rights?|units?)\s*$", exchange_name):
            classification = (
                "WARRANT" if "warrant" in exchange_name
                else "RIGHT" if "right" in exchange_name
                else "UNIT"
            )
            method = "exchange_security_name_explicit"
        elif "preferred" in exchange_name and re.search(r"\b(stock|securities|share)", exchange_name):
            classification = "PREFERRED"
            method = "exchange_security_name_explicit"
        elif re.search(r"\b(american depositary shares?|ads|adrs?)\b", exchange_name):
            classification = "ADR"
            method = "exchange_security_name_explicit"
        elif re.search(r"\b(common stock|common shares?|ordinary shares?)\b", exchange_name):
            classification = "COMMON_EQUITY"
            method = "exchange_security_name_explicit"
        else:
            classification = "UNKNOWN"
            method = None
    return {
        "security_type": classification,
        "classification_method": method,
        "heuristic_signal": heuristic,
        "authoritative": bool(metadata and method),
        "evidence": evidence,
    }


def classify_sec_security_type(symbol: str, name: str, raw_ticker: str) -> str:
    """Compatibility wrapper for the non-authoritative secondary signal."""
    result = _heuristic_security_type(symbol, name, raw_ticker)
    return "ETF_OR_FUND" if result == "ETF" else result


def _sec_same_issuer_derivative_type(
    symbol: str, issuer_symbols: set[str]
) -> str | None:
    suffixes = {
        "W": ("WARRANT", ("U", "R")),
        "U": ("UNIT", ("W", "R")),
        "R": ("RIGHT", ("U", "W")),
        "-WT": ("WARRANT", ("-UN", "-RI")),
        "-UN": ("UNIT", ("-WT", "-RI")),
        "-RI": ("RIGHT", ("-UN", "-WT")),
    }
    for suffix, (security_type, paired_suffixes) in suffixes.items():
        if not symbol.endswith(suffix) or len(symbol) <= len(suffix):
            continue
        root = symbol[: -len(suffix)]
        if any(f"{root}{paired}" in issuer_symbols for paired in paired_suffixes):
            return security_type
    return None


def _parse_nasdaq_alert_date(value: str) -> datetime.date:
    for date_format in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            continue
    raise ValueError(f"invalid Nasdaq corporate-action date: {value!r}")


def _nasdaq_alert_security_type(issue: str) -> str:
    issue_lower = issue.lower()
    if re.search(r"\bright(?:s)?\b", issue_lower):
        return "RIGHT"
    if re.search(r"\bunit(?:s)?\b", issue_lower):
        return "UNIT"
    if re.search(r"\betn(?:s)?\b", issue_lower):
        return "ETN"
    if re.search(r"american depositary|\badr(?:s)?\b|\bads\b", issue_lower):
        return "ADR"
    if "preferred" in issue_lower or "preference" in issue_lower:
        return "PREFERRED"
    if re.search(r"\b(?:common stock|common shares?|ordinary shares?)\b", issue_lower):
        return "COMMON_EQUITY"
    return "UNKNOWN"


def _nasdaq_lifecycle_claim_digest(events: Sequence[Mapping[str, Any]]) -> str:
    claims = [
        {
            "symbol": str(event["symbol"]),
            "event": str(event["event"]),
            "effective_date": str(event["effective_date"]),
            "last_trading_date": str(event["last_trading_date"]),
            "security_type": str(event["security_type"]),
        }
        for event in sorted(events, key=lambda item: str(item["symbol"]))
    ]
    return _evidence_sha256(claims)


def parse_nasdaq_corporate_action_alert(
    text: str,
    *,
    source_id: str = NASDAQ_CORPORATE_ACTION_ALERT_SOURCE_ID,
    source_url: str = NASDAQ_CORPORATE_ACTION_ALERT_URL,
    payload_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Parse a hash-verified Nasdaq Trader corporate-action alert.

    Only claims with both a last-trading date and an effective suspension date
    become lifecycle events.  A replacement listing without those claims (for
    example WATR in ECA2026-576) is deliberately not inherited as a lifecycle
    event for the replaced symbol.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nasdaq corporate-action alert is empty")
    raw_payload_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    effective_payload_sha256 = payload_sha256 or raw_payload_sha256
    if not _SHA256.fullmatch(effective_payload_sha256):
        raise ValueError("Nasdaq corporate-action payload hash is invalid")
    if expected_payload_sha256 and effective_payload_sha256 != expected_payload_sha256:
        raise ValueError("Nasdaq corporate-action payload hash mismatch")

    source_text = text
    viewstate_match = re.search(
        r'name=["\']__VIEWSTATE["\'][^>]*value=["\']([^"\']+)',
        text,
        flags=re.IGNORECASE,
    )
    if viewstate_match:
        try:
            decoded_viewstate = base64.b64decode(viewstate_match.group(1), validate=True)
            source_text = f"{source_text}\n{decoded_viewstate.decode('utf-8', errors='ignore')}"
        except (ValueError, base64.binascii.Error):
            raise ValueError("Nasdaq corporate-action ASP.NET state is invalid")
    clean_text = html.unescape(re.sub(r"<[^>]+>", " ", source_text))
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    entry_pattern = re.compile(
        r"Company Name/Issue:\s*(?P<issue>.+?)\s+CUSIP(?:\s+Number)?\s*#?\s*:?\s*\S+\s+"
        r"Symbol:\s*(?P<symbol>[A-Z][A-Z0-9-]*)\s*(?P<details>.*?)(?=Company Name/Issue:|$)",
        flags=re.IGNORECASE,
    )
    events: list[dict[str, Any]] = []
    source_identity = f"{source_id}:{effective_payload_sha256}"
    for match in entry_pattern.finditer(clean_text):
        issue = re.sub(r"\s+", " ", match.group("issue")).strip()
        symbol = validate_us_ticker(match.group("symbol"))
        details = match.group("details")
        last_match = re.search(
            r"Last Trading Date:\s*(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})",
            details,
            flags=re.IGNORECASE,
        )
        effective_match = re.search(
            r"Marketplace Effective Date(?: for Suspension)?:\s*"
            r"(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})",
            details,
            flags=re.IGNORECASE,
        )
        if last_match is None or effective_match is None:
            continue
        security_type = _nasdaq_alert_security_type(issue)
        if security_type == "UNKNOWN":
            raise ValueError(f"Nasdaq corporate-action security type is unknown for {symbol}")
        last_trading_date = _parse_nasdaq_alert_date(last_match.group("date"))
        effective_date = _parse_nasdaq_alert_date(effective_match.group("date"))
        claim = {
            "symbol": symbol,
            "event": "suspended",
            "effective_date": effective_date.isoformat(),
            "last_trading_date": last_trading_date.isoformat(),
            "security_type": security_type,
            "source": source_url,
            "source_id": source_id,
            "source_identity": source_identity,
            "payload_sha256": effective_payload_sha256,
        }
        claim["evidence_sha256"] = _nasdaq_lifecycle_claim_digest([claim])
        events.append(claim)

    if not events:
        raise ValueError("Nasdaq corporate-action alert contains no lifecycle events")
    unique_events: dict[str, dict[str, Any]] = {}
    for event in events:
        previous = unique_events.get(event["symbol"])
        if previous is not None and previous != event:
            raise ValueError(f"conflicting Nasdaq corporate-action events for {event['symbol']}")
        unique_events[event["symbol"]] = event
    source = {
        "source": source_url,
        "source_id": source_id,
        "source_identity": source_identity,
        "payload_sha256": effective_payload_sha256,
        "evidence_sha256": _nasdaq_lifecycle_claim_digest(list(unique_events.values())),
        "status": "healthy",
    }
    return {"events": list(unique_events.values()), "sources": [source]}


def fetch_us_lifecycle_events(
    target_market_date: datetime.date | None,
    *,
    timeout: int = 15,
    fetch_text=None,
) -> dict[str, Any]:
    """Fetch the pinned Nasdaq corporate-action source used by US closure.

    The source is intentionally hash-pinned.  A changed alert fails closed so
    a changed public page cannot silently mutate the denominator contract.
    """
    if target_market_date is None or target_market_date < NASDAQ_CORPORATE_ACTION_ALERT_EFFECTIVE_DATE:
        return {"events": [], "sources": []}

    if fetch_text is not None:
        content = fetch_text(NASDAQ_CORPORATE_ACTION_ALERT_URL)
        if isinstance(content, str):
            raw_content = content.encode("utf-8")
        elif isinstance(content, bytes):
            raw_content = content
        else:
            raise ValueError("Nasdaq corporate-action fetcher returned invalid content")
    else:
        request = urllib.request.Request(
            NASDAQ_CORPORATE_ACTION_ALERT_URL,
            headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_content = response.read(NASDAQ_CORPORATE_ACTION_ALERT_MAX_BYTES + 1)
    if len(raw_content) > NASDAQ_CORPORATE_ACTION_ALERT_MAX_BYTES:
        raise RuntimeError("Nasdaq corporate-action alert response is too large")
    payload_sha256 = hashlib.sha256(raw_content).hexdigest()
    document = parse_nasdaq_corporate_action_alert(
        raw_content.decode("utf-8-sig"),
        source_id=NASDAQ_CORPORATE_ACTION_ALERT_SOURCE_ID,
        source_url=NASDAQ_CORPORATE_ACTION_ALERT_URL,
        payload_sha256=payload_sha256,
    )
    if document["sources"][0]["evidence_sha256"] != NASDAQ_CORPORATE_ACTION_ALERT_CLAIMS_SHA256:
        raise RuntimeError("Nasdaq corporate-action claims hash mismatch")
    document["events"] = [
        event
        for event in document["events"]
        if datetime.date.fromisoformat(event["effective_date"]) <= target_market_date
    ]
    return document


def parse_sec_us_universe_with_metadata(
    document: dict,
    *,
    scope: str = "EQUITY_OBSERVATION",
    security_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    security_metadata_sources: Sequence[Mapping[str, Any]] | None = None,
    target_market_date: datetime.date | None = None,
    lifecycle_events: Sequence[Mapping[str, Any]] | None = None,
) -> USUniverseBreakdown:
    if not isinstance(document, dict):
        raise ValueError("invalid SEC universe document")
    fields = document.get("fields")
    rows = document.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("invalid SEC universe schema")
    required = {"name", "ticker", "exchange"}
    if not required.issubset(fields):
        raise ValueError("SEC universe fields are incomplete")
    positions = {name: fields.index(name) for name in required}
    cik_position = fields.index("cik") if "cik" in fields else None

    symbols = set()
    exchange_counts: dict[str, int] = {}
    derivative_counts: dict[str, int] = {
        "ETN": 0, "WARRANT": 0, "UNIT": 0, "PREFERRED": 0, "RIGHT": 0
    }
    exclusions: dict[str, dict[str, Any]] = {}
    eligibility: dict[str, dict[str, Any]] = {}
    security_type_counts: dict[str, int] = {}

    excluded_exchange = 0
    excluded_crypto = 0
    excluded_invalid = 0
    excluded_derivative = 0
    eligible_listed_count = 0
    document_evidence_sha256 = _evidence_sha256({"fields": fields, "data": rows})
    issuer_symbols: dict[str, set[str]] = {}
    if cik_position is not None:
        for row in rows:
            if not isinstance(row, list) or len(row) < len(fields):
                continue
            issuer = str(row[cik_position] or "").strip()
            ticker = str(row[positions["ticker"]] or "").strip().upper().replace(".", "-")
            if not issuer:
                continue
            try:
                issuer_symbols.setdefault(issuer, set()).add(validate_us_ticker(ticker))
            except ValueError:
                continue
    security_records = {
        str(symbol).upper().replace(".", "-"): dict(record)
        for symbol, record in (security_metadata or {}).items()
        if isinstance(record, Mapping)
    }
    security_metadata_status = "partial" if security_records else "unavailable"

    lifecycle_status = "lifecycle_evidence_unavailable"
    lifecycle_count: int | None = None
    lifecycle_by_symbol: dict[str, dict[str, Any]] = {}
    lifecycle_sources: dict[str, dict[str, Any]] = {}
    if lifecycle_events is not None and target_market_date is not None:
        lifecycle_status = "available"
        lifecycle_by_symbol = {}
        for raw_event in lifecycle_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("US lifecycle event is invalid")
            try:
                event_symbol = validate_us_ticker(raw_event["symbol"])
                event_type = str(raw_event["event"]).strip().lower()
                effective_date = datetime.date.fromisoformat(str(raw_event["effective_date"]))
                source = str(raw_event["source"]).strip()
                source_id = str(raw_event.get("source_id") or "").strip()
                source_identity = str(raw_event["source_identity"]).strip()
                evidence_hash = str(raw_event["evidence_sha256"]).strip()
                payload_hash = str(raw_event.get("payload_sha256") or "").strip()
                security_type = str(raw_event.get("security_type") or "UNKNOWN").strip().upper()
                last_trading_date_value = raw_event.get("last_trading_date")
                last_trading_date = (
                    datetime.date.fromisoformat(str(last_trading_date_value))
                    if last_trading_date_value is not None
                    else None
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("US lifecycle event is invalid") from exc
            if (
                event_type not in {"delisted", "terminated", "suspended"}
                or not source
                or not source_identity
                or not _SHA256.fullmatch(evidence_hash)
                or (payload_hash and not _SHA256.fullmatch(payload_hash))
                or security_type not in SUPPORTED_SECURITY_TYPES
                or (last_trading_date is not None and last_trading_date > effective_date)
            ):
                raise ValueError("US lifecycle event evidence is invalid")
            if effective_date > target_market_date:
                continue
            event = {
                "symbol": event_symbol,
                "event": event_type,
                "effective_date": effective_date.isoformat(),
                "source": source,
                "source_identity": source_identity,
                "evidence_sha256": evidence_hash,
            }
            if source_id:
                event["source_id"] = source_id
            if payload_hash:
                event["payload_sha256"] = payload_hash
            if security_type != "UNKNOWN":
                event["security_type"] = security_type
            if last_trading_date is not None:
                event["last_trading_date"] = last_trading_date.isoformat()
            previous = lifecycle_by_symbol.get(event_symbol)
            if previous is not None and previous != event:
                raise ValueError(f"conflicting US lifecycle events for {event_symbol}")
            lifecycle_by_symbol[event_symbol] = event
            lifecycle_sources[source_identity] = {
                "source": source,
                "source_identity": source_identity,
                "evidence_sha256": evidence_hash,
            }
        lifecycle_count = len(lifecycle_by_symbol)

    def _sec_exclusion(
        *,
        symbol: str,
        reason: str,
        classification: str | None = None,
        source: str | None = None,
        source_identity: str | None = None,
        as_of: str | None = None,
        evidence_sha256: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        source_value = source or str(document.get("source_id") or SEC_US_UNIVERSE_URL)
        identity_value = source_identity or str(
            document.get("source_identity")
            or f"{source_value}:{document.get('_payload_sha256') or document_evidence_sha256}"
        )
        result: dict[str, Any] = {
            "symbol": symbol,
            "reason": reason,
            "classification": classification or "UNKNOWN",
            "source": source_value,
            "source_identity": identity_value,
            "effective_date": as_of,
            "as_of": as_of,
            "evidence_sha256": evidence_sha256
            or str(document.get("_payload_sha256") or document_evidence_sha256),
        }
        result.update(extra)
        return result

    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        exchange_val = str(row[positions["exchange"]] or "").strip().upper()
        name_val = str(row[positions["name"]] or "").strip()
        raw_ticker = str(row[positions["ticker"]] or "").strip().upper()

        if exchange_val not in US_ACCEPTED_EXCHANGES:
            excluded_exchange += 1
            if raw_ticker:
                exclusions[raw_ticker] = _sec_exclusion(
                    symbol=raw_ticker,
                    reason="excluded_non_major_exchange",
                    as_of=document.get("as_of"),
                    exchange=exchange_val,
                    name=name_val,
                )
            continue

        if any(term in name_val.lower() for term in CRYPTO_SECURITY_TERMS):
            excluded_crypto += 1
            if raw_ticker:
                exclusions[raw_ticker] = _sec_exclusion(
                    symbol=raw_ticker,
                    reason="excluded_crypto_term",
                    as_of=document.get("as_of"),
                    exchange=exchange_val,
                    name=name_val,
                )
            continue

        ticker = raw_ticker.replace(".", "-")
        try:
            valid_sym = validate_us_ticker(ticker)
        except ValueError:
            excluded_invalid += 1
            if raw_ticker:
                exclusions[raw_ticker] = _sec_exclusion(
                    symbol=raw_ticker,
                    reason="excluded_invalid_ticker",
                    as_of=document.get("as_of"),
                    exchange=exchange_val,
                    name=name_val,
                )
            continue

        eligible_listed_count += 1
        lifecycle_event = lifecycle_by_symbol.get(valid_sym)
        metadata = security_records.get(valid_sym)
        classification = _authoritative_security_classification(
            valid_sym, name_val, raw_ticker, metadata
        )
        issuer = str(row[cik_position] or "").strip() if cik_position is not None else ""
        paired_type = _sec_same_issuer_derivative_type(
            valid_sym, issuer_symbols.get(issuer, set())
        )
        if paired_type and not classification.get("authoritative"):
            source = str(document.get("source_id") or SEC_US_UNIVERSE_URL)
            source_identity = str(
                document.get("source_identity")
                or f"{source}:{document.get('_payload_sha256') or document_evidence_sha256}"
            )
            classification = {
                **classification,
                "security_type": paired_type,
                "classification_method": "sec_same_issuer_derivative_pair",
                "authoritative": True,
                "evidence": {
                    "source_id": source,
                    "source_identity": source_identity,
                    "as_of": document.get("as_of"),
                    "evidence_sha256": str(
                        document.get("_payload_sha256") or document_evidence_sha256
                    ),
                },
            }
        event_security_type = str((lifecycle_event or {}).get("security_type") or "UNKNOWN").upper()
        if event_security_type != "UNKNOWN":
            if classification.get("authoritative") and classification["security_type"] != event_security_type:
                raise ValueError(f"conflicting US security classifications for {valid_sym}")
            if not classification.get("authoritative") or classification["security_type"] == "UNKNOWN":
                classification = {
                    **classification,
                    "security_type": event_security_type,
                    "classification_method": "authoritative_lifecycle_event",
                    "authoritative": True,
                    "evidence": lifecycle_event,
                }
        sec_type = classification["security_type"]
        security_type_counts[sec_type] = security_type_counts.get(sec_type, 0) + 1
        metadata_evidence = classification.get("evidence") or {}
        eligibility[valid_sym] = {
            "symbol": valid_sym,
            "security_type": sec_type,
            "classification_method": classification.get("classification_method"),
            "authoritative": classification.get("authoritative", False),
            "heuristic_signal": classification.get("heuristic_signal"),
            "eligible": True,
            "source": metadata_evidence.get("source_id") or SEC_US_UNIVERSE_URL,
            "source_identity": metadata_evidence.get("source_identity") or SEC_US_UNIVERSE_URL,
            "as_of": metadata_evidence.get("as_of")
            or (lifecycle_event or {}).get("effective_date")
            or document.get("as_of"),
            "effective_date": (lifecycle_event or {}).get("effective_date")
            or metadata_evidence.get("as_of")
            or document.get("as_of"),
            "evidence_sha256": metadata_evidence.get("evidence_sha256") or document_evidence_sha256,
        }

        if scope == "EQUITY_OBSERVATION" and sec_type in DERIVATIVE_SECURITY_TYPES and classification.get("authoritative"):
            excluded_derivative += 1
            derivative_counts[sec_type] += 1
            eligibility[valid_sym]["eligible"] = False
            eligibility[valid_sym]["reason"] = f"excluded_authoritative_security_type_{sec_type.lower()}"
            exclusions[valid_sym] = _sec_exclusion(
                symbol=valid_sym,
                reason=f"excluded_authoritative_security_type_{sec_type.lower()}",
                classification=sec_type,
                source=metadata_evidence.get("source_id") or SEC_US_UNIVERSE_URL,
                source_identity=metadata_evidence.get("source_identity") or SEC_US_UNIVERSE_URL,
                as_of=metadata_evidence.get("as_of") or document.get("as_of"),
                evidence_sha256=metadata_evidence.get("evidence_sha256") or document_evidence_sha256,
                exchange=exchange_val,
                name=name_val,
                classification_method=classification.get("classification_method"),
            )
            continue

        if lifecycle_event is not None:
            eligibility[valid_sym]["eligible"] = False
            eligibility[valid_sym]["reason"] = "excluded_effective_lifecycle_event"
            exclusions[valid_sym] = _sec_exclusion(
                symbol=valid_sym,
                reason="excluded_effective_lifecycle_event",
                classification=sec_type,
                source=lifecycle_event["source"],
                source_id=lifecycle_event.get("source_id"),
                source_identity=lifecycle_event["source_identity"],
                as_of=lifecycle_event["effective_date"],
                evidence_sha256=lifecycle_event["evidence_sha256"],
                event=lifecycle_event["event"],
                exchange=exchange_val,
                name=name_val,
            )
            continue

        symbols.add(valid_sym)
        exchange_counts[exchange_val] = exchange_counts.get(exchange_val, 0) + 1

    if security_records:
        security_metadata_status = (
            "healthy"
            if all(symbol in security_records for symbol in eligibility)
            else "partial"
        )

    if not symbols:
        raise ValueError("SEC universe contains no supported US symbols")

    sorted_symbols = sorted(symbols)
    return USUniverseBreakdown(
        configured_listed_count=len(rows),
        eligible_listed_count=eligible_listed_count,
        active_universe_count=len(sorted_symbols),
        excluded_exchange_count=excluded_exchange,
        excluded_crypto_count=excluded_crypto,
        excluded_invalid_count=excluded_invalid,
        excluded_derivative_count=excluded_derivative,
        derivative_breakdown=derivative_counts,
        terminated_delisted_count=lifecycle_count,
        exchange_counts=exchange_counts,
        symbols=sorted_symbols,
        exclusions_by_symbol=exclusions,
        security_metadata_status=security_metadata_status,
        security_metadata_sources=[dict(item) for item in (security_metadata_sources or [])],
        security_type_counts=security_type_counts,
        security_eligibility_by_symbol=eligibility,
        lifecycle_evidence_status=lifecycle_status,
        lifecycle_evidence_sources=list(lifecycle_sources.values()),
        lifecycle_events_by_symbol=lifecycle_by_symbol,
    )


def parse_sec_us_universe(document: dict) -> list[str]:
    return parse_sec_us_universe_with_metadata(document).symbols


def fetch_sec_us_universe_json() -> dict:
    req = urllib.request.Request(
        SEC_US_UNIVERSE_URL,
        headers={"User-Agent": "ABSORB-Research/1.0 (contact@absorb.local)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        if len(content) > SEC_US_UNIVERSE_MAX_BYTES:
            raise RuntimeError("SEC universe response is too large")
        return json.loads(content.decode("utf-8"))


def read_us_universe_cache(cache_path: Path) -> dict | None:
    try:
        cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        symbols = [validate_us_ticker(item) for item in cached.get("symbols", [])]
        as_of = cached.get("as_of")
        if not symbols or not isinstance(as_of, str):
            return None
        return {
            "as_of": as_of,
            "source": cached.get("source", "cache"),
            "symbols": sorted(set(symbols)),
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None


def get_us_universe_breakdown(
    root: str | Path,
    *,
    fetch_json=None,
    fetch_exchange_metadata=None,
    fetch_lifecycle_events=None,
    now: datetime.datetime | None = None,
    scope: str = "EQUITY_OBSERVATION",
    target_market_date: datetime.date | None = None,
) -> USUniverseBreakdown:
    checked_at = now or datetime.datetime.now(TAIPEI)
    cache_dir = Path(root) / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"us-universe-{scope.lower()}.json"
    cached = None
    target_iso = target_market_date.isoformat() if target_market_date else None
    if cache_path.exists():
        try:
            cached_doc = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached_doc.get("contract_version") == US_UNIVERSE_CACHE_CONTRACT_VERSION
                and cached_doc.get("as_of") == checked_at.date().isoformat()
                and isinstance(cached_doc.get("symbols"), list)
                and cached_doc.get("active_universe_count")
                and cached_doc.get("security_metadata_status")
                and cached_doc.get("lifecycle_evidence_status")
                and (
                    target_market_date is None
                    or target_market_date < NASDAQ_CORPORATE_ACTION_ALERT_EFFECTIVE_DATE
                    or cached_doc.get("lifecycle_evidence_status") == "available"
                )
                and isinstance(cached_doc.get("security_eligibility_by_symbol"), dict)
                and cached_doc.get("target_market_date") == target_iso
            ):
                cached = USUniverseBreakdown(
                    configured_listed_count=cached_doc.get("configured_listed_count", len(cached_doc["symbols"])),
                    eligible_listed_count=cached_doc.get("eligible_listed_count", len(cached_doc["symbols"])),
                    active_universe_count=cached_doc.get("active_universe_count", len(cached_doc["symbols"])),
                    excluded_exchange_count=cached_doc.get("excluded_exchange_count", 0),
                    excluded_crypto_count=cached_doc.get("excluded_crypto_count", 0),
                    excluded_invalid_count=cached_doc.get("excluded_invalid_count", 0),
                    excluded_derivative_count=cached_doc.get("excluded_derivative_count", 0),
                    derivative_breakdown=cached_doc.get("derivative_breakdown", {}),
                    terminated_delisted_count=cached_doc.get("terminated_delisted_count"),
                    exchange_counts=cached_doc.get("exchange_counts", {}),
                    symbols=cached_doc["symbols"],
                    exclusions_by_symbol=cached_doc.get("exclusions_by_symbol", {}),
                    security_metadata_status=cached_doc["security_metadata_status"],
                    security_metadata_sources=cached_doc.get("security_metadata_sources", []),
                    security_type_counts=cached_doc.get("security_type_counts", {}),
                    security_eligibility_by_symbol=cached_doc.get("security_eligibility_by_symbol", {}),
                    lifecycle_evidence_status=cached_doc["lifecycle_evidence_status"],
                    lifecycle_evidence_sources=cached_doc.get("lifecycle_evidence_sources", []),
                    lifecycle_events_by_symbol=cached_doc.get("lifecycle_events_by_symbol", {}),
                )
        except Exception:
            cached = None

    if cached is not None:
        return cached

    try:
        doc = (fetch_json or fetch_sec_us_universe_json)()
        if not isinstance(doc, dict):
            raise ValueError("SEC universe document is invalid")
        doc = dict(doc)
        doc.setdefault("as_of", checked_at.date().isoformat())

        metadata_document = None
        metadata_sources: list[dict[str, Any]] = []
        if fetch_exchange_metadata is not None:
            metadata_document = fetch_exchange_metadata()
        elif fetch_json is not None:
            metadata_document = doc.get("security_metadata")
        else:
            try:
                metadata_document = fetch_official_us_security_metadata(
                    target_market_date=target_market_date
                )
            except Exception as exc:
                metadata_document = {
                    "records": {},
                    "sources": [
                        {
                            "source_id": "us:first_party_security_metadata",
                            "source_url": "https://www.nasdaqtrader.com/;https://ftp.nyse.com/NYSESymbolMapping/",
                            "status": "unavailable",
                            "error_type": type(exc).__name__,
                        }
                    ],
                    "status": "unavailable",
                }
        if isinstance(metadata_document, Mapping) and "records" in metadata_document:
            security_metadata = metadata_document.get("records") or {}
            metadata_sources = [dict(item) for item in (metadata_document.get("sources") or [])]
        elif isinstance(metadata_document, Mapping):
            security_metadata = metadata_document
        else:
            security_metadata = None

        lifecycle_events = doc.get("lifecycle_events")
        lifecycle_sources: list[dict[str, Any]] = []
        if fetch_lifecycle_events is not None:
            lifecycle_document = fetch_lifecycle_events(target_market_date)
            if isinstance(lifecycle_document, Mapping):
                lifecycle_events = lifecycle_document.get("events")
                lifecycle_sources = [dict(item) for item in (lifecycle_document.get("sources") or [])]
            else:
                lifecycle_events = lifecycle_document
        elif (
            fetch_json is None
            and target_market_date is not None
            and target_market_date >= NASDAQ_CORPORATE_ACTION_ALERT_EFFECTIVE_DATE
        ):
            lifecycle_document = fetch_us_lifecycle_events(target_market_date)
            lifecycle_events = lifecycle_document.get("events")
            lifecycle_sources = [
                dict(item) for item in (lifecycle_document.get("sources") or [])
            ]

        breakdown = parse_sec_us_universe_with_metadata(
            doc,
            scope=scope,
            security_metadata=security_metadata,
            security_metadata_sources=metadata_sources,
            target_market_date=target_market_date,
            lifecycle_events=lifecycle_events,
        )
        if lifecycle_sources:
            breakdown = USUniverseBreakdown(
                **{
                    **breakdown.__dict__,
                    "lifecycle_evidence_sources": lifecycle_sources,
                }
            )
        source = SEC_US_UNIVERSE_URL
    except Exception as exc:
        if cached:
            return cached
        raise RuntimeError("US universe is unavailable") from exc

    payload = {
        "schema_version": 2,
        "contract_version": US_UNIVERSE_CACHE_CONTRACT_VERSION,
        "market": "US",
        "scope": scope,
        "as_of": checked_at.date().isoformat(),
        "target_market_date": target_iso,
        "source": source,
        "configured_listed_count": breakdown.configured_listed_count,
        "eligible_listed_count": breakdown.eligible_listed_count,
        "active_universe_count": breakdown.active_universe_count,
        "excluded_exchange_count": breakdown.excluded_exchange_count,
        "excluded_crypto_count": breakdown.excluded_crypto_count,
        "excluded_invalid_count": breakdown.excluded_invalid_count,
        "excluded_derivative_count": breakdown.excluded_derivative_count,
        "derivative_breakdown": breakdown.derivative_breakdown,
        "terminated_delisted_count": breakdown.terminated_delisted_count,
        "lifecycle_evidence_status": breakdown.lifecycle_evidence_status,
        "lifecycle_evidence_sources": breakdown.lifecycle_evidence_sources,
        "lifecycle_events_by_symbol": breakdown.lifecycle_events_by_symbol,
        "exchange_counts": breakdown.exchange_counts,
        "symbol_count": breakdown.active_universe_count,
        "symbols": breakdown.symbols,
        "exclusions_by_symbol": breakdown.exclusions_by_symbol,
        "security_metadata_status": breakdown.security_metadata_status,
        "security_metadata_sources": breakdown.security_metadata_sources,
        "security_type_counts": breakdown.security_type_counts,
        "security_eligibility_by_symbol": breakdown.security_eligibility_by_symbol,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_bytes(raw)
    temp_path.replace(cache_path)
    return breakdown


def get_us_symbols(
    root: str | Path,
    *,
    fetch_json=None,
    now: datetime.datetime | None = None,
    scope: str = "EQUITY_OBSERVATION",
) -> list[str]:
    return get_us_universe_breakdown(root, fetch_json=fetch_json, now=now, scope=scope).symbols
