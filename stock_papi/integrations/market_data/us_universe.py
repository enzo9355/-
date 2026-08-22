"""Authoritative US stock market universe from official SEC exchange listings."""

import datetime
import hashlib
import json
import re
import urllib.request
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
SEC_US_UNIVERSE_MAX_BYTES = 15 * 1024 * 1024
NASDAQ_SYMBOL_DIRECTORY_MAX_BYTES = 8 * 1024 * 1024
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
        "WARRANT",
        "RIGHT",
        "UNIT",
        "PREFERRED",
        "UNKNOWN",
    }
)
DERIVATIVE_SECURITY_TYPES = frozenset({"WARRANT", "RIGHT", "UNIT", "PREFERRED"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        raw_symbol = values[positions[symbol_field]].strip().upper()
        if not raw_symbol:
            continue
        symbol = raw_symbol.replace(".", "-")
        try:
            symbol = validate_us_ticker(symbol)
        except ValueError:
            continue
        record = {
            "symbol": symbol,
            "security_name": values[positions["Security Name"]].strip(),
            "exchange_code": values[positions["Exchange"]].strip() if "Exchange" in positions else None,
            "etf": values[positions["ETF"]].strip().upper() if "ETF" in positions else None,
            "test_issue": values[positions["Test Issue"]].strip().upper() if "Test Issue" in positions else None,
            "financial_status": values[positions["Financial Status"]].strip().upper() if "Financial Status" in positions else None,
        }
        enriched = _metadata_source_record(
            source_id=source_id,
            source_url=source_url,
            payload_sha256=payload_sha256,
            as_of=effective_as_of,
            record=record,
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


def _heuristic_security_type(symbol: str, name: str, raw_ticker: str) -> str:
    """Return a non-authoritative secondary signal for audit only."""
    name_lower = name.lower()
    sym_upper = symbol.upper()
    raw_upper = raw_ticker.upper()
    if "warrant" in name_lower or sym_upper.endswith(("-WT", "-WTA", "-WTB", "-WTC", "WS")) or (len(sym_upper) == 5 and sym_upper.endswith("W")):
        return "WARRANT"
    if "right" in name_lower or sym_upper.endswith("-RI") or (len(sym_upper) == 5 and sym_upper.endswith("R")):
        return "RIGHT"
    if "unit" in name_lower or sym_upper.endswith("-UN") or (len(sym_upper) == 5 and sym_upper.endswith("U")):
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

    symbols = set()
    exchange_counts: dict[str, int] = {}
    derivative_counts: dict[str, int] = {
        "WARRANT": 0, "UNIT": 0, "PREFERRED": 0, "RIGHT": 0
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
                source_identity = str(raw_event["source_identity"]).strip()
                evidence_hash = str(raw_event["evidence_sha256"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("US lifecycle event is invalid") from exc
            if event_type not in {"delisted", "terminated"} or not source or not source_identity or not _SHA256.fullmatch(evidence_hash):
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
        metadata = security_records.get(valid_sym)
        classification = _authoritative_security_classification(
            valid_sym, name_val, raw_ticker, metadata
        )
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
            "as_of": metadata_evidence.get("as_of") or document.get("as_of"),
            "effective_date": metadata_evidence.get("as_of") or document.get("as_of"),
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

        lifecycle_event = lifecycle_by_symbol.get(valid_sym)
        if lifecycle_event is not None:
            eligibility[valid_sym]["eligible"] = False
            eligibility[valid_sym]["reason"] = "excluded_effective_lifecycle_event"
            exclusions[valid_sym] = _sec_exclusion(
                symbol=valid_sym,
                reason="excluded_effective_lifecycle_event",
                classification=sec_type,
                source=lifecycle_event["source"],
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
                cached_doc.get("contract_version") == 2
                and cached_doc.get("as_of") == checked_at.date().isoformat()
                and isinstance(cached_doc.get("symbols"), list)
                and cached_doc.get("active_universe_count")
                and cached_doc.get("security_metadata_status")
                and cached_doc.get("lifecycle_evidence_status")
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
                metadata_document = fetch_nasdaq_security_directory()
            except Exception as exc:
                metadata_document = {
                    "records": {},
                    "sources": [
                        {
                            "source_id": source_id,
                            "source_url": url,
                            "status": "unavailable",
                            "error_type": type(exc).__name__,
                        }
                        for url, source_id in NASDAQ_SYMBOL_DIRECTORY_URLS
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
        "contract_version": 2,
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
