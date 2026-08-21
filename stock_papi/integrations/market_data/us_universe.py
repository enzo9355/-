"""Authoritative US stock market universe from official SEC exchange listings."""

import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path
import zoneinfo

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")

SEC_US_UNIVERSE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NASDAQ_US_UNIVERSE_URLS = (
    "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json",
)
SEC_US_UNIVERSE_MAX_BYTES = 15 * 1024 * 1024
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


from dataclasses import dataclass


@dataclass(frozen=True)
class USUniverseBreakdown:
    configured_listed_count: int
    active_universe_count: int
    excluded_exchange_count: int
    excluded_crypto_count: int
    excluded_invalid_count: int
    terminated_delisted_count: int
    exchange_counts: dict[str, int]
    symbols: list[str]


def parse_sec_us_universe_with_metadata(document: dict) -> USUniverseBreakdown:
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
    excluded_exchange = 0
    excluded_crypto = 0
    excluded_invalid = 0
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        exchange_val = str(row[positions["exchange"]] or "").strip().upper()
        name_val = str(row[positions["name"]] or "").strip().lower()
        raw_ticker = str(row[positions["ticker"]] or "").strip().upper()
        if exchange_val not in US_ACCEPTED_EXCHANGES:
            excluded_exchange += 1
            continue
        if any(term in name_val for term in CRYPTO_SECURITY_TERMS):
            excluded_crypto += 1
            continue
        ticker = raw_ticker.replace(".", "-")
        try:
            valid_sym = validate_us_ticker(ticker)
            symbols.add(valid_sym)
            exchange_counts[exchange_val] = exchange_counts.get(exchange_val, 0) + 1
        except ValueError:
            excluded_invalid += 1
            continue
    if not symbols:
        raise ValueError("SEC universe contains no supported US symbols")
    sorted_symbols = sorted(symbols)
    return USUniverseBreakdown(
        configured_listed_count=len(rows),
        active_universe_count=len(sorted_symbols),
        excluded_exchange_count=excluded_exchange,
        excluded_crypto_count=excluded_crypto,
        excluded_invalid_count=excluded_invalid,
        terminated_delisted_count=0,
        exchange_counts=exchange_counts,
        symbols=sorted_symbols,
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
    now: datetime.datetime | None = None,
) -> USUniverseBreakdown:
    checked_at = now or datetime.datetime.now(TAIPEI)
    cache_dir = Path(root) / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "us-universe.json"
    cached = None
    if cache_path.exists():
        try:
            cached_doc = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached_doc.get("as_of") == checked_at.date().isoformat()
                and isinstance(cached_doc.get("symbols"), list)
                and cached_doc.get("active_universe_count")
            ):
                cached = USUniverseBreakdown(
                    configured_listed_count=cached_doc.get("configured_listed_count", len(cached_doc["symbols"])),
                    active_universe_count=cached_doc.get("active_universe_count", len(cached_doc["symbols"])),
                    excluded_exchange_count=cached_doc.get("excluded_exchange_count", 0),
                    excluded_crypto_count=cached_doc.get("excluded_crypto_count", 0),
                    excluded_invalid_count=cached_doc.get("excluded_invalid_count", 0),
                    terminated_delisted_count=cached_doc.get("terminated_delisted_count", 0),
                    exchange_counts=cached_doc.get("exchange_counts", {}),
                    symbols=cached_doc["symbols"],
                )
        except Exception:
            cached = None

    if cached is not None:
        return cached

    try:
        doc = (fetch_json or fetch_sec_us_universe_json)()
        breakdown = parse_sec_us_universe_with_metadata(doc)
        source = SEC_US_UNIVERSE_URL
    except Exception as exc:
        if cached:
            return cached
        raise RuntimeError("US universe is unavailable") from exc

    payload = {
        "schema_version": 1,
        "market": "US",
        "as_of": checked_at.date().isoformat(),
        "source": source,
        "configured_listed_count": breakdown.configured_listed_count,
        "active_universe_count": breakdown.active_universe_count,
        "excluded_exchange_count": breakdown.excluded_exchange_count,
        "excluded_crypto_count": breakdown.excluded_crypto_count,
        "excluded_invalid_count": breakdown.excluded_invalid_count,
        "terminated_delisted_count": breakdown.terminated_delisted_count,
        "exchange_counts": breakdown.exchange_counts,
        "symbol_count": breakdown.active_universe_count,
        "symbols": breakdown.symbols,
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
) -> list[str]:
    return get_us_universe_breakdown(root, fetch_json=fetch_json, now=now).symbols
