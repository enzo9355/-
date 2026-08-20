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


def parse_sec_us_universe(document: dict) -> list[str]:
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
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        exchange_val = str(row[positions["exchange"]] or "").strip().upper()
        name_val = str(row[positions["name"]] or "").strip().lower()
        raw_ticker = str(row[positions["ticker"]] or "").strip().upper()
        if exchange_val not in US_ACCEPTED_EXCHANGES:
            continue
        if any(term in name_val for term in CRYPTO_SECURITY_TERMS):
            continue
        ticker = raw_ticker.replace(".", "-")
        try:
            symbols.add(validate_us_ticker(ticker))
        except ValueError:
            continue
    if not symbols:
        raise ValueError("SEC universe contains no supported US symbols")
    return sorted(symbols)


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


def get_us_symbols(
    root: str | Path,
    *,
    fetch_json=None,
    now: datetime.datetime | None = None,
) -> list[str]:
    checked_at = now or datetime.datetime.now(TAIPEI)
    cache_dir = Path(root) / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "us-universe.json"
    cached = read_us_universe_cache(cache_path) if cache_path.exists() else None
    today_iso = checked_at.date().isoformat()
    if cached and cached["as_of"] == today_iso:
        return cached["symbols"]

    try:
        doc = (fetch_json or fetch_sec_us_universe_json)()
        symbols = parse_sec_us_universe(doc)
        source = SEC_US_UNIVERSE_URL
    except Exception as exc:
        if cached:
            return cached["symbols"]
        raise RuntimeError("US universe is unavailable") from exc

    payload = {
        "schema_version": 1,
        "market": "US",
        "as_of": today_iso,
        "source": source,
        "symbol_count": len(symbols),
        "symbols": symbols,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_bytes(raw)
    temp_path.replace(cache_path)
    return symbols
