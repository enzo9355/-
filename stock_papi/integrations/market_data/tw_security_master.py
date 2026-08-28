"""Canonical Taiwan security master and point-in-time name resolution.

The third-party ``twstock`` catalogue is useful as a compatibility fallback,
but it is not a lifecycle source.  This module keeps official TWSE/TPEx
security-master data at the boundary and gives every consumer one resolver.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import html
import re
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping


TWSE_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TWSE_ETF_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap47_L"
TWSE_QUOTE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_COMPANY_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TPEX_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
DEFAULT_TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MASTER_SCHEMA_VERSION = "tw-security-master-v1"
_TAIWAN_SYMBOL_RE = re.compile(r"[0-9]{4,5}[0-9A-Z]?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TaiwanSecurityMasterError(RuntimeError):
    """Base class for authoritative Taiwan security-master failures."""


class TaiwanSecurityMasterUnavailable(TaiwanSecurityMasterError):
    """The official security master could not be loaded."""


class TaiwanSecurityMasterConflict(TaiwanSecurityMasterError):
    """Official sources contain conflicting identity data."""


class TaiwanSecurityNotFound(TaiwanSecurityMasterError):
    """A symbol is not present in the authoritative master."""


def normalize_taiwan_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not _TAIWAN_SYMBOL_RE.fullmatch(symbol):
        raise ValueError("invalid Taiwan security symbol")
    return symbol


def is_taiwan_symbol(value: Any) -> bool:
    try:
        normalize_taiwan_symbol(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_display_name(value: Any) -> str:
    """Remove official quote-status decorations without changing the name."""

    text = html.unescape(str(value or "")).replace("\xa0", " ").strip()
    # TPEx's CompanyAbbreviation appends '*' to indicate a quote state.  It
    # is not part of the security's canonical display name.
    return text.rstrip("*").strip()


def _parse_date(value: Any) -> _datetime.date | None:
    text = re.sub(r"\D", "", str(value or ""))
    if not text:
        return None
    if len(text) == 6:
        text = str(int(text[:2]) + 1911) + text[2:]
    elif len(text) == 7:
        text = str(int(text[:3]) + 1911) + text[3:]
    if len(text) != 8:
        return None
    try:
        return _datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = normalize_display_name(row.get(key))
        if value:
            return value
    return ""


@dataclass(frozen=True)
class NameChange:
    symbol: str
    old_name: str
    new_name: str
    effective_date: _datetime.date
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_taiwan_symbol(self.symbol))
        old_name = normalize_display_name(self.old_name)
        new_name = normalize_display_name(self.new_name)
        if not old_name or not new_name or old_name == new_name:
            raise ValueError("Taiwan name change is invalid")
        if not isinstance(self.effective_date, _datetime.date) or isinstance(
            self.effective_date, _datetime.datetime
        ):
            raise TypeError("name change effective_date must be a date")
        if not str(self.source_id).strip():
            raise ValueError("Taiwan name change source is invalid")
        object.__setattr__(self, "old_name", old_name)
        object.__setattr__(self, "new_name", new_name)
        object.__setattr__(self, "source_id", str(self.source_id).strip())


@dataclass(frozen=True)
class TaiwanSecurity:
    symbol: str
    name: str
    exchange: str
    instrument_type: str
    listed_date: _datetime.date | None = None
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_taiwan_symbol(self.symbol))
        name = normalize_display_name(self.name)
        if not name:
            raise ValueError("Taiwan security name is empty")
        if self.exchange not in {"TWSE", "TPEx"}:
            raise ValueError("Taiwan security exchange is invalid")
        if self.instrument_type not in {"STOCK", "ETF", "OTHER"}:
            raise ValueError("Taiwan instrument type is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_id", str(self.source_id).strip())


@dataclass(frozen=True)
class TaiwanSecurityMaster:
    as_of: _datetime.date
    entries: Mapping[str, TaiwanSecurity]
    source_hashes: Mapping[str, str]
    active_symbols: frozenset[str] = frozenset()
    name_changes: tuple[NameChange, ...] = ()
    schema_version: str = MASTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, _datetime.date) or isinstance(
            self.as_of, _datetime.datetime
        ):
            raise TypeError("Taiwan security master as_of must be a date")
        entries = {
            normalize_taiwan_symbol(symbol): entry
            for symbol, entry in dict(self.entries).items()
        }
        if any(entry.symbol != symbol for symbol, entry in entries.items()):
            raise TaiwanSecurityMasterConflict("security master symbol identity conflicts")
        active_symbols = frozenset(
            normalize_taiwan_symbol(symbol) for symbol in self.active_symbols
        )
        if not active_symbols.issubset(entries):
            raise TaiwanSecurityMasterConflict("active security is missing from master")
        hashes = dict(self.source_hashes)
        if any(not _SHA256_RE.fullmatch(str(value)) for value in hashes.values()):
            raise TaiwanSecurityMasterConflict("security master source hash is invalid")
        changes = tuple(sorted(self.name_changes, key=lambda item: (item.symbol, item.effective_date, item.source_id)))
        if any(change.symbol not in entries for change in changes):
            raise TaiwanSecurityMasterConflict("name change references an unknown symbol")
        object.__setattr__(self, "entries", MappingProxyType(entries))
        object.__setattr__(self, "active_symbols", active_symbols)
        object.__setattr__(self, "source_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "name_changes", changes)

    def resolve_name(
        self,
        symbol: str,
        target_date: _datetime.date | None = None,
    ) -> str:
        symbol = normalize_taiwan_symbol(symbol)
        target = self.as_of if target_date is None else target_date
        if not isinstance(target, _datetime.date) or isinstance(target, _datetime.datetime):
            raise TypeError("target_date must be a date")
        entry = self.entries.get(symbol)
        if entry is None:
            raise TaiwanSecurityNotFound(symbol)
        if entry.listed_date is not None and target < entry.listed_date:
            raise TaiwanSecurityNotFound(f"{symbol} was not listed on {target.isoformat()}")

        name = entry.name
        changes = [change for change in self.name_changes if change.symbol == symbol]
        if target < self.as_of:
            for change in reversed(changes):
                if target < change.effective_date <= self.as_of:
                    if name == change.new_name:
                        name = change.old_name
                    elif name != change.old_name:
                        raise TaiwanSecurityMasterConflict(
                            f"name history does not reconcile for {symbol}"
                        )
        else:
            for change in changes:
                if self.as_of < change.effective_date <= target:
                    if name == change.old_name:
                        name = change.new_name
                    elif name != change.new_name:
                        raise TaiwanSecurityMasterConflict(
                            f"future name history does not reconcile for {symbol}"
                        )
        return name

    def resolve(self, symbol: str) -> TaiwanSecurity:
        symbol = normalize_taiwan_symbol(symbol)
        try:
            return self.entries[symbol]
        except KeyError:
            raise TaiwanSecurityNotFound(symbol) from None

    def search(self, keyword: str) -> tuple[str, str] | None:
        needle = str(keyword or "").strip().upper()
        if not needle:
            return None
        exact = self.entries.get(needle)
        if exact is not None:
            return exact.symbol, exact.name
        matches = [
            (entry.symbol, entry.name)
            for entry in self.entries.values()
            if needle in entry.name.upper()
        ]
        return max(matches, key=lambda item: len(item[1])) if matches else None


def _add_entry(
    entries: dict[str, TaiwanSecurity],
    row: Mapping[str, Any],
    *,
    exchange: str,
    instrument_type: str,
    source_id: str,
    symbol_keys: tuple[str, ...],
    name_keys: tuple[str, ...],
    listed_keys: tuple[str, ...],
) -> None:
    raw_symbol = _first_text(row, *symbol_keys)
    if not raw_symbol:
        return
    try:
        symbol = normalize_taiwan_symbol(raw_symbol)
    except ValueError:
        return
    name = _first_text(row, *name_keys)
    if not name:
        return
    listed_date = _parse_date(next((row.get(key) for key in listed_keys if row.get(key)), None))
    entry = TaiwanSecurity(
        symbol=symbol,
        name=name,
        exchange=exchange,
        instrument_type=instrument_type,
        listed_date=listed_date,
        source_id=source_id,
    )
    previous = entries.get(symbol)
    if previous is None:
        entries[symbol] = entry
        return
    if previous == entry:
        return
    # Company/fund masters outrank the quote feed, which may carry a market
    # status suffix or a shorter transient name.
    if previous.exchange != entry.exchange or previous.name != entry.name:
        preferred = {"twse_company", "twse_etf", "tpex_company"}
        if previous.source_id in preferred and entry.source_id not in preferred:
            return
        if entry.source_id in preferred and previous.source_id not in preferred:
            entries[symbol] = entry
            return
        raise TaiwanSecurityMasterConflict(f"conflicting official identity for {symbol}")


def _row_symbol(row: Mapping[str, Any], *keys: str) -> str | None:
    raw_symbol = _first_text(row, *keys)
    if not raw_symbol:
        return None
    try:
        return normalize_taiwan_symbol(raw_symbol)
    except ValueError:
        return None


def build_taiwan_security_master(
    *,
    as_of: _datetime.date,
    twse_company_rows: Iterable[Mapping[str, Any]],
    twse_etf_rows: Iterable[Mapping[str, Any]],
    tpex_company_rows: Iterable[Mapping[str, Any]],
    tpex_quote_rows: Iterable[Mapping[str, Any]],
    twse_quote_rows: Iterable[Mapping[str, Any]] = (),
    source_hashes: Mapping[str, str] | None = None,
    name_changes: Iterable[NameChange] = (),
) -> TaiwanSecurityMaster:
    twse_quote_rows = tuple(twse_quote_rows or ())
    tpex_quote_rows = tuple(tpex_quote_rows or ())
    entries: dict[str, TaiwanSecurity] = {}
    for row in twse_company_rows or ():
        _add_entry(
            entries,
            row,
            exchange="TWSE",
            instrument_type="STOCK",
            source_id="twse_company",
            symbol_keys=("公司代號", "Code"),
            name_keys=("公司簡稱", "Name", "Company"),
            listed_keys=("上市日期", "ListingDate"),
        )
    for row in twse_etf_rows or ():
        _add_entry(
            entries,
            row,
            exchange="TWSE",
            instrument_type="ETF",
            source_id="twse_etf",
            symbol_keys=("基金代號", "Code"),
            name_keys=("基金簡稱", "Name"),
            listed_keys=("上市日期", "ListingDate"),
        )
    for row in tpex_company_rows or ():
        _add_entry(
            entries,
            row,
            exchange="TPEx",
            instrument_type="STOCK",
            source_id="tpex_company",
            symbol_keys=("SecuritiesCompanyCode", "Code"),
            name_keys=("CompanyAbbreviation", "Name", "CompanyName"),
            listed_keys=("DateOfListing", "ListingDate"),
        )
    for row in twse_quote_rows or ():
        _add_entry(
            entries,
            row,
            exchange="TWSE",
            instrument_type="ETF" if str(row.get("Code") or "").startswith("00") else "OTHER",
            source_id="twse_quote",
            symbol_keys=("Code", "基金代號"),
            name_keys=("Name", "基金簡稱"),
            listed_keys=("ListingDate",),
        )
    for row in tpex_quote_rows or ():
        _add_entry(
            entries,
            row,
            exchange="TPEx",
            instrument_type="ETF" if str(row.get("SecuritiesCompanyCode") or "").startswith("00") else "OTHER",
            source_id="tpex_quote",
            symbol_keys=("SecuritiesCompanyCode", "Code"),
            name_keys=("CompanyName", "CompanyAbbreviation", "Name"),
            listed_keys=("DateOfListing", "ListingDate"),
        )
    if not entries:
        raise TaiwanSecurityMasterUnavailable("official security master is empty")
    active_symbols = frozenset(
        {
            symbol
            for row in twse_quote_rows
            if (symbol := _row_symbol(row, "Code", "基金代號")) is not None
        }
        | {
            symbol
            for row in tpex_quote_rows
            if (symbol := _row_symbol(row, "SecuritiesCompanyCode", "Code"))
            is not None
        }
    )
    return TaiwanSecurityMaster(
        as_of=as_of,
        entries=entries,
        source_hashes=source_hashes or {},
        active_symbols=active_symbols,
        name_changes=tuple(name_changes),
    )


def _request_json(session: Any, url: str, timeout: int) -> tuple[list[dict[str, Any]], str]:
    try:
        response = session.get(url, timeout=timeout, headers={"User-Agent": "ABSORB/1.0"})
    except Exception as exc:
        raise TaiwanSecurityMasterUnavailable(f"official security master transport: {type(exc).__name__}") from None
    status = int(getattr(response, "status_code", 0))
    if status != 200:
        raise TaiwanSecurityMasterUnavailable(f"official security master HTTP {status}")
    content = bytes(getattr(response, "content", b""))
    if not content or len(content) > MAX_RESPONSE_BYTES:
        raise TaiwanSecurityMasterUnavailable("official security master response is invalid")
    try:
        payload = response.json() if callable(getattr(response, "json", None)) else None
        if payload is None:
            import json
            payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeError, ValueError, TypeError):
        raise TaiwanSecurityMasterUnavailable("official security master JSON is invalid") from None
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TaiwanSecurityMasterUnavailable("official security master schema is invalid")
    return payload, hashlib.sha256(content).hexdigest()


def fetch_taiwan_security_master(
    *,
    session: Any = None,
    as_of: _datetime.date | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    name_changes: Iterable[NameChange] = (),
) -> TaiwanSecurityMaster:
    if session is None:
        import requests
        session = requests.Session()
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("security master timeout is invalid")
    rows = {}
    for source_id, url in (
        ("twse_company", TWSE_COMPANY_URL),
        ("twse_etf", TWSE_ETF_URL),
        ("twse_quote", TWSE_QUOTE_URL),
        ("tpex_company", TPEX_COMPANY_URL),
        ("tpex_quote", TPEX_QUOTE_URL),
    ):
        rows[source_id], digest = _request_json(session, url, timeout)
        rows[f"{source_id}_sha256"] = digest
    source_hashes = {
        source_id: rows[f"{source_id}_sha256"]
        for source_id in ("twse_company", "twse_etf", "twse_quote", "tpex_company", "tpex_quote")
    }
    snapshot_date = as_of or _datetime.datetime.now(_datetime.timezone.utc).date()
    return build_taiwan_security_master(
        as_of=snapshot_date,
        twse_company_rows=rows["twse_company"],
        twse_etf_rows=rows["twse_etf"],
        twse_quote_rows=rows["twse_quote"],
        tpex_company_rows=rows["tpex_company"],
        tpex_quote_rows=rows["tpex_quote"],
        source_hashes=source_hashes,
        name_changes=name_changes,
    )


def _legacy_name(info: Any) -> str:
    return normalize_display_name(getattr(info, "name", ""))


def _legacy_exchange(info: Any) -> str | None:
    value = str(getattr(info, "data_source", "") or "").lower()
    return {"twse": "TWSE", "tpex": "TPEx"}.get(value)


class TaiwanSecurityMasterResolver:
    """Thread-safe lazy official resolver with an explicit degraded state."""

    def __init__(
        self,
        loader: Callable[[], TaiwanSecurityMaster],
        *,
        fallback_registry: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        ttl_seconds: float = 6 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.loader = loader
        self.fallback_registry = fallback_registry
        self.ttl_seconds = float(ttl_seconds)
        self.clock = clock
        self._master: TaiwanSecurityMaster | None = None
        self._loaded_at = 0.0
        self._last_error: str | None = None
        self._lock = threading.RLock()

    def _registry(self) -> Mapping[str, Any]:
        registry = self.fallback_registry() if callable(self.fallback_registry) else self.fallback_registry
        return registry or {}

    def get_master(self, *, required: bool = False) -> TaiwanSecurityMaster | None:
        with self._lock:
            now = self.clock()
            if self._master is not None and now - self._loaded_at < self.ttl_seconds:
                return self._master
            try:
                master = self.loader()
                if not isinstance(master, TaiwanSecurityMaster):
                    raise TaiwanSecurityMasterUnavailable("security master loader returned an invalid object")
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if required:
                    if isinstance(exc, TaiwanSecurityMasterError):
                        raise
                    raise TaiwanSecurityMasterUnavailable(self._last_error) from None
                return self._master
            self._master = master
            self._loaded_at = now
            self._last_error = None
            return master

    def refresh(self) -> TaiwanSecurityMaster:
        with self._lock:
            self._loaded_at = 0.0
        master = self.get_master(required=True)
        assert master is not None
        return master

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": "authoritative" if self._master is not None and self._last_error is None else "degraded_legacy_fallback",
                "schema_version": self._master.schema_version if self._master else None,
                "as_of": self._master.as_of.isoformat() if self._master else None,
                "source_hashes": dict(self._master.source_hashes) if self._master else {},
                "last_error": self._last_error,
            }

    def resolve_name(
        self,
        symbol: str,
        target_date: _datetime.date | None = None,
        *,
        require_authoritative: bool = False,
    ) -> str:
        symbol = normalize_taiwan_symbol(symbol)
        master = self.get_master(required=require_authoritative)
        if master is not None:
            entry = master.entries.get(symbol)
            if entry is not None:
                return master.resolve_name(symbol, target_date)
            if require_authoritative:
                raise TaiwanSecurityNotFound(symbol)
            # An authoritative master has loaded successfully; do not hide a
            # missing listing behind a stale third-party name.
            return symbol
        info = self._registry().get(symbol)
        name = _legacy_name(info) if info is not None else ""
        return name or symbol

    def resolve_exchange(
        self,
        symbol: str,
        *,
        require_authoritative: bool = False,
    ) -> str:
        symbol = normalize_taiwan_symbol(symbol)
        master = self.get_master(required=require_authoritative)
        if master is not None:
            entry = master.entries.get(symbol)
            if entry is None:
                if require_authoritative:
                    raise TaiwanSecurityNotFound(symbol)
                raise TaiwanSecurityMasterUnavailable(f"official exchange metadata is missing for {symbol}")
            return entry.exchange
        exchange = _legacy_exchange(self._registry().get(symbol))
        if exchange:
            return exchange
        raise TaiwanSecurityMasterUnavailable(f"exchange metadata is missing for {symbol}")

    def contains(self, symbol: str) -> bool:
        try:
            symbol = normalize_taiwan_symbol(symbol)
        except ValueError:
            return False
        master = self.get_master()
        if master is not None:
            return symbol in master.entries
        return symbol in self._registry()

    def search(self, keyword: str) -> tuple[str, str] | None:
        needle = str(keyword or "").strip().upper()
        master = self.get_master()
        if master is not None:
            return master.search(needle)
        registry = self._registry()
        if is_taiwan_symbol(needle) and needle in registry:
            return needle, _legacy_name(registry[needle]) or needle
        matches = [
            (str(symbol), _legacy_name(info))
            for symbol, info in registry.items()
            if _legacy_name(info) and needle in _legacy_name(info).upper()
        ]
        return max(matches, key=lambda item: len(item[1])) if matches else None


def _runtime_exchange(info: Any) -> str | None:
    return _legacy_exchange(info)


def audit_taiwan_universe(
    runtime_registry: Mapping[str, Any],
    master: TaiwanSecurityMaster,
    *,
    configured_symbols: Iterable[str] | None = None,
    runtime_snapshot_date: _datetime.date | None = None,
) -> dict[str, Any]:
    """Compare a production universe with the official master.

    ``configured_symbols`` is the actual production selector (for ABSORB it
    is ``industry_map['全市場']``), not the entire third-party catalogue.  A
    missing official symbol is split into a genuinely new listing and a
    structural universe exclusion so the latter cannot be mistaken for a
    provider failure.
    """

    runtime = {normalize_taiwan_symbol(symbol): info for symbol, info in (runtime_registry or {}).items() if is_taiwan_symbol(symbol)}
    configured = {
        normalize_taiwan_symbol(symbol)
        for symbol in (configured_symbols if configured_symbols is not None else runtime)
        if is_taiwan_symbol(symbol)
    }
    result: dict[str, Any] = {
        "NAME_MISMATCH": {},
        "MARKET_MISMATCH": {},
        "MISSING_NEW_LISTING": {},
        "STALE_DELISTED": {},
        "STRUCTURAL_EXCLUSION": {},
        "SECURITY_TYPE_CONFUSION": {},
        "runtime_universe_count": len(configured),
        "official_master_count": len(master.entries),
    }
    for symbol in sorted(configured & set(master.entries)):
        info = runtime.get(symbol)
        official = master.entries[symbol]
        runtime_name = _legacy_name(info) if info is not None else ""
        if runtime_name and runtime_name != official.name:
            result["NAME_MISMATCH"][symbol] = {
                "runtime_name": runtime_name,
                "official_name": official.name,
                "official_source": official.source_id,
            }
        runtime_exchange = _runtime_exchange(info)
        if runtime_exchange and runtime_exchange != official.exchange:
            result["MARKET_MISMATCH"][symbol] = {
                "runtime_exchange": runtime_exchange,
                "official_exchange": official.exchange,
            }
        runtime_type = normalize_display_name(getattr(info, "type", "")) if info is not None else ""
        if runtime_type and official.instrument_type == "ETF" and "ETF" not in runtime_type.upper():
            result["SECURITY_TYPE_CONFUSION"][symbol] = {
                "runtime_type": runtime_type,
                "official_type": official.instrument_type,
            }
    for symbol in sorted(configured - set(master.entries)):
        info = runtime.get(symbol)
        result["STALE_DELISTED"][symbol] = {
            "runtime_name": _legacy_name(info) if info is not None else "",
            "runtime_exchange": _runtime_exchange(info),
        }
    for symbol in sorted(set(master.entries) - configured):
        entry = master.entries[symbol]
        detail = {
            "official_name": entry.name,
            "official_exchange": entry.exchange,
            "listed_date": entry.listed_date.isoformat() if entry.listed_date else None,
            "official_source": entry.source_id,
        }
        if (
            entry.listed_date is not None
            and runtime_snapshot_date is not None
            and entry.listed_date > runtime_snapshot_date
        ):
            result["MISSING_NEW_LISTING"][symbol] = detail
        else:
            result["STRUCTURAL_EXCLUSION"][symbol] = detail
    return result
