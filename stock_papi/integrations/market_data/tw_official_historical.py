"""Date-addressable TWSE/TPEx bulk reports and bounded snapshot series."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialDailySnapshot,
    OfficialRequestBudget,
    OfficialSourceDefinition,
    OfficialSourceFailure,
    OfficialSourceResult,
    normalize_market_date,
    normalize_symbol,
    parse_number,
    parse_tpex_institutional,
    parse_twse_institutional,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_source,
    store_cached_source,
)

SOURCE_MODE = "tw_official_bulk_v2"
SOURCE_SCHEMA_VERSION = "tw-official-historical-v2"
PARSER_VERSION = "tw-official-historical-parser-v2"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_ATTEMPTS = 2
MAX_CATCHUP_SESSIONS = 10
DEFAULT_MINIMUM_SOURCE_SYMBOLS = MappingProxyType({
    "twse_institutional": 500,
    "twse_margin": 400,
    "tpex_institutional": 300,
    "tpex_margin": 300,
})
TWSE_MARGIN_FIELDS = (
    "代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額",
    "次一營業日限額", "買進", "賣出", "現券償還", "前日餘額", "今日餘額",
    "次一營業日限額", "資券互抵", "註記",
)
TPEX_MARGIN_FIELDS = (
    "代號", "名稱", "前資餘額(張)", "資買", "資賣", "現償", "資餘額",
    "資屬證金", "資使用率(%)", "資限額", "前券餘額(張)", "券賣", "券買",
    "券償", "券餘額", "券屬證金", "券使用率(%)", "券限額",
    "資券相抵(張)", "備註",
)


@dataclass(frozen=True)
class OfficialSnapshotSeries:
    target_date: _datetime.date
    snapshots: Mapping[_datetime.date, OfficialDailySnapshot]
    manifest_sha256: str
    request_count: int
    request_budget: OfficialRequestBudget
    source_mode: str = SOURCE_MODE
    source_schema_version: str = SOURCE_SCHEMA_VERSION

    @property
    def dates(self) -> tuple[_datetime.date, ...]:
        return tuple(sorted(self.snapshots))


HISTORICAL_SOURCE_DEFINITIONS: dict[str, OfficialSourceDefinition] = {
    "twse_price": OfficialSourceDefinition(
        "twse_price", "TWSE", "price",
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", "twse_tables",
        30 * 1024 * 1024,
    ),
    "twse_institutional": OfficialSourceDefinition(
        "twse_institutional", "TWSE", "institutional",
        "https://www.twse.com.tw/rwd/zh/fund/T86", "twse_report",
    ),
    "twse_margin": OfficialSourceDefinition(
        "twse_margin", "TWSE", "margin",
        "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN", "twse_tables",
    ),
    "tpex_price": OfficialSourceDefinition(
        "tpex_price", "TPEx", "price",
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes", "tpex_tables",
        30 * 1024 * 1024,
    ),
    "tpex_institutional": OfficialSourceDefinition(
        "tpex_institutional", "TPEx", "institutional",
        "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php", "tpex_tables",
        15 * 1024 * 1024,
    ),
    "tpex_margin": OfficialSourceDefinition(
        "tpex_margin", "TPEx", "margin",
        "https://www.tpex.org.tw/www/zh-tw/margin/balance", "tpex_tables",
        15 * 1024 * 1024,
    ),
}


def roc_date_text(value: _datetime.date) -> str:
    return f"{value.year - 1911:03d}/{value.month:02d}/{value.day:02d}"


def _params(source_id: str, target_date: _datetime.date) -> dict[str, str]:
    ymd = target_date.strftime("%Y%m%d")
    roc = roc_date_text(target_date)
    if source_id == "twse_price":
        return {"date": ymd, "type": "ALLBUT0999", "response": "json"}
    if source_id == "twse_institutional":
        return {"date": ymd, "selectType": "ALLBUT0999", "response": "json"}
    if source_id == "twse_margin":
        return {"date": ymd, "selectType": "STOCK", "response": "json"}
    if source_id == "tpex_price":
        return {"date": target_date.strftime("%Y/%m/%d"), "response": "json"}
    if source_id == "tpex_institutional":
        return {"l": "zh-tw", "o": "json", "se": "EW", "t": "D", "d": roc, "s": "0,asc"}
    if source_id == "tpex_margin":
        return {"date": target_date.strftime("%Y/%m/%d"), "response": "json"}
    raise ValueError("unknown historical source")


def _status_date(payload: Any, target_date: _datetime.date, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} schema is invalid")
    if str(payload.get("stat") or "").upper() != "OK":
        raise ValueError(f"{label} status is invalid")
    if normalize_market_date(payload.get("date")) != target_date:
        raise ValueError(f"{label} target date mismatch")
    return payload


def _table(payload: Mapping[str, Any], predicate: Callable[[list[str], Mapping[str, Any]], bool], label: str) -> tuple[list[str], list[Any], Mapping[str, Any]]:
    matches = []
    for candidate in payload.get("tables") or []:
        if not isinstance(candidate, dict):
            continue
        fields = candidate.get("fields")
        rows = candidate.get("data")
        if isinstance(fields, list) and isinstance(rows, list):
            names = [str(item) for item in fields]
            if predicate(names, candidate):
                matches.append((names, rows, candidate))
    if len(matches) != 1:
        raise ValueError(f"{label} target table is ambiguous")
    return matches[0]


def _number(value: Any, *, allow_empty: bool = False) -> float | None:
    return parse_number(re.sub(r"<[^>]*>", "", str(value or "")).replace("&nbsp;", " ").strip(), allow_empty=allow_empty)


def _dedupe(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]]) -> tuple[dict[str, Any], ...]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = key(row)
        previous = result.get(identity)
        if previous is not None and previous != row:
            raise ValueError("official source contains conflicting duplicate rows")
        result[identity] = row
    return tuple(result[item] for item in sorted(result))


def _price_row(target_date: _datetime.date, source: Sequence[Any], *, symbol_index: int, volume_index: int, open_index: int, high_index: int, low_index: int, close_index: int) -> dict[str, Any] | None:
    symbol = normalize_symbol(re.sub(r"<[^>]*>", "", str(source[symbol_index] or "")).strip())
    values = {
        "date": target_date.isoformat(),
        "stock_id": symbol,
        "open": _number(source[open_index], allow_empty=True),
        "max": _number(source[high_index], allow_empty=True),
        "min": _number(source[low_index], allow_empty=True),
        "close": _number(source[close_index], allow_empty=True),
        "Trading_Volume": _number(source[volume_index], allow_empty=True),
    }
    if any(values[name] is None for name in ("open", "max", "min", "close", "Trading_Volume")):
        return None
    open_value = float(values["open"])
    high = float(values["max"])
    low = float(values["min"])
    close = float(values["close"])
    volume = float(values["Trading_Volume"])
    if min(open_value, high, low, close) <= 0 or volume < 0:
        raise ValueError("official price values are invalid")
    if high < max(open_value, close, low) or low > min(open_value, close, high):
        raise ValueError("official OHLC relationship is invalid")
    return values


def parse_twse_price_report(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    document = _status_date(payload, target_date, "TWSE price")
    fields, data, _ = _table(
        document,
        lambda names, _candidate: names == [
            "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價",
            "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差", "最後揭示買價",
            "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比",
        ],
        "TWSE price",
    )
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) != len(fields):
            continue
        try:
            row = _price_row(target_date, source, symbol_index=0, volume_index=2, open_index=5, high_index=6, low_index=7, close_index=8)
        except ValueError:
            if not re.fullmatch(r"\d{4,6}", re.sub(r"<[^>]*>", "", str(source[0] or "")).strip()):
                continue
            raise
        if row is not None:
            rows.append(row)
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def parse_twse_margin_report(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    document = _status_date(payload, target_date, "TWSE margin")
    fields, data, _ = _table(
        document,
        lambda names, candidate: tuple(names) == TWSE_MARGIN_FIELDS and "股票" in str(candidate.get("title") or ""),
        "TWSE margin",
    )
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) != len(fields):
            continue
        try:
            symbol = normalize_symbol(source[0])
        except ValueError:
            continue
        rows.append({
            "date": target_date.isoformat(),
            "stock_id": symbol,
            "MarginPurchaseTodayBalance": float(_number(source[6])),
            "ShortSaleTodayBalance": float(_number(source[12])),
        })
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def parse_tpex_price_report(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    document = _status_date(payload, target_date, "TPEx price")
    fields, data, table = _table(
        document,
        lambda names, candidate: names[:9] == ["代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低", "均價", "成交股數"] and "上櫃股票行情" in str(candidate.get("title") or ""),
        "TPEx price",
    )
    if normalize_market_date(table.get("date")) != target_date:
        raise ValueError("TPEx price table date mismatch")
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) != len(fields):
            continue
        try:
            row = _price_row(target_date, source, symbol_index=0, volume_index=8, open_index=4, high_index=5, low_index=6, close_index=2)
        except ValueError:
            if not re.fullmatch(r"\d{4,6}", re.sub(r"<[^>]*>", "", str(source[0] or "")).strip()):
                continue
            raise
        if row is not None:
            rows.append(row)
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


def parse_tpex_margin_report(payload: Any, target_date: _datetime.date) -> tuple[dict[str, Any], ...]:
    document = _status_date(payload, target_date, "TPEx margin")
    fields, data, table = _table(
        document,
        lambda names, candidate: tuple(names) == TPEX_MARGIN_FIELDS and "融資融券餘額" in str(candidate.get("title") or ""),
        "TPEx margin",
    )
    if normalize_market_date(table.get("date")) != target_date:
        raise ValueError("TPEx margin table date mismatch")
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) != len(fields):
            continue
        try:
            symbol = normalize_symbol(re.sub(r"<[^>]*>", "", str(source[0] or "")).strip())
        except ValueError:
            continue
        rows.append({
            "date": target_date.isoformat(),
            "stock_id": symbol,
            "MarginPurchaseTodayBalance": float(_number(source[6])),
            "ShortSaleTodayBalance": float(_number(source[14])),
        })
    return _dedupe(rows, lambda row: (row["stock_id"], row["date"]))


HISTORICAL_PARSERS: Mapping[str, Callable[[Any, _datetime.date], tuple[dict[str, Any], ...]]] = MappingProxyType({
    "twse_price": parse_twse_price_report,
    "twse_institutional": parse_twse_institutional,
    "twse_margin": parse_twse_margin_report,
    "tpex_price": parse_tpex_price_report,
    "tpex_institutional": parse_tpex_institutional,
    "tpex_margin": parse_tpex_margin_report,
})


def _request_headers(source_id: str) -> dict[str, str]:
    headers = {"User-Agent": "ABSORB/1.0"}
    if source_id == "tpex_price":
        headers["X-Requested-With"] = "XMLHttpRequest"
    return headers


def _request_payload(definition: OfficialSourceDefinition, target_date: _datetime.date, *, session: Any, timeout: int, retry_attempts: int, sleep_fn: Callable[[float], None]) -> tuple[Any, int, int]:
    attempts = 0
    for attempt in range(retry_attempts):
        attempts += 1
        try:
            response = session.get(
                definition.url,
                params=_params(definition.source_id, target_date),
                headers=_request_headers(definition.source_id),
                timeout=timeout,
            )
        except Exception as exc:
            if attempt + 1 < retry_attempts:
                sleep_fn(0.5 * (2 ** attempt))
                continue
            raise OfficialSourceFailure(definition.source_id, "transport_error", retryable=True, safe_message=type(exc).__name__) from None
        status = int(getattr(response, "status_code", 0))
        if status in {500, 502, 503, 504} and attempt + 1 < retry_attempts:
            sleep_fn(0.5 * (2 ** attempt))
            continue
        if status != 200:
            raise OfficialSourceFailure(definition.source_id, "http_error", http_status=status, retryable=status >= 500, safe_message=f"HTTP {status}")
        content = bytes(getattr(response, "content", b""))
        header = getattr(response, "headers", {}).get("Content-Length") if hasattr(response, "headers") else None
        if header:
            try:
                if int(header) > definition.max_bytes:
                    raise OfficialSourceFailure(definition.source_id, "response_too_large", safe_message="content length exceeds limit")
            except ValueError:
                raise OfficialSourceFailure(definition.source_id, "invalid_headers", safe_message="invalid content length") from None
        if not content or len(content) > definition.max_bytes:
            raise OfficialSourceFailure(definition.source_id, "response_too_large", safe_message="response size is invalid")
        try:
            return json.loads(content.decode("utf-8-sig")), len(content), attempts
        except (UnicodeError, ValueError):
            raise OfficialSourceFailure(definition.source_id, "invalid_json", safe_message="response JSON is invalid") from None
    raise AssertionError("unreachable")


def _coverage(dataset: str, rows: Sequence[Mapping[str, Any]], minimum: int) -> int:
    symbols = {str(row.get("stock_id")) for row in rows if row.get("stock_id")}
    if len(symbols) < minimum:
        raise ValueError(f"official {dataset} coverage is below minimum")
    return len(symbols)


def build_historical_daily_snapshot(
    root: Any,
    target_date: _datetime.date,
    *,
    session: Any = None,
    now: _datetime.datetime | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
    minimum_price_symbols: Mapping[str, int] | None = None,
    minimum_chip_symbols: int | None = None,
) -> OfficialDailySnapshot:
    if not isinstance(target_date, _datetime.date) or isinstance(target_date, _datetime.datetime):
        raise TypeError("target_date must be a date")
    if session is None:
        import requests
        session = requests.Session()
    checked_at = now or _datetime.datetime.now(_datetime.timezone.utc)
    minimum_price_symbols = dict(minimum_price_symbols or {"TWSE": 500, "TPEx": 500})
    if (
        minimum_chip_symbols is not None
        and (
            isinstance(minimum_chip_symbols, bool)
            or not isinstance(minimum_chip_symbols, int)
            or minimum_chip_symbols < 1
        )
    ):
        raise ValueError("minimum chip symbol count is invalid")

    def minimum_for_source(
        source_id: str,
        definition: OfficialSourceDefinition,
    ) -> int:
        if definition.dataset == "price":
            return int(minimum_price_symbols.get(definition.market, 1))
        if minimum_chip_symbols is not None:
            return minimum_chip_symbols
        return int(DEFAULT_MINIMUM_SOURCE_SYMBOLS[source_id])

    results: dict[str, OfficialSourceResult] = {}
    request_count = 0
    cold_sources = 0

    for source_id, definition in HISTORICAL_SOURCE_DEFINITIONS.items():
        try:
            cached = load_cached_source(root, source_id=source_id, target_date=target_date, parser_version=PARSER_VERSION)
        except OfficialCacheError as exc:
            raise OfficialSourceFailure(source_id, "cache_invalid", safe_message=str(exc)) from None
        if cached is None:
            cold_sources += 1
            payload, response_size, attempts = _request_payload(
                definition,
                target_date,
                session=session,
                timeout=timeout,
                retry_attempts=retry_attempts,
                sleep_fn=sleep_fn,
            )
            request_count += attempts
            try:
                rows = HISTORICAL_PARSERS[source_id](payload, target_date)
                minimum = minimum_for_source(source_id, definition)
                symbol_count = _coverage(definition.dataset, rows, int(minimum))
            except (KeyError, TypeError, ValueError) as exc:
                raise OfficialSourceFailure(source_id, "schema_validation", safe_message=str(exc)) from None
            cached = store_cached_source(
                root,
                source_id=source_id,
                target_date=target_date,
                rows=rows,
                symbol_count=symbol_count,
                parser_version=PARSER_VERSION,
                source_url=definition.url,
                fetched_at=checked_at,
                date_verification="explicit",
            )
            cache_hit = False
        else:
            rows = cached.rows
            response_size = cached.compressed_size
            cache_hit = True
        minimum = minimum_for_source(source_id, definition)
        try:
            symbol_count = _coverage(definition.dataset, rows, int(minimum))
        except ValueError as exc:
            raise OfficialSourceFailure(
                source_id,
                "schema_validation",
                safe_message=str(exc),
            ) from None
        results[source_id] = OfficialSourceResult(
            source_id=source_id,
            market=definition.market,
            dataset=definition.dataset,
            target_date=target_date,
            rows=tuple(dict(row) for row in rows),
            symbol_count=symbol_count,
            content_sha256=cached.content_sha256,
            response_size_bytes=response_size,
            cache_hit=cache_hit,
            date_verification="explicit",
        )

    price_symbols_by_market = {
        market: {
            str(row.get("stock_id"))
            for result in results.values()
            if result.market == market and result.dataset == "price"
            for row in result.rows
            if row.get("stock_id")
        }
        for market in ("TWSE", "TPEx")
    }
    for source_id, result in results.items():
        if result.dataset == "price":
            continue
        source_symbols = {
            str(row.get("stock_id"))
            for row in result.rows
            if row.get("stock_id")
        }
        minimum_overlap = minimum_for_source(
            source_id,
            HISTORICAL_SOURCE_DEFINITIONS[source_id],
        )
        if len(source_symbols & price_symbols_by_market[result.market]) < int(minimum_overlap):
            raise OfficialSourceFailure(
                source_id,
                "cross_source_identity",
                safe_message="official source overlap is below minimum",
            )

    price: dict[str, dict[str, Any]] = {}
    institutional: dict[str, list[dict[str, Any]]] = {}
    institutional_keys: set[tuple[str, str]] = set()
    margin: dict[str, dict[str, Any]] = {}
    for result in results.values():
        for source_row in result.rows:
            row = dict(source_row)
            symbol = row["stock_id"]
            if result.dataset == "price":
                if symbol in price and price[symbol] != row:
                    raise OfficialSourceFailure(result.source_id, "cross_source_duplicate", safe_message=f"duplicate price symbol {symbol}")
                price[symbol] = row
            elif result.dataset == "institutional":
                identity = (symbol, str(row["name"]))
                if identity in institutional_keys:
                    raise OfficialSourceFailure(
                        result.source_id,
                        "cross_source_duplicate",
                        safe_message=f"duplicate institutional symbol/category {symbol}/{row['name']}",
                    )
                institutional_keys.add(identity)
                institutional.setdefault(symbol, []).append(row)
            else:
                if symbol in margin and margin[symbol] != row:
                    raise OfficialSourceFailure(result.source_id, "cross_source_duplicate", safe_message=f"duplicate margin symbol {symbol}")
                margin[symbol] = row

    manifest_document = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "target_date": target_date.isoformat(),
        "validation": {
            "minimum_price_symbols": dict(sorted(minimum_price_symbols.items())),
            "minimum_chip_symbols": minimum_chip_symbols,
            "default_minimum_source_symbols": dict(DEFAULT_MINIMUM_SOURCE_SYMBOLS),
        },
        "sources": {
            source_id: {
                "content_sha256": result.content_sha256,
                "symbol_count": result.symbol_count,
                "date_verification": result.date_verification,
            }
            for source_id, result in sorted(results.items())
        },
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    budget = OfficialRequestBudget(
        planned_minimum_requests=cold_sources,
        planned_worst_case_requests=cold_sources * retry_attempts,
        official_requests=cold_sources,
        finmind_requests=0,
        capacity_proven=True,
        reason="capacity_proven",
    )
    return OfficialDailySnapshot(
        target_date=target_date,
        price_by_symbol=MappingProxyType({symbol: MappingProxyType(row) for symbol, row in price.items()}),
        institutional_by_symbol=MappingProxyType({
            symbol: tuple(MappingProxyType(item) for item in sorted(rows, key=lambda item: item["name"]))
            for symbol, rows in institutional.items()
        }),
        margin_by_symbol=MappingProxyType({symbol: MappingProxyType(row) for symbol, row in margin.items()}),
        source_results=MappingProxyType(dict(results)),
        manifest_sha256=manifest_sha256,
        request_count=request_count,
        request_budget=budget,
        source_mode=SOURCE_MODE,
        source_schema_version=SOURCE_SCHEMA_VERSION,
    )


def build_official_snapshot_series(
    root: Any,
    trading_dates: Iterable[_datetime.date],
    *,
    snapshot_builder: Callable[..., OfficialDailySnapshot] = build_historical_daily_snapshot,
    **kwargs: Any,
) -> OfficialSnapshotSeries:
    dates = tuple(sorted(set(trading_dates)))
    if not dates or len(dates) > MAX_CATCHUP_SESSIONS:
        raise ValueError("official catch-up session count is invalid")
    if any(not isinstance(item, _datetime.date) or isinstance(item, _datetime.datetime) for item in dates):
        raise TypeError("trading_dates must contain dates")
    snapshots: dict[_datetime.date, OfficialDailySnapshot] = {}
    request_count = 0
    minimum = 0
    worst_case = 0
    for value in dates:
        snapshot = snapshot_builder(root, value, **kwargs)
        if snapshot.target_date != value:
            raise ValueError("official snapshot series date mismatch")
        if snapshot.source_schema_version != SOURCE_SCHEMA_VERSION:
            raise ValueError("official snapshot schema version mismatch")
        snapshots[value] = snapshot
        request_count += snapshot.request_count
        minimum += snapshot.request_budget.planned_minimum_requests
        worst_case += snapshot.request_budget.planned_worst_case_requests
    manifest_document = {
        "source_mode": SOURCE_MODE,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "target_date": dates[-1].isoformat(),
        "snapshots": [
            {"date": value.isoformat(), "manifest_sha256": snapshots[value].manifest_sha256}
            for value in dates
        ],
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return OfficialSnapshotSeries(
        target_date=dates[-1],
        snapshots=MappingProxyType(dict(snapshots)),
        manifest_sha256=manifest_sha256,
        request_count=request_count,
        request_budget=OfficialRequestBudget(
            planned_minimum_requests=minimum,
            planned_worst_case_requests=worst_case,
            official_requests=minimum,
            finmind_requests=0,
            capacity_proven=True,
            reason="capacity_proven",
        ),
    )
