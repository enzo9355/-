"""Hardened date-addressable TW official reports with fail-closed source guards."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from stock_papi.integrations.market_data import tw_official_historical as _legacy
from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialDailySnapshot,
    OfficialRequestBudget,
    OfficialSourceFailure,
    OfficialSourceResult,
    normalize_market_date,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialCacheError,
    load_cached_source,
    store_cached_source,
)

SOURCE_MODE = "tw_official_bulk_v2"
SOURCE_SCHEMA_VERSION = "tw-official-historical-v2"
PARSER_VERSION = "tw-official-historical-parser-v2"
DEFAULT_TIMEOUT_SECONDS = _legacy.DEFAULT_TIMEOUT_SECONDS
DEFAULT_RETRY_ATTEMPTS = _legacy.DEFAULT_RETRY_ATTEMPTS
MAX_CATCHUP_SESSIONS = _legacy.MAX_CATCHUP_SESSIONS
MIN_SOURCE_PRICE_OVERLAP_RATIO = 0.90

DEFAULT_PRICE_MINIMUMS: Mapping[str, int] = MappingProxyType(
    {"TWSE": 500, "TPEx": 500}
)
DEFAULT_CORE_MINIMUMS: Mapping[str, int] = MappingProxyType(
    {
        "twse_institutional": 500,
        "twse_margin": 400,
        "tpex_institutional": 300,
        "tpex_margin": 300,
    }
)

TWSE_MARGIN_FIELDS = (
    "代號",
    "名稱",
    "買進",
    "賣出",
    "現金償還",
    "前日餘額",
    "今日餘額",
    "次一營業日限額",
    "買進",
    "賣出",
    "現券償還",
    "前日餘額",
    "今日餘額",
    "次一營業日限額",
    "資券互抵",
    "註記",
)
TPEX_MARGIN_FIELDS = (
    "代號",
    "名稱",
    "前資餘額(張)",
    "資買",
    "資賣",
    "現償",
    "資餘額",
    "資屬證金",
    "資使用率(%)",
    "資限額",
    "前券餘額(張)",
    "券賣",
    "券買",
    "券償",
    "券餘額",
    "券屬證金",
    "券使用率(%)",
    "券限額",
    "資券相抵(張)",
    "備註",
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


def _exact_table(
    payload: Any,
    *,
    expected_fields: Sequence[str],
    title_token: str,
    label: str,
    target_date: _datetime.date | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} schema is invalid")
    matches = []
    for candidate in payload.get("tables") or []:
        if not isinstance(candidate, dict):
            continue
        fields = candidate.get("fields")
        rows = candidate.get("data")
        if (
            isinstance(fields, list)
            and tuple(str(item) for item in fields) == tuple(expected_fields)
            and isinstance(rows, list)
            and title_token in str(candidate.get("title") or "")
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"{label} schema fingerprint is invalid")
    table = matches[0]
    if target_date is not None and normalize_market_date(table.get("date")) != target_date:
        raise ValueError(f"{label} table date mismatch")
    return table


def parse_twse_margin_report(
    payload: Any,
    target_date: _datetime.date,
) -> tuple[dict[str, Any], ...]:
    _exact_table(
        payload,
        expected_fields=TWSE_MARGIN_FIELDS,
        title_token="股票",
        label="TWSE margin",
    )
    return _legacy.parse_twse_margin_report(payload, target_date)


def parse_tpex_margin_report(
    payload: Any,
    target_date: _datetime.date,
) -> tuple[dict[str, Any], ...]:
    _exact_table(
        payload,
        expected_fields=TPEX_MARGIN_FIELDS,
        title_token="融資融券餘額",
        label="TPEx margin",
        target_date=target_date,
    )
    return _legacy.parse_tpex_margin_report(payload, target_date)


HISTORICAL_SOURCE_DEFINITIONS = _legacy.HISTORICAL_SOURCE_DEFINITIONS
HISTORICAL_PARSERS: Mapping[
    str,
    Callable[[Any, _datetime.date], tuple[dict[str, Any], ...]],
] = MappingProxyType(
    {
        "twse_price": _legacy.parse_twse_price_report,
        "twse_institutional": _legacy.parse_twse_institutional,
        "twse_margin": parse_twse_margin_report,
        "tpex_price": _legacy.parse_tpex_price_report,
        "tpex_institutional": _legacy.parse_tpex_institutional,
        "tpex_margin": parse_tpex_margin_report,
    }
)


def _validated_minimum(
    values: Mapping[str, int],
    *,
    source_id: str,
    market: str,
) -> int:
    value = values.get(source_id, values.get(market))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"official minimum is invalid for {source_id}")
    return value


def _coverage(dataset: str, rows: Sequence[Mapping[str, Any]], minimum: int) -> int:
    return _legacy._coverage(dataset, rows, minimum)


def _source_symbols(result: OfficialSourceResult) -> set[str]:
    return {
        str(row.get("stock_id"))
        for row in result.rows
        if row.get("stock_id")
    }


def _validate_cross_source_identity(
    results: Mapping[str, OfficialSourceResult],
    *,
    minimum_overlap_ratio: float,
) -> None:
    if (
        isinstance(minimum_overlap_ratio, bool)
        or not isinstance(minimum_overlap_ratio, (int, float))
        or not 0 < float(minimum_overlap_ratio) <= 1
    ):
        raise ValueError("source overlap ratio is invalid")
    for prefix in ("twse", "tpex"):
        price_result = results[f"{prefix}_price"]
        price_symbols = _source_symbols(price_result)
        for suffix in ("institutional", "margin"):
            source_id = f"{prefix}_{suffix}"
            source_symbols = _source_symbols(results[source_id])
            overlap = len(source_symbols & price_symbols)
            ratio = overlap / len(source_symbols) if source_symbols else 0.0
            if ratio < float(minimum_overlap_ratio):
                raise OfficialSourceFailure(
                    source_id,
                    "cross_source_identity",
                    safe_message=(
                        "same-market price overlap "
                        f"{ratio:.3f} is below {float(minimum_overlap_ratio):.3f}"
                    ),
                )


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
    minimum_core_symbols: Mapping[str, int] | None = None,
    minimum_overlap_ratio: float = MIN_SOURCE_PRICE_OVERLAP_RATIO,
) -> OfficialDailySnapshot:
    if (
        not isinstance(target_date, _datetime.date)
        or isinstance(target_date, _datetime.datetime)
    ):
        raise TypeError("target_date must be a date")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")
    if (
        isinstance(retry_attempts, bool)
        or not isinstance(retry_attempts, int)
        or retry_attempts < 1
    ):
        raise ValueError("retry_attempts must be a positive integer")
    if session is None:
        import requests

        session = requests.Session()

    checked_at = now or _datetime.datetime.now(_datetime.timezone.utc)
    price_minimums = dict(minimum_price_symbols or DEFAULT_PRICE_MINIMUMS)
    core_minimums = dict(minimum_core_symbols or DEFAULT_CORE_MINIMUMS)
    results: dict[str, OfficialSourceResult] = {}
    request_count = 0
    cold_sources = 0

    for source_id, definition in HISTORICAL_SOURCE_DEFINITIONS.items():
        try:
            cached = load_cached_source(
                root,
                source_id=source_id,
                target_date=target_date,
                parser_version=PARSER_VERSION,
            )
        except OfficialCacheError as exc:
            raise OfficialSourceFailure(
                source_id,
                "cache_invalid",
                safe_message=str(exc),
            ) from None

        if definition.dataset == "price":
            minimum = _validated_minimum(
                price_minimums,
                source_id=source_id,
                market=definition.market,
            )
        else:
            minimum = _validated_minimum(
                core_minimums,
                source_id=source_id,
                market=definition.market,
            )

        if cached is None:
            cold_sources += 1
            payload, response_size, attempts = _legacy._request_payload(
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
                symbol_count = _coverage(definition.dataset, rows, minimum)
            except (KeyError, TypeError, ValueError) as exc:
                raise OfficialSourceFailure(
                    source_id,
                    "schema_validation",
                    safe_message=str(exc),
                ) from None
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

        try:
            symbol_count = _coverage(definition.dataset, rows, minimum)
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

    _validate_cross_source_identity(
        results,
        minimum_overlap_ratio=minimum_overlap_ratio,
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
                if symbol in price:
                    raise OfficialSourceFailure(
                        result.source_id,
                        "cross_source_duplicate",
                        safe_message=f"duplicate price symbol {symbol}",
                    )
                price[symbol] = row
            elif result.dataset == "institutional":
                identity = (symbol, str(row["name"]))
                if identity in institutional_keys:
                    raise OfficialSourceFailure(
                        result.source_id,
                        "cross_source_duplicate",
                        safe_message=(
                            "duplicate institutional symbol/category "
                            f"{symbol}/{row['name']}"
                        ),
                    )
                institutional_keys.add(identity)
                institutional.setdefault(symbol, []).append(row)
            else:
                if symbol in margin:
                    raise OfficialSourceFailure(
                        result.source_id,
                        "cross_source_duplicate",
                        safe_message=f"duplicate margin symbol {symbol}",
                    )
                margin[symbol] = row

    manifest_document = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "target_date": target_date.isoformat(),
        "validation": {
            "minimum_price_symbols": dict(sorted(price_minimums.items())),
            "minimum_core_symbols": dict(sorted(core_minimums.items())),
            "minimum_overlap_ratio": float(minimum_overlap_ratio),
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
        json.dumps(
            manifest_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
        price_by_symbol=MappingProxyType(
            {
                symbol: MappingProxyType(row)
                for symbol, row in price.items()
            }
        ),
        institutional_by_symbol=MappingProxyType(
            {
                symbol: tuple(
                    MappingProxyType(item)
                    for item in sorted(rows, key=lambda item: item["name"])
                )
                for symbol, rows in institutional.items()
            }
        ),
        margin_by_symbol=MappingProxyType(
            {
                symbol: MappingProxyType(row)
                for symbol, row in margin.items()
            }
        ),
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
    snapshot_builder: Callable[..., OfficialDailySnapshot] = (
        build_historical_daily_snapshot
    ),
    **kwargs: Any,
) -> OfficialSnapshotSeries:
    dates = tuple(sorted(set(trading_dates)))
    if not dates or len(dates) > MAX_CATCHUP_SESSIONS:
        raise ValueError("official catch-up session count is invalid")
    if any(
        not isinstance(item, _datetime.date)
        or isinstance(item, _datetime.datetime)
        for item in dates
    ):
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
            {
                "date": value.isoformat(),
                "manifest_sha256": snapshots[value].manifest_sha256,
            }
            for value in dates
        ],
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
