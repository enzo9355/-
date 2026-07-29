"""Shared contracts and canonical institutional parsers for TW official data."""

from __future__ import annotations

import datetime as _datetime
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

SOURCE_SCHEMA_VERSION = "tw-official-common-v1"
DEFAULT_RETRY_ATTEMPTS = 2
MAX_FINMIND_FALLBACK_REQUESTS = 20


class OfficialSourceFailure(RuntimeError):
    """Safe structured failure for a public official market source."""

    def __init__(
        self,
        source_id: str,
        category: str,
        *,
        http_status: int | None = None,
        retryable: bool = False,
        safe_message: str | None = None,
    ):
        self.source_id = str(source_id)
        self.category = str(category)
        self.http_status = int(http_status) if http_status is not None else None
        self.retryable = bool(retryable)
        self.safe_message = str(safe_message or category)
        super().__init__(f"{self.source_id}: {self.safe_message}")


@dataclass(frozen=True)
class OfficialSourceDefinition:
    source_id: str
    market: str
    dataset: str
    url: str
    response_kind: str
    max_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True)
class OfficialSourceResult:
    source_id: str
    market: str
    dataset: str
    target_date: _datetime.date
    rows: tuple[dict[str, Any], ...]
    symbol_count: int
    content_sha256: str
    response_size_bytes: int
    cache_hit: bool
    date_verification: str = "explicit"


@dataclass(frozen=True)
class OfficialRequestBudget:
    planned_minimum_requests: int
    planned_worst_case_requests: int
    official_requests: int
    finmind_requests: int
    capacity_proven: bool
    reason: str


@dataclass(frozen=True)
class OfficialDailySnapshot:
    target_date: _datetime.date
    price_by_symbol: Mapping[str, Mapping[str, Any]]
    institutional_by_symbol: Mapping[
        str, tuple[Mapping[str, Any], ...]
    ]
    margin_by_symbol: Mapping[str, Mapping[str, Any]]
    source_results: Mapping[str, OfficialSourceResult]
    manifest_sha256: str
    request_count: int
    request_budget: OfficialRequestBudget
    source_mode: str = "tw_official_bulk_v1"
    source_schema_version: str = SOURCE_SCHEMA_VERSION
    trading_status_by_symbol: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    terminated_by_symbol: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )


_EMPTY_MARKERS = {
    "",
    "--",
    "---",
    "----",
    "-",
    "N/A",
    "NA",
    "null",
    "None",
    "除權息",
    "暫停交易",
}
_SYMBOL_RE = re.compile(r"\d{4,6}")
TPEX_INSTITUTIONAL_FIELDS = (
    "代號", "名稱",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "買進股數", "賣出股數", "買賣超股數",
    "三大法人買賣超股數合計",
)


def normalize_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if not _SYMBOL_RE.fullmatch(text):
        raise ValueError("official source symbol is invalid")
    return text


def normalize_market_date(value: Any) -> _datetime.date:
    if isinstance(value, _datetime.datetime):
        return value.date()
    if isinstance(value, _datetime.date):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return _datetime.date(
            int(text[:4]), int(text[4:6]), int(text[6:8])
        )
    if re.fullmatch(r"\d{7}", text):
        return _datetime.date(
            int(text[:3]) + 1911,
            int(text[3:5]),
            int(text[5:7]),
        )
    match = re.fullmatch(
        r"(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})", text
    )
    if match:
        year, month, day = map(int, match.groups())
        if year < 1911:
            year += 1911
        return _datetime.date(year, month, day)
    raise ValueError("official source date is invalid")


def parse_number(
    value: Any, *, allow_empty: bool = False
) -> float | None:
    if value is None:
        if allow_empty:
            return None
        raise ValueError("official source number is missing")
    if isinstance(value, bool):
        raise ValueError("official source number is invalid")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = (
            str(value)
            .strip()
            .replace(",", "")
            .replace("＋", "+")
            .replace("－", "-")
        )
        if text in _EMPTY_MARKERS:
            if allow_empty:
                return None
            raise ValueError("official source number is missing")
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()
        number = float(text)
        if negative:
            number = -number
    if not math.isfinite(number):
        raise ValueError("official source number is not finite")
    return number


def _institutional_rows(
    target_date: _datetime.date,
    symbol: str,
    foreign: tuple[float, float],
    trust: tuple[float, float],
    dealer: tuple[float, float],
) -> list[dict[str, Any]]:
    result = []
    for name, pair in (
        ("Foreign", foreign),
        ("InvestmentTrust", trust),
        ("Dealer", dealer),
    ):
        buy, sell = pair
        if buy < 0 or sell < 0:
            raise ValueError("official institutional buy/sell is negative")
        result.append(
            {
                "date": target_date.isoformat(),
                "stock_id": symbol,
                "name": name,
                "buy": buy,
                "sell": sell,
            }
        )
    return result


def _dedupe(
    rows: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> tuple[dict[str, Any], ...]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = key(row)
        previous = result.get(identity)
        if previous is not None and previous != row:
            raise ValueError(
                "official source contains conflicting duplicate rows"
            )
        result[identity] = row
    return tuple(result[item] for item in sorted(result))


def _table_rows(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[Any]]:
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("official tabular report schema is invalid")
    return [str(item) for item in fields], data


def parse_twse_institutional(
    payload: Any, target_date: _datetime.date
) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise ValueError("TWSE institutional status is invalid")
    if normalize_market_date(payload.get("date")) != target_date:
        raise ValueError("TWSE institutional target date mismatch")
    fields, data = _table_rows(payload)
    position = {name: index for index, name in enumerate(fields)}
    required = (
        "證券代號",
        "外陸資買進股數(不含外資自營商)",
        "外陸資賣出股數(不含外資自營商)",
        "外資自營商買進股數",
        "外資自營商賣出股數",
        "投信買進股數",
        "投信賣出股數",
        "自營商買進股數(自行買賣)",
        "自營商賣出股數(自行買賣)",
        "自營商買進股數(避險)",
        "自營商賣出股數(避險)",
    )
    if any(name not in position for name in required):
        raise ValueError("TWSE institutional fields are incomplete")
    rows = []
    for source in data:
        if not isinstance(source, list) or len(source) < len(fields):
            continue
        try:
            symbol = normalize_symbol(source[position["證券代號"]])
        except ValueError:
            continue

        def number(name: str) -> float:
            return float(parse_number(source[position[name]]))

        rows.extend(
            _institutional_rows(
                target_date,
                symbol,
                (
                    number("外陸資買進股數(不含外資自營商)")
                    + number("外資自營商買進股數"),
                    number("外陸資賣出股數(不含外資自營商)")
                    + number("外資自營商賣出股數"),
                ),
                (number("投信買進股數"), number("投信賣出股數")),
                (
                    number("自營商買進股數(自行買賣)")
                    + number("自營商買進股數(避險)"),
                    number("自營商賣出股數(自行買賣)")
                    + number("自營商賣出股數(避險)"),
                ),
            )
        )
    return _dedupe(
        rows,
        lambda row: (
            row["stock_id"],
            row["date"],
            row["name"],
        ),
    )


def _strip_html(value: Any) -> str:
    text = re.sub(r"<[^>]*>", "", str(value or ""))
    return text.replace("&nbsp;", " ").strip()


def _find_tpex_table_rows(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], _datetime.date]:
    actual_date = normalize_market_date(payload.get("date"))
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("TPEx institutional tables are missing")
    matches = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_date = table.get("date")
        if table_date is not None and normalize_market_date(table_date) != actual_date:
            raise ValueError("TPEx institutional table date mismatch")
        if (
            table.get("title") == "三大法人買賣明細資訊"
            and table.get("columnNum") == 25
            and tuple(str(item) for item in table.get("fields") or ())
            == TPEX_INSTITUTIONAL_FIELDS
            and isinstance(table.get("data"), list)
            and table.get("data")
        ):
            matches.append(table)
    if len(matches) != 1:
        raise ValueError("TPEx institutional schema fingerprint is invalid")
    return matches[0], actual_date


def parse_tpex_institutional(
    payload: Any, target_date: _datetime.date
) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(payload, dict)
        or str(payload.get("stat") or "OK").upper() not in {"OK", "0"}
    ):
        raise ValueError("TPEx institutional status is invalid")
    table, actual_date = _find_tpex_table_rows(payload)
    data = table["data"]
    if actual_date != target_date:
        raise ValueError("TPEx institutional target date mismatch")
    rows = []
    for source in data:
        if isinstance(source, dict):
            aliases = {
                "symbol": (
                    "SecuritiesCompanyCode",
                    "代號",
                    "證券代號",
                ),
                "foreign_buy": (
                    "ForeignInvestorsBuy",
                    "外資及陸資買進股數",
                ),
                "foreign_sell": (
                    "ForeignInvestorsSell",
                    "外資及陸資賣出股數",
                ),
                "trust_buy": ("InvestmentTrustBuy", "投信買進股數"),
                "trust_sell": ("InvestmentTrustSell", "投信賣出股數"),
                "dealer_buy": ("DealerBuy", "自營商買進股數"),
                "dealer_sell": ("DealerSell", "自營商賣出股數"),
            }

            def known(names: Sequence[str]) -> Any:
                hits = [name for name in names if name in source]
                if len(hits) != 1:
                    raise ValueError(
                        "TPEx institutional field is missing or ambiguous"
                    )
                return source[hits[0]]

            try:
                symbol = normalize_symbol(known(aliases["symbol"]))
            except ValueError:
                continue
            foreign = (
                float(parse_number(known(aliases["foreign_buy"]))),
                float(parse_number(known(aliases["foreign_sell"]))),
            )
            trust = (
                float(parse_number(known(aliases["trust_buy"]))),
                float(parse_number(known(aliases["trust_sell"]))),
            )
            dealer = (
                float(parse_number(known(aliases["dealer_buy"]))),
                float(parse_number(known(aliases["dealer_sell"]))),
            )
        elif (
            isinstance(source, list)
            and len(source) == len(TPEX_INSTITUTIONAL_FIELDS)
        ):
            try:
                symbol = normalize_symbol(_strip_html(source[0]))
            except ValueError:
                continue
            foreign = (
                float(parse_number(_strip_html(source[8]))),
                float(parse_number(_strip_html(source[9]))),
            )
            trust = (
                float(parse_number(_strip_html(source[11]))),
                float(parse_number(_strip_html(source[12]))),
            )
            dealer = (
                float(parse_number(_strip_html(source[20]))),
                float(parse_number(_strip_html(source[21]))),
            )
        else:
            continue
        rows.extend(
            _institutional_rows(
                target_date,
                symbol,
                foreign,
                trust,
                dealer,
            )
        )
    return _dedupe(
        rows,
        lambda row: (
            row["stock_id"],
            row["date"],
            row["name"],
        ),
    )


def plan_official_request_budget(
    *,
    cold_source_count: int,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    fallback_symbols: int = 0,
    fallback_requests_per_symbol: int = 3,
    fallback_enabled: bool = False,
    max_finmind_fallback_requests: int = MAX_FINMIND_FALLBACK_REQUESTS,
) -> OfficialRequestBudget:
    values = (
        cold_source_count,
        retry_attempts,
        fallback_symbols,
        fallback_requests_per_symbol,
        max_finmind_fallback_requests,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in values
    ):
        raise ValueError(
            "request budget values must be non-negative integers"
        )
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be positive")
    finmind_requests = (
        fallback_symbols * fallback_requests_per_symbol
        if fallback_enabled
        else 0
    )
    if fallback_symbols and not fallback_enabled:
        return OfficialRequestBudget(
            cold_source_count,
            cold_source_count * retry_attempts,
            cold_source_count,
            0,
            False,
            "fallback_disabled",
        )
    if finmind_requests > max_finmind_fallback_requests:
        return OfficialRequestBudget(
            cold_source_count + finmind_requests,
            cold_source_count * retry_attempts + finmind_requests,
            cold_source_count,
            finmind_requests,
            False,
            "fallback_budget_exceeded",
        )
    return OfficialRequestBudget(
        cold_source_count + finmind_requests,
        cold_source_count * retry_attempts + finmind_requests,
        cold_source_count,
        finmind_requests,
        True,
        "capacity_proven",
    )


def assert_fallback_capacity(budget: OfficialRequestBudget) -> None:
    if not budget.capacity_proven:
        raise OfficialSourceFailure(
            "finmind_fallback",
            "capacity_not_proven",
            safe_message=budget.reason,
        )
