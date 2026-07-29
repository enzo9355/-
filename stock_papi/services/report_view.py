"""Normalize verified report metadata before it reaches Jinja."""

from dataclasses import dataclass
import re
from typing import Any, Mapping

from reporting.exceptions import ReportWebError
from stock_papi.batch.observation_products import (
    validate_trading_status_observations,
)


_REPORT_LABELS = {
    "post_close": "盤後觀察",
    "pre_market": "盤前風險更新",
}
_CORE_LIST_KEYS = (
    "industry_observations",
    "heatmap",
    "stock_events",
    "etf_observations",
    "daily_focus",
)


def _finite_number(value: Any, *, optional=False) -> bool:
    if value is None:
        return optional
    return type(value) in (int, float) and value == value and abs(value) != float("inf")


def _valid_core_items(value: dict[str, Any]) -> bool:
    market = value["market_observation"]
    quality = value["data_quality"]
    if not all(
        _finite_number(market.get(key), optional=True)
        for key in (
            "return_1d_pct",
            "ma20_breadth_pct",
            "realized_volatility_20d_pct",
        )
    ) or not all(
        _finite_number(market.get(key))
        for key in ("advancing_count", "declining_count")
    ):
        return False
    if (
        not _finite_number(quality.get("coverage"), optional=True)
        or not _finite_number(quality.get("symbol_count"), optional=True)
        or not _finite_number(quality.get("failure_count"), optional=True)
    ):
        return False
    if not all(isinstance(item, str) for item in value["daily_focus"][:20]):
        return False
    for item in value["industry_observations"][:100]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not _finite_number(item.get("available_count"))
            or not _finite_number(item.get("component_count"))
            or not _finite_number(item.get("relative_return_5d_pct"), optional=True)
        ):
            return False
    for item in value["stock_events"][:200]:
        if (
            not isinstance(item, dict)
            or re.fullmatch(r"[A-Z0-9.-]{1,16}", str(item.get("symbol") or "")) is None
            or not all(isinstance(item.get(key), str) for key in ("name", "observation", "as_of", "unit"))
            or not _finite_number(item.get("metric_value"), optional=True)
        ):
            return False
    for item in value["etf_observations"][:100]:
        if (
            not isinstance(item, dict)
            or re.fullmatch(r"[A-Z0-9.-]{1,16}", str(item.get("symbol") or "")) is None
            or not isinstance(item.get("name"), str)
            or not all(
                _finite_number(item.get(key), optional=True)
                for key in ("price", "return_1d_pct", "return_5d_pct")
            )
        ):
            return False
    return True


@dataclass(frozen=True)
class ObservationReportView:
    report_type: str
    label: str
    title: str
    source_market_date: str
    applicable_trading_date: str
    published_at: str
    summary: tuple[str, ...]
    warnings: tuple[str, ...]
    core: Mapping[str, Any]
    overnight_overlay: Mapping[str, Any] | None


def _observation_core(value: Any, source_market_date: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportWebError("Observation report core 不合法")
    if not isinstance(value.get("market_observation"), dict):
        raise ReportWebError("Observation report market 不合法")
    if not isinstance(value.get("data_quality"), dict):
        raise ReportWebError("Observation report quality 不合法")
    quality = dict(value["data_quality"])
    available_count = quality.get("available_count")
    symbol_count = quality.get("symbol_count")
    if (
        available_count is not None
        and symbol_count is not None
        and available_count != symbol_count
    ):
        raise ReportWebError("Observation report quality count 不一致")
    quality["symbol_count"] = (
        available_count if available_count is not None else symbol_count
    )
    normalized = dict(value)
    normalized["data_quality"] = quality
    for key in _CORE_LIST_KEYS:
        if not isinstance(value.get(key), list):
            raise ReportWebError(f"Observation report {key} 不合法")
    if not _valid_core_items(normalized):
        raise ReportWebError("Observation report content schema 不合法")
    try:
        statuses = validate_trading_status_observations(
            value.get("trading_status_observations"), source_market_date
        )
    except ValueError as exc:
        raise ReportWebError(str(exc)) from None
    return {
        "market_observation": dict(value["market_observation"]),
        "industry_observations": list(value["industry_observations"]),
        "heatmap": list(value["heatmap"]),
        "stock_events": list(value["stock_events"]),
        "trading_status_observations": statuses,
        "etf_observations": list(value["etf_observations"]),
        "daily_focus": list(value["daily_focus"]),
        "data_quality": quality,
    }


def _overnight_overlay(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportWebError("Pre-market overlay 不合法")
    available = value.get("available")
    unavailable = value.get("unavailable")
    if (
        value.get("status") not in {"risk_on", "risk_off", "mixed", "insufficient"}
        or not isinstance(value.get("message"), str)
        or not isinstance(value.get("as_of"), str)
        or not isinstance(available, list)
        or not isinstance(unavailable, list)
        or len(available) > 50
        or len(unavailable) > 50
    ):
        raise ReportWebError("Pre-market overlay schema 不合法")
    return {
        "status": value["status"],
        "message": value["message"],
        "available": list(available),
        "unavailable": list(unavailable),
        "as_of": value["as_of"],
    }


def build_observation_report_view(
    metadata: Any, *, expected_base_metadata_sha256: str | None = None
) -> ObservationReportView:
    if not isinstance(metadata, dict) or metadata.get("product_mode") != "observation":
        raise ReportWebError("報告不在 Observation 服務範圍")
    report_type = metadata.get("report_type")
    content = metadata.get("content")
    overlay = None
    source_market_date = str(metadata.get("source_market_date") or "")
    if report_type == "post_close":
        core = _observation_core(content, source_market_date)
    elif report_type == "pre_market":
        base_metadata_sha256 = (
            content.get("base_metadata_sha256")
            if isinstance(content, dict)
            else None
        )
        if (
            not isinstance(content, dict)
            or re.fullmatch(r"[0-9a-f]{64}", str(base_metadata_sha256 or ""))
            is None
            or expected_base_metadata_sha256 is None
            or base_metadata_sha256 != expected_base_metadata_sha256
        ):
            raise ReportWebError("Pre-market report base 不合法")
        core = _observation_core(content.get("core"), source_market_date)
        overlay = _overnight_overlay(content.get("overnight_overlay"))
    else:
        raise ReportWebError("Observation report type 不支援")
    return ObservationReportView(
        report_type=report_type,
        label=_REPORT_LABELS[report_type],
        title=str(metadata["title"]),
        source_market_date=str(metadata["source_market_date"]),
        applicable_trading_date=str(metadata["applicable_trading_date"]),
        published_at=str(metadata["published_at"]),
        summary=tuple(str(value) for value in metadata.get("summary") or ()),
        warnings=tuple(str(value) for value in metadata.get("warnings") or ()),
        core=core,
        overnight_overlay=overlay,
    )
