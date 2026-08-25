"""US presentation helpers - centralized view-model mapping for US pages."""
from __future__ import annotations
from typing import Any

MARKET_OBSERVATION_LABELS = {
    "advancing_count": "上漲家數",
    "declining_count": "下跌家數",
    "unchanged_count": "平盤家數",
    "ma20_breadth_pct": "站上 MA20 比例",
    "ma60_breadth_pct": "站上 MA60 比例",
    "new_high_20d_count": "20 日新高家數",
    "new_low_20d_count": "20 日新低家數",
    "realized_volatility_20d_pct": "20 日已實現波動率",
    "return_1d_pct": "單日報酬",
    "return_5d_pct": "5 日報酬",
    "return_20d_pct": "20 日報酬",
    "return_60d_pct": "60 日報酬",
    "median_volume_ratio": "中位量比",
    "risk_state": "風險狀態",
    "market_state": "市場狀態",
}

RISK_STATE_LABELS = {
    "normal": "一般",
    "cautious": "謹慎",
    "elevated": "升高",
    "defensive": "防守",
}

INDUSTRY_FIELD_LABELS = {
    "relative_return_5d_pct": "5 日相對大盤",
    "coverage": "資料覆蓋率",
    "available_count": "有效樣本",
    "component_count": "成分數",
    "rank": "排名",
    "name": "產業",
}

VALIDATION_GATE_LABELS = {
    "ranking": "模型排名",
    "calibration": "模型校準",
    "quality": "品質 Gate",
    "transaction_value": "交易價值驗證",
    "promotion": "模型 Promotion",
}

GATE_UNAVAILABLE_TEXT = "尚未提供驗證資料"
GATE_BLOCKED_TEXT = "暫不發布"


def format_market_value(key: str, value: Any) -> str:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"none", "null", ""}):
        return "尚無已驗證資料"
    if isinstance(value, bool):
        return "尚無已驗證資料"
    if key == "risk_state" and isinstance(value, str):
        return RISK_STATE_LABELS.get(value, value)
    if key.endswith("_pct") and isinstance(value, (int, float)):
        return f"{value:+.2f}%" if "return" in key else f"{value:.1f}%"
    if key == "median_volume_ratio" and isinstance(value, (int, float)):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_us_market_observation_view(market_section: dict) -> list[dict[str, Any]]:
    if not isinstance(market_section, dict) or market_section.get("status") != "available":
        return []
    data = market_section.get("data") or {}
    if not isinstance(data, dict):
        return []
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)) and not isinstance(value, str):
            continue
        label = MARKET_OBSERVATION_LABELS.get(key, key)
        formatted = format_market_value(key, value)
        rows.append({"key": key, "label": label, "value": value, "formatted": formatted, "available": formatted != "尚無已驗證資料"})
    # order by label priority
    order = list(MARKET_OBSERVATION_LABELS.keys())
    rows.sort(key=lambda x: order.index(x["key"]) if x["key"] in order else 999)
    return rows


def build_us_industries_view(industries_section: dict) -> list[dict[str, Any]]:
    if not isinstance(industries_section, dict) or industries_section.get("status") != "available":
        return []
    data = industries_section.get("data") or {}
    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return []
    result = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "未命名"
        result.append({
            "name": name,
            "rank": item.get("rank"),
            "relative_return_5d_pct": item.get("relative_return_5d_pct"),
            "coverage": item.get("coverage"),
            "available_count": item.get("available_count"),
            "component_count": item.get("component_count"),
        })
    return result


def format_percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "尚無已驗證資料"
    return f"{value:+.2f}%"


def format_coverage(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "尚無已驗證資料"
    return f"{value*100:.0f}%"


def validation_gate_display(gate_name: str, gate_value: str) -> tuple[str, str]:
    label = VALIDATION_GATE_LABELS.get(gate_name, gate_name)
    if gate_value == "BLOCKED":
        return label, GATE_BLOCKED_TEXT
    if gate_value == "UNAVAILABLE":
        return label, GATE_UNAVAILABLE_TEXT
    return label, gate_value


def stock_display_name(name: str, symbol: str) -> str:
    n = (name or "").strip()
    s = (symbol or "").strip().upper()
    if not n or n.upper() == s:
        return s
    return f"{n} · {s}"
