"""Market-map and sector-signal payload services."""

import datetime

from stock_papi.shared.formatting import clamp as _clamp
from stock_papi.shared.formatting import safe_float as _safe_float
from stock_papi.integrations.market_data.tw_security_master import is_taiwan_symbol


def sector_signal_score(data):
    bt = data.get("bt") or {}
    foreign = data.get("foreign_flow") or {}
    prob = _safe_float(data.get("prob"))
    strat_bonus = _clamp(_safe_float(bt.get("strat_cum")), -20.0, 20.0) * 0.35
    foreign_bonus = _clamp(
        _safe_float(foreign.get("net_5")) / 1000.0, -5.0, 5.0
    )
    drawdown_penalty = min(abs(_safe_float(bt.get("mdd"))), 30.0) * 0.15
    return round(prob + strat_bonus + foreign_bonus - drawdown_penalty, 2)


def sector_candidates(category, codes, limit=20, activity=None):
    selected = []
    seen = set()
    for code in codes:
        code = str(code).strip()
        if code in seen or not is_taiwan_symbol(code):
            continue
        if category != "ETF專區" and code.startswith("00"):
            continue
        selected.append(code)
        seen.add(code)
    if activity:
        selected.sort(
            key=lambda code: (
                _safe_float((activity.get(code) or {}).get("trade_value")),
                _safe_float((activity.get(code) or {}).get("trade_volume")),
            ),
            reverse=True,
        )
    return selected[:limit]


def sector_signal_item(code, data, *, get_stock_name):
    if not data or not isinstance(data.get("as_of"), str):
        return None
    bt = data.get("bt") or {}
    foreign = data.get("foreign_flow") or {}
    name = (
        get_stock_name(code)
        if is_taiwan_symbol(code)
        else data.get("name") or get_stock_name(code)
    )
    return {
        "code": code,
        "name": name,
        "price": _safe_float(data.get("price")),
        "prob": int(round(_safe_float(data.get("prob")))),
        "trend": data.get("trend") or "中性",
        "score": sector_signal_score(data),
        "strat_cum": _safe_float(bt.get("strat_cum")),
        "mdd": _safe_float(bt.get("mdd")),
        "foreign_net_5": _safe_float(foreign.get("net_5")),
        "as_of": data["as_of"],
    }


def build_sector_signal_snapshot(
    market_map,
    analyze_fn,
    *,
    now=None,
    activity=None,
    scan_limit=20,
    display_limit=10,
    get_stock_name,
):
    now = now or datetime.datetime.utcnow()
    sectors = {}
    dates = []
    candidate_limit = max(scan_limit, display_limit * 6)
    for category, codes in market_map.items():
        items = []
        for code in sector_candidates(
            category, codes, limit=candidate_limit, activity=activity
        ):
            try:
                item = sector_signal_item(
                    code, analyze_fn(code), get_stock_name=get_stock_name
                )
            except Exception:
                item = None
            if item:
                items.append(item)
                dates.append(item["as_of"])
                if len(items) >= display_limit:
                    break
        items.sort(key=lambda item: item["score"], reverse=True)
        sectors[category] = items
    return {
        "as_of": max(dates) if dates else now.date().isoformat(),
        "generated_at": now.replace(microsecond=0).isoformat() + "Z",
        "sectors": sectors,
    }


def build_market_map(codes, theme_sectors):
    market = {"全市場": [], "ETF專區": []}
    for theme in theme_sectors:
        market[theme] = []
    from stock_papi.shared.symbol import get_instrument_type
    for code, info in codes.items():
        if not is_taiwan_symbol(code):
            continue
        if len(code) == 6 and get_instrument_type(code) != "ETF":
            continue
        group = getattr(info, "group", None) or getattr(info, "type", None)
        if group and isinstance(group, str) and group.strip():
            market["全市場"].append(code)
            if get_instrument_type(code) == "ETF":
                market["ETF專區"].append(code)
            for theme, names in theme_sectors.items():
                if info.name in names and code not in market[theme]:
                    market[theme].append(code)
    return {key: value for key, value in market.items() if value}


def find_industry_peers(code, market_map, limit=5):
    selected = str(code).upper()
    for category, codes in market_map.items():
        if category in {"全市場", "ETF專區"}:
            continue
        normalized = [str(item).upper() for item in codes]
        if selected in normalized:
            return {
                "category": category,
                "codes": [item for item in normalized if item != selected][:limit],
            }
    return {"category": "", "codes": []}
