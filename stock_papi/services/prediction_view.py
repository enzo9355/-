"""Presentation-safe view of a verified five-session prediction."""

import datetime
import math


def _date(value):
    try:
        result = datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return result if result.isoformat() == value else None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def prediction_for(snapshot, market, symbol, observation_as_of):
    schema_version = snapshot.get("schema_version") if isinstance(snapshot, dict) else None
    research = schema_version == 2 and snapshot.get("validation_mode") == "research"
    promoted = schema_version == 1 and isinstance(snapshot.get("backtest_sha256"), str)
    if (
        not isinstance(snapshot, dict)
        or not (research or promoted)
        or snapshot.get("kind") != "absorb-five-session-predictions"
        or snapshot.get("market") != market
        or not isinstance(snapshot.get("entities"), dict)
    ):
        return None
    entity = snapshot["entities"].get(symbol)
    observed = _date(observation_as_of)
    as_of = _date(snapshot.get("as_of"))
    if not isinstance(entity, dict) or observed is None or as_of is None or as_of > observed:
        return None
    probability = _number(entity.get("up_probability"))
    current = _number(entity.get("current_price"))
    predicted_return = _number(entity.get("predicted_return_5d"))
    predicted_price = _number(entity.get("predicted_price"))
    change = _number(entity.get("predicted_change_pct"))
    target = _date(entity.get("target_session"))
    if (
        entity.get("symbol") != symbol
        or _date(entity.get("as_of")) != as_of
        or target is None
        or target <= as_of
        or None in (probability, current, predicted_return, predicted_price, change)
        or not 0 <= probability <= 1
        or current <= 0
        or predicted_price <= 0
        or not math.isclose(predicted_price, current * (1 + predicted_return), rel_tol=1e-9)
        or not math.isclose(change, predicted_return * 100, rel_tol=1e-9)
    ):
        return None
    status = "current" if as_of == observed else "previous"
    prefix = "前次 " if status == "previous" else ""
    result = {
        "status": status,
        "validation_mode": "research" if research else "promoted",
        "label": prefix + ("AI 五日研究推估" if research else "AI 五日預測"),
        "probability_label": "模型推估上漲機率（未校準）" if research else "五日上漲機率",
        "as_of": as_of.isoformat(),
        "target_session": target.isoformat(),
        "current_price": current,
        "probability_pct": round(probability * 100, 1),
        "predicted_return_5d": predicted_return,
        "predicted_price": predicted_price,
        "predicted_change_pct": change,
        "model_version": snapshot.get("model_version"),
        "backtest_sha256": None if research else snapshot.get("backtest_sha256"),
        "line": [
            {"time": as_of.isoformat(), "value": current},
            {"time": target.isoformat(), "value": predicted_price},
        ],
    }
    if entity.get("entity_type") == "market_index" and isinstance(entity.get("candles"), list):
        result["candles"] = list(entity["candles"])
    return result
