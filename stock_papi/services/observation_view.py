"""Public stock view built only from verified actual-market fields."""

import datetime
import json
import math

from stock_papi.integrations.market_data.tw_trading_status import (
    validate_status_evidence,
)


_STATUS_LABELS = {
    "officially_suspended": "停止買賣",
    "official_no_regular_trade": "當日無正常交易",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _date_text(value):
    text = str(value or "").split("T", 1)[0]
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _sum(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _risk_events(rows):
    latest = rows[-1]
    events = []
    close = _number(latest.get("Close"))
    if len(rows) > 1:
        previous = _number(rows[-2].get("Close"))
        if close is not None and previous is not None and previous > 0:
            change = (close / previous - 1) * 100
            if abs(change) >= 5:
                events.append(
                    "單日漲幅異常" if change > 0 else "單日跌幅異常"
                )
    volume = _number(latest.get("VOL_RATIO"))
    if volume is not None and volume >= 2:
        events.append("量能異常放大")
    elif volume is not None and volume <= 0.5:
        events.append("量能明顯收縮")
    rsi = _number(latest.get("RSI"))
    if rsi is not None and rsi >= 70:
        events.append("RSI 進入過熱區")
    elif rsi is not None and rsi <= 30:
        events.append("RSI 進入超賣區")
    closes = [
        _number(row.get("Close"))
        for row in rows[-20:]
        if _number(row.get("Close")) is not None
    ]
    if close is not None and len(closes) >= 20:
        if close >= max(closes):
            events.append("收盤創 20 日新高")
        elif close <= min(closes):
            events.append("收盤創 20 日新低")
    if (_number(latest.get("DATA_PRICE_WARNING")) or 0) > 0:
        events.append("資料來源價差警示")
    return events or ["未觸發額外風險事件"]


def _status_observation(snapshot, rows):
    target = _date_text(snapshot.get("target_market_date"))
    observed = _date_text(snapshot.get("observation_as_of"))
    latest_regular = _date_text(snapshot.get("latest_regular_price_date"))
    latest_row_date = _date_text(rows[-1].get("Date"))
    status = snapshot.get("observation_kind")
    if (
        snapshot.get("market") != "TW"
        or status not in _STATUS_LABELS
        or target is None
        or target != observed
        or latest_regular is None
        or latest_regular >= target
        or latest_regular != snapshot.get("as_of")
        or latest_row_date != latest_regular
    ):
        return None
    try:
        evidence = validate_status_evidence(
            snapshot.get("trading_status_evidence"),
            symbol=snapshot.get("symbol"),
            target_date=datetime.date.fromisoformat(target),
        )
    except ValueError:
        return None
    if evidence["status"] != status:
        return None
    result = {
        "code": str(snapshot.get("symbol") or ""),
        "name": str(snapshot.get("name") or snapshot.get("symbol") or ""),
        "market": "TW",
        "observation_kind": status,
        "status_label": _STATUS_LABELS[status],
        "observation_as_of": observed,
        "latest_regular_price_date": latest_regular,
        "evidence_sha256": evidence["evidence_sha256"],
        "quant_source": "已驗證官方交易狀態",
        "prediction_status": "AI 預測研究中",
    }
    last_close = _number(rows[-1].get("Close"))
    if last_close is not None and last_close > 0:
        result["last_regular_close"] = last_close
    return result


def build_stock_observation(snapshot):
    schema_version = snapshot.get("schema_version") if isinstance(snapshot, dict) else None
    if (
        not isinstance(snapshot, dict)
        or schema_version not in {1, 2}
        or snapshot.get("market") not in {"TW", "US"}
        or snapshot.get("sample_data") is True
        or not isinstance(snapshot.get("daily"), list)
        or not snapshot["daily"]
    ):
        return None
    rows = [row for row in snapshot["daily"] if isinstance(row, dict)]
    if len(rows) != len(snapshot["daily"]):
        return None
    if schema_version == 2 and snapshot.get("observation_kind") in _STATUS_LABELS:
        return _status_observation(snapshot, rows)
    latest = rows[-1]
    as_of = _date_text(latest.get("Date"))
    if as_of != snapshot.get("as_of"):
        return None
    if schema_version == 2 and (
        snapshot.get("market") not in {"TW", "US"}
        or snapshot.get("observation_kind") != "regular_price"
        or snapshot.get("trading_status_evidence") is not None
        or _date_text(snapshot.get("target_market_date")) != as_of
        or _date_text(snapshot.get("observation_as_of")) != as_of
        or _date_text(snapshot.get("latest_regular_price_date")) != as_of
    ):
        return None
    close = _number(latest.get("Close"))
    if close is None:
        return None
    ma20 = _number(latest.get("MA20"))
    ma60 = _number(latest.get("MA60"))
    if ma20 is not None and ma60 is not None and close >= ma20 >= ma60:
        trend = "above_ma20_ma60"
    elif ma20 is not None and close >= ma20:
        trend = "above_ma20"
    elif ma60 is not None and close < ma60:
        trend = "below_ma60"
    else:
        trend = "mixed"

    candles = []
    ma20_line = []
    for row in rows[-260:]:
        date_text = _date_text(row.get("Date"))
        row_close = _number(row.get("Close"))
        if date_text is None or row_close is None:
            continue
        open_value = _number(row.get("Open"))
        high = _number(row.get("High"))
        low = _number(row.get("Low"))
        open_value = row_close if open_value is None else open_value
        high = max(value for value in (open_value, high, low, row_close) if value is not None)
        low = min(value for value in (open_value, high, low, row_close) if value is not None)
        candles.append(
            {
                "time": date_text,
                "open": open_value,
                "high": high,
                "low": low,
                "close": row_close,
            }
        )
        moving_average = _number(row.get("MA20"))
        if moving_average is not None:
            ma20_line.append({"time": date_text, "value": moving_average})
    return {
        "code": str(snapshot.get("symbol") or ""),
        "name": str(snapshot.get("name") or snapshot.get("symbol") or ""),
        "market": snapshot["market"],
        "observation_kind": "regular_price",
        "observation_as_of": as_of,
        "latest_regular_price_date": as_of,
        "price": close,
        "as_of": as_of,
        "quant_source": "已驗證本地快照",
        "prediction_status": "AI 預測研究中",
        "trend_observation": trend,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": _number(latest.get("RSI")),
        "macd_osc": _number(latest.get("MACD_OSC")),
        "k": _number(latest.get("K")),
        "d": _number(latest.get("D")),
        "volume_ratio": _number(latest.get("VOL_RATIO")),
        "institution_net_ratio_pct": (
            None
            if _number(latest.get("INST_NET_RATIO")) is None
            else _number(latest.get("INST_NET_RATIO")) * 100
        ),
        "foreign_net_5": _sum(rows[-5:], "ForeignNet"),
        "foreign_net_20": _sum(rows[-20:], "ForeignNet"),
        "data_quality_warning": (
            (_number(latest.get("DATA_PRICE_WARNING")) or 0) > 0
        ),
        "risk_events": _risk_events(rows),
        "candles": json.dumps(
            candles, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        "ma20_line": json.dumps(
            ma20_line,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "news": [],
    }
