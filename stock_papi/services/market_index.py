"""Fail-closed reader for the official TWSE TAIEX daily OHLC series."""

from __future__ import annotations

import datetime
import math
import time

TWSE_INDEX_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
TWSE_INDEX_FIELDS = ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"]
MARKET_INDEX_CACHE = {}
MARKET_INDEX_CACHE_SECONDS = 3600
MAX_RESPONSE_BYTES = 2_000_000


def _month_start(value):
    return datetime.date(value.year, value.month, 1)


def _previous_month(value):
    return _month_start(value) - datetime.timedelta(days=1)


def _number(value):
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        raise ValueError("TWSE index value is invalid") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError("TWSE index value is invalid")
    return result


def _roc_date(value):
    parts = str(value or "").strip().split("/")
    if len(parts) != 3:
        raise ValueError("TWSE index date is invalid")
    try:
        return datetime.date(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
    except (TypeError, ValueError):
        raise ValueError("TWSE index date is invalid") from None


def parse_twse_index_month(document, expected_month):
    """Validate one official monthly response and return normalized candles."""
    expected_month = _month_start(expected_month)
    if (
        not isinstance(document, dict)
        or document.get("stat") != "OK"
        or document.get("date") != expected_month.strftime("%Y%m01")
        or document.get("fields") != TWSE_INDEX_FIELDS
        or "發行量加權股價指數歷史資料" not in str(document.get("title") or "")
        or not isinstance(document.get("data"), list)
    ):
        raise ValueError("TWSE index schema is invalid")
    candles = []
    seen = set()
    for source in document["data"]:
        if not isinstance(source, list) or len(source) != 5:
            raise ValueError("TWSE index row schema is invalid")
        date = _roc_date(source[0])
        if _month_start(date) != expected_month or date in seen:
            raise ValueError("TWSE index date is invalid")
        open_value, high, low, close = (_number(item) for item in source[1:])
        if high < max(open_value, close, low) or low > min(open_value, close, high):
            raise ValueError("TWSE index OHLC relationship is invalid")
        seen.add(date)
        candles.append(
            {
                "time": date.isoformat(),
                "open": open_value,
                "high": high,
                "low": low,
                "close": close,
            }
        )
    if not candles:
        raise ValueError("TWSE index month is empty")
    return sorted(candles, key=lambda item: item["time"])


def _return_pct(candles, periods):
    if len(candles) <= periods:
        return None
    current = candles[-1]["close"]
    previous = candles[-1 - periods]["close"]
    return round((current / previous - 1) * 100, 2)


def fetch_twse_index_snapshot(
    target_date,
    *,
    http_get,
    cache=MARKET_INDEX_CACHE,
    now=time.time,
):
    """Return TAIEX OHLC only when the official series reaches target_date."""
    if not isinstance(target_date, datetime.date) or isinstance(
        target_date, datetime.datetime
    ):
        return None
    key = target_date.isoformat()
    cached = cache.get(key)
    timestamp = now()
    if cached and timestamp - cached[1] < MARKET_INDEX_CACHE_SECONDS:
        return cached[0]

    month = _month_start(target_date)
    candles = []
    try:
        for _ in range(12):
            response = http_get(
                TWSE_INDEX_URL,
                params={"date": month.strftime("%Y%m01"), "response": "json"},
                headers={"User-Agent": "ABSORB/1.0"},
                timeout=12,
            )
            content = bytes(getattr(response, "content", b""))
            if (
                int(getattr(response, "status_code", 0)) != 200
                or not content
                or len(content) > MAX_RESPONSE_BYTES
            ):
                return None
            candles.extend(parse_twse_index_month(response.json(), month))
            month = _month_start(_previous_month(month))
    except Exception:
        return None

    candles = sorted(
        (item for item in candles if item["time"] <= target_date.isoformat()),
        key=lambda item: item["time"],
    )
    if (
        len(candles) < 40
        or candles[-1]["time"] != target_date.isoformat()
        or len({item["time"] for item in candles}) != len(candles)
    ):
        return None
    candles = candles[-90:]
    ma20 = []
    closes = [item["close"] for item in candles]
    for index in range(19, len(candles)):
        ma20.append(
            {
                "time": candles[index]["time"],
                "value": round(sum(closes[index - 19 : index + 1]) / 20, 2),
            }
        )
    latest = candles[-1]
    previous_close = candles[-2]["close"]
    change = latest["close"] - previous_close
    result = {
        "symbol": "TAIEX",
        "name": "加權指數",
        "source": "臺灣證券交易所",
        "as_of": latest["time"],
        "price": latest["close"],
        "change": round(change, 2),
        "change_pct": round(change / previous_close * 100, 2),
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "candles": candles,
        "ma20": ma20,
        "returns": {
            "1d": _return_pct(candles, 1),
            "5d": _return_pct(candles, 5),
            "20d": _return_pct(candles, 20),
            "60d": _return_pct(candles, 60),
        },
    }
    cache[key] = (result, timestamp)
    return result
