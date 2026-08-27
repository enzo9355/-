"""Build deterministic market observations without reading model outputs."""

import datetime
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

from stock_papi.integrations.market_data.tw_trading_status import (
    validate_status_evidence,
)


MIN_SOURCE_COVERAGE = 0.95
MAX_SOURCE_AGE_DAYS = 7
TRADING_STATUS_LABELS = {
    "officially_suspended": "停止買賣",
    "official_no_regular_trade": "當日無正常交易",
}
_STATUS_OBSERVATION_KEYS = frozenset(
    {
        "symbol",
        "name",
        "status",
        "label",
        "observation_as_of",
        "latest_regular_price_date",
        "evidence_sha256",
    }
)
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "ai_p",
        "prob",
        "probability",
        "direction_score",
        "score",
        "recommendation",
        "top_picks",
        "model_version",
        "backtest_version",
    }
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def validate_trading_status_observations(value, observation_as_of):
    """Validate the compact, hash-bound status view shared by report surfaces."""
    if value is None:
        return []
    try:
        source_date = datetime.date.fromisoformat(str(observation_as_of))
    except (TypeError, ValueError):
        raise ValueError("trading status observation date is invalid") from None
    if not isinstance(value, list) or len(value) > 10_000:
        raise ValueError("trading status observations schema is invalid")
    normalized = []
    seen = set()
    for item in value:
        keys = set(item) if isinstance(item, dict) else set()
        if keys not in (
            set(_STATUS_OBSERVATION_KEYS),
            set(_STATUS_OBSERVATION_KEYS) | {"last_regular_close"},
        ):
            raise ValueError("trading status observations schema is invalid")
        try:
            observed = datetime.date.fromisoformat(str(item["observation_as_of"]))
            latest = datetime.date.fromisoformat(
                str(item["latest_regular_price_date"])
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("trading status observations schema is invalid") from None
        symbol = str(item.get("symbol") or "")
        status = item.get("status")
        last_close = item.get("last_regular_close")
        if (
            re.fullmatch(r"[A-Z0-9.-]{1,16}", symbol) is None
            or symbol in seen
            or not isinstance(item.get("name"), str)
            or not item["name"].strip()
            or status not in TRADING_STATUS_LABELS
            or item.get("label") != TRADING_STATUS_LABELS[status]
            or observed != source_date
            or latest >= observed
            or re.fullmatch(
                r"[0-9a-f]{64}", str(item.get("evidence_sha256") or "")
            )
            is None
            or (
                "last_regular_close" in item
                and (_number(last_close) is None or float(last_close) <= 0)
            )
        ):
            raise ValueError("trading status observations schema is invalid")
        seen.add(symbol)
        normalized.append(dict(item))
    return normalized


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _rounded(value, digits=2):
    return None if value is None else round(float(value), digits)


def _return_pct(stock, periods):
    if len(stock.daily) <= periods:
        return None
    current = _number(stock.daily[-1].get("Close"))
    previous = _number(stock.daily[-1 - periods].get("Close"))
    if current is None or previous is None or previous <= 0:
        return None
    return (current / previous - 1.0) * 100.0


def _finite_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation source must contain finite JSON")
        return
    if isinstance(value, list):
        for item in value:
            _finite_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("observation source keys must be strings")
            _finite_json(item)
        return
    raise ValueError("observation source contains unsupported JSON")


def _validate_source(source, today):
    manifest = getattr(source, "manifest", None)
    stocks = getattr(source, "stocks", None)
    if (
        manifest is None
        or manifest.schema_version not in {2, 3, 4}
        or manifest.market not in ("TW", "US")
        or not isinstance(manifest.market_as_of, datetime.date)
        or not isinstance(stocks, list)
        or not stocks
        or any(
            type(value) is not int or value < 0
            for value in (
                manifest.universe_count,
                manifest.symbol_count,
                manifest.failure_count,
            )
        )
        or manifest.symbol_count != len(stocks)
        or type(manifest.coverage) not in (int, float)
        or not math.isfinite(manifest.coverage)
        or manifest.coverage <= MIN_SOURCE_COVERAGE
        or type(manifest.failure_rate) not in (int, float)
        or not math.isfinite(manifest.failure_rate)
        or not 0 <= manifest.failure_rate < 0.05
    ):
        raise ValueError("observation source coverage or schema is invalid")
    target_date = manifest.market_as_of
    if manifest.schema_version in (3, 4):
        regular_count = manifest.regular_price_symbol_count
        status_count = (
            manifest.expected_non_price_symbol_count
            if manifest.schema_version == 3
            else manifest.verified_non_price_symbol_count
        )
        operational_count = manifest.operational_failure_count
        denominator = manifest.regular_price_denominator
        expected = manifest.expected_non_price_symbols
        if (
            manifest.target_market_date != target_date
            or manifest.observation_as_of != target_date
            or any(
                type(value) is not int or value < 0
                for value in (
                    regular_count,
                    status_count,
                    operational_count,
                    denominator,
                )
            )
            or not isinstance(expected, dict)
            or len(expected) != status_count
            or regular_count + status_count != manifest.symbol_count
            or operational_count != manifest.operational_failure_count
            or denominator <= 0
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (
                    manifest.regular_price_coverage,
                    manifest.observation_coverage,
                )
            )
            or not math.isclose(
                manifest.regular_price_coverage,
                regular_count / denominator,
            )
            or not math.isclose(
                manifest.observation_coverage,
                manifest.symbol_count / manifest.universe_count,
            )
            or not math.isclose(manifest.coverage, manifest.observation_coverage)
        ):
            raise ValueError("observation source counts are invalid")
        if manifest.schema_version == 3:
            if (
                operational_count != manifest.failure_count
                or manifest.universe_count
                != manifest.symbol_count + operational_count
                or denominator != manifest.universe_count - status_count
            ):
                raise ValueError("observation source v3 counts are invalid")
        else:
            unavailable_count = manifest.unavailable_count
            unavailable = manifest.unavailable_symbols
            if (
                operational_count != 0
                or unavailable_count is None
                or unavailable is None
                or type(unavailable_count) is not int
                or unavailable_count < 0
                or not isinstance(unavailable, list)
                or len(unavailable) != unavailable_count
                or manifest.universe_count
                != manifest.symbol_count + unavailable_count
                or denominator != manifest.symbol_count - status_count
                or manifest.symbol_count * 100 <= manifest.universe_count * 95
            ):
                raise ValueError("observation source v4 counts are invalid")
    age = today - manifest.market_as_of
    if not 0 <= age.days <= MAX_SOURCE_AGE_DAYS:
        raise ValueError("observation source is stale or from the future")
    if (
        re.fullmatch(
            r"manifests/(?:TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json",
            str(manifest.manifest_path),
        )
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(manifest.manifest_sha256)) is None
    ):
        raise ValueError("observation source manifest identity is invalid")
    seen = set()
    regular_symbols = set()
    status_symbols = set()
    for stock in stocks:
        if (
            stock.sample_data
            or stock.market not in ("TW", "US")
            or stock.symbol in seen
            or not isinstance(stock.daily, list)
            or not stock.daily
        ):
            label = "sample" if stock.sample_data else "stock"
            raise ValueError(f"observation {label} source is invalid")
        seen.add(stock.symbol)
        for row in stock.daily:
            if not isinstance(row, dict):
                raise ValueError("observation stock rows are invalid")
            _finite_json(row)
        latest_row_date = str(stock.daily[-1].get("Date") or "").split("T", 1)[0]
        if manifest.schema_version == 2:
            if (
                stock.as_of != target_date
                or latest_row_date != target_date.isoformat()
                or stock.observation_kind != "regular_price"
                or stock.trading_status_evidence is not None
            ):
                raise ValueError("observation stock date is invalid")
            regular_symbols.add(stock.symbol)
            continue
        if (
            stock.observation_as_of != target_date
            or stock.latest_regular_price_date != stock.as_of
            or latest_row_date != stock.as_of.isoformat()
        ):
            raise ValueError("observation stock date is invalid")
        if stock.observation_kind == "regular_price":
            if stock.as_of != target_date or stock.trading_status_evidence is not None:
                raise ValueError("observation regular price source is invalid")
            regular_symbols.add(stock.symbol)
            continue
        if (
            stock.observation_kind not in TRADING_STATUS_LABELS
            or stock.as_of >= target_date
        ):
            raise ValueError("observation status source is invalid")
        try:
            evidence = validate_status_evidence(
                stock.trading_status_evidence,
                symbol=stock.symbol,
                target_date=target_date,
            )
        except ValueError:
            raise ValueError("observation status evidence is invalid") from None
        expected = manifest.expected_non_price_symbols.get(stock.symbol)
        if (
            not isinstance(expected, dict)
            or evidence["status"] != stock.observation_kind
            or expected.get("status") != stock.observation_kind
            or expected.get("evidence_sha256") != evidence["evidence_sha256"]
            or expected.get("artifact_sha256") != stock.sha256
            or expected.get("latest_regular_price_date")
            != stock.as_of.isoformat()
        ):
            raise ValueError("observation status binding is invalid")
        status_symbols.add(stock.symbol)
    if manifest.schema_version in (3, 4) and (
        len(regular_symbols) != manifest.regular_price_symbol_count
        or len(status_symbols)
        != (
            manifest.expected_non_price_symbol_count
            if manifest.schema_version == 3
            else manifest.verified_non_price_symbol_count
        )
        or status_symbols != set(manifest.expected_non_price_symbols)
    ):
        raise ValueError("observation source partition is invalid")


def _trading_status_observations(stocks):
    observations = []
    for stock in sorted(stocks, key=lambda item: item.symbol):
        evidence = stock.trading_status_evidence
        item = {
            "symbol": stock.symbol,
            "name": stock.name,
            "status": stock.observation_kind,
            "label": TRADING_STATUS_LABELS[stock.observation_kind],
            "observation_as_of": stock.observation_as_of.isoformat(),
            "latest_regular_price_date": stock.as_of.isoformat(),
            "evidence_sha256": evidence["evidence_sha256"],
        }
        close = _number(stock.daily[-1].get("Close"))
        if close is not None and close > 0:
            item["last_regular_close"] = close
        observations.append(item)
    return observations


def _market_daily_returns(stocks):
    by_date = defaultdict(list)
    for stock in stocks:
        for previous, current in zip(stock.daily, stock.daily[1:]):
            previous_close = _number(previous.get("Close"))
            current_close = _number(current.get("Close"))
            date_text = str(current.get("Date") or "").split("T", 1)[0]
            if (
                previous_close is not None
                and previous_close > 0
                and current_close is not None
                and date_text
            ):
                by_date[date_text].append(current_close / previous_close - 1.0)
    return [
        statistics.median(by_date[date_text])
        for date_text in sorted(by_date)[-20:]
        if by_date[date_text]
    ]


def _market_observation(stocks):
    returns = {
        f"return_{periods}d_pct": _rounded(
            _median([_return_pct(stock, periods) for stock in stocks])
        )
        for periods in (1, 5, 20, 60)
    }
    advancing = declining = unchanged = 0
    above_ma20 = []
    above_ma60 = []
    new_highs = new_lows = 0
    volume_ratios = []
    institution_ratios = []
    for stock in stocks:
        one_day = _return_pct(stock, 1)
        if one_day is not None:
            if one_day > 0:
                advancing += 1
            elif one_day < 0:
                declining += 1
            else:
                unchanged += 1
        latest = stock.daily[-1]
        close = _number(latest.get("Close"))
        ma20 = _number(latest.get("MA20"))
        ma60 = _number(latest.get("MA60"))
        if close is not None and ma20 is not None:
            above_ma20.append(close >= ma20)
        if close is not None and ma60 is not None:
            above_ma60.append(close >= ma60)
        closes = [
            _number(row.get("Close"))
            for row in stock.daily[-20:]
            if _number(row.get("Close")) is not None
        ]
        if close is not None and len(closes) >= 20:
            if close >= max(closes):
                new_highs += 1
            if close <= min(closes):
                new_lows += 1
        volume = _number(latest.get("VOL_RATIO"))
        institution = _number(latest.get("INST_NET_RATIO"))
        if volume is not None:
            volume_ratios.append(volume)
        if institution is not None:
            institution_ratios.append(institution)

    daily_returns = _market_daily_returns(stocks)
    volatility = (
        statistics.pstdev(daily_returns) * math.sqrt(252) * 100
        if len(daily_returns) >= 5
        else None
    )
    if declining > advancing and (
        new_lows > new_highs or (volatility is not None and volatility >= 25)
    ):
        risk_state = "elevated"
    elif declining > advancing or (volatility is not None and volatility >= 20):
        risk_state = "cautious"
    else:
        risk_state = "normal"
    return {
        **returns,
        "advancing_count": advancing,
        "declining_count": declining,
        "unchanged_count": unchanged,
        "ma20_breadth_pct": _rounded(
            _mean([100.0 if value else 0.0 for value in above_ma20])
        ),
        "ma60_breadth_pct": _rounded(
            _mean([100.0 if value else 0.0 for value in above_ma60])
        ),
        "new_high_20d_count": new_highs,
        "new_low_20d_count": new_lows,
        "median_volume_ratio": _rounded(_median(volume_ratios)),
        "median_institution_net_ratio_pct": _rounded(
            None
            if not institution_ratios
            else statistics.median(institution_ratios) * 100
        ),
        "realized_volatility_20d_pct": _rounded(volatility),
        "risk_state": risk_state,
    }


def _phase(relative_5d, relative_20d):
    if relative_5d is None or relative_20d is None:
        return "insufficient"
    if relative_5d > 0 and relative_20d > 0:
        return "strong"
    if relative_5d > 0:
        return "strengthening"
    if relative_20d > 0:
        return "weakening"
    return "weak"


def _attention_companies(stocks):
    items = []
    for stock in stocks:
        return_5d = _return_pct(stock, 5)
        if return_5d is None:
            continue
        latest = stock.daily[-1]
        close = _number(latest.get("Close"))
        ma20 = _number(latest.get("MA20"))
        volume_ratio = _number(latest.get("VOL_RATIO"))
        items.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "price": _rounded(close),
                "return_5d_pct": _rounded(return_5d),
                "above_ma20": close is not None and ma20 is not None and close >= ma20,
                "volume_ratio": _rounded(volume_ratio),
                "as_of": stock.as_of.isoformat(),
            }
        )
    items.sort(
        key=lambda item: (
            -item["return_5d_pct"],
            not item["above_ma20"],
            item["volume_ratio"] is None,
            -(item["volume_ratio"] or 0),
            item["symbol"],
        )
    )
    return items[:5]


def _industry_observations(industry_map, stock_by_symbol, market):
    observations = []
    for name, raw_symbols in industry_map.items():
        if name in {"全市場", "ETF專區"}:
            continue
        symbols = sorted({str(symbol) for symbol in raw_symbols})
        stocks = [
            stock_by_symbol[symbol]
            for symbol in symbols
            if symbol in stock_by_symbol
        ]
        if not symbols:
            continue
        return_1d = _mean([_return_pct(stock, 1) for stock in stocks])
        return_5d = _mean([_return_pct(stock, 5) for stock in stocks])
        return_20d = _mean([_return_pct(stock, 20) for stock in stocks])
        relative_5d = (
            return_5d - market["return_5d_pct"]
            if return_5d is not None and market["return_5d_pct"] is not None
            else None
        )
        relative_20d = (
            return_20d - market["return_20d_pct"]
            if return_20d is not None and market["return_20d_pct"] is not None
            else None
        )
        advancing = [
            _return_pct(stock, 1) > 0
            for stock in stocks
            if _return_pct(stock, 1) is not None
        ]
        above_ma20 = []
        volume_ratios = []
        institution_ratios = []
        for stock in stocks:
            latest = stock.daily[-1]
            close = _number(latest.get("Close"))
            ma20 = _number(latest.get("MA20"))
            if close is not None and ma20 is not None:
                above_ma20.append(close >= ma20)
            volume = _number(latest.get("VOL_RATIO"))
            institution = _number(latest.get("INST_NET_RATIO"))
            if volume is not None:
                volume_ratios.append(volume)
            if institution is not None:
                institution_ratios.append(institution)
        observations.append(
            {
                "name": str(name),
                "component_count": len(symbols),
                "available_count": len(stocks),
                "coverage": _rounded(len(stocks) / len(symbols), 4),
                "return_1d_pct": _rounded(return_1d),
                "return_5d_pct": _rounded(return_5d),
                "return_20d_pct": _rounded(return_20d),
                "relative_return_5d_pct": _rounded(relative_5d),
                "relative_return_20d_pct": _rounded(relative_20d),
                "advancing_ratio_pct": _rounded(
                    _mean([100.0 if value else 0.0 for value in advancing])
                ),
                "ma20_breadth_pct": _rounded(
                    _mean([100.0 if value else 0.0 for value in above_ma20])
                ),
                "median_volume_ratio": _rounded(_median(volume_ratios)),
                "median_institution_net_ratio_pct": _rounded(
                    None
                    if not institution_ratios
                    else statistics.median(institution_ratios) * 100
                ),
                "phase": _phase(relative_5d, relative_20d),
                "ranking_basis": "actual_momentum",
                "attention_companies": _attention_companies(stocks),
            }
        )
    observations.sort(
        key=lambda item: (
            item["relative_return_5d_pct"] is None,
            -(
                item["relative_return_5d_pct"]
                if item["relative_return_5d_pct"] is not None
                else 0
            ),
            item["name"],
        )
    )
    for position, item in enumerate(observations, 1):
        item["display_order"] = position
    return observations


def _stock_events(stocks):
    events = []
    severity_order = {"high": 0, "medium": 1, "low": 2}

    def add(stock, event_type, severity, metric_value, unit, observation):
        events.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "event_type": event_type,
                "severity": severity,
                "metric_value": _rounded(metric_value),
                "unit": unit,
                "observation": observation,
                "as_of": stock.as_of.isoformat(),
            }
        )

    for stock in sorted(stocks, key=lambda item: item.symbol):
        latest = stock.daily[-1]
        one_day = _return_pct(stock, 1)
        if one_day is not None and abs(one_day) >= 5:
            add(
                stock,
                "price_move",
                "high",
                one_day,
                "pct",
                "單日漲幅異常" if one_day > 0 else "單日跌幅異常",
            )
        volume = _number(latest.get("VOL_RATIO"))
        if volume is not None and volume >= 2:
            add(stock, "volume_surge", "medium", volume, "ratio", "量能異常放大")
        elif volume is not None and volume <= 0.5:
            add(stock, "volume_dry_up", "low", volume, "ratio", "量能明顯收縮")
        rsi = _number(latest.get("RSI"))
        if rsi is not None and rsi >= 70:
            add(stock, "rsi_overbought", "medium", rsi, "index", "RSI 進入過熱區")
        elif rsi is not None and rsi <= 30:
            add(stock, "rsi_oversold", "medium", rsi, "index", "RSI 進入超賣區")
        close = _number(latest.get("Close"))
        closes = [
            _number(row.get("Close"))
            for row in stock.daily[-20:]
            if _number(row.get("Close")) is not None
        ]
        if close is not None and len(closes) >= 20:
            if close >= max(closes):
                add(stock, "new_high_20d", "medium", close, "price", "收盤創 20 日新高")
            elif close <= min(closes):
                add(stock, "new_low_20d", "medium", close, "price", "收盤創 20 日新低")
        institution = _number(latest.get("INST_NET_RATIO"))
        if institution is not None and abs(institution) >= 0.02:
            add(
                stock,
                "institution_flow",
                "medium",
                institution * 100,
                "pct",
                "機構淨流入偏高" if institution > 0 else "機構淨流出偏高",
            )
        if (_number(latest.get("DATA_PRICE_WARNING")) or 0) > 0:
            add(stock, "data_warning", "high", 1, "flag", "資料來源價差警示")
    events.sort(
        key=lambda item: (
            severity_order[item["severity"]],
            -abs(item["metric_value"] or 0),
            item["symbol"],
            item["event_type"],
        )
    )
    return events[:30]


def _etf_observations(stocks):
    observations = []
    for stock in sorted(stocks, key=lambda item: item.symbol):
        latest = stock.daily[-1]
        close = _number(latest.get("Close"))
        ma20 = _number(latest.get("MA20"))
        ma60 = _number(latest.get("MA60"))
        if close is None:
            trend = "insufficient"
        elif ma20 is not None and ma60 is not None and close >= ma20 >= ma60:
            trend = "above_ma20_ma60"
        elif ma20 is not None and close >= ma20:
            trend = "above_ma20"
        elif ma60 is not None and close < ma60:
            trend = "below_ma60"
        else:
            trend = "mixed"
        observations.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "price": _rounded(close),
                "return_1d_pct": _rounded(_return_pct(stock, 1)),
                "return_5d_pct": _rounded(_return_pct(stock, 5)),
                "return_20d_pct": _rounded(_return_pct(stock, 20)),
                "volume_ratio": _rounded(_number(latest.get("VOL_RATIO"))),
                "trend_observation": trend,
                "as_of": stock.as_of.isoformat(),
            }
        )
    return observations


def _heatmap(industries):
    items = []
    for industry in industries:
        value = industry["relative_return_5d_pct"]
        if value is None:
            continue
        items.append(
            {
                "name": industry["name"],
                "metric_name": "relative_return_5d_pct",
                "metric_value_pct": value,
                "available_count": industry["available_count"],
                "coverage": industry["coverage"],
                "tone": "hot" if value >= 2 else "cold" if value <= -2 else "steady",
            }
        )
    return items


def _daily_focus(market, industries, events):
    focus = [f"市場風險狀態：{market['risk_state']}"]
    if industries:
        strongest = industries[0]
        value = strongest["relative_return_5d_pct"]
        if value is not None:
            focus.append(f"{strongest['name']} 5 日相對大盤 {value:+.2f}%")
    if events:
        focus.append(
            f"{events[0]['name']}（{events[0]['symbol']}）：{events[0]['observation']}"
        )
    return focus


def validate_observation_dashboard(document):
    try:
        generated = datetime.datetime.fromisoformat(
            str(document["generated_at"]).replace("Z", "+00:00")
        )
        datetime.date.fromisoformat(str(document["observation_as_of"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observation dashboard dates are invalid") from exc
    if (
        document.get("schema_version") != 2
        or document.get("kind") != "absorb-observation-dashboard"
        or document.get("product_mode") != "observation"
        or document.get("market") not in ("TW", "US")
        or generated.tzinfo is None
        or generated.utcoffset() is None
        or re.fullmatch(
            r"quant/v1/manifests/(?:TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json",
            str(document.get("source_manifest") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(document.get("source_manifest_sha256") or ""),
        )
        is None
        or not isinstance(document.get("prediction_capability"), dict)
        or not isinstance(document.get("market_observation"), dict)
        or not isinstance(document.get("industry_observations"), list)
        or not isinstance(document.get("heatmap"), list)
        or not isinstance(document.get("stock_events"), list)
        or not isinstance(document.get("etf_observations"), list)
        or not isinstance(document.get("daily_focus"), list)
        or not isinstance(document.get("data_quality"), dict)
        or not isinstance(document.get("gates"), dict)
    ):
        raise ValueError("observation dashboard schema is invalid")
    capability = document["prediction_capability"]
    if (
        capability.get("mode") != "research"
        or capability.get("observation_enabled") is not True
        or capability.get("probability_allowed") is not False
        or capability.get("ranking_allowed") is not False
        or capability.get("strong_action_allowed") is not False
        or capability.get("performance_endorsement_allowed") is not False
    ):
        raise ValueError("observation dashboard capability is invalid")
    statuses = validate_trading_status_observations(
        document.get("trading_status_observations"),
        document["observation_as_of"],
    )
    quality = document["data_quality"]
    count_keys = {
        "universe_count",
        "available_count",
        "failure_count",
        "regular_price_count",
        "verified_status_count",
        "operational_failure_count",
    }
    if {
        "regular_price_count",
        "verified_status_count",
        "operational_failure_count",
    }.intersection(quality):
        if (
            not count_keys.issubset(quality)
            or any(type(quality[key]) is not int or quality[key] < 0 for key in count_keys)
            or quality["regular_price_count"] + quality["verified_status_count"]
            != quality["available_count"]
            or quality["available_count"] + quality["operational_failure_count"]
            != quality["universe_count"]
            or quality["failure_count"] != quality["operational_failure_count"]
            or quality["verified_status_count"] != len(statuses)
        ):
            raise ValueError("observation dashboard quality counts are invalid")

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in FORBIDDEN_OUTPUT_KEYS:
                    raise ValueError("observation dashboard contains prediction fields")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(document)
    try:
        json.dumps(document, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation dashboard must contain finite JSON") from exc
    return document


def build_observation_dashboard(
    source,
    industry_map,
    prediction_capability,
    *,
    generated_at=None,
    today=None,
):
    """Return an Observation-only dashboard bound to a verified source manifest."""
    current_date = today or datetime.date.today()
    timestamp = generated_at or datetime.datetime.now(datetime.timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if (
        not prediction_capability.observation_enabled
        or prediction_capability.probability_allowed
        or prediction_capability.ranking_allowed
        or prediction_capability.strong_action_allowed
        or prediction_capability.performance_endorsement_allowed
    ):
        raise ValueError("observation builder requires prediction capabilities disabled")
    if not isinstance(industry_map, dict):
        raise ValueError("industry map is invalid")
    _validate_source(source, current_date)

    regular_stocks = [
        stock for stock in source.stocks if stock.observation_kind == "regular_price"
    ]
    status_stocks = [
        stock for stock in source.stocks if stock.observation_kind in TRADING_STATUS_LABELS
    ]
    status_symbols = {stock.symbol for stock in status_stocks}
    stock_by_symbol = {
        stock.symbol: stock for stock in sorted(regular_stocks, key=lambda item: item.symbol)
    }
    etf_symbols = {
        str(symbol) for symbol in industry_map.get("ETF專區", [])
    }
    market_stocks = [
        stock
        for symbol, stock in stock_by_symbol.items()
        if symbol not in etf_symbols
    ]
    if not market_stocks:
        raise ValueError("observation market universe is empty")
    market = _market_observation(market_stocks)
    price_industry_map = {
        name: [str(symbol) for symbol in symbols if str(symbol) not in status_symbols]
        for name, symbols in industry_map.items()
    }
    industries = _industry_observations(price_industry_map, stock_by_symbol, market)
    events = _stock_events(market_stocks)
    etfs = _etf_observations(
        [
            stock_by_symbol[symbol]
            for symbol in sorted(etf_symbols)
            if symbol in stock_by_symbol
        ]
    )
    manifest = source.manifest
    document = {
        "schema_version": 2,
        "kind": "absorb-observation-dashboard",
        "product_mode": "observation",
        "market": manifest.market,
        "observation_as_of": manifest.market_as_of.isoformat(),
        "generated_at": timestamp.astimezone(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_manifest": f"quant/v1/{manifest.manifest_path}",
        "source_manifest_sha256": manifest.manifest_sha256,
        "prediction_capability": prediction_capability.to_document(),
        "market_observation": market,
        "industry_observations": industries,
        "heatmap": _heatmap(industries),
        "stock_events": events,
        "trading_status_observations": _trading_status_observations(status_stocks),
        "etf_observations": etfs,
        "daily_focus": _daily_focus(market, industries, events),
        "data_quality": {
            "universe_count": manifest.universe_count,
            "available_count": manifest.symbol_count,
            "failure_count": manifest.failure_count,
            "coverage": _rounded(manifest.coverage, 6),
            "failure_rate": _rounded(manifest.failure_rate, 6),
            "source_age_days": (current_date - manifest.market_as_of).days,
            "failed_symbols": list(manifest.failed_symbols),
            "regular_price_count": len(regular_stocks),
            "verified_status_count": len(status_stocks),
            "operational_failure_count": manifest.failure_count,
            "regular_price_coverage": _rounded(
                manifest.regular_price_coverage
                if manifest.schema_version in (3, 4)
                else manifest.coverage,
                6,
            ),
            "observation_coverage": _rounded(manifest.coverage, 6),
            "operational_failed_symbols": list(manifest.failed_symbols),
            "unavailable_symbols": list(
                manifest.unavailable_symbols
                if getattr(manifest, "unavailable_symbols", None) is not None
                else manifest.failed_symbols
            ),
            "unavailable_count": (
                manifest.unavailable_count
                if getattr(manifest, "unavailable_count", None) is not None
                else manifest.failure_count
            ),
        },
        "gates": {
            "source_identity": "PASS",
            "source_schema": "PASS",
            "finite_json": "PASS",
            "sample_data": "PASS",
            "coverage": "PASS",
            "prediction_separation": "PASS",
        },
    }
    return validate_observation_dashboard(document)


def _canonical(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("immutable observation candidate conflict")
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_observation_candidate(root, report_metadata, dashboard):
    from reporting.schemas import ReportMetadataV2

    dashboard = validate_observation_dashboard(dict(dashboard))
    report = ReportMetadataV2.from_document(dict(report_metadata)).to_document()
    if (
        report.get("product_mode") != "observation"
        or report["source_market_date"] != dashboard["observation_as_of"]
        or report["source_manifest"] != dashboard["source_manifest"]
        or report["source_manifest_sha256"]
        != dashboard["source_manifest_sha256"]
        or report["prediction_capability"]
        != dashboard["prediction_capability"]
    ):
        raise ValueError("observation candidate identity mismatch")
    documents = {
        "dashboard-snapshot.json": dashboard,
        "post-close-report-v2.json": report,
    }
    identity = _canonical(documents)
    candidate_id = hashlib.sha256(identity).hexdigest()[:16]
    directory = (
        Path(root)
        / "outputs"
        / "observation"
        / "candidates"
        / f"{dashboard['observation_as_of']}-{candidate_id}"
    )
    files = {}
    for name, document in documents.items():
        content = _canonical(document)
        _write_immutable(directory / name, content)
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    manifest = {
        "schema_version": 1,
        "kind": "absorb-observation-candidate",
        "product_mode": "observation",
        "observation_as_of": dashboard["observation_as_of"],
        "files": files,
    }
    _write_immutable(directory / "candidate.json", _canonical(manifest))
    return directory


def _read_observation_candidate(directory):
    directory = Path(directory)
    try:
        manifest = json.loads(
            (directory / "candidate.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("observation candidate manifest is invalid") from exc
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "absorb-observation-candidate"
        or manifest.get("product_mode") != "observation"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("observation candidate manifest is invalid")
    documents = {}
    for name in ("dashboard-snapshot.json", "post-close-report-v2.json"):
        expected = manifest["files"].get(name)
        if not isinstance(expected, dict):
            raise ValueError("observation candidate manifest is invalid")
        path = directory / name
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError("observation candidate file is unavailable") from exc
        if (
            len(content) != expected.get("size")
            or hashlib.sha256(content).hexdigest() != expected.get("sha256")
        ):
            raise ValueError("observation candidate hash mismatch")
        try:
            documents[name] = json.loads(content)
        except ValueError as exc:
            raise ValueError("observation candidate JSON is invalid") from exc
    validate_observation_dashboard(documents["dashboard-snapshot.json"])
    return documents


def _prepare_observation_dashboard(root, document):
    document = validate_observation_dashboard(dict(document))
    content = _canonical(document)
    if len(content) > 5_000_000:
        raise ValueError("observation dashboard is too large")
    digest = hashlib.sha256(content).hexdigest()
    publish = Path(root) / "publish" / "dashboard" / "v1"
    _write_immutable(publish / "objects" / f"{digest}.json", content)
    market = document["market"]
    latest = {
        "schema_version": 2,
        "kind": "absorb-observation-dashboard",
        "product_mode": "observation",
        "market": market,
        "observation_as_of": document["observation_as_of"],
        "generated_at": document["generated_at"],
        "source_manifest": document["source_manifest"],
        "source_manifest_sha256": document["source_manifest_sha256"],
        "path": f"objects/{digest}.json",
        "sha256": digest,
        "size": len(content),
    }
    return publish / f"latest-{market}.json", _canonical(latest)


def _restore(path, previous):
    path = Path(path)
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _write_atomic(path, previous)


def promote_observation_candidate(root, directory):
    from reporting.publisher import publish_report_v2
    from reporting.professional_builder import build_professional_post_close_artifact
    from reporting import git_commit_sha

    root = Path(root)
    documents = _read_observation_candidate(directory)
    dashboard_latest, dashboard_latest_bytes = _prepare_observation_dashboard(
        root, documents["dashboard-snapshot.json"]
    )
    market = documents.get("dashboard-snapshot.json", {}).get("market") or documents.get("post-close-report-v2.json", {}).get("market") or "TW"
    report_publish = root / "publish" / "reports" / "v2"
    report_index = report_publish / f"index-{market}.json"
    report_latest = report_publish / f"latest-{market}-post_close.json"
    previous = {
        report_index: report_index.read_bytes() if report_index.exists() else None,
        report_latest: report_latest.read_bytes() if report_latest.exists() else None,
        dashboard_latest: (
            dashboard_latest.read_bytes() if dashboard_latest.exists() else None
        ),
    }

    report_metadata = documents["post-close-report-v2.json"]
    from reporting.regression_input_builder import (
        AGGREGATE_MANIFEST_INTERVAL_VALIDATION_READY,
        PRODUCTION_REGRESSION_ARTIFACT_AVAILABLE,
        PRODUCTION_REGRESSION_INPUT_READY,
        PRODUCTION_REGRESSION_SOURCE_ADAPTER_READY,
    )
    regression_ready = all(
        (
            PRODUCTION_REGRESSION_SOURCE_ADAPTER_READY,
            PRODUCTION_REGRESSION_INPUT_READY,
            PRODUCTION_REGRESSION_ARTIFACT_AVAILABLE,
            AGGREGATE_MANIFEST_INTERVAL_VALIDATION_READY,
        )
    )
    if not regression_ready and report_metadata.get("regression_research") is not None:
        raise ValueError("production regression is not ready; metadata pointer is forbidden")
    commit_sha = git_commit_sha()
    import re
    if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-f]{7,64}", commit_sha):
        raise ValueError("promote_observation_candidate failed to get a valid code_commit_sha")

    prof_report = build_professional_post_close_artifact(
        report_metadata, code_commit_sha=commit_sha
    )

    try:
        report_result = publish_report_v2(
            root, report_metadata, professional_report=prof_report
        )
        _write_atomic(dashboard_latest, dashboard_latest_bytes)
    except Exception:
        for path, content in previous.items():
            _restore(path, content)
        raise
    return {
        "report_latest": str(report_result),
        "dashboard_latest": str(dashboard_latest),
    }
