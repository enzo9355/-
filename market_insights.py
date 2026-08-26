import datetime
import math
import re


TAIPEI = datetime.timezone(datetime.timedelta(hours=8), "Asia/Taipei")

ETF_CATALOG = (
    {"ticker": "0050.TW", "name": "元大台灣50", "market": "TW"},
    {"ticker": "00878.TW", "name": "國泰永續高股息", "market": "TW"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "market": "US"},
    {"ticker": "QQQ", "name": "Invesco QQQ", "market": "US"},
    {"ticker": "1321.T", "name": "NEXT FUNDS Nikkei 225", "market": "JP"},
    {"ticker": "1306.T", "name": "NEXT FUNDS TOPIX", "market": "JP"},
)

SUPPLY_CHAINS = (
    {
        "id": "semiconductor", "name": "半導體供應鏈",
        "stages": (
            ("IC 設計", (("2454", "聯發科", "TW"), ("NVDA", "NVIDIA", "US"), ("AMD", "AMD", "US"))),
            ("晶圓製造", (("2330", "台積電", "TW"), ("2303", "聯電", "TW"))),
            ("設備材料", (("AMAT", "Applied Materials", "US"), ("8035.T", "東京威力科創", "JP"), ("4063.T", "信越化學", "JP"))),
            ("封裝測試", (("3711", "日月光投控", "TW"), ("6857.T", "愛德萬測試", "JP"), ("6146.T", "DISCO", "JP"))),
        ),
    },
    {
        "id": "ai-server", "name": "AI 伺服器供應鏈",
        "stages": (
            ("AI 晶片", (("NVDA", "NVIDIA", "US"), ("AMD", "AMD", "US"), ("2330", "台積電", "TW"))),
            ("伺服器 ODM", (("2317", "鴻海", "TW"), ("2382", "廣達", "TW"), ("3231", "緯創", "TW"), ("6669", "緯穎", "TW"))),
            ("網通與電源", (("2345", "智邦", "TW"), ("2308", "台達電", "TW"), ("AVGO", "Broadcom", "US"))),
            ("檢測設備", (("6857.T", "愛德萬測試", "JP"), ("8035.T", "東京威力科創", "JP"))),
        ),
    },
    {
        "id": "ev", "name": "電動車與能源供應鏈",
        "stages": (
            ("整車平台", (("TSLA", "Tesla", "US"), ("7203.T", "豐田汽車", "JP"))),
            ("電池材料", (("ALB", "Albemarle", "US"), ("6752.T", "Panasonic", "JP"))),
            ("電源零組件", (("2308", "台達電", "TW"), ("2317", "鴻海", "TW"))),
        ),
    },
)


def _clean_text(value, limit=500):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _roc_datetime(date_value, time_value):
    date_text = re.sub(r"\D", "", str(date_value or ""))
    time_text = re.sub(r"\D", "", str(time_value or "")).zfill(6)
    if len(date_text) != 7 or len(time_text) != 6:
        raise ValueError("invalid MOPS date")
    year = int(date_text[:3]) + 1911
    return datetime.datetime(
        year, int(date_text[3:5]), int(date_text[5:7]),
        int(time_text[:2]), int(time_text[2:4]), int(time_text[4:6]),
        tzinfo=TAIPEI,
    )


def parse_mops_items(items, source, limit=100):
    result = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        code = _clean_text(item.get("公司代號") or item.get("SecuritiesCompanyCode"), 8)
        title = _clean_text(item.get("主旨 ") or item.get("主旨"), 500)
        if not re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", code.upper()) or not title:
            continue
        try:
            published = _roc_datetime(item.get("發言日期"), item.get("發言時間"))
        except (TypeError, ValueError):
            continue
        key = (code, title, published.date().isoformat())
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "code": code,
            "name": _clean_text(item.get("公司名稱") or item.get("CompanyName"), 80),
            "title": title,
            "rule": _clean_text(item.get("符合條款"), 40),
            "published_at": published.isoformat(),
            "source": _clean_text(source, 30),
        })
    return sorted(result, key=lambda row: row["published_at"], reverse=True)[:limit]


def normalize_etf_holdings(rows, etf, limit=10):
    holdings = []
    for row in rows or []:
        try:
            weight = float(row.get("weight"))
        except (AttributeError, TypeError, ValueError):
            continue
        symbol = _clean_text(row.get("symbol"), 16).upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,15}", symbol) or not 0 <= weight <= 1:
            continue
        holdings.append({
            "symbol": symbol.removesuffix(".TW"),
            "name": _clean_text(row.get("name"), 100) or symbol,
            "weight": round(weight * 100, 2),
        })
    holdings.sort(key=lambda row: row["weight"], reverse=True)
    return {
        "ticker": _clean_text(etf.get("ticker"), 16).upper(),
        "name": _clean_text(etf.get("name"), 100),
        "market": _clean_text(etf.get("market"), 3).upper(),
        "holdings": holdings[:limit],
    }


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _score(value):
    return max(0, min(10, int(round(value))))


def _signals(metric):
    signals = []
    probability = _number(metric.get("prob"))
    institutional = _number(metric.get("inst_ratio"))
    volume = _number(metric.get("volume_ratio"))
    if probability is not None and probability >= 60:
        signals.append("AI偏多")
    elif probability is not None and probability <= 40:
        signals.append("AI偏空")
    if institutional is not None and institutional > 0:
        signals.append("法人偏多")
    elif institutional is not None and institutional < 0:
        signals.append("法人偏空")
    if volume is not None and volume >= 1.2:
        signals.append("量能升溫")
    return signals


def build_industries(theme_map, metrics, limit=5):
    industries = []
    for name, symbols in (theme_map or {}).items():
        if name in {"全市場", "ETF專區"}:
            continue
        leaders = []
        for symbol in symbols:
            metric = (metrics or {}).get(str(symbol).upper())
            if not metric:
                continue
            leaders.append({"symbol": str(symbol).upper(), **metric, "signals": _signals(metric)})
        leaders.sort(key=lambda row: float(row.get("prob") or 0), reverse=True)
        if leaders:
            probabilities = [_number(row.get("prob")) for row in leaders]
            returns = [_number(row.get("return_1d")) for row in leaders]
            institutional = [_number(row.get("inst_ratio")) for row in leaders]
            margin = [_number(row.get("margin_change")) for row in leaders]
            volume = [_number(row.get("volume_ratio")) for row in leaders]
            probabilities = [value for value in probabilities if value is not None]
            returns = [value for value in returns if value is not None]
            institutional = [value for value in institutional if value is not None]
            margin = [value for value in margin if value is not None]
            volume = [value for value in volume if value is not None]
            average_prob = round(sum(probabilities) / len(probabilities), 1) if probabilities else 0.0
            average_return = round(sum(returns) / len(returns), 2) if returns else 0.0
            scored = [row for row in leaders if _number(row.get("prob")) is not None]
            bullish_ratio = round(
                sum(1 for row in scored if row.get("trend") == "多頭") / len(scored) * 100,
                1,
            ) if scored else 0.0
            industries.append({
                "name": str(name),
                "leaders": leaders[:limit],
                "average_prob": average_prob,
                "average_return": average_return,
                "bullish_ratio": bullish_ratio,
                "coverage": len(scored),
                "candidate_count": len(leaders),
                "heat_tone": (
                    "surge" if average_return >= 1.5 else "rise" if average_return > 0
                    else "fall" if average_return <= -1.5 else "weak" if average_return < 0
                    else "flat"
                ),
                "heat_size": "lg" if len(leaders) >= 5 else "md" if len(leaders) >= 3 else "sm",
                "chips": [
                    {"label": "法人", "score": _score(5 + sum(institutional) / len(institutional) * 2) if institutional else None},
                    {"label": "融資", "score": _score(5 + sum(margin) / len(margin) * 50) if margin else None},
                    {"label": "量能", "score": _score(5 + (sum(volume) / len(volume) - 1) * 5) if volume else None},
                ],
            })
    industries.sort(key=lambda row: row["average_prob"], reverse=True)
    return industries


def build_supply_chains(metrics):
    chains = []
    for chain in SUPPLY_CHAINS:
        stages = []
        for stage_name, catalog_nodes in chain["stages"]:
            nodes = []
            for symbol, name, market in catalog_nodes:
                metric = (metrics or {}).get(symbol, {})
                nodes.append({
                    "symbol": symbol, "name": name, "market": market,
                    "prob": metric.get("prob"),
                    "trend": metric.get("trend") or "資料待更新",
                    "as_of": metric.get("as_of") or "",
                    "close": metric.get("close"),
                    "return_1d": metric.get("return_1d"),
                    "signals": _signals(metric),
                })
            stages.append({"name": stage_name, "nodes": nodes})
        chains.append({"id": chain["id"], "name": chain["name"], "stages": stages})
    return chains
