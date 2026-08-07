"""Stock analysis orchestration without Flask or LINE dependencies."""

import math

from reporting.interpretation import interpret_backtest
from stock_papi.services.recommendation_engine import recommend_analysis


def snapshot_dataframe(snapshot, *, pd):
    try:
        frame = pd.DataFrame(snapshot["daily"])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
        required = [
            "Open", "High", "Low", "Close", "MA20", "RSI", "Volat",
            "MACD_OSC", "K", "D", "ForeignNet",
        ]
        if frame.empty or not set(required).issubset(frame.columns):
            return None
        persisted_latest_date = frame.index[-1]

        def finite_numeric(value):
            if value is None or isinstance(value, (bool, str, bytes)):
                return False
            try:
                return math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                return False

        complete = pd.Series(True, index=frame.index)
        for name in required:
            complete &= frame[name].map(finite_numeric)
        frame = frame.loc[complete]
        if len(frame) < 200 or frame.index[-1] != persisted_latest_date:
            return None
        if "AI_P" not in frame.columns or not finite_numeric(frame["AI_P"].iloc[-1]):
            return None
        return frame
    except (KeyError, TypeError, ValueError):
        return None


def analyze_uncached(
    code, *, fetch_snapshot, build_snapshot_frame, get_data, calc_all,
    run_ai_engine, get_stock_name, get_news, analyze_sentiment_detail,
    summarize_foreign_flow, calculate_projection, pd, json, datetime,
):
    snapshot = fetch_snapshot(code)
    frame = build_snapshot_frame(snapshot) if snapshot else None
    if frame is None:
        frame = get_data(code)
        if frame.empty or len(frame) < 200:
            return None
        frame = calc_all(frame)
        backtest = run_ai_engine(frame)
        quant_source = "即時計算"
    else:
        backtest = snapshot["backtest"]
        quant_source = "本地回測快照"
    if not backtest:
        return None

    last = frame.iloc[-1]
    name = get_stock_name(code)
    sentiment = analyze_sentiment_detail(get_news(name, code))
    news = sentiment["items"]
    probability = int(max(0, min(100, last["AI_P"])))
    trend = "多頭" if last["Close"] > last["MA20"] else "空頭"
    foreign_flow = summarize_foreign_flow(frame)

    chart = frame.copy().reset_index()
    chart["Date"] = chart["Date"].dt.strftime("%Y-%m-%d")
    chart["Open"] = chart["Open"].fillna(chart["Close"])
    chart["High"] = chart["High"].fillna(chart["Close"])
    chart["Low"] = chart["Low"].fillna(chart["Close"])
    chart["High_corr"] = chart[["Open", "High", "Low", "Close"]].max(axis=1)
    chart["Low_corr"] = chart[["Open", "High", "Low", "Close"]].min(axis=1)

    last_volatility = frame["Volat"].iloc[-1] if pd.notna(frame["Volat"].iloc[-1]) else 0.02
    drift = ((probability - 50) / 50.0) * (last_volatility * last["Close"])
    prediction = [{"time": chart["Date"].iloc[-1], "value": last["Close"]}]
    current_date = frame.index[-1]
    current_price = last["Close"]
    for _ in range(5):
        current_date += datetime.timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += datetime.timedelta(days=1)
        current_price += drift
        prediction.append({
            "time": current_date.strftime("%Y-%m-%d"),
            "value": round(current_price, 2),
        })

    result = {
        "code": code, "name": name, "price": last["Close"], "prob": probability,
        "as_of": frame.index[-1].date().isoformat(),
        "quant_source": quant_source,
        "bt": backtest, "news": news, "trend": trend,
        "rsi": last["RSI"], "ma20": last["MA20"],
        "volume_ratio": last.get("VOL_RATIO"),
        "volatility": last.get("Volat"),
        "data_quality_warning": bool(last.get("DATA_PRICE_WARNING", 0)),
        "macd_osc": last["MACD_OSC"], "k": last["K"], "d": last["D"],
        "foreign_flow": foreign_flow,
        "s_score": sentiment["score"], "s_status": sentiment["status"],
        "news_count": sentiment["count"],
        "news_positive_ratio": sentiment["positive_ratio"],
        "news_negative_ratio": sentiment["negative_ratio"],
        "news_neutral_ratio": sentiment["neutral_ratio"],
        "news_confidence_score": sentiment["confidence_score"],
        "news_confidence": sentiment["confidence"],
        "news_source_count": sentiment["source_count"],
        "news_publisher_count": sentiment["publisher_count"],
        "social_sample_size": sentiment["social_sample_size"],
        "news_weighted_volatility": sentiment["weighted_volatility"],
        "news_momentum": sentiment["momentum"],
        "news_momentum_data_sufficient": sentiment["momentum_data_sufficient"],
        "news_disagreement": sentiment["disagreement"],
        "news_effective_sample_size": sentiment["effective_sample_size"],
        "news_missing_metadata_ratio": sentiment["missing_metadata_ratio"],
        "news_extreme_score_flag": sentiment["extreme_score_flag"],
        "sentiment_window_days": sentiment["window_days"],
        "candles": json.dumps(chart[["Date", "Open", "High_corr", "Low_corr", "Close"]].rename(columns={"Date": "time", "Open": "open", "High_corr": "high", "Low_corr": "low", "Close": "close"}).to_dict("records")),
        "ma20_line": json.dumps(chart[["Date", "MA20"]].dropna().rename(columns={"Date": "time", "MA20": "value"}).to_dict("records")),
        "prob_h": json.dumps(chart[["Date", "AI_P"]].dropna().rename(columns={"Date": "time", "AI_P": "value"}).to_dict("records")),
        "pred": json.dumps(prediction),
    }
    result["projection"] = calculate_projection(100000, result)
    result["recommendation"] = recommend_analysis(result).to_dict()
    result["backtest_interpretation"] = interpret_backtest(backtest)
    return result


def analyze_cached(code, *, cache, expiry_seconds, now, analyze_fn):
    timestamp = now()
    if code in cache:
        cached_data, cached_at = cache[code]
        if timestamp - cached_at < expiry_seconds:
            return cached_data
    data = analyze_fn(code)
    if data:
        cache[code] = (data, timestamp)
    return data
