"""Quant market-data normalization and feature context assembly."""

from stock_papi.quant.constants import (
    DATA_QUALITY_FEATURES,
    MARKET_FEATURES,
    OPTION_FEATURES,
    PRICE_DIFF_WARNING_THRESHOLD,
)


def foreign_flow_mask(frame, *, pd):
    mask = pd.Series(False, index=frame.index)
    for column in ("name", "institutional_investor", "institutional_investors", "type"):
        if column in frame:
            mask |= frame[column].astype(str).str.contains(
                "Foreign|外資", case=False, regex=True, na=False
            )
    return mask


def merge_chip_data(price, institutional=None, margin=None, *, pd):
    result = price.copy()
    if institutional is not None and not institutional.empty:
        flows = institutional.copy()
        flows["Date"] = pd.to_datetime(flows["date"], errors="coerce")
        flows["buy"] = pd.to_numeric(flows["buy"], errors="coerce").fillna(0)
        flows["sell"] = pd.to_numeric(flows["sell"], errors="coerce").fillna(0)
        flows["InstitutionalNet"] = flows["buy"] - flows["sell"]
        mask = foreign_flow_mask(flows, pd=pd)
        foreign = flows.loc[mask] if mask.any() else flows
        foreign = foreign.groupby("Date", as_index=False)["InstitutionalNet"].sum()
        foreign = foreign.rename(columns={"InstitutionalNet": "ForeignNet"})
        flows = flows.groupby("Date", as_index=False)["InstitutionalNet"].sum()
        result = result.merge(flows, on="Date", how="left")
        result = result.merge(foreign, on="Date", how="left")
    if margin is not None and not margin.empty:
        balances = margin.copy()
        balances["Date"] = pd.to_datetime(balances["date"], errors="coerce")
        balances = balances.rename(columns={
            "MarginPurchaseTodayBalance": "MarginBalance",
            "ShortSaleTodayBalance": "ShortBalance",
        })
        balances = balances[["Date", "MarginBalance", "ShortBalance"]]
        balances[["MarginBalance", "ShortBalance"]] = balances[
            ["MarginBalance", "ShortBalance"]
        ].apply(pd.to_numeric, errors="coerce")
        balances = balances.groupby("Date", as_index=False).last()
        result = result.merge(balances, on="Date", how="left")
    for column in ["InstitutionalNet", "ForeignNet", "MarginBalance", "ShortBalance"]:
        if column not in result:
            result[column] = 0.0
        result[column] = result[column].fillna(0.0)
    return result


def neutral_market_features(frame):
    result = frame.copy()
    for column in MARKET_FEATURES:
        result[column] = 0.0
    return result


def market_feature_frame(market, prefix, *, pd):
    if market is None or market.empty or "Date" not in market or "Close" not in market:
        return pd.DataFrame()
    frame = market[["Date", "Close"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    daily_return = close.pct_change(fill_method=None)
    frame[f"{prefix}_RET_1"] = daily_return
    frame[f"{prefix}_RET_5"] = close.pct_change(5, fill_method=None)
    frame[f"{prefix}_RET_20"] = close.pct_change(20, fill_method=None)
    frame[f"{prefix}_VOL_20"] = daily_return.rolling(20).std()
    return frame.drop(columns=["Close"]).dropna(subset=["Date"])


def add_market_context_features(price, market=None, etf50=None, *, pd, np):
    if price is None or price.empty:
        return price
    result = price.copy()
    if "Date" not in result:
        return neutral_market_features(result)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    market_frame = market_feature_frame(market, "MARKET", pd=pd)
    if not market_frame.empty:
        result = result.merge(market_frame, on="Date", how="left")
    etf_frame = market_feature_frame(etf50, "ETF50", pd=pd)
    if not etf_frame.empty and "ETF50_RET_5" in etf_frame:
        result = result.merge(etf_frame[["Date", "ETF50_RET_5"]], on="Date", how="left")
    stock_ret_5 = (
        pd.to_numeric(result["Close"], errors="coerce").pct_change(5, fill_method=None)
        if "Close" in result else pd.Series(0.0, index=result.index)
    )
    market_ret_5 = (
        pd.to_numeric(result["MARKET_RET_5"], errors="coerce")
        if "MARKET_RET_5" in result else pd.Series(0.0, index=result.index)
    )
    result["STOCK_VS_MARKET_5"] = stock_ret_5 - market_ret_5
    for column in MARKET_FEATURES:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], 0
        ).fillna(0.0)
    return result


def option_close_frame(frame, column, *, pd):
    if frame is None or frame.empty or "Date" not in frame or "Close" not in frame:
        return pd.DataFrame(columns=["Date", column])
    result = frame[["Date", "Close"]].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").astype("datetime64[ns]")
    result[column] = pd.to_numeric(result["Close"], errors="coerce")
    return result.drop(columns=["Close"]).dropna(subset=["Date"]).sort_values(
        "Date"
    ).drop_duplicates("Date", keep="last")


def add_option_context_features(price, vix=None, vix9d=None, vix3m=None, *, pd, np):
    if price is None or price.empty:
        return price
    result = price.copy()
    for column in OPTION_FEATURES[:-1]:
        result[column] = 0.0
    result["OPTION_DATA_MISSING"] = 1.0
    if "Date" not in result:
        return result
    option = option_close_frame(vix, "VIX", pd=pd)
    if option.empty:
        return result
    option["OPTION_IV_LEVEL"] = option["VIX"] / 100.0
    option["OPTION_IV_CHG_1"] = option["VIX"].pct_change(fill_method=None)
    option["OPTION_IV_CHG_5"] = option["VIX"].pct_change(5, fill_method=None)
    for frame, column in ((vix9d, "VIX9D"), (vix3m, "VIX3M")):
        extra = option_close_frame(frame, column, pd=pd)
        if not extra.empty:
            option = option.merge(extra, on="Date", how="left")
    if "VIX9D" not in option:
        option["VIX9D"] = np.nan
    if "VIX3M" not in option:
        option["VIX3M"] = np.nan
    option["OPTION_IV_TERM_9D_3M"] = option["VIX9D"] / option["VIX3M"] - 1.0
    option["OPTION_DATA_MISSING"] = 0.0
    option = option[["Date"] + OPTION_FEATURES].sort_values("Date")
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").astype("datetime64[ns]")
    result = result.drop(columns=OPTION_FEATURES).sort_values("Date")
    result = pd.merge_asof(
        result, option, on="Date", direction="backward", tolerance=pd.Timedelta(days=4)
    )
    for column in OPTION_FEATURES[:-1]:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
    result["OPTION_DATA_MISSING"] = pd.to_numeric(
        result["OPTION_DATA_MISSING"], errors="coerce"
    ).fillna(1.0)
    return result


def add_price_quality_features(price, yf_price=None, *, pd, np):
    result = price.copy()
    result["YF_CLOSE"] = 0.0
    result["DATA_PRICE_DIFF_PCT"] = 0.0
    result["DATA_PRICE_WARNING"] = 0.0
    if yf_price is None or yf_price.empty or "Date" not in yf_price or "Close" not in yf_price:
        return result
    left = result[["Date", "Close"]].copy()
    left["Date"] = pd.to_datetime(left["Date"], errors="coerce")
    right = yf_price[["Date", "Close"]].copy()
    right["Date"] = pd.to_datetime(right["Date"], errors="coerce")
    right = right.rename(columns={"Close": "YF_CLOSE"})
    merged = left.merge(right, on="Date", how="left")
    yf_close = pd.to_numeric(merged["YF_CLOSE"], errors="coerce")
    close = pd.to_numeric(merged["Close"], errors="coerce")
    diff = ((yf_close - close).abs() / (close.abs() + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    )
    result["YF_CLOSE"] = yf_close.fillna(0.0).to_numpy()
    result["DATA_PRICE_DIFF_PCT"] = diff.fillna(0.0).to_numpy()
    result["DATA_PRICE_WARNING"] = (
        result["DATA_PRICE_DIFF_PCT"] > PRICE_DIFF_WARNING_THRESHOLD
    ).astype(float)
    return result


def clean_df(frame, *, pd, np):
    frame[["Open", "High", "Low", "Close"]] = frame[
        ["Open", "High", "Low", "Close"]
    ].replace(0, np.nan)
    numeric_columns = [
        "Volume", "InstitutionalNet", "ForeignNet", "MarginBalance", "ShortBalance",
    ] + MARKET_FEATURES + OPTION_FEATURES + DATA_QUALITY_FEATURES
    for column in numeric_columns:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], 0
        ).fillna(0.0)
    frame = frame.dropna(subset=["Date", "Close"])
    return frame.sort_values("Date").drop_duplicates(
        subset=["Date"], keep="last"
    ).set_index("Date")


def summarize_foreign_flow(frame, *, pd):
    if "ForeignNet" not in frame or frame["ForeignNet"].abs().sum() == 0:
        return {"available": False, "net_5": 0.0, "net_20": 0.0, "status": "資料不足", "source": "外資"}
    net = pd.to_numeric(frame["ForeignNet"], errors="coerce").fillna(0)
    net_5 = float(net.tail(5).sum())
    net_20 = float(net.tail(20).sum())
    status = "外資偏多" if net_5 > 0 and net_20 > 0 else "外資偏空" if net_5 < 0 and net_20 < 0 else "外資中性"
    return {"available": True, "net_5": net_5, "net_20": net_20, "status": status, "source": "外資"}


def get_data(
    code, days=730, *, as_of=None, datetime, pd, is_us_ticker, twstock_codes,
    fetch_yfinance, fetch_finmind, fetch_option_context,
    add_price_quality, add_market_context, add_option_context,
    merge_chip, clean, taiwan_security_master=None,
):
    if as_of is None:
        now = datetime.datetime.now()
        start_date = (now - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
    else:
        if type(as_of) is not datetime.date:
            raise TypeError("as_of must be a date")
        start_date = (as_of - datetime.timedelta(days=days)).isoformat()
        end_date = (as_of + datetime.timedelta(days=1)).isoformat()

    def finish(frame):
        result = clean(merge_chip(frame))
        return result if as_of is None else result[result.index.date <= as_of]

    if is_us_ticker(code):
        price = fetch_yfinance(code, start_date, end_date)
        if price.empty:
            return pd.DataFrame()
        price = add_price_quality(price)
        market = fetch_yfinance("^GSPC", start_date, end_date)
        spy = fetch_yfinance("SPY", start_date, end_date)
        price = add_market_context(price, market, spy)
        price = add_option_context(price, *fetch_option_context(start_date, end_date))
        return finish(price)
    raw = fetch_finmind("TaiwanStockPrice", code, start_date, end_date)
    yf_price = pd.DataFrame()
    if code != "TAIEX":
        if taiwan_security_master is not None:
            master = taiwan_security_master.get_master()
            if master is not None:
                exchange = taiwan_security_master.resolve_exchange(
                    code,
                    require_authoritative=True,
                )
            else:
                info = twstock_codes.get(code)
                exchange = "TPEx" if getattr(info, "data_source", "") == "tpex" else "TWSE"
        else:
            info = twstock_codes.get(code)
            exchange = "TPEx" if getattr(info, "data_source", "") == "tpex" else "TWSE"
        suffix = ".TWO" if exchange == "TPEx" else ".TW"
        yf_price = fetch_yfinance([f"{code}{suffix}"], start_date, end_date)
    price = None
    if not raw.empty:
        price = pd.DataFrame({
            "Date": pd.to_datetime(raw["date"], errors="coerce"),
            "Open": pd.to_numeric(raw["open"], errors="coerce"),
            "High": pd.to_numeric(raw["max"], errors="coerce"),
            "Low": pd.to_numeric(raw["min"], errors="coerce"),
            "Close": pd.to_numeric(raw["close"], errors="coerce"),
            "Volume": pd.to_numeric(raw.get("Trading_Volume", 0), errors="coerce"),
        })
    if price is None:
        price = fetch_yfinance("^TWII", start_date, end_date) if code == "TAIEX" else yf_price
    if price is None or price.empty:
        return pd.DataFrame()
    price = add_price_quality(price, yf_price)
    market = fetch_yfinance("^TWII", start_date, end_date)
    etf50 = fetch_yfinance("0050.TW", start_date, end_date)
    price = add_market_context(price, market, etf50)
    price = add_option_context(price, *fetch_option_context(start_date, end_date))
    institutional = margin = None
    if code != "TAIEX":
        institutional = fetch_finmind(
            "TaiwanStockInstitutionalInvestorsBuySell", code, start_date, end_date
        )
        margin = fetch_finmind(
            "TaiwanStockMarginPurchaseShortSale", code, start_date, end_date
        )
    result = clean(merge_chip(price, institutional, margin))
    return result if as_of is None else result[result.index.date <= as_of]
