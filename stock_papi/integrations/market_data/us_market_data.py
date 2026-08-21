"""Production US stock market data fetcher with schema validation and technical indicators."""

import datetime
import math
import zoneinfo
import numpy as np
import pandas as pd
import yfinance as yf

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")


def compute_us_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard technical indicators for US stock daily history."""
    df = df.copy()
    if df.empty or len(df) < 5:
        return df

    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["MA5"] = close.rolling(window=5, min_periods=1).mean()
    df["MA20"] = close.rolling(window=20, min_periods=1).mean()
    df["MA60"] = close.rolling(window=60, min_periods=1).mean()

    # Volume ratio (today volume / 5d average volume)
    vol_ma5 = volume.rolling(window=5, min_periods=1).mean()
    df["VOL_RATIO"] = np.where(vol_ma5 > 0, volume / vol_ma5, 1.0)

    # RSI (14-period standard)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_OSC"] = df["MACD"] - df["MACD_SIGNAL"]

    # KD (9, 3, 3)
    low9 = df["Low"].rolling(window=9, min_periods=1).min()
    high9 = df["High"].rolling(window=9, min_periods=1).max()
    rsv = np.where(high9 > low9, (close - low9) / (high9 - low9) * 100.0, 50.0)
    k_series = pd.Series(50.0, index=df.index)
    d_series = pd.Series(50.0, index=df.index)
    for i in range(len(df)):
        if i == 0:
            k_series.iloc[i] = 50.0 * (2/3) + rsv[i] * (1/3)
            d_series.iloc[i] = 50.0 * (2/3) + k_series.iloc[i] * (1/3)
        else:
            k_series.iloc[i] = k_series.iloc[i-1] * (2/3) + rsv[i] * (1/3)
            d_series.iloc[i] = d_series.iloc[i-1] * (2/3) + k_series.iloc[i] * (1/3)
    df["K"] = k_series
    df["D"] = d_series

    return df


def fetch_us_stock_history(
    symbol: str,
    days: int = 730,
    *,
    target_market_date: datetime.date | None = None,
    mock_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch and validate daily price history for a US stock."""
    symbol = str(symbol).strip().upper()
    if mock_df is not None:
        raw_df = mock_df.copy()
    else:
        # Ticker query using yfinance
        yf_ticker = yf.Ticker(symbol)
        period_str = "2y" if days >= 500 else "1y" if days >= 250 else "3mo"
        raw_df = yf_ticker.history(period=period_str, auto_adjust=False)

    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    # Standardize column names
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(raw_df.columns)):
        raise ValueError(f"US price schema is incomplete for {symbol}")

    df = raw_df.copy()

    # Normalize index to America/New_York date
    if df.index.tz is not None:
        df.index = df.index.tz_convert(NEW_YORK)
    
    # Extract date
    dates = [
        val.date() if isinstance(val, (pd.Timestamp, datetime.datetime)) else val
        for val in df.index
    ]
    df["Date"] = dates
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date").sort_index()

    # Filter out invalid numbers
    df = df.dropna(subset=["Close", "Open", "High", "Low"])
    if df.empty:
        return pd.DataFrame()

    # Integrity validations (fail-closed, no silent mutation of market facts)
    if ((df["Close"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Open"] <= 0)).any():
        raise ValueError(f"US price bar integrity violation for {symbol}: non-positive price values")
    if (df["Volume"] < 0).any():
        raise ValueError(f"US price bar integrity violation for {symbol}: negative volume")

    invalid_high = df["High"] < df[["Open", "Close", "Low"]].max(axis=1)
    invalid_low = df["Low"] > df[["Open", "Close", "High"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"US price bar integrity violation for {symbol}: High/Low outside Open/Close range")

    if target_market_date is not None:
        df = df[df.index <= target_market_date]
        if df.empty:
            return pd.DataFrame()

    # Compute technical indicators
    df = compute_us_technical_indicators(df)
    return df
