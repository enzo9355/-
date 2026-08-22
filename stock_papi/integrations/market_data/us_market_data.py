"""Production US stock market data fetcher with schema validation and technical indicators."""

import datetime
import math
import zoneinfo
import numpy as np
import pandas as pd
import yfinance as yf

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")


class USObservationError(Exception):
    """Base class for all US market data observation exceptions."""


class USObservationUnavailable(USObservationError):
    """Legitimate absence of observation data (M partition). Provider healthy, symbol valid, but no trades."""


class USProviderOperationalError(USObservationError):
    """Operational failure communicating with market data provider (network, timeout, transport)."""


class USRateLimitError(USProviderOperationalError):
    """Provider capacity limit / HTTP 429 encountered."""


class USSchemaError(USObservationError):
    """Malformed provider response schema or missing required columns."""


class USIntegrityError(USObservationError):
    """OHLC price bar integrity violation (e.g. High < Open/Close or Low > Open/Close)."""


class USCalendarError(USObservationError):
    """Invalid session or calendar date resolution error."""


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


import json
import urllib.request
import urllib.error


def fetch_direct_yahoo_chart(symbol: str, range_str: str = "2y", max_retries: int = 3) -> pd.DataFrame:
    """Fetch daily chart history directly from Yahoo chart API with zero crumb rate limits and retry backoff."""
    import time
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_str}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
    )
    last_exc = None
    doc = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return pd.DataFrame()
            if e.code == 429:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise USRateLimitError(f"HTTP 429 rate limited for {symbol}") from e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise USProviderOperationalError(f"HTTP {e.code} operational error for {symbol}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise USProviderOperationalError(f"Network transport error for {symbol}: {e}") from e

    if not doc:
        if last_exc:
            raise USProviderOperationalError(f"Network transport error for {symbol}: {last_exc}") from last_exc
        return pd.DataFrame()

    entry = res[0]
    timestamps = entry.get("timestamp")
    if not timestamps:
        return pd.DataFrame()

    quote = entry.get("indicators", {}).get("quote", [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    n = len(timestamps)
    records = []
    for i in range(n):
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else 0.0
        if o is None or h is None or l is None or c is None:
            continue
        try:
            o, h, l, c, v = float(o), float(h), float(l), float(c), float(v or 0.0)
        except (TypeError, ValueError):
            continue
        if o <= 0 or h <= 0 or l <= 0 or c <= 0 or v < 0:
            continue
        if h < max(o, c, l) - 1e-4 or l > min(o, c, h) + 1e-4:
            continue
        dt = datetime.datetime.fromtimestamp(timestamps[i], tz=datetime.timezone.utc).astimezone(NEW_YORK).date()
        records.append({
            "Date": dt,
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date").sort_index()
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
        range_str = "2y" if days >= 500 else "1y" if days >= 250 else "3mo"
        try:
            raw_df = fetch_direct_yahoo_chart(symbol, range_str=range_str)
        except USObservationError:
            raise
        except Exception as e:
            # Fallback to yfinance if direct chart endpoint encounters unexpected error
            try:
                yf_ticker = yf.Ticker(symbol)
                raw_df = yf_ticker.history(period=range_str, auto_adjust=False)
            except Exception as yf_err:
                raise USProviderOperationalError(f"Provider failure for {symbol}: {yf_err}") from yf_err

    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    # Standardize column names
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(raw_df.columns)):
        raise USSchemaError(f"US price schema is incomplete for {symbol}")

    df = raw_df.copy()

    # Normalize index to America/New_York date if needed
    if not isinstance(df.index, pd.DatetimeIndex) and "Date" in df.columns:
        dates = [
            val.date() if isinstance(val, (pd.Timestamp, datetime.datetime)) else val
            for val in df["Date"]
        ]
        df["Date"] = dates
        df = df.drop_duplicates(subset=["Date"], keep="last")
        df = df.set_index("Date").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(NEW_YORK)
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
        raise USIntegrityError(f"US price bar integrity violation for {symbol}: non-positive price values")
    if (df["Volume"] < 0).any():
        raise USIntegrityError(f"US price bar integrity violation for {symbol}: negative volume")

    invalid_high = df["High"] < (df[["Open", "Close", "Low"]].max(axis=1) - 1e-4)
    invalid_low = df["Low"] > (df[["Open", "Close", "High"]].min(axis=1) + 1e-4)
    if invalid_high.any() or invalid_low.any():
        raise USIntegrityError(f"US price bar integrity violation for {symbol}: High/Low outside Open/Close range")

    if target_market_date is not None:
        df = df[df.index <= target_market_date]
        if df.empty:
            return pd.DataFrame()

    # Compute technical indicators
    df = compute_us_technical_indicators(df)
    return df
