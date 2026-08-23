"""Production US stock market data fetcher with schema validation and technical indicators."""

import datetime
import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zoneinfo
from collections.abc import Mapping

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
    if df.empty:
        return df

    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["MA5"] = close.rolling(window=5, min_periods=5).mean()
    df["MA20"] = close.rolling(window=20, min_periods=20).mean()
    df["MA60"] = close.rolling(window=60, min_periods=60).mean()

    # Volume ratio (today volume / 5d average volume)
    vol_ma5 = volume.rolling(window=5, min_periods=5).mean()
    df["VOL_RATIO"] = np.where(vol_ma5 > 0, volume / vol_ma5, 1.0)

    # RSI (14-period standard)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rsi = pd.Series(np.nan, index=df.index, dtype=float)
    valid_rsi = avg_gain.notna() & avg_loss.notna()
    positive_loss = valid_rsi & avg_loss.gt(0)
    rsi.loc[positive_loss] = 100.0 - (
        100.0 / (1.0 + avg_gain.loc[positive_loss] / avg_loss.loc[positive_loss])
    )
    no_loss_gain = valid_rsi & avg_loss.eq(0) & avg_gain.gt(0)
    rsi.loc[no_loss_gain] = 100.0
    flat_window = valid_rsi & avg_loss.eq(0) & avg_gain.eq(0)
    rsi.loc[flat_window] = 50.0
    df["RSI"] = rsi

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_OSC"] = df["MACD"] - df["MACD_SIGNAL"]

    # KD (9, 3, 3)
    low9 = df["Low"].rolling(window=9, min_periods=9).min()
    high9 = df["High"].rolling(window=9, min_periods=9).max()
    rsv = pd.Series(np.nan, index=df.index, dtype=float)
    valid_kd = low9.notna() & high9.notna()
    non_flat_kd = valid_kd & high9.gt(low9)
    rsv.loc[non_flat_kd] = (
        (close.loc[non_flat_kd] - low9.loc[non_flat_kd])
        / (high9.loc[non_flat_kd] - low9.loc[non_flat_kd])
        * 100.0
    )
    rsv.loc[valid_kd & high9.eq(low9)] = 50.0
    k_series = pd.Series(np.nan, index=df.index, dtype=float)
    d_series = pd.Series(np.nan, index=df.index, dtype=float)
    previous_k = 50.0
    previous_d = 50.0
    for i in range(len(df)):
        if pd.isna(rsv.iloc[i]):
            continue
        previous_k = previous_k * (2 / 3) + float(rsv.iloc[i]) * (1 / 3)
        previous_d = previous_d * (2 / 3) + previous_k * (1 / 3)
        k_series.iloc[i] = previous_k
        d_series.iloc[i] = previous_d
    df["K"] = k_series
    df["D"] = d_series

    return df

_US_REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _scalar_is_missing(value) -> bool:
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def is_explicit_non_observation_placeholder(row: Mapping[str, object]) -> bool:
    """Return whether a provider row explicitly contains no market observation.

    This predicate is deliberately narrower than generic null handling: every
    OHLC field must be scalar-missing, and Volume must be scalar-missing or a
    numeric zero.  Partial rows, non-numeric values, and positive-volume rows
    are never treated as placeholders.
    """
    if not isinstance(row, Mapping) or not all(
        field in row for field in _US_REQUIRED_PRICE_COLUMNS
    ):
        return False
    if not all(_scalar_is_missing(row[field]) for field in _US_REQUIRED_PRICE_COLUMNS[:4]):
        return False
    volume = row["Volume"]
    if _scalar_is_missing(volume):
        return True
    if isinstance(volume, (bool, np.bool_)):
        return False
    try:
        volume_number = float(volume)
    except (TypeError, ValueError):
        return False
    return math.isfinite(volume_number) and volume_number == 0.0


def _normalise_us_date(value, symbol: str) -> datetime.date:
    try:
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None:
                value = value.tz_convert(NEW_YORK)
            return value.date()
        if isinstance(value, datetime.datetime):
            if value.tzinfo is not None:
                value = value.astimezone(NEW_YORK)
            return value.date()
        if isinstance(value, datetime.date):
            return value
        return datetime.date.fromisoformat(str(value).split("T", 1)[0])
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        raise USSchemaError(f"US price row date is invalid for {symbol}: {value!r}") from exc


def _coerce_us_number(value, *, symbol: str, field: str, date: datetime.date) -> float:
    if isinstance(value, (bool, np.bool_)) or value is None or _scalar_is_missing(value):
        raise USSchemaError(
            f"US price row has missing {field} for {symbol} on {date.isoformat()}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise USSchemaError(
            f"US price row has non-numeric {field} for {symbol} on {date.isoformat()}"
        ) from exc
    if not math.isfinite(number):
        raise USSchemaError(
            f"US price row has non-finite {field} for {symbol} on {date.isoformat()}"
        )
    return number


def _validate_us_price_values(
    *, symbol: str, date: datetime.date, values: dict[str, float]
) -> None:
    o, h, l, c, v = (values[column] for column in _US_REQUIRED_PRICE_COLUMNS)
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        raise USIntegrityError(
            f"US price bar integrity violation for {symbol} on {date.isoformat()}: "
            "non-positive price values"
        )
    if v < 0:
        raise USIntegrityError(
            f"US price bar integrity violation for {symbol} on {date.isoformat()}: "
            "negative volume"
        )
    if h < max(o, c, l) - 1e-4 or l > min(o, c, h) + 1e-4:
        raise USIntegrityError(
            f"US price bar integrity violation for {symbol} on {date.isoformat()}: "
            "High/Low outside Open/Close range"
        )


def _parse_nasdaq_historical_number(
    value, *, symbol: str, field: str, date: datetime.date
) -> float:
    if value is None or (isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA", "--"}):
        raise USSchemaError(
            f"Nasdaq historical row has missing {field} for {symbol} on {date.isoformat()}"
        )
    if isinstance(value, bool):
        raise USSchemaError(
            f"Nasdaq historical row has invalid {field} for {symbol} on {date.isoformat()}"
        )
    text = str(value).strip().replace(",", "")
    if text.startswith("$"):
        text = text[1:].strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise USSchemaError(
            f"Nasdaq historical row has non-numeric {field} for {symbol} on {date.isoformat()}"
        ) from exc
    if not math.isfinite(number):
        raise USSchemaError(
            f"Nasdaq historical row has non-finite {field} for {symbol} on {date.isoformat()}"
        )
    return number


def fetch_nasdaq_historical_chart(
    symbol: str,
    *,
    target_market_date: datetime.date,
    days: int = 730,
    timeout: int = 30,
    max_retries: int = 2,
    fetch_json=None,
) -> pd.DataFrame:
    """Fetch exact-symbol daily history from Nasdaq's official historical API.

    The endpoint is used only as a secondary source after a primary data
    integrity/schema failure.  Incomplete historical rows are discarded with
    a counted provenance record; an incomplete target row is always rejected.
    No price or volume is synthesized.
    """
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol or not all(
        character.isalnum() or character == "-" for character in normalized_symbol
    ):
        raise USSchemaError(f"invalid Nasdaq historical symbol: {symbol!r}")
    if not isinstance(target_market_date, datetime.date):
        raise USSchemaError("Nasdaq historical target date is invalid")
    if days <= 0:
        raise USSchemaError("Nasdaq historical lookback is invalid")
    if max_retries <= 0:
        raise USSchemaError("Nasdaq historical retry count is invalid")

    from_date = target_market_date - datetime.timedelta(days=days)
    query = urllib.parse.urlencode(
        {
            "assetclass": "stocks",
            "fromdate": from_date.isoformat(),
            "todate": target_market_date.isoformat(),
            "limit": "5000",
        }
    )
    source_url = (
        f"https://api.nasdaq.com/api/quote/{urllib.parse.quote(normalized_symbol)}/historical?{query}"
    )

    if fetch_json is not None:
        try:
            document = fetch_json()
        except TypeError:
            document = fetch_json(source_url)
        try:
            payload_sha256 = hashlib.sha256(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            raise USSchemaError("Nasdaq historical test payload is not canonical JSON") from exc
    else:
        request = urllib.request.Request(
            source_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Encoding": "identity",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
            },
        )
        last_exc = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw_payload = response.read(8 * 1024 * 1024 + 1)
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in {408, 425, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    continue
                raise USProviderOperationalError(
                    f"Nasdaq historical HTTP {exc.code} for {normalized_symbol}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    continue
                raise USProviderOperationalError(
                    f"Nasdaq historical transport error for {normalized_symbol}: {exc}"
                ) from exc
        else:
            raise USProviderOperationalError(
                f"Nasdaq historical transport error for {normalized_symbol}: {last_exc}"
            ) from last_exc
        if len(raw_payload) > 8 * 1024 * 1024:
            raise USSchemaError("Nasdaq historical response is too large")
        payload_sha256 = hashlib.sha256(raw_payload).hexdigest()
        try:
            document = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise USSchemaError("Nasdaq historical response is not valid JSON") from exc

    if not isinstance(document, Mapping) or not isinstance(document.get("data"), Mapping):
        raise USSchemaError(f"Nasdaq historical payload schema is invalid for {normalized_symbol}")
    data = document["data"]
    provider_symbol = str(data.get("symbol") or "").strip().upper()
    if provider_symbol != normalized_symbol:
        raise USSchemaError(
            f"Nasdaq historical symbol identity mismatch: requested {normalized_symbol}, got {provider_symbol or '<missing>'}"
        )
    table = data.get("tradesTable")
    rows = table.get("rows") if isinstance(table, Mapping) else None
    if not isinstance(rows, list):
        raise USSchemaError(f"Nasdaq historical rows schema is invalid for {normalized_symbol}")

    records: list[dict[str, object]] = []
    skipped_incomplete_rows = 0
    target_row_sha256: str | None = None
    field_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise USSchemaError(
                f"Nasdaq historical row {row_number} is invalid for {normalized_symbol}"
            )
        date_text = str(row.get("date") or row.get("Date") or "").strip()
        try:
            row_date = datetime.datetime.strptime(date_text, "%m/%d/%Y").date()
        except (TypeError, ValueError) as exc:
            raise USSchemaError(
                f"Nasdaq historical date is invalid for {normalized_symbol}: {date_text!r}"
            ) from exc
        if row_date > target_market_date:
            raise USSchemaError(
                f"Nasdaq historical future row for {normalized_symbol}: {row_date.isoformat()}"
            )
        raw_values = {field: row.get(source_field) for field, source_field in field_map.items()}
        incomplete = any(
            value is None
            or (isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA", "--"})
            for value in raw_values.values()
        )
        if incomplete:
            if row_date == target_market_date:
                raise USSchemaError(
                    f"Nasdaq historical target row is incomplete for {normalized_symbol}"
                )
            skipped_incomplete_rows += 1
            continue
        numeric = {
            field: _parse_nasdaq_historical_number(
                value,
                symbol=normalized_symbol,
                field=field,
                date=row_date,
            )
            for field, value in raw_values.items()
        }
        _validate_us_price_values(symbol=normalized_symbol, date=row_date, values=numeric)
        if row_date == target_market_date:
            target_row_sha256 = hashlib.sha256(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        records.append({"Date": row_date, **numeric})

    prepared = _prepare_us_history_frame(pd.DataFrame.from_records(records), normalized_symbol)
    prepared = compute_us_technical_indicators(prepared)
    prepared.attrs.update(
        {
            "source_schema_version": "nasdaq-historical-v1",
            "source_id": "nasdaqtrader:historical",
            "source_url": source_url,
            "source_identity": f"nasdaqtrader:historical:{payload_sha256}",
            "payload_sha256": payload_sha256,
            "provider_symbol": provider_symbol,
            "target_market_date": target_market_date.isoformat(),
            "target_observation": "present" if target_market_date in prepared.index else "absent",
            "target_row_sha256": target_row_sha256,
            "skipped_incomplete_rows": skipped_incomplete_rows,
        }
    )
    return prepared


def _prepare_us_history_frame(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize and validate every provider row before any filtering or deduplication."""
    if raw_df is None or raw_df.empty:
        empty = pd.DataFrame()
        empty.attrs["dropped_non_observation_placeholder_count"] = 0
        return empty
    if not isinstance(raw_df, pd.DataFrame):
        raise USSchemaError(f"US price payload is not tabular for {symbol}")
    if not set(_US_REQUIRED_PRICE_COLUMNS).issubset(raw_df.columns):
        raise USSchemaError(f"US price schema is incomplete for {symbol}")

    df = raw_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        date_values = list(df.index)
    elif "Date" in df.columns:
        date_values = list(df["Date"])
    elif len(df.index) and all(
        isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp))
        for value in df.index
    ):
        date_values = list(df.index)
    else:
        raise USSchemaError(f"US price payload has no date column for {symbol}")

    if len(date_values) != len(df):
        raise USSchemaError(f"US price date column length is invalid for {symbol}")

    df = df.reset_index(drop=True)
    normalized_dates = [_normalise_us_date(value, symbol) for value in date_values]
    df["Date"] = normalized_dates

    dropped_placeholder_rows = 0
    dropped_row_numbers: list[int] = []
    for row_number in range(len(df)):
        date = normalized_dates[row_number]
        raw_values = {
            field: df.at[row_number, field]
            for field in _US_REQUIRED_PRICE_COLUMNS
        }
        if is_explicit_non_observation_placeholder(raw_values):
            dropped_placeholder_rows += 1
            dropped_row_numbers.append(row_number)
            continue
        values = {}
        for field in _US_REQUIRED_PRICE_COLUMNS:
            number = _coerce_us_number(
                df.at[row_number, field],
                symbol=symbol,
                field=field,
                date=date,
            )
            values[field] = number
            df.at[row_number, field] = number
        _validate_us_price_values(symbol=symbol, date=date, values=values)

    if dropped_row_numbers:
        df = df.drop(index=dropped_row_numbers)

    duplicate_dates = df[df.duplicated(subset=["Date"], keep=False)]
    if not duplicate_dates.empty:
        for date, group in duplicate_dates.groupby("Date", sort=False):
            if group[list(_US_REQUIRED_PRICE_COLUMNS)].drop_duplicates().shape[0] > 1:
                raise USIntegrityError(
                    f"US price payload contains conflicting duplicate rows for "
                    f"{symbol} on {date.isoformat()}"
                )
        df = df.drop_duplicates(subset=["Date"], keep="first")

    prepared = df.set_index("Date").sort_index()
    prepared.attrs["dropped_non_observation_placeholder_count"] = dropped_placeholder_rows
    return prepared


def fetch_direct_yahoo_chart(
    symbol: str,
    range_str: str = "2y",
    max_retries: int = 3,
    *,
    target_market_date: datetime.date | None = None,
) -> pd.DataFrame:
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

    if not isinstance(doc, dict) or not isinstance(doc.get("chart"), dict):
        raise USSchemaError(f"Yahoo chart payload schema is invalid for {symbol}")
    res = doc["chart"].get("result")
    if not res:
        return pd.DataFrame()
    if not isinstance(res, list) or len(res) != 1 or not isinstance(res[0], dict):
        raise USSchemaError(f"Yahoo chart result schema is invalid for {symbol}")

    entry = res[0]
    timestamps = entry.get("timestamp")
    if not timestamps:
        empty = pd.DataFrame()
        empty.attrs["dropped_non_observation_placeholder_count"] = 0
        return empty
    if not isinstance(timestamps, list):
        raise USSchemaError(f"Yahoo chart timestamps are invalid for {symbol}")

    indicators = entry.get("indicators")
    quote_list = indicators.get("quote") if isinstance(indicators, dict) else None
    quote = quote_list[0] if isinstance(quote_list, list) and quote_list else None
    if not isinstance(quote, dict):
        raise USSchemaError(f"Yahoo chart quote schema is invalid for {symbol}")
    quote_fields = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    arrays = {field: quote.get(source_field) for field, source_field in quote_fields.items()}
    for field, values in arrays.items():
        if not isinstance(values, list) or len(values) < len(timestamps):
            raise USSchemaError(
                f"Yahoo chart {field} array is incomplete for {symbol}"
            )

    n = len(timestamps)
    records = []
    dropped_placeholder_count = 0
    for i in range(n):
        try:
            dt = datetime.datetime.fromtimestamp(
                timestamps[i], tz=datetime.timezone.utc
            ).astimezone(NEW_YORK).date()
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise USSchemaError(
                f"Yahoo chart timestamp is invalid for {symbol} at row {i}"
            ) from exc
        values = {
            field: arrays[field][i]
            for field in _US_REQUIRED_PRICE_COLUMNS
        }
        if is_explicit_non_observation_placeholder(values):
            dropped_placeholder_count += 1
            continue
        numeric = {
            field: _coerce_us_number(
                value,
                symbol=symbol,
                field=field,
                date=dt,
            )
            for field, value in values.items()
        }
        _validate_us_price_values(symbol=symbol, date=dt, values=numeric)
        records.append({
            "Date": dt,
            **numeric,
        })

    prepared = _prepare_us_history_frame(pd.DataFrame.from_records(records), symbol)
    prepared.attrs["dropped_non_observation_placeholder_count"] = (
        prepared.attrs.get("dropped_non_observation_placeholder_count", 0)
        + dropped_placeholder_count
    )
    return prepared


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
            raw_df = fetch_direct_yahoo_chart(
                symbol,
                range_str=range_str,
                target_market_date=target_market_date,
            )
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
        empty = pd.DataFrame()
        empty.attrs["dropped_non_observation_placeholder_count"] = 0
        return empty

    upstream_dropped_placeholder_count = int(
        getattr(raw_df, "attrs", {}).get(
            "dropped_non_observation_placeholder_count", 0
        )
        or 0
    )
    df = _prepare_us_history_frame(raw_df, symbol)
    dropped_placeholder_count = upstream_dropped_placeholder_count + int(
        df.attrs.get("dropped_non_observation_placeholder_count", 0)
    )
    if df.empty:
        df.attrs["dropped_non_observation_placeholder_count"] = dropped_placeholder_count
        return df

    if target_market_date is not None:
        df = df[df.index <= target_market_date].copy()
        df.attrs["dropped_non_observation_placeholder_count"] = dropped_placeholder_count
        if df.empty:
            return df

    # Compute technical indicators
    df = compute_us_technical_indicators(df)
    df.attrs["dropped_non_observation_placeholder_count"] = dropped_placeholder_count
    return df
