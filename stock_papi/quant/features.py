"""Technical feature and prediction-target construction."""

from stock_papi.quant.constants import (
    DATA_QUALITY_FEATURES,
    MARKET_FEATURES,
    OPTION_FEATURES,
    PREDICTION_HORIZON,
)


CALCULATED_COLUMNS: tuple[str, ...] = (
    "MA_5", "MA20", "RET_1", "RET_5", "RET_20", "RANGE_PCT",
    "VOL_RATIO", "VOL_CHG", "INST_NET_RATIO", "MARGIN_CHG",
    "SHORT_CHG", "RSI", "Volat", "MACD_DIF", "MACD",
    "MACD_OSC", "K", "D", "BB_UP", "BB_DN",
)


def calc_all(frame, *, pd, np):
    frame = frame.copy()
    numeric_columns = [
        "Volume", "InstitutionalNet", "ForeignNet", "MarginBalance", "ShortBalance",
    ] + MARKET_FEATURES + OPTION_FEATURES + DATA_QUALITY_FEATURES
    for column in numeric_columns:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], 0
        ).fillna(0.0)
    close = frame["Close"]
    frame["MA_5"] = close.rolling(5).mean()
    frame["MA20"] = close.rolling(20).mean()
    frame["RET_1"] = close.pct_change(fill_method=None)
    frame["RET_5"] = close.pct_change(5, fill_method=None)
    frame["RET_20"] = close.pct_change(20, fill_method=None)
    frame["RANGE_PCT"] = (frame["High"] - frame["Low"]) / (close.abs() + 1e-9)
    frame["VOL_RATIO"] = frame["Volume"].rolling(5).mean() / (
        frame["Volume"].rolling(20).mean() + 1e-9
    )
    frame["VOL_CHG"] = frame["Volume"].pct_change(fill_method=None).replace(
        [np.inf, -np.inf], 0
    ).fillna(0).clip(-5, 5)
    frame["INST_NET_RATIO"] = (
        frame["InstitutionalNet"] / (frame["Volume"] + 1e-9)
    ).clip(-5, 5)
    frame["MARGIN_CHG"] = frame["MarginBalance"].replace(0, np.nan).pct_change(
        fill_method=None
    ).replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)
    frame["SHORT_CHG"] = frame["ShortBalance"].replace(0, np.nan).pct_change(
        fill_method=None
    ).replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    frame["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    frame["Volat"] = frame["RET_1"].rolling(20).std()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["MACD_DIF"] = ema12 - ema26
    frame["MACD"] = frame["MACD_DIF"].ewm(span=9, adjust=False).mean()
    frame["MACD_OSC"] = frame["MACD_DIF"] - frame["MACD"]
    high9 = frame["High"].rolling(9).max()
    low9 = frame["Low"].rolling(9).min()
    rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
    frame["K"] = rsv.ewm(com=2, adjust=False).mean()
    frame["D"] = frame["K"].ewm(com=2, adjust=False).mean()
    std20 = close.rolling(20).std()
    frame["BB_UP"] = frame["MA20"] + 2 * std20
    frame["BB_DN"] = frame["MA20"] - 2 * std20
    return frame.dropna()


def add_prediction_target(frame, *, np):
    result = frame.copy()
    future = result["Close"].shift(-PREDICTION_HORIZON) / result["Close"] - 1
    result["FUTURE_RET_5"] = future
    result["T"] = np.where(future.notna(), (future > 0).astype(float), np.nan)
    return result
