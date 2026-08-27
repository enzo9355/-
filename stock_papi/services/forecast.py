"""Five-session price and direction forecast from point-in-time OHLC history."""

from __future__ import annotations

import copy
import datetime
import math
from functools import lru_cache


MODEL_VERSION = "lgbm-ohlc-5d-v1"
HORIZON_SESSIONS = 5
MIN_TRAINING_ROWS = 120
FEATURES = (
    "ret_1",
    "ret_5",
    "ret_20",
    "ma_5_gap",
    "ma_20_gap",
    "ma_60_gap",
    "vol_10",
    "vol_20",
    "range_pct",
)


def _number(value):
    if isinstance(value, bool):
        raise ValueError("forecast number is invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("forecast number is invalid")
    return result


def _normalize_rows(rows):
    if not isinstance(rows, list):
        return None
    normalized = []
    previous = None
    try:
        for row in rows:
            if not isinstance(row, dict):
                return None
            date = datetime.date.fromisoformat(
                str(row.get("Date") or row.get("time") or "").split("T", 1)[0]
            )
            close = _number(row.get("Close", row.get("close")))
            open_value = _number(row.get("Open", row.get("open", close)))
            high = _number(row.get("High", row.get("high", close)))
            low = _number(row.get("Low", row.get("low", close)))
            if previous is not None and date <= previous:
                return None
            if high < max(open_value, close) or low > min(open_value, close):
                return None
            normalized.append((date.isoformat(), open_value, high, low, close))
            previous = date
    except (TypeError, ValueError, OverflowError):
        return None
    return tuple(normalized)


def _future_weekdays(last_date):
    result = []
    candidate = last_date
    while len(result) < HORIZON_SESSIONS:
        candidate += datetime.timedelta(days=1)
        if candidate.weekday() < 5:
            result.append(candidate)
    return result


@lru_cache(maxsize=512)
def _cached_forecast(rows, market):
    del market
    if len(rows) < MIN_TRAINING_ROWS + 65:
        return None
    try:
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.model_selection import TimeSeriesSplit

        frame = pd.DataFrame(
            rows, columns=("date", "open", "high", "low", "close")
        )
        close = frame["close"]
        features = pd.DataFrame(index=frame.index)
        features["ret_1"] = close.pct_change(1)
        features["ret_5"] = close.pct_change(5)
        features["ret_20"] = close.pct_change(20)
        features["ma_5_gap"] = close / close.rolling(5).mean() - 1
        features["ma_20_gap"] = close / close.rolling(20).mean() - 1
        features["ma_60_gap"] = close / close.rolling(60).mean() - 1
        features["vol_10"] = features["ret_1"].rolling(10).std()
        features["vol_20"] = features["ret_1"].rolling(20).std()
        features["range_pct"] = (frame["high"] - frame["low"]) / close
        future_return = close.shift(-HORIZON_SESSIONS) / close - 1
        training = features.copy()
        training["future_return"] = future_return
        training = training.dropna()
        latest = features.iloc[[-1]].dropna()
        if (
            len(training) < MIN_TRAINING_ROWS
            or latest.empty
            or (training["future_return"] > 0).nunique() < 2
        ):
            return None

        classifier_options = {
            "n_estimators": 80,
            "learning_rate": 0.04,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 15,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 1,
        }
        regressor_options = {
            **classifier_options,
            "objective": "huber",
        }
        oos_probability = pd.Series(np.nan, index=training.index, dtype=float)
        oos_return = pd.Series(np.nan, index=training.index, dtype=float)
        splitter = TimeSeriesSplit(n_splits=5, gap=HORIZON_SESSIONS)
        for train_index, test_index in splitter.split(training):
            train = training.iloc[train_index]
            if (train["future_return"] > 0).nunique() < 2:
                continue
            classifier = LGBMClassifier(**classifier_options)
            regressor = LGBMRegressor(**regressor_options)
            classifier.fit(train[list(FEATURES)], train["future_return"] > 0)
            regressor.fit(train[list(FEATURES)], train["future_return"])
            test_features = training.iloc[test_index][list(FEATURES)]
            oos_probability.iloc[test_index] = classifier.predict_proba(
                test_features
            )[:, 1]
            oos_return.iloc[test_index] = regressor.predict(test_features)

        valid = oos_probability.notna() & oos_return.notna()
        if valid.sum() < 30:
            return None
        actual_return = training.loc[valid, "future_return"]
        actual_direction = (actual_return > 0).astype(int)
        direction_accuracy = float(
            ((oos_probability.loc[valid] >= 0.5).astype(int) == actual_direction).mean()
            * 100
        )
        brier = float(
            ((oos_probability.loc[valid] - actual_direction) ** 2).mean()
        )
        price_mae = float(
            (oos_return.loc[valid] - actual_return).abs().mean() * 100
        )

        classifier = LGBMClassifier(**classifier_options)
        regressor = LGBMRegressor(**regressor_options)
        classifier.fit(training[list(FEATURES)], training["future_return"] > 0)
        regressor.fit(training[list(FEATURES)], training["future_return"])
        probability = float(classifier.predict_proba(latest[list(FEATURES)])[0, 1])
        predicted_return = float(regressor.predict(latest[list(FEATURES)])[0])
        lower = max(float(training["future_return"].quantile(0.05)), -0.25)
        upper = min(float(training["future_return"].quantile(0.95)), 0.25)
        predicted_return = min(max(predicted_return, lower), upper)

        probability_pct = round(min(max(probability * 100, 1.0), 99.0), 1)
        current_price = float(frame.iloc[-1]["close"])
        target_price = round(current_price * (1 + predicted_return), 2)
        expected_return_pct = round(predicted_return * 100, 2)
        if probability_pct >= 55 and predicted_return > 0:
            direction = "up"
        elif probability_pct <= 45 and predicted_return < 0:
            direction = "down"
        else:
            direction = "neutral"

        last_date = datetime.date.fromisoformat(str(frame.iloc[-1]["date"]))
        future_dates = _future_weekdays(last_date)
        points = [{"time": last_date.isoformat(), "value": current_price}]
        for index, date in enumerate(future_dates, 1):
            value = current_price + (target_price - current_price) * (
                index / HORIZON_SESSIONS
            )
            points.append({"time": date.isoformat(), "value": round(value, 2)})

        return {
            "status": "published",
            "as_of": last_date.isoformat(),
            "horizon_sessions": HORIZON_SESSIONS,
            "direction": direction,
            "probability_up_pct": probability_pct,
            "target_price": target_price,
            "expected_return_pct": expected_return_pct,
            "model_version": MODEL_VERSION,
            "validation": {
                "method": "five-fold expanding time-series split with five-session gap",
                "oos_samples": int(valid.sum()),
                "direction_accuracy_pct": round(direction_accuracy, 1),
                "brier": round(brier, 3),
                "price_mae_pct": round(price_mae, 2),
            },
            "points": points,
        }
    except Exception:
        return None


def build_five_session_forecast(rows, *, market):
    """Return a deterministic, OOS-scored forecast or ``None``."""
    if market not in {"TW", "US"}:
        return None
    normalized = _normalize_rows(rows)
    if normalized is None:
        return None
    return copy.deepcopy(_cached_forecast(normalized, market))
