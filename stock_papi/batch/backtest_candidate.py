"""Build immutable full-backtest candidates from completed OOS evidence only."""

from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

from stock_papi.batch.backtest_store import BacktestStore, BacktestStoreError
from stock_papi.batch.runtime import job_namespace
from stock_papi.services.recommendation_engine import RECOMMENDATION_POLICY_VERSION


def _canonical(document):
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BacktestStoreError("backtest evidence is not finite JSON") from exc


def _load_object(path, label):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BacktestStoreError(f"{label} is unreadable") from exc
    if not isinstance(document, dict):
        raise BacktestStoreError(f"{label} must be an object")
    return document


def _write_immutable(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != content:
            raise BacktestStoreError("immutable OOS artifact conflict")
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _gzip(content):
    target = io.BytesIO()
    with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as stream:
        stream.write(content)
    return target.getvalue()


def _date(value, label):
    try:
        return datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise BacktestStoreError(f"invalid {label}") from exc


def _job_type(market):
    if market not in {"TW", "US"}:
        raise BacktestStoreError("unsupported backtest market")
    return "full_backtest" if market == "TW" else "full_backtest_us"


def _checkpoint(root, market):
    return _load_object(
        job_namespace(root, _job_type(market)).checkpoint,
        "full backtest checkpoint",
    )


def _validate_checkpoint(document, market):
    items = document.get("completed_items")
    item_count = document.get("item_count")
    if (
        document.get("schema_version") != 1
        or document.get("job_type") != _job_type(market)
        or (market == "US" and document.get("market") != market)
        or document.get("status") != "completed"
        or type(item_count) is not int
        or item_count < 1
        or document.get("next_index") != item_count
        or not isinstance(items, list)
        or len(items) != item_count
        or len(items) != len(set(items))
        or not all(re.fullmatch(r"[A-Z0-9.-]{1,12}", str(item)) for item in items)
    ):
        raise BacktestStoreError("full backtest is not complete")
    return tuple(items)


def _valid_prediction(value, symbol, cutoff):
    if not isinstance(value, dict):
        raise BacktestStoreError("OOS prediction is invalid")
    source_date = _date(value.get("source_market_date"), "OOS source_market_date")
    probability = value.get("probability")
    predicted_return = value.get("predicted_return")
    future_return = value.get("future_return")
    direction = value.get("direction")
    fold_index = value.get("fold_index")
    if (
        source_date > cutoff
        or type(probability) not in (int, float)
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
        or type(predicted_return) not in (int, float)
        or not math.isfinite(predicted_return)
        or type(future_return) not in (int, float)
        or not math.isfinite(future_return)
        or type(direction) is not int
        or direction not in (0, 1)
        or type(fold_index) is not int
        or fold_index < 0
    ):
        raise BacktestStoreError("OOS prediction is invalid")
    return {
        "symbol": symbol,
        "source_market_date": source_date.isoformat(),
        "probability": float(probability),
        "predicted_return": float(predicted_return),
        "future_return": float(future_return),
        "direction": direction,
        "fold_index": fold_index,
    }


def build_candidate(root, *, git_sha, market="TW", now=None):
    """Create an immutable candidate, never a ``latest`` promotion pointer."""
    if re.fullmatch(r"[0-9a-f]{40}", str(git_sha)) is None:
        raise BacktestStoreError("git_sha is invalid")
    root = Path(root)
    checkpoint = _checkpoint(root, market)
    items = _validate_checkpoint(checkpoint, market)
    cutoff = _date(checkpoint.get("cutoff"), "cutoff")
    required = ("dataset_manifest", "dataset_sha256", "model_version", "feature_schema_version")
    if any(key not in checkpoint for key in required):
        raise BacktestStoreError("full backtest checkpoint identity is invalid")
    result_root = job_namespace(root, _job_type(market)).output / checkpoint["dataset_sha256"] / "symbols"
    predictions = []
    fold_counts = []
    seen = set()
    for symbol in items:
        result = _load_object(result_root / f"{symbol}.json", f"full backtest result {symbol}")
        if any(result.get(key) != checkpoint[key] for key in required[:3]) or result.get("cutoff") != cutoff.isoformat():
            raise BacktestStoreError("full backtest result identity mismatch")
        backtest = result.get("backtest")
        oos = result.get("oos_predictions")
        if (
            not isinstance(backtest, dict)
            or backtest.get("five_session_gap") is not True
            or type(backtest.get("fold_count")) is not int
            or backtest["fold_count"] < 1
            or not isinstance(oos, list)
            or len(oos) < 30
        ):
            raise BacktestStoreError("full backtest OOS evidence is unavailable")
        fold_counts.append(backtest["fold_count"])
        for value in oos:
            prediction = _valid_prediction(value, symbol, cutoff)
            identity = (prediction["symbol"], prediction["source_market_date"])
            if identity in seen:
                raise BacktestStoreError("duplicate OOS prediction")
            seen.add(identity)
            predictions.append(prediction)
    if len(predictions) < 30:
        raise BacktestStoreError("insufficient OOS observations")
    predictions.sort(key=lambda item: (item["source_market_date"], item["symbol"]))
    target = [item["direction"] for item in predictions]
    probability = [item["probability"] for item in predictions]
    correct = sum(int((value >= 0.5) == label) for value, label in zip(probability, target))
    brier = sum((value - label) ** 2 for value, label in zip(probability, target)) / len(target)
    price_errors = [
        item["predicted_return"] - item["future_return"] for item in predictions
    ]
    price_mae = sum(abs(value) for value in price_errors) / len(price_errors)
    naive_mae = sum(abs(item["future_return"]) for item in predictions) / len(predictions)
    oos_document = {
        "schema_version": 1,
        "market": market,
        "dataset_manifest": checkpoint["dataset_manifest"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "model_version": checkpoint["model_version"],
        "cutoff": cutoff.isoformat(),
        "five_session_gap": True,
        "predictions": predictions,
    }
    compressed = _gzip(_canonical(oos_document))
    oos_sha = hashlib.sha256(compressed).hexdigest()
    store = BacktestStore(root, market)
    _write_immutable(store.root / "oos" / f"{oos_sha}.json.gz", compressed)
    generated_at = now or datetime.datetime.now(datetime.timezone.utc)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise BacktestStoreError("generated_at must be timezone-aware")
    candidate = {
        "schema_version": 1,
        "market": market,
        "dataset_manifest": checkpoint["dataset_manifest"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "model_version": checkpoint["model_version"],
        "feature_schema_version": checkpoint["feature_schema_version"],
        "recommendation_policy_version": RECOMMENDATION_POLICY_VERSION,
        "cutoff": cutoff.isoformat(),
        "data_start": predictions[0]["source_market_date"],
        "data_end": predictions[-1]["source_market_date"],
        "fold_count": min(fold_counts),
        "five_session_gap": True,
        "oos_observations": len(predictions),
        "oos_predictions_path": f"backtests/v1/oos/{oos_sha}.json.gz",
        "oos_predictions_sha256": oos_sha,
        "metrics": {
            "accuracy": correct / len(target) * 100.0,
            "brier": brier,
            "oos_observations": len(predictions),
            "price_mae": price_mae,
            "price_rmse": math.sqrt(
                sum(value * value for value in price_errors) / len(price_errors)
            ),
            "price_naive_mae": naive_mae,
            "price_mae_ratio": price_mae / naive_mae if naive_mae else 0.0,
        },
        "generated_at": generated_at.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
    }
    digest = store.write_candidate(candidate)
    return {**candidate, "candidate_sha256": digest}
