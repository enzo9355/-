"""PIT-safe TEJ challenger features kept separate from the live model schema."""

from __future__ import annotations

import datetime

from .challengers import DIRECTION_FEATURES, run_feature_challenger
from .evaluation import evaluate_prediction_result
from .tej import TejSchemaError, validate_factor_snapshot


BASELINE_FEATURES = tuple(DIRECTION_FEATURES)
TEJ_CHALLENGER_MODEL_VERSION = "tej-challenger-lgbm-v1"
TEJ_FEATURE_SCHEMA_VERSION = 1


def _date(value):
    return datetime.date.fromisoformat(str(value).split("T", 1)[0])


def _timestamp(value):
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("TEJ factor available_at must include timezone")
    return parsed.astimezone(datetime.timezone.utc)


def build_tej_challenger_frame(price_frame, factor_document):
    """Join only factor observations visible at each price row's as-of date.

    A date-only price row is treated as visible at the beginning of that UTC
    date. This is conservative: an announcement later on the same day cannot
    leak into a close-derived training row without an explicit timestamp.
    """

    required = {"symbol", "source_market_date", *BASELINE_FEATURES}
    missing = sorted(required - set(price_frame.columns))
    if missing:
        return {
            "status": "BLOCKED",
            "reason": f"baseline price dataset is missing columns: {', '.join(missing)}",
            "frame": price_frame.copy(),
            "features": [],
            "eligible_count": 0,
            "coverage": 0.0,
        }
    try:
        validate_factor_snapshot(factor_document)
    except TejSchemaError:
        raise
    rows = list(factor_document["rows"])
    features = tuple(factor_document["feature_manifest"])
    result = price_frame.copy()
    for feature in features:
        result[feature] = float("nan")
    result["_tej_missing_count"] = len(features)
    if not features:
        return {
            "status": "BLOCKED",
            "reason": "no PIT-safe TEJ factor features are available",
            "frame": result,
            "features": [],
            "eligible_count": 0,
            "coverage": 0.0,
            "factor_lineage": {
                key: factor_document[key]
                for key in (
                    "source_normalized_sha256",
                    "field_map_sha256",
                    "entity_map_sha256",
                )
            },
        }

    by_symbol = {}
    for row in rows:
        try:
            symbol = str(row["symbol"])
            effective = _date(row["effective_date"])
            available = _timestamp(row["available_at"])
            factor_as_of = _timestamp(row["factor_as_of"])
        except (KeyError, TypeError, ValueError):
            continue
        by_symbol.setdefault(symbol, []).append(
            (effective, available, factor_as_of, row)
        )
    for index, source in result.iterrows():
        symbol = str(source["symbol"])
        market_date = _date(source["source_market_date"])
        prediction_time = datetime.datetime(
            market_date.year,
            market_date.month,
            market_date.day,
            tzinfo=datetime.timezone.utc,
        )
        candidates = [
            candidate
            for candidate in by_symbol.get(symbol, [])
            if candidate[0] <= market_date
            and candidate[1] <= prediction_time
            and candidate[2] <= prediction_time
        ]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2],
                str(candidate[3].get("source_payload_sha256") or ""),
            ),
        )
        count = 0
        for feature in features:
            value = (selected[3].get("factors") or {}).get(feature)
            if value is None:
                continue
            result.at[index, feature] = float(value)
            count += 1
        result.at[index, "_tej_missing_count"] = len(features) - count
    eligible = result[result["_tej_missing_count"] == 0]
    coverage_by_date = {}
    for market_date, group in result.groupby("source_market_date", sort=True):
        coverage_by_date[str(market_date)] = {
            "row_count": int(len(group)),
            "eligible_count": int((group["_tej_missing_count"] == 0).sum()),
        }
    return {
        "status": "available" if len(eligible) else "BLOCKED",
        "reason": None if len(eligible) else "no price rows have complete PIT TEJ coverage",
        "frame": result,
        "features": list(features),
        "eligible_count": int(len(eligible)),
        "coverage": float(len(eligible) / len(result)) if len(result) else 0.0,
        "coverage_by_date": coverage_by_date,
        "missing_count_column": "_tej_missing_count",
        "factor_lineage": {
            key: factor_document[key]
            for key in (
                "source_normalized_sha256",
                "field_map_sha256",
                "entity_map_sha256",
            )
        },
    }


def build_tej_challenger_spec(features):
    challenger_features = tuple(features or ())
    return {
        "schema_version": TEJ_FEATURE_SCHEMA_VERSION,
        "kind": "absorb-tej-challenger-feature-schema",
        "model_version": TEJ_CHALLENGER_MODEL_VERSION,
        "baseline_features": list(BASELINE_FEATURES),
        "challenger_features": list(challenger_features),
        "feature_count": len(challenger_features),
        "pit_policy": "available_at and factor_as_of <= source_market_date midnight UTC",
        "production_eligible": False,
        "promotion_allowed": False,
        "production_model_changed": False,
        "survivorship_risk": "UNVERIFIED_WITHOUT_PIT_UNIVERSE_AUDIT",
    }


def run_tej_challenger(
    joined,
    split_plan,
    *,
    model_factory=None,
    bootstrap_iterations=500,
):
    """Evaluate baseline and TEJ challenger on the identical eligible rows."""

    if joined.get("status") != "available":
        return {
            "status": "BLOCKED",
            "reason": joined.get("reason"),
            "model_version": TEJ_CHALLENGER_MODEL_VERSION,
            "production_eligible": False,
        }
    frame = joined["frame"]
    eligible = frame[frame["_tej_missing_count"] == 0].copy()
    features = tuple(joined.get("features") or ())
    if not features:
        return {
            "status": "BLOCKED",
            "reason": "TEJ challenger feature manifest is empty",
            "model_version": TEJ_CHALLENGER_MODEL_VERSION,
            "production_eligible": False,
        }
    baseline_raw = run_feature_challenger(
        eligible,
        split_plan,
        features=BASELINE_FEATURES,
        model_factory=model_factory,
        name="baseline_official_features",
    )
    challenger_raw = run_feature_challenger(
        eligible,
        split_plan,
        features=BASELINE_FEATURES + features,
        model_factory=model_factory,
        name="tej_challenger_lightgbm",
    )
    baseline = evaluate_prediction_result(
        eligible,
        baseline_raw,
        bootstrap_iterations=bootstrap_iterations,
    )
    challenger = evaluate_prediction_result(
        eligible,
        challenger_raw,
        bootstrap_iterations=bootstrap_iterations,
    )
    return {
        "status": "RUN"
        if baseline.get("status") == "RUN" and challenger.get("status") == "RUN"
        else "NOT_RUN",
        "model_version": TEJ_CHALLENGER_MODEL_VERSION,
        "feature_schema": build_tej_challenger_spec(features),
        "eligible_row_count": int(len(eligible)),
        "coverage": joined.get("coverage", 0.0),
        "coverage_by_date": joined.get("coverage_by_date", {}),
        "factor_lineage": joined.get("factor_lineage", {}),
        "survivorship_risk": "UNVERIFIED_WITHOUT_PIT_UNIVERSE_AUDIT",
        "baseline": baseline,
        "challenger": challenger,
        "promotion": {
            "production_eligible": False,
            "automatic_promotion": "DISABLED",
            "reason": "TEJ remains a research challenger until explicit PIT/OOS gates pass",
        },
    }


__all__ = [
    "BASELINE_FEATURES",
    "TEJ_CHALLENGER_MODEL_VERSION",
    "build_tej_challenger_frame",
    "build_tej_challenger_spec",
    "run_tej_challenger",
]
