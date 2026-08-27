"""Build and validate promoted five-session prediction products."""

import datetime
import math
import re


INDEX_SYMBOLS = {
    "TW": frozenset({"TAIEX"}),
    "US": frozenset({"^GSPC", "^IXIC", "^DJI"}),
}
PROMOTION_GATES = frozenset(
    {"parity", "leakage", "calibration", "schema", "security", "quality", "price_quality"}
)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"invalid {label}")
    return result


def _date(value, label):
    try:
        result = datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if result.isoformat() != value:
        raise ValueError(f"invalid {label}")
    return result


def _timestamp(value):
    try:
        result = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid generated_at") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("invalid generated_at")
    return result


def _valid_symbol(market, symbol):
    if symbol in INDEX_SYMBOLS[market]:
        return True
    if market == "TW":
        return re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", symbol) is not None
    return re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z0-9]+)?", symbol) is not None


def _validate_entity(market, symbol, value, as_of):
    if not isinstance(value, dict) or value.get("symbol") != symbol or not _valid_symbol(market, symbol):
        raise ValueError("prediction symbol is invalid")
    observed = _date(value.get("as_of"), "prediction as_of")
    target = _date(value.get("target_session"), "prediction target_session")
    current = _number(value.get("current_price"), "current_price")
    probability = _number(value.get("up_probability"), "up_probability")
    predicted_return = _number(value.get("predicted_return_5d"), "predicted_return_5d")
    predicted_price = _number(value.get("predicted_price"), "predicted_price")
    change_pct = _number(value.get("predicted_change_pct"), "predicted_change_pct")
    expected_type = "market_index" if symbol in INDEX_SYMBOLS[market] else "security"
    if (
        observed != as_of
        or target <= observed
        or value.get("entity_type") != expected_type
        or current <= 0
        or not 0 <= probability <= 1
        or predicted_price <= 0
        or not math.isclose(predicted_price, current * (1 + predicted_return), rel_tol=1e-9)
        or not math.isclose(change_pct, predicted_return * 100, rel_tol=1e-9)
    ):
        raise ValueError("prediction entity is invalid")


def validate_prediction_product(document):
    if not isinstance(document, dict):
        raise ValueError("prediction product must be an object")
    market = document.get("market")
    entities = document.get("entities")
    if market not in INDEX_SYMBOLS or not isinstance(entities, dict) or not entities:
        raise ValueError("prediction product schema is invalid")
    as_of = _date(document.get("as_of"), "as_of")
    _timestamp(document.get("generated_at"))
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "absorb-five-session-predictions"
        or document.get("horizon_sessions") != 5
        or re.fullmatch(
            rf"quant/v1/manifests/{market}-[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{12}}\.json",
            str(document.get("source_manifest") or ""),
        ) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(document.get("source_manifest_sha256") or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(document.get("backtest_sha256") or "")) is None
        or not isinstance(document.get("model_version"), str)
        or not document["model_version"]
        or type(document.get("feature_schema_version")) is not int
        or document["feature_schema_version"] < 1
    ):
        raise ValueError("prediction product schema is invalid")
    for symbol, value in entities.items():
        _validate_entity(market, symbol, value, as_of)
    return document


def build_prediction_product(
    market,
    quant_manifest,
    snapshots,
    promoted_backtest,
    *,
    next_session,
    generated_at,
):
    if market not in INDEX_SYMBOLS or not isinstance(quant_manifest, dict):
        raise ValueError("prediction input is invalid")
    if (
        quant_manifest.get("market") != market
        or quant_manifest.get("schema_version") != 4
        or not isinstance(quant_manifest.get("symbols"), dict)
        or not isinstance(snapshots, dict)
        or not snapshots
    ):
        raise ValueError("quant manifest is invalid")
    if not isinstance(promoted_backtest, dict):
        raise ValueError("prediction promotion is invalid")
    gates = promoted_backtest.get("gates")
    if (
        promoted_backtest.get("market") != market
        or not isinstance(gates, dict)
        or set(gates) != PROMOTION_GATES
        or not all(value is True for value in gates.values())
        or re.fullmatch(r"[0-9a-f]{64}", str(promoted_backtest.get("candidate_sha256") or "")) is None
    ):
        raise ValueError("prediction promotion is invalid")
    as_of = _date(quant_manifest.get("observation_as_of"), "observation_as_of")
    generated = _timestamp(generated_at)
    target = next_session(market, as_of, 5)
    if not isinstance(target, datetime.date) or isinstance(target, datetime.datetime) or target <= as_of:
        raise ValueError("prediction target session is invalid")
    model_version = promoted_backtest.get("model_version")
    feature_schema = promoted_backtest.get("feature_schema_version")
    entities = {}
    for symbol, snapshot in snapshots.items():
        if not _valid_symbol(market, symbol):
            raise ValueError("prediction symbol is invalid")
        if symbol not in quant_manifest["symbols"] or not isinstance(snapshot, dict):
            raise ValueError("prediction source is invalid")
        rows = snapshot.get("daily")
        latest = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else None
        if (
            latest is None
            or snapshot.get("market") != market
            or snapshot.get("symbol") != symbol
            or snapshot.get("as_of") != as_of.isoformat()
            or quant_manifest["symbols"][symbol].get("as_of") != as_of.isoformat()
            or snapshot.get("model_version") != model_version
            or snapshot.get("feature_schema_version") != feature_schema
            or str(latest.get("Date"))[:10] != as_of.isoformat()
        ):
            raise ValueError("prediction source is invalid")
        current = _number(latest.get("Close"), "current price")
        probability = _number(latest.get("AI_P"), "prediction probability") / 100
        predicted_return = _number(latest.get("AI_PRED_RET_5"), "prediction return")
        predicted_price = _number(latest.get("AI_PRED_PRICE_5"), "prediction price")
        if not math.isclose(predicted_price, current * (1 + predicted_return), rel_tol=1e-9):
            raise ValueError("prediction price is inconsistent")
        entities[symbol] = {
            "symbol": symbol,
            "entity_type": "market_index" if symbol in INDEX_SYMBOLS[market] else "security",
            "as_of": as_of.isoformat(),
            "target_session": target.isoformat(),
            "current_price": current,
            "up_probability": probability,
            "predicted_return_5d": predicted_return,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_return * 100,
        }
    product = {
        "schema_version": 1,
        "kind": "absorb-five-session-predictions",
        "market": market,
        "as_of": as_of.isoformat(),
        "generated_at": generated.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizon_sessions": 5,
        "source_manifest": quant_manifest.get("source_manifest"),
        "source_manifest_sha256": quant_manifest.get("source_manifest_sha256"),
        "backtest_sha256": promoted_backtest["candidate_sha256"],
        "model_version": model_version,
        "feature_schema_version": feature_schema,
        "entities": entities,
    }
    return validate_prediction_product(product)
