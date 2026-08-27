"""Immutable full-backtest candidates and gated promotion."""

import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path


REQUIRED_PROMOTION_GATES = frozenset(
    {
        "parity", "leakage", "calibration", "schema", "security", "quality",
        "price_quality",
    }
)


class BacktestStoreError(ValueError):
    """Backtest candidate 或 promotion 不合法。"""


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
        raise BacktestStoreError("backtest document is not finite JSON") from exc


def _date(value, label):
    try:
        parsed = datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise BacktestStoreError(f"invalid {label}") from exc
    if parsed.isoformat() != value:
        raise BacktestStoreError(f"invalid {label}")
    return parsed


def _timestamp(value):
    if (
        not isinstance(value, datetime.datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise BacktestStoreError("promoted_at must be timezone-aware")
    return value.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_candidate(document, market):
    if not isinstance(document, dict):
        raise BacktestStoreError("candidate must be an object")
    sha = r"[0-9a-f]{64}"
    manifest = rf"quant/v1/manifests/{market}-[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{12}}\.json"
    oos_path = rf"backtests/v1/oos/{sha}\.json\.gz"
    start = _date(document.get("data_start"), "data_start")
    end = _date(document.get("data_end"), "data_end")
    cutoff = _date(document.get("cutoff"), "cutoff")
    metrics = document.get("metrics")
    if (
        document.get("schema_version") != 1
        or document.get("market") != market
        or re.fullmatch(manifest, str(document.get("dataset_manifest") or "")) is None
        or re.fullmatch(sha, str(document.get("dataset_sha256") or "")) is None
        or not isinstance(document.get("model_version"), str)
        or not 1 <= len(document["model_version"]) <= 100
        or type(document.get("feature_schema_version")) is not int
        or document["feature_schema_version"] < 1
        or not isinstance(document.get("recommendation_policy_version"), str)
        or not 1 <= len(document["recommendation_policy_version"]) <= 100
        or not start <= end <= cutoff
        or type(document.get("fold_count")) is not int
        or document["fold_count"] < 1
        or document.get("five_session_gap") is not True
        or type(document.get("oos_observations")) is not int
        or document["oos_observations"] < 30
        or re.fullmatch(oos_path, str(document.get("oos_predictions_path") or "")) is None
        or re.fullmatch(sha, str(document.get("oos_predictions_sha256") or "")) is None
        or not isinstance(metrics, dict)
        or not metrics
        or not all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
        or re.fullmatch(r"[0-9a-f]{40}", str(document.get("git_sha") or "")) is None
    ):
        raise BacktestStoreError("candidate schema is invalid")
    try:
        generated = datetime.datetime.fromisoformat(
            str(document.get("generated_at")).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise BacktestStoreError("candidate generated_at is invalid") from exc
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise BacktestStoreError("candidate generated_at is invalid")
    _canonical(document)
    return document


def _write_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != content:
            raise BacktestStoreError("immutable candidate conflict")
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class BacktestStore:
    def __init__(self, root, market):
        if market not in {"TW", "US"}:
            raise BacktestStoreError("unsupported backtest market")
        self.market = market
        self.root = Path(root) / "publish" / "backtests" / "v1"
        self.latest_path = self.root / f"latest-{market}.json"

    def candidate_path(self, digest):
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise BacktestStoreError("invalid candidate hash")
        return self.root / "candidates" / f"{digest}.json"

    def write_candidate(self, document, candidate_id=None):
        _validate_candidate(document, self.market)
        content = _canonical(document)
        digest = hashlib.sha256(content).hexdigest()
        selected = digest if candidate_id is None else candidate_id
        path = self.candidate_path(selected)
        if selected != digest:
            if path.exists() and path.read_bytes() != content:
                raise BacktestStoreError("immutable candidate conflict")
            raise BacktestStoreError("candidate hash mismatch")
        _write_exclusive(path, content)
        return digest

    def _candidate(self, digest):
        path = self.candidate_path(digest)
        try:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise BacktestStoreError("candidate hash mismatch")
            document = json.loads(content)
        except FileNotFoundError as exc:
            raise BacktestStoreError("candidate does not exist") from exc
        except (OSError, ValueError) as exc:
            raise BacktestStoreError("candidate is unreadable") from exc
        return _validate_candidate(document, self.market)

    def promote(self, digest, *, gates, promoted_at):
        if (
            not isinstance(gates, dict)
            or set(gates) != REQUIRED_PROMOTION_GATES
            or not all(value is True for value in gates.values())
        ):
            raise BacktestStoreError("all promotion gates must pass")
        candidate = self._candidate(digest)
        pointer = {
            "schema_version": 1,
            "market": self.market,
            "candidate_path": f"backtests/v1/candidates/{digest}.json",
            "candidate_sha256": digest,
            "model_version": candidate["model_version"],
            "feature_schema_version": candidate["feature_schema_version"],
            "recommendation_policy_version": candidate[
                "recommendation_policy_version"
            ],
            "dataset_sha256": candidate["dataset_sha256"],
            "cutoff": candidate["cutoff"],
            "promoted_at": _timestamp(promoted_at),
            "gates": dict(sorted(gates.items())),
        }
        _write_atomic(self.latest_path, _canonical(pointer))
        return {**candidate, **pointer}

    def load_latest(self):
        if not self.latest_path.exists():
            return None
        try:
            pointer = json.loads(self.latest_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise BacktestStoreError("latest pointer is unreadable") from exc
        digest = str(pointer.get("candidate_sha256") or "")
        expected_path = f"backtests/v1/candidates/{digest}.json"
        if (
            pointer.get("schema_version") != 1
            or pointer.get("market") != self.market
            or pointer.get("candidate_path") != expected_path
        ):
            raise BacktestStoreError("latest pointer schema is invalid")
        candidate = self._candidate(digest)
        if (
            pointer.get("model_version") != candidate["model_version"]
            or pointer.get("feature_schema_version")
            != candidate["feature_schema_version"]
            or pointer.get("recommendation_policy_version")
            != candidate["recommendation_policy_version"]
            or pointer.get("dataset_sha256") != candidate["dataset_sha256"]
            or pointer.get("cutoff") != candidate["cutoff"]
        ):
            raise BacktestStoreError("latest pointer does not match candidate")
        return {**candidate, **pointer}


def assess_backtest_compatibility(
    backtest,
    *,
    expected_model_version,
    expected_feature_schema_version=1,
    expected_recommendation_policy_version="recommendation-v1",
):
    if (
        not isinstance(backtest, dict)
        or not isinstance(expected_model_version, str)
        or type(expected_feature_schema_version) is not int
        or not isinstance(expected_recommendation_policy_version, str)
    ):
        raise BacktestStoreError("invalid compatibility input")
    gates = backtest.get("gates")
    try:
        promoted_at = datetime.datetime.fromisoformat(
            str(backtest.get("promoted_at")).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        promoted_at = None
    promotion_verified = (
        re.fullmatch(r"[0-9a-f]{64}", str(backtest.get("candidate_sha256") or ""))
        is not None
        and isinstance(gates, dict)
        and set(gates) == REQUIRED_PROMOTION_GATES
        and all(value is True for value in gates.values())
        and promoted_at is not None
        and promoted_at.tzinfo is not None
        and promoted_at.utcoffset() is not None
    )
    if not promotion_verified:
        return {
            "compatible": False,
            "confidence_cap": "low",
            "strong_action_allowed": False,
            "reason": "backtest_not_promoted",
            "mismatch_fields": ["promotion_gates"],
        }
    mismatches = []
    if backtest.get("model_version") != expected_model_version:
        mismatches.append("model_version")
    if backtest.get("feature_schema_version") != expected_feature_schema_version:
        mismatches.append("feature_schema_version")
    if (
        backtest.get("recommendation_policy_version")
        != expected_recommendation_policy_version
    ):
        mismatches.append("recommendation_policy_version")
    if not mismatches:
        return {
            "compatible": True,
            "confidence_cap": "normal",
            "strong_action_allowed": True,
            "reason": None,
            "mismatch_fields": [],
            "backtest_as_of": backtest.get("cutoff"),
            "backtest_version": backtest.get("candidate_sha256"),
        }
    return {
        "compatible": False,
        "confidence_cap": "low",
        "strong_action_allowed": False,
        "reason": f"{mismatches[0]}_mismatch",
        "mismatch_fields": mismatches,
    }
