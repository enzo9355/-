"""Independent baselines and LightGBM challengers fitted from PIT dataset rows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


DIRECTION_FEATURES = (
    "return_1",
    "momentum_5",
    "momentum_20",
    "volatility_20",
    "volume_ratio_20",
)
DATASET_COLUMNS = {
    "symbol",
    "source_market_date",
    "close",
    "volume",
    *DIRECTION_FEATURES,
    "future_return_5",
    "direction_5",
}
RANKING_DEPENDENCIES = (
    "tradable_universe",
    "listing_delisting",
    "suspension",
)


def load_dataset(manifest_path):
    """Read and hash-verify one immutable research dataset."""

    import numpy as np
    import pandas as pd

    manifest_path = Path(manifest_path)
    try:
        manifest_content = manifest_path.read_bytes()
        manifest = json.loads(manifest_content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("research dataset manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "absorb-pit-price-dataset"
        or re.fullmatch(r"[0-9a-f]{64}", manifest_path.stem) is None
        or hashlib.sha256(manifest_content).hexdigest() != manifest_path.stem
        or re.fullmatch(
            r"datasets/[0-9a-f]{64}\.jsonl\.gz",
            str(manifest.get("dataset_path") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get("dataset_sha256") or "")
        )
        is None
    ):
        raise ValueError("research dataset manifest identity is invalid")
    pit_root = manifest_path.parent.parent
    dataset_path = (pit_root / manifest["dataset_path"]).resolve()
    try:
        dataset_path.relative_to(pit_root.resolve())
    except ValueError as exc:
        raise ValueError("research dataset path escaped PIT root") from exc
    try:
        dataset_content = dataset_path.read_bytes()
    except OSError as exc:
        raise ValueError("research dataset is unavailable") from exc
    if (
        len(dataset_content) != manifest.get("dataset_size")
        or hashlib.sha256(dataset_content).hexdigest()
        != manifest["dataset_sha256"]
    ):
        raise ValueError("research dataset hash or size mismatch")
    frame = pd.read_json(
        dataset_path,
        lines=True,
        compression="gzip",
        dtype={
            "symbol": "string",
            "source_market_date": "string",
            "direction_5": "int8",
        },
        convert_dates=False,
    )
    if (
        set(frame.columns) != DATASET_COLUMNS
        or len(frame) != manifest.get("row_count")
        or frame["symbol"].nunique() != manifest.get("symbol_count")
        or not bool(frame["direction_5"].isin((0, 1)).all())
        or frame.duplicated(["symbol", "source_market_date"]).any()
    ):
        raise ValueError("research dataset schema is invalid")
    numeric = frame[
        [
            "close",
            "volume",
            *DIRECTION_FEATURES,
            "future_return_5",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("research dataset contains non-finite values")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["source_market_date"] = frame["source_market_date"].astype(str)
    frame["direction_5"] = frame["direction_5"].astype(int)
    return frame.reset_index(drop=True), manifest


class _ConstantPrior:
    def fit(self, values, target):
        import numpy as np

        target = np.asarray(target, dtype=float)
        self.probability = float(
            np.clip(target.mean(), 1e-6, 1.0 - 1e-6)
        )
        return self

    def predict_proba(self, values):
        import numpy as np

        probability = np.full(len(values), self.probability, dtype=float)
        return np.column_stack((1.0 - probability, probability))


class _NewtonLogistic:
    def __init__(self, regularization=1e-4, maximum_iterations=30):
        self.regularization = regularization
        self.maximum_iterations = maximum_iterations

    def fit(self, values, target):
        import numpy as np

        values = np.asarray(values, dtype=float)
        target = np.asarray(target, dtype=float)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        standardized = (values - self.mean) / self.scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        prevalence = float(np.clip(target.mean(), 1e-6, 1.0 - 1e-6))
        beta = np.zeros(design.shape[1], dtype=float)
        beta[0] = math.log(prevalence / (1.0 - prevalence))
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        for _ in range(self.maximum_iterations):
            linear = np.clip(design @ beta, -30.0, 30.0)
            fitted = 1.0 / (1.0 + np.exp(-linear))
            weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
            gradient = design.T @ (target - fitted) - penalty @ beta
            information = (
                design.T @ (weights[:, None] * design) + penalty
            )
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError as exc:
                raise ValueError("baseline logistic fit is singular") from exc
            beta += step
            if float(np.max(np.abs(step))) < 1e-7:
                break
        self.beta = beta
        return self

    def predict_proba(self, values):
        import numpy as np

        values = np.asarray(values, dtype=float)
        standardized = (values - self.mean) / self.scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        linear = np.clip(design @ self.beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return np.column_stack((1.0 - probability, probability))


def _indices_for_dates(frame, dates):
    import numpy as np

    return np.flatnonzero(frame["source_market_date"].isin(dates).to_numpy())


def _predict_positive_probability(model, values):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^X does not have valid feature names, but LGBMClassifier "
                r"was fitted with feature names$"
            ),
            category=UserWarning,
        )
        return model.predict_proba(values)[:, 1]


def _run_classifier(frame, plan, *, name, features, model_factory):
    import numpy as np

    validation_indices = []
    validation_probability = []
    fit_rows = []
    for fold in plan["walk_forward_folds"]:
        train_indices = _indices_for_dates(frame, fold["train_dates"])
        selected_indices = _indices_for_dates(
            frame, fold["validation_dates"]
        )
        model = model_factory()
        model.fit(
            frame.iloc[train_indices][list(features)].to_numpy(dtype=float),
            frame.iloc[train_indices]["direction_5"].to_numpy(dtype=int),
        )
        probability = _predict_positive_probability(
            model,
            frame.iloc[selected_indices][list(features)].to_numpy(dtype=float)
        )
        validation_indices.append(selected_indices)
        validation_probability.append(probability)
        fit_rows.append(int(len(train_indices)))

    development_indices = _indices_for_dates(
        frame, plan["development_dates"]
    )
    holdout_indices = _indices_for_dates(
        frame, plan["final_holdout_dates"]
    )
    final_model = model_factory()
    final_model.fit(
        frame.iloc[development_indices][list(features)].to_numpy(dtype=float),
        frame.iloc[development_indices]["direction_5"].to_numpy(dtype=int),
    )
    holdout_probability = _predict_positive_probability(
        final_model,
        frame.iloc[holdout_indices][list(features)].to_numpy(dtype=float)
    )
    return {
        "name": name,
        "status": "RUN",
        "fit_source": "dataset_features",
        "features": list(features),
        "fit_evidence": {
            "walk_forward_fit_rows": fit_rows,
            "final_fit_rows": int(len(development_indices)),
            "fit_count": len(fit_rows) + 1,
            "selection_uses_final_holdout": False,
        },
        "validation": {
            "indices": np.concatenate(validation_indices),
            "probability": np.concatenate(validation_probability),
        },
        "holdout": {
            "indices": holdout_indices,
            "probability": holdout_probability,
        },
    }


def run_baselines(frame, plan):
    local = frame.copy()
    local["mean_reversion_1"] = -local["return_1"]
    local["mean_reversion_5"] = -local["momentum_5"]
    return {
        "constant_prior": _run_classifier(
            local,
            plan,
            name="constant_prior",
            features=(),
            model_factory=_ConstantPrior,
        ),
        "momentum_logistic": _run_classifier(
            local,
            plan,
            name="momentum_logistic",
            features=("momentum_5", "momentum_20"),
            model_factory=_NewtonLogistic,
        ),
        "mean_reversion_logistic": _run_classifier(
            local,
            plan,
            name="mean_reversion_logistic",
            features=("mean_reversion_1", "mean_reversion_5"),
            model_factory=_NewtonLogistic,
        ),
    }


def _default_direction_factory():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm dependency is unavailable") from exc
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=20260717,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def run_direction_lightgbm(frame, plan, *, model_factory=None):
    return run_feature_challenger(
        frame,
        plan,
        features=DIRECTION_FEATURES,
        model_factory=model_factory,
        name="direction_lightgbm",
    )


def run_feature_challenger(
    frame,
    plan,
    *,
    features,
    model_factory=None,
    name="feature_challenger",
):
    """Run the existing purged walk-forward classifier on an explicit schema.

    The production direction model continues to call this with
    ``DIRECTION_FEATURES``. TEJ callers pass a separate feature tuple, so the
    baseline schema is not modified implicitly.
    """

    features = tuple(features or ())
    if not features or any(feature not in frame.columns for feature in features):
        return {
            "name": name,
            "status": "NOT_RUN",
            "reason": "explicit challenger feature schema is unavailable",
            "dependencies": list(features),
        }
    factory = model_factory or _default_direction_factory
    try:
        return _run_classifier(
            frame,
            plan,
            name=name,
            features=features,
            model_factory=factory,
        )
    except RuntimeError as exc:
        return {
            "name": name,
            "status": "NOT_RUN",
            "reason": str(exc),
            "dependencies": ["lightgbm"],
        }


def run_ranking_lightgbm(
    frame,
    plan,
    availability_audit,
    *,
    model_factory=None,
):
    requirements = availability_audit.get("requirements") or {}
    unavailable = [
        name
        for name in RANKING_DEPENDENCIES
        if (requirements.get(name) or {}).get("status") != "available"
    ]
    if unavailable:
        return {
            "name": "ranking_lightgbm",
            "status": "NOT_RUN",
            "reason": (
                "cross-sectional ranking requires frozen PIT universe, "
                "listing/delisting and suspension history"
            ),
            "dependencies": unavailable,
        }
    if model_factory is None:
        try:
            import lightgbm as lgb
        except ImportError:
            return {
                "name": "ranking_lightgbm",
                "status": "NOT_RUN",
                "reason": "lightgbm dependency is unavailable",
                "dependencies": ["lightgbm"],
            }
        factory = lambda: lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=60,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            random_state=20260717,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    else:
        factory = model_factory
    return _run_ranker(
        frame,
        plan,
        name="ranking_lightgbm",
        features=DIRECTION_FEATURES,
        model_factory=factory,
    )


def _ranking_fit_data(frame, indices, features):
    import numpy as np

    selected = frame.iloc[indices].sort_values(
        ["source_market_date", "symbol"]
    )
    percentile = selected.groupby("source_market_date")[
        "future_return_5"
    ].rank(method="first", pct=True)
    relevance = np.minimum(
        4, np.maximum(0, np.ceil(percentile.to_numpy() * 5).astype(int) - 1)
    )
    groups = (
        selected.groupby("source_market_date", sort=True)
        .size()
        .to_numpy(dtype=int)
    )
    return (
        selected.index.to_numpy(dtype=int),
        selected[list(features)].to_numpy(dtype=float),
        relevance,
        groups,
    )


def _run_ranker(frame, plan, *, name, features, model_factory):
    import numpy as np

    validation_indices = []
    validation_scores = []
    fit_rows = []
    for fold in plan["walk_forward_folds"]:
        train = _indices_for_dates(frame, fold["train_dates"])
        validation = _indices_for_dates(frame, fold["validation_dates"])
        _, train_values, relevance, groups = _ranking_fit_data(
            frame, train, features
        )
        ordered_validation, validation_values, _, _ = _ranking_fit_data(
            frame, validation, features
        )
        model = model_factory()
        model.fit(train_values, relevance, group=groups)
        validation_indices.append(ordered_validation)
        validation_scores.append(model.predict(validation_values))
        fit_rows.append(int(len(train_values)))

    development = _indices_for_dates(frame, plan["development_dates"])
    holdout = _indices_for_dates(frame, plan["final_holdout_dates"])
    _, train_values, relevance, groups = _ranking_fit_data(
        frame, development, features
    )
    ordered_holdout, holdout_values, _, _ = _ranking_fit_data(
        frame, holdout, features
    )
    model = model_factory()
    model.fit(train_values, relevance, group=groups)
    return {
        "name": name,
        "status": "RUN",
        "fit_source": "dataset_features",
        "features": list(features),
        "fit_evidence": {
            "walk_forward_fit_rows": fit_rows,
            "final_fit_rows": int(len(train_values)),
            "fit_count": len(fit_rows) + 1,
            "selection_uses_final_holdout": False,
            "relevance_labels": "within-date return quintiles 0-4",
        },
        "validation": {
            "indices": np.concatenate(validation_indices),
            "score": np.concatenate(validation_scores),
        },
        "holdout": {
            "indices": ordered_holdout,
            "score": model.predict(holdout_values),
        },
    }
