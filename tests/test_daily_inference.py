import logging
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from stock_papi.quant.constants import MODEL_FEATURES
from stock_papi.quant.model import run_ai_engine, run_latest_inference


class FakeClassifier:
    fit_sizes = []

    def __init__(self, **_settings):
        self.training_size = 0
        self.feature_importances_ = np.arange(1, len(MODEL_FEATURES) + 1)

    def fit(self, features, _target):
        self.training_size = len(features)
        self.fit_sizes.append(self.training_size)
        return self

    def predict_proba(self, features):
        probability = 0.55 + self.training_size / 10000
        return np.tile([1 - probability, probability], (len(features), 1))


class FakeRegressor:
    fit_sizes = []

    def __init__(self, **_settings):
        self.training_size = 0

    def fit(self, features, _target):
        self.training_size = len(features)
        self.fit_sizes.append(self.training_size)
        return self

    def predict(self, features):
        return np.full(len(features), 0.03)


def frame_and_target():
    rows = 140
    frame = pd.DataFrame(
        {
            feature: np.linspace(index, index + 1, rows)
            for index, feature in enumerate(MODEL_FEATURES)
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="B"),
    )
    frame["Close"] = np.linspace(100.0, 120.0, rows)

    def add_target(value):
        result = value.copy()
        result["FUTURE_RET_5"] = np.where(np.arange(rows) % 2, 0.02, -0.02)
        result["T"] = np.arange(rows) % 2
        return result

    return frame, add_target


class DailyInferenceTests(unittest.TestCase):
    def setUp(self):
        FakeClassifier.fit_sizes = []
        FakeRegressor.fit_sizes = []
        self.lightgbm = SimpleNamespace(
            LGBMClassifier=FakeClassifier,
            LGBMRegressor=FakeRegressor,
        )
        self.logger = logging.getLogger("test-daily-inference")

    def test_fast_lane_fits_once_and_never_builds_walk_forward_folds(self):
        frame, add_target = frame_and_target()
        with patch.dict(sys.modules, {"lightgbm": self.lightgbm}):
            result = run_latest_inference(
                frame,
                add_prediction_target=add_target,
                pd=pd,
                np=np,
                logger=self.logger,
            )

        self.assertEqual(FakeClassifier.fit_sizes, [140])
        self.assertEqual(result["model_version"], "lgbm-5d-v1")
        self.assertAlmostEqual(frame["AI_P"].iloc[-1], result["probability"])
        self.assertEqual(frame["AI_P"].notna().sum(), 1)
        self.assertAlmostEqual(result["predicted_return_5d"], 0.03)
        self.assertAlmostEqual(result["predicted_price_5d"], 123.6)
        self.assertAlmostEqual(frame["AI_PRED_RET_5"].iloc[-1], 0.03)
        self.assertAlmostEqual(frame["AI_PRED_PRICE_5"].iloc[-1], 123.6)
        self.assertEqual(FakeRegressor.fit_sizes, [140])

    def test_fast_lane_and_full_engine_have_identical_latest_probability(self):
        fast_frame, add_target = frame_and_target()
        full_frame = fast_frame.copy()

        with patch.dict(sys.modules, {"lightgbm": self.lightgbm}):
            fast = run_latest_inference(
                fast_frame,
                add_prediction_target=add_target,
                pd=pd,
                np=np,
                logger=self.logger,
            )
            metrics = run_ai_engine(
                full_frame,
                add_prediction_target=add_target,
                build_time_splits=lambda _size: [(np.arange(80), np.arange(90, 120))],
                score_oos_predictions=lambda _returns, _probabilities: {
                    "trades": 0,
                    "strat_cum": 0.0,
                    "bh_cum": 0.0,
                    "sharpe": 0.0,
                    "mdd": 0.0,
                },
                pd=pd,
                np=np,
                logger=self.logger,
            )

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(fast["probability"], full_frame["AI_P"].iloc[-1])
        self.assertAlmostEqual(
            fast["predicted_price_5d"], full_frame["AI_PRED_PRICE_5"].iloc[-1]
        )
        self.assertEqual(FakeClassifier.fit_sizes, [140, 80, 140])
        self.assertEqual(FakeRegressor.fit_sizes, [140, 80, 140])

    def test_full_engine_can_return_immutable_oos_evidence_without_changing_default(self):
        frame, add_target = frame_and_target()
        with patch.dict(sys.modules, {"lightgbm": self.lightgbm}):
            metrics = run_ai_engine(
                frame,
                add_prediction_target=add_target,
                build_time_splits=lambda _size: [(np.arange(80), np.arange(90, 120))],
                score_oos_predictions=lambda _returns, _probabilities: {
                    "trades": 0,
                    "strat_cum": 0.0,
                    "bh_cum": 0.0,
                    "sharpe": 0.0,
                    "mdd": 0.0,
                },
                pd=pd,
                np=np,
                logger=self.logger,
                include_oos=True,
            )

        self.assertEqual(len(metrics["oos_predictions"]), 30)
        self.assertEqual(metrics["fold_count"], 1)
        self.assertTrue(metrics["five_session_gap"])
        self.assertEqual(metrics["oos_predictions"][0]["fold_index"], 0)
        self.assertAlmostEqual(
            metrics["oos_predictions"][0]["predicted_return"], 0.03
        )
        self.assertEqual(metrics["price_metrics"]["sample_count"], 30)
        self.assertAlmostEqual(metrics["price_metrics"]["mae"], 0.03)
        self.assertAlmostEqual(
            metrics["price_metrics"]["rmse"], 0.03605551275463989
        )
        self.assertAlmostEqual(metrics["price_metrics"]["naive_mae"], 0.02)
        self.assertAlmostEqual(metrics["price_metrics"]["mae_ratio"], 1.5)


if __name__ == "__main__":
    unittest.main()
