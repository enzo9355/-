"""LightGBM training and out-of-sample evaluation."""

from stock_papi.quant.constants import MODEL_FEATURES


MODEL_VERSION = "lgbm-5d-v1"
FEATURE_SCHEMA_VERSION = 1
MODEL_SETTINGS = {
    "n_estimators": 80,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": 42,
    "verbose": -1,
}
FEATURE_NAMES = {
    "MA_5": "5日均線動能", "MA20": "月線趨勢支撐",
    "RET_1": "單日反轉動能", "RET_5": "5日價格動能",
    "RET_20": "月報酬動能", "RSI": "RSI 強弱度",
    "Volat": "波動收斂度", "RANGE_PCT": "日內振幅",
    "VOL_RATIO": "成交量趨勢", "VOL_CHG": "成交量變化",
    "INST_NET_RATIO": "法人買賣超", "MARGIN_CHG": "融資變化",
    "SHORT_CHG": "融券變化", "MACD_OSC": "MACD 柱狀體動能",
    "K": "KD K值", "D": "KD D值",
    "MARKET_RET_1": "大盤單日動能", "MARKET_RET_5": "大盤5日動能",
    "MARKET_RET_20": "大盤月動能", "MARKET_VOL_20": "大盤波動度",
    "ETF50_RET_5": "0050五日動能", "STOCK_VS_MARKET_5": "個股相對大盤強度",
    "DATA_PRICE_DIFF_PCT": "資料源價差", "DATA_PRICE_WARNING": "資料品質警示",
}


def _training_frame(frame, add_prediction_target):
    training = add_prediction_target(frame).dropna(
        subset=MODEL_FEATURES + ["FUTURE_RET_5", "T"]
    )
    if len(training) < 100 or training["T"].nunique() < 2:
        return None
    return training


def _fit_latest(frame, training, classifier, regressor):
    model = classifier(**MODEL_SETTINGS)
    model.fit(training[MODEL_FEATURES], training["T"].astype(int))
    probability = float(
        model.predict_proba(frame.iloc[[-1]][MODEL_FEATURES])[0, 1] * 100
    )
    price_model = regressor(**MODEL_SETTINGS)
    price_model.fit(training[MODEL_FEATURES], training["FUTURE_RET_5"])
    predicted_return = float(
        price_model.predict(frame.iloc[[-1]][MODEL_FEATURES])[0]
    )
    current_price = float(frame.iloc[-1]["Close"])
    predicted_price = current_price * (1.0 + predicted_return)
    importances = model.feature_importances_
    total_importance = max(float(importances.sum()), 1.0)
    top_features = [
        f"{FEATURE_NAMES.get(feature, feature)} (貢獻度: {importance / total_importance * 100:.1f}%)"
        for feature, importance in sorted(
            zip(MODEL_FEATURES, importances), key=lambda item: item[1], reverse=True
        )[:3]
    ]
    return probability, predicted_return, predicted_price, top_features


def run_latest_inference(frame, *, add_prediction_target, pd, np, logger):
    del pd
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        training = _training_frame(frame, add_prediction_target)
        if training is None:
            return None
        probability, predicted_return, predicted_price, top_features = _fit_latest(
            frame, training, LGBMClassifier, LGBMRegressor
        )
        if not all(map(np.isfinite, (probability, predicted_return, predicted_price))):
            return None
        frame["AI_P"] = np.nan
        frame["AI_PRED_RET_5"] = np.nan
        frame["AI_PRED_PRICE_5"] = np.nan
        frame.loc[frame.index[-1], "AI_P"] = probability
        frame.loc[frame.index[-1], "AI_PRED_RET_5"] = predicted_return
        frame.loc[frame.index[-1], "AI_PRED_PRICE_5"] = predicted_price
        return {
            "model_version": MODEL_VERSION,
            "probability": probability,
            "predicted_return_5d": predicted_return,
            "predicted_price_5d": predicted_price,
            "top_features": top_features,
            "training_observations": len(training),
        }
    except Exception:
        logger.error("最新推論失敗")
        return None


def run_ai_engine(
    frame, *, add_prediction_target, build_time_splits,
    score_oos_predictions, pd, np, logger, include_oos=False,
):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        training = _training_frame(frame, add_prediction_target)
        if training is None:
            return None
        oos_prob = pd.Series(np.nan, index=training.index, dtype=float)
        oos_return = pd.Series(np.nan, index=training.index, dtype=float)
        oos_fold = pd.Series(-1, index=training.index, dtype=int)
        for fold_index, (train_index, test_index) in enumerate(
            build_time_splits(len(training))
        ):
            fold = training.iloc[train_index]
            if fold["T"].nunique() < 2:
                continue
            model = LGBMClassifier(**MODEL_SETTINGS)
            model.fit(fold[MODEL_FEATURES], fold["T"].astype(int))
            oos_prob.iloc[test_index] = model.predict_proba(
                training.iloc[test_index][MODEL_FEATURES]
            )[:, 1]
            price_model = LGBMRegressor(**MODEL_SETTINGS)
            price_model.fit(fold[MODEL_FEATURES], fold["FUTURE_RET_5"])
            oos_return.iloc[test_index] = price_model.predict(
                training.iloc[test_index][MODEL_FEATURES]
            )
            oos_fold.iloc[test_index] = fold_index
        valid = oos_prob.notna() & oos_return.notna()
        if valid.sum() < 30:
            return None
        metrics = score_oos_predictions(
            training.loc[valid, "FUTURE_RET_5"], oos_prob.loc[valid]
        )
        latest_probability, latest_return, latest_price, top_features = _fit_latest(
            frame, training, LGBMClassifier, LGBMRegressor
        )
        actual_return = training.loc[valid, "FUTURE_RET_5"].astype(float)
        errors = oos_return.loc[valid] - actual_return
        naive_mae = float(actual_return.abs().mean())
        price_metrics = {
            "sample_count": int(valid.sum()),
            "mae": float(errors.abs().mean()),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "naive_mae": naive_mae,
            "mae_ratio": (
                float(errors.abs().mean() / naive_mae) if naive_mae > 0 else None
            ),
        }
        if not all(
            np.isfinite(value)
            for value in price_metrics.values()
            if value is not None
        ):
            return None
        frame["AI_P"] = np.nan
        frame["AI_PRED_RET_5"] = np.nan
        frame["AI_PRED_PRICE_5"] = np.nan
        frame.loc[oos_prob.loc[valid].index, "AI_P"] = oos_prob.loc[valid] * 100
        frame.loc[frame.index[-1], "AI_P"] = latest_probability
        frame.loc[oos_return.loc[valid].index, "AI_PRED_RET_5"] = oos_return.loc[valid]
        frame.loc[frame.index[-1], "AI_PRED_RET_5"] = latest_return
        frame.loc[oos_return.loc[valid].index, "AI_PRED_PRICE_5"] = (
            training.loc[valid, "Close"] * (1.0 + oos_return.loc[valid])
        )
        frame.loc[frame.index[-1], "AI_PRED_PRICE_5"] = latest_price
        metrics["top_features"] = top_features
        metrics["price_metrics"] = price_metrics
        if include_oos:
            # 僅供 immutable full-backtest artifact 使用；預設 daily 回傳不變。
            metrics["oos_predictions"] = [
                {
                    "source_market_date": pd.Timestamp(index).date().isoformat(),
                    "probability": float(oos_prob.loc[index]),
                    "predicted_return": float(oos_return.loc[index]),
                    "future_return": float(training.loc[index, "FUTURE_RET_5"]),
                    "direction": int(training.loc[index, "T"]),
                    "fold_index": int(oos_fold.loc[index]),
                }
                for index in oos_prob.loc[valid].index
            ]
            metrics["fold_count"] = len(
                {int(value) for value in oos_fold.loc[valid].tolist()}
            )
            metrics["five_session_gap"] = True
        if metrics["trades"] == 0:
            metrics["conclusion"] = "⏸️ 訊號空窗：模型未發現高勝率進場點，選擇空手觀望。"
        elif metrics["strat_cum"] > metrics["bh_cum"]:
            metrics["conclusion"] = (
                "✅ 策略優勢：高報酬且風險控制優異。"
                if metrics["sharpe"] > 1 else "✅ 擊敗大盤：能創造超額報酬。"
            )
        else:
            metrics["conclusion"] = (
                "🛡️ 下檔保護：大跌時具備避險作用。"
                if metrics["mdd"] > -15 else "⚠️ 模型失真：容易追高殺低。"
            )
        return metrics
    except Exception:
        logger.error("回測引擎失敗")
        return None
