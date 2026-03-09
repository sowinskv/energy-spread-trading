from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnsembleAnalyst:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.models = {}
        self.weights = None
        self.scaler = StandardScaler()
        self._init_models()

    def _init_models(self) -> None:
        for model_name in self.config.ensemble.models:
            if model_name == "xgboost":
                self.models["xgboost"] = xgb.XGBRegressor(
                    n_estimators=self.config.model_params.xgboost.n_estimators,
                    max_depth=self.config.model_params.xgboost.max_depth,
                    learning_rate=self.config.model_params.xgboost.learning_rate,
                    subsample=self.config.model_params.xgboost.subsample,
                    colsample_bytree=self.config.model_params.xgboost.colsample_bytree,
                    random_state=42,
                    objective="reg:squarederror",
                )

            elif model_name == "random_forest":
                self.models["random_forest"] = RandomForestRegressor(
                    n_estimators=self.config.model_params.random_forest.n_estimators,
                    max_depth=self.config.model_params.random_forest.max_depth,
                    min_samples_split=self.config.model_params.random_forest.min_samples_split,
                    min_samples_leaf=self.config.model_params.random_forest.min_samples_leaf,
                    max_features=self.config.model_params.random_forest.max_features,
                    random_state=44,
                    n_jobs=-1,
                )

            elif model_name == "ridge":
                self.models["ridge"] = Ridge(alpha=self.config.model_params.ridge.alpha, random_state=45)

            elif model_name == "extra_trees":
                self.models["extra_trees"] = ExtraTreesRegressor(
                    n_estimators=self.config.model_params.extra_trees.n_estimators,
                    max_depth=self.config.model_params.extra_trees.max_depth,
                    min_samples_split=self.config.model_params.extra_trees.min_samples_split,
                    min_samples_leaf=self.config.model_params.extra_trees.min_samples_leaf,
                    max_features=self.config.model_params.extra_trees.max_features,
                    random_state=47,
                    n_jobs=-1,
                )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> EnsembleAnalyst:
        X_scaled = self.scaler.fit_transform(X)

        individual_predictions = {}
        model_performance = {}

        for name, model in self.models.items():
            try:
                if name in ["ridge"]:
                    model.fit(X_scaled, y)
                    individual_predictions[name] = model.predict(X_scaled)
                else:
                    model.fit(X, y)
                    individual_predictions[name] = model.predict(X)

                pred = individual_predictions[name]
                mse = np.mean((y - pred) ** 2)
                mae = np.mean(np.abs(y - pred))
                r2 = 1 - (np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

                model_performance[name] = {"mse": mse, "mae": mae, "r2": r2, "score": 1.0 / (1.0 + mse)}

            except Exception as e:
                logger.error("model %s failed: %s", name, e)
                del self.models[name]

        if individual_predictions:
            self.weights = self._calculate_oof_weights(X, y)

        if verbose:
            from src.ui.display import model_table

            model_data = []
            ensemble_r2 = 0
            ensemble_mse = 0

            for name in sorted(model_performance.keys()):
                perf = model_performance[name]
                weight = self.weights.get(name, 0) if self.weights else 1 / len(model_performance)
                model_data.append({"name": name, "r2": perf["r2"], "mse": perf["mse"], "weight": weight})
                ensemble_r2 += weight * perf["r2"]
                ensemble_mse += weight * perf["mse"]

            model_table(model_data, ensemble_r2, ensemble_mse)

        try:
            mlflow.log_params(
                {
                    f"ensemble_models_{len(self.models)}": list(self.models.keys()),
                    f"ensemble_weights_{hash(str(self.weights))}": self.weights,
                }
            )
        except Exception as e:
            logger.warning("mlflow logging failed: %s", e)

        return self

    def _calculate_oof_weights(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Weight ensemble members by out-of-fold MSE (TimeSeriesSplit, 3 folds).

        Avoids rewarding memorization: weights are based on held-out performance,
        not training-set error.
        """
        tscv = TimeSeriesSplit(n_splits=3)
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # scaler is already fitted on full training X in fit() — reuse it
        X_scaled = self.scaler.transform(X_np)
        y_np = np.asarray(y)

        oof_mse: dict[str, list[float]] = {name: [] for name in self.models}

        for tr_idx, val_idx in tscv.split(X_np):
            X_tr, X_val = X_np[tr_idx], X_np[val_idx]
            X_tr_sc, X_val_sc = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y_np[tr_idx], y_np[val_idx]

            for name, model in self.models.items():
                try:
                    m = clone(model)
                    if name == "ridge":
                        m.fit(X_tr_sc, y_tr)
                        pred = m.predict(X_val_sc)
                    else:
                        m.fit(X_tr, y_tr)
                        pred = m.predict(X_val)
                    oof_mse[name].append(float(np.mean((y_val - pred) ** 2)))
                except Exception as e:
                    logger.warning("OOF fold failed for %s: %s", name, e)
                    oof_mse[name].append(float("inf"))

        avg_mse = {name: np.mean(errs) for name, errs in oof_mse.items() if errs}
        scores = {name: 1.0 / (1.0 + mse) for name, mse in avg_mse.items()}
        total = sum(scores.values())
        weights = {name: s / total for name, s in scores.items()}
        min_w = 0.1 / len(weights)
        weights = {name: max(w, min_w) for name, w in weights.items()}
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        if not self.models:
            raise ValueError("no trained models available")

        X_scaled = self.scaler.transform(X)

        predictions = {}
        for name, model in self.models.items():
            try:
                if name in ["ridge"]:
                    predictions[name] = model.predict(X_scaled)
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning("prediction failed for %s: %s", name, e)
                continue

        if not predictions:
            raise ValueError("no models could generate predictions")

        if self.weights is None:
            weights = {name: 1.0 / len(predictions) for name in predictions}
        else:
            weights = {name: self.weights.get(name, 0) for name in predictions}

        final_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            final_pred += weights[name] * pred

        return final_pred

    def get_individual_predictions(self, X: pd.DataFrame) -> dict[str, NDArray[np.floating]]:
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for name, model in self.models.items():
            try:
                if name in ["ridge"]:
                    predictions[name] = model.predict(X_scaled)
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning("prediction failed for %s: %s", name, e)
                predictions[name] = np.zeros(len(X))

        return predictions


class EnsembleClassifier:
    """Classification ensemble for directional prediction (P(spread > 0)).

    Mirrors EnsembleAnalyst structure but uses classifiers.
    Tree models are wrapped in CalibratedClassifierCV(cv=3, method='isotonic')
    so that output probabilities are reliable conviction signals.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.models: dict = {}
        self._raw_models: dict = {}
        self.weights: dict[str, float] | None = None
        self.scaler = StandardScaler()
        self._init_models()

    def _init_models(self) -> None:
        for model_name in self.config.ensemble.models:
            if model_name == "xgboost":
                raw = xgb.XGBClassifier(
                    n_estimators=self.config.model_params.xgboost.n_estimators,
                    max_depth=self.config.model_params.xgboost.max_depth,
                    learning_rate=self.config.model_params.xgboost.learning_rate,
                    subsample=self.config.model_params.xgboost.subsample,
                    colsample_bytree=self.config.model_params.xgboost.colsample_bytree,
                    min_child_weight=self.config.model_params.xgboost.get("min_child_weight", 1),
                    random_state=42,
                    objective="binary:logistic",
                    eval_metric="logloss",
                )
                self._raw_models["xgboost"] = raw
                self.models["xgboost"] = CalibratedClassifierCV(raw, cv=3, method="isotonic")

            elif model_name == "random_forest":
                raw = RandomForestClassifier(
                    n_estimators=self.config.model_params.random_forest.n_estimators,
                    max_depth=self.config.model_params.random_forest.max_depth,
                    min_samples_split=self.config.model_params.random_forest.min_samples_split,
                    min_samples_leaf=self.config.model_params.random_forest.min_samples_leaf,
                    max_features=self.config.model_params.random_forest.max_features,
                    random_state=44,
                    n_jobs=-1,
                )
                self._raw_models["random_forest"] = raw
                self.models["random_forest"] = CalibratedClassifierCV(raw, cv=3, method="isotonic")

            elif model_name == "extra_trees":
                raw = ExtraTreesClassifier(
                    n_estimators=self.config.model_params.extra_trees.n_estimators,
                    max_depth=self.config.model_params.extra_trees.max_depth,
                    min_samples_split=self.config.model_params.extra_trees.min_samples_split,
                    min_samples_leaf=self.config.model_params.extra_trees.min_samples_leaf,
                    max_features=self.config.model_params.extra_trees.max_features,
                    random_state=47,
                    n_jobs=-1,
                )
                self._raw_models["extra_trees"] = raw
                self.models["extra_trees"] = CalibratedClassifierCV(raw, cv=3, method="isotonic")

            elif model_name == "ridge":
                # logistic regression with C = 1/alpha — already well-calibrated, no wrapper needed
                alpha = self.config.model_params.ridge.alpha
                raw = LogisticRegression(
                    C=1.0 / max(alpha, 1e-6),
                    random_state=45,
                    max_iter=1000,
                )
                self._raw_models["logistic"] = raw
                self.models["logistic"] = raw

    def fit(self, X: pd.DataFrame, y: pd.Series, *, verbose: bool = True) -> EnsembleClassifier:
        y_binary = (y > 0).astype(int)
        X_scaled = self.scaler.fit_transform(X)

        individual_probs: dict[str, NDArray[np.floating]] = {}
        model_performance: dict[str, dict] = {}

        for name, model in list(self.models.items()):
            try:
                if name in ("logistic",):
                    model.fit(X_scaled, y_binary)
                    individual_probs[name] = model.predict_proba(X_scaled)[:, 1]
                else:
                    model.fit(X, y_binary)
                    individual_probs[name] = model.predict_proba(X)[:, 1]

                prob = individual_probs[name]
                accuracy = float(np.mean((prob > 0.5) == (y_binary == 1)))
                log_loss_val = float(
                    -np.mean(y_binary * np.log(prob + 1e-10) + (1 - y_binary) * np.log(1 - prob + 1e-10))
                )
                model_performance[name] = {"accuracy": accuracy, "log_loss": log_loss_val}

            except Exception as e:
                logger.error("classifier %s failed: %s", name, e)
                del self.models[name]

        if individual_probs:
            self.weights = self._calculate_oof_weights(X, y_binary)

        if verbose:
            from src.ui.display import model_table

            model_data = []
            ensemble_acc = 0.0

            for name in sorted(model_performance.keys()):
                perf = model_performance[name]
                weight = self.weights.get(name, 0) if self.weights else 1 / len(model_performance)
                model_data.append({"name": name, "r2": perf["accuracy"], "mse": perf["log_loss"], "weight": weight})
                ensemble_acc += weight * perf["accuracy"]

            model_table(model_data, ensemble_acc, 0)

        return self

    def _calculate_oof_weights(self, X: pd.DataFrame, y_binary: NDArray) -> dict[str, float]:
        """Weight classifier members by out-of-fold accuracy (TimeSeriesSplit, 3 folds).

        Uses raw (uncalibrated) base estimators for speed — calibration is only
        applied to the final fitted models, not during weight search.
        """
        tscv = TimeSeriesSplit(n_splits=3)
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # fit a fresh scaler on this X for OOF (the main scaler is fit in fit())
        X_scaled = StandardScaler().fit_transform(X_np)
        y_np = np.asarray(y_binary)

        oof_accs: dict[str, list[float]] = {name: [] for name in self._raw_models}

        for tr_idx, val_idx in tscv.split(X_np):
            X_tr, X_val = X_np[tr_idx], X_np[val_idx]
            X_tr_sc, X_val_sc = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y_np[tr_idx], y_np[val_idx]

            for name, base_model in self._raw_models.items():
                try:
                    m = clone(base_model)
                    if name == "logistic":
                        m.fit(X_tr_sc, y_tr)
                        prob = m.predict_proba(X_val_sc)[:, 1]
                    else:
                        m.fit(X_tr, y_tr)
                        prob = m.predict_proba(X_val)[:, 1]
                    acc = float(np.mean((prob > 0.5) == (y_val == 1)))
                    oof_accs[name].append(acc)
                except Exception as e:
                    logger.warning("OOF classifier fold failed for %s: %s", name, e)
                    oof_accs[name].append(0.5)

        avg_acc = {name: np.mean(accs) for name, accs in oof_accs.items() if accs}
        # weight by improvement over random (0.5 baseline)
        scores = {name: max(acc - 0.50, 1e-6) for name, acc in avg_acc.items()}
        total = sum(scores.values())
        if total <= 0:
            return {name: 1.0 / len(self._raw_models) for name in self._raw_models}
        weights = {name: s / total for name, s in scores.items()}
        min_w = 0.1 / len(weights)
        weights = {name: max(w, min_w) for name, w in weights.items()}
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.floating]:
        """Returns P(spread > 0) for each sample."""
        X_scaled = self.scaler.transform(X)

        probs: dict[str, NDArray[np.floating]] = {}
        for name, model in self.models.items():
            try:
                if name in ("logistic",):
                    probs[name] = model.predict_proba(X_scaled)[:, 1]
                else:
                    probs[name] = model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.warning("predict_proba failed for %s: %s", name, e)

        if not probs:
            return np.full(len(X), 0.5)

        weights = self.weights or {name: 1.0 / len(probs) for name in probs}

        final_prob = np.zeros(len(X))
        for name, prob in probs.items():
            final_prob += weights.get(name, 0) * prob

        return final_prob


class MultiHorizonEnsemble:
    def __init__(self, config: DictConfig, horizons: list[int] | None = None) -> None:
        self.config = config
        self.horizons = horizons or [1, 4, 12, 24]
        self.horizon_ensembles = {}
        self.horizon_weights = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, timestamps: pd.DatetimeIndex, *, verbose: bool = True
    ) -> MultiHorizonEnsemble:
        horizon_performance = {}

        for horizon in self.horizons:
            X_horizon = self._create_horizon_features(X, horizon)

            ensemble = EnsembleAnalyst(self.config)
            ensemble.fit(X_horizon, y, verbose=False)

            pred_horizon = ensemble.predict(X_horizon)
            mse = np.mean((y - pred_horizon) ** 2)
            r2 = 1 - (np.sum((y - pred_horizon) ** 2) / np.sum((y - np.mean(y)) ** 2))

            horizon_performance[horizon] = {"mse": mse, "r2": r2, "n_models": len(ensemble.models)}

            self.horizon_ensembles[horizon] = ensemble

        self.horizon_weights = self._calculate_horizon_weights(X, y, timestamps)

        if verbose:
            from src.ui.display import horizon_table

            horizon_data = []
            total_r2 = 0
            total_mse = 0

            for horizon in sorted(horizon_performance.keys()):
                perf = horizon_performance[horizon]
                weight = self.horizon_weights.get(horizon, 1 / len(self.horizons))
                horizon_data.append({"label": f"{horizon}h", "r2": perf["r2"], "mse": perf["mse"], "weight": weight})
                total_r2 += weight * perf["r2"]
                total_mse += weight * perf["mse"]

            horizon_table(horizon_data, total_r2, total_mse, len(self.horizon_ensembles), len(X))

        return self

    def _create_horizon_features(self, X: pd.DataFrame, horizon: int) -> pd.DataFrame:
        X_horizon = X.copy()

        rolling_cols = [col for col in X.columns if "rolling" in col or "lag" in col]

        for col in rolling_cols:
            if horizon > 1:
                X_horizon[f"{col}_h{horizon}"] = X[col].shift(horizon)

        return X_horizon.ffill().fillna(0)

    def _calculate_horizon_weights(
        self, X: pd.DataFrame, y: pd.Series, timestamps: pd.DatetimeIndex
    ) -> dict[int, float]:
        returns = y.diff().abs()
        recent_vol = returns.rolling(24).std().iloc[-1] if len(returns) > 24 else returns.std()

        if recent_vol > returns.quantile(0.7):
            weights = {1: 0.4, 4: 0.3, 12: 0.2, 24: 0.1}
        elif recent_vol < returns.quantile(0.3):
            weights = {1: 0.1, 4: 0.2, 12: 0.3, 24: 0.4}
        else:
            weights = {1: 0.25, 4: 0.25, 12: 0.25, 24: 0.25}

        return {h: weights.get(h, 0.25) for h in self.horizons}

    def predict(self, X: pd.DataFrame, timestamps: pd.DatetimeIndex) -> NDArray[np.floating]:
        horizon_predictions = {}

        for horizon, ensemble in self.horizon_ensembles.items():
            X_horizon = self._create_horizon_features(X, horizon)
            horizon_predictions[horizon] = ensemble.predict(X_horizon)

        if self.horizon_weights is None:
            weights = {h: 1.0 / len(self.horizons) for h in self.horizons}
        else:
            weights = self.horizon_weights

        final_pred = np.zeros(len(X))
        for horizon, pred in horizon_predictions.items():
            final_pred += weights[horizon] * pred

        return final_pred
