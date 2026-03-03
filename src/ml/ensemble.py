from __future__ import annotations

import logging
import pickle

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
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

            elif model_name == "lightgbm":
                self.models["lightgbm"] = lgb.LGBMRegressor(
                    n_estimators=self.config.model_params.lightgbm.n_estimators,
                    max_depth=self.config.model_params.lightgbm.max_depth,
                    learning_rate=self.config.model_params.lightgbm.learning_rate,
                    num_leaves=self.config.model_params.lightgbm.num_leaves,
                    subsample=self.config.model_params.lightgbm.subsample,
                    colsample_bytree=self.config.model_params.lightgbm.colsample_bytree,
                    min_child_samples=self.config.model_params.lightgbm.min_child_samples,
                    min_split_gain=self.config.model_params.lightgbm.min_split_gain,
                    random_state=43,
                    objective="regression",
                    boosting_type="gbdt",
                    verbosity=-1,
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
            self.weights = self._calculate_performance_weights(X, y, individual_predictions)

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

    def _calculate_performance_weights(
        self, X: pd.DataFrame, y: pd.Series, predictions: dict[str, NDArray[np.floating]]
    ) -> dict[str, float]:
        if len(predictions) < 2:
            return {name: 1.0 for name in self.models}

        recent_size = max(100, int(len(y) * 0.3))
        y_recent = y.iloc[-recent_size:]

        errors = {}
        performance_scores = {}

        for name, pred in predictions.items():
            if len(pred) >= recent_size:
                recent_pred = pred[-recent_size:]
                mse = np.mean((y_recent - recent_pred) ** 2)
                mae = np.mean(np.abs(y_recent - recent_pred))
                score = 1.0 / (1.0 + mse)

                errors[name] = score
                performance_scores[name] = {"mse": mse, "mae": mae, "score": score}

        total_weight = sum(errors.values())
        weights = {name: weight / total_weight for name, weight in errors.items()}

        min_weight = 0.1 / len(weights)
        for name in weights:
            weights[name] = max(weights[name], min_weight)

        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

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

    def save(self, filepath: str) -> None:
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except OSError as e:
            logger.error("failed to save ensemble to %s: %s", filepath, e)
            raise

    @classmethod
    def load(cls, filepath: str) -> EnsembleAnalyst:
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"ensemble file not found: {filepath}") from None
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"corrupted ensemble file '{filepath}': {e}") from e


class MultiHorizonEnsemble:
    def __init__(self, config: DictConfig, horizons: list[int] | None = None) -> None:
        self.config = config
        self.horizons = horizons or [1, 4, 12, 24]
        self.horizon_ensembles = {}
        self.horizon_weights = None

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.DatetimeIndex) -> MultiHorizonEnsemble:
        horizon_performance = {}

        for horizon in self.horizons:
            X_horizon = self._create_horizon_features(X, horizon)

            ensemble = EnsembleAnalyst(self.config)
            ensemble.fit(X_horizon, y, verbose=False)  # Suppress individual prints

            pred_horizon = ensemble.predict(X_horizon)
            mse = np.mean((y - pred_horizon) ** 2)
            r2 = 1 - (np.sum((y - pred_horizon) ** 2) / np.sum((y - np.mean(y)) ** 2))

            horizon_performance[horizon] = {"mse": mse, "r2": r2, "n_models": len(ensemble.models)}

            self.horizon_ensembles[horizon] = ensemble

        self.horizon_weights = self._calculate_horizon_weights(X, y, timestamps)

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
