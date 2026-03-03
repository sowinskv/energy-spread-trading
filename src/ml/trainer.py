from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from ensemble_models import EnsembleAnalyst, MultiHorizonEnsemble
from features import EnergyFeatureEngineer, TimeSeriesImputer
from src.trading.metrics import asymmetric_trading_loss

AnalystModel = EnsembleAnalyst | MultiHorizonEnsemble | xgb.XGBRegressor


class FoldTrainer:
    def __init__(self, config: DictConfig, bool_cols: list[str], numeric_cols: list[str]) -> None:
        self.config = config
        self.bool_cols = bool_cols
        self.numeric_cols = numeric_cols
        self.preprocessor: Pipeline | None = None
        self.analyst: AnalystModel | None = None
        self.manager: xgb.XGBClassifier | None = None
        self.individual_preds_meta: dict[str, NDArray[np.floating]] = {}
        self.individual_preds_test: dict[str, NDArray[np.floating]] = {}

    def create_preprocessor(self) -> Pipeline:
        self.preprocessor = Pipeline(
            [
                ("imputer", TimeSeriesImputer(bool_cols=self.bool_cols, numeric_cols=self.numeric_cols)),
                ("feature_engineer", EnergyFeatureEngineer()),
            ]
        )
        return self.preprocessor

    def split_train_data(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_meta_idx = len(train_df) - (self.config.cv.meta_train_days * 24)
        primary_df = train_df.iloc[:split_meta_idx]
        meta_df = train_df.iloc[split_meta_idx:]
        return primary_df, meta_df

    def prepare_fold_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
        primary_df, meta_df = self.split_train_data(train_df)
        self.create_preprocessor()

        leakage_cols = self.config.data.leakage_cols

        X_prim = self.preprocessor.fit_transform(primary_df.drop(columns=leakage_cols))
        y_prim = primary_df[self.config.data.target_col]

        X_meta = self.preprocessor.transform(meta_df.drop(columns=leakage_cols))
        y_meta = meta_df[self.config.data.target_col]

        X_test = self.preprocessor.transform(test_df.drop(columns=leakage_cols))
        y_test = test_df[self.config.data.target_col]

        return {
            "primary_df": primary_df,
            "meta_df": meta_df,
            "X_prim": X_prim,
            "y_prim": y_prim,
            "X_meta": X_meta,
            "y_meta": y_meta,
            "X_test": X_test,
            "y_test": y_test,
        }

    def train_analyst(
        self,
        X_prim: pd.DataFrame,
        y_prim: pd.Series,
        primary_index: pd.DatetimeIndex | None = None,
        analyst_config: DictConfig | None = None,
    ) -> AnalystModel:
        cfg = analyst_config or self.config

        if cfg.ensemble.enable and cfg.ensemble.multi_horizon:
            horizons = cfg.ensemble.get("horizons", [1, 4, 12, 24])
            self.analyst = MultiHorizonEnsemble(cfg, horizons=horizons)
            self.analyst.fit(X_prim, y_prim, primary_index)
        elif cfg.ensemble.enable:
            self.analyst = EnsembleAnalyst(cfg)
            self.analyst.fit(X_prim, y_prim)
        else:
            self.analyst = xgb.XGBRegressor(**cfg.model, objective=asymmetric_trading_loss)
            self.analyst.fit(X_prim, y_prim)

        return self.analyst

    def predict_analyst(self, X: pd.DataFrame, timestamps: pd.DatetimeIndex | None = None) -> NDArray[np.floating]:
        if isinstance(self.analyst, MultiHorizonEnsemble):
            return self.analyst.predict(X, timestamps)
        return self.analyst.predict(X)

    def get_individual_predictions(self, X: pd.DataFrame) -> dict[str, NDArray[np.floating]]:
        if isinstance(self.analyst, EnsembleAnalyst):
            return self.analyst.get_individual_predictions(X)
        return {}

    @staticmethod
    def create_meta_labels(
        analyst_preds: NDArray[np.floating], y_true: pd.Series, cost_per_mwh: float = 0.5
    ) -> NDArray[np.integer]:
        raw_pnl = np.sign(analyst_preds) * y_true - cost_per_mwh
        return (raw_pnl > 0).astype(int)

    @staticmethod
    def enhance_features(
        X: pd.DataFrame,
        analyst_preds: NDArray[np.floating],
        individual_preds: dict[str, NDArray[np.floating]] | None = None,
        pred_col: str = "analyst_pred_ensemble",
    ) -> pd.DataFrame:
        X_enhanced = X.copy()
        X_enhanced[pred_col] = analyst_preds

        if individual_preds:
            for model_name, preds in individual_preds.items():
                X_enhanced[f"analyst_pred_{model_name}"] = preds

            pred_values = list(individual_preds.values())
            X_enhanced["prediction_variance"] = np.var(pred_values, axis=0)
            X_enhanced["prediction_range"] = np.max(pred_values, axis=0) - np.min(pred_values, axis=0)
            X_enhanced["model_consensus"] = np.mean([np.sign(pred) for pred in pred_values], axis=0)

        return X_enhanced

    def train_manager(
        self,
        X_meta_enhanced: pd.DataFrame,
        meta_labels: NDArray[np.integer],
        manager_params: dict[str, int | float] | None = None,
    ) -> xgb.XGBClassifier:
        if manager_params is None:
            params = dict(self.config.meta_model)
            params.pop("confidence_threshold", None)
        else:
            params = manager_params

        self.manager = xgb.XGBClassifier(**params)
        self.manager.fit(X_meta_enhanced, meta_labels)
        return self.manager

    def run_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        analyst_config: DictConfig | None = None,
    ) -> dict[str, pd.Series | NDArray[np.floating] | pd.DatetimeIndex | pd.DataFrame]:
        data = self.prepare_fold_data(train_df, test_df)

        self.train_analyst(
            data["X_prim"],
            data["y_prim"],
            primary_index=data["primary_df"].index,
            analyst_config=analyst_config,
        )

        meta_preds = self.predict_analyst(data["X_meta"], data["meta_df"].index)
        test_preds = self.predict_analyst(data["X_test"], test_df.index)

        cfg = analyst_config or self.config
        use_individual = cfg.ensemble.enable and not cfg.ensemble.get("multi_horizon", False)

        meta_individual = self.get_individual_predictions(data["X_meta"]) if use_individual else {}
        test_individual = self.get_individual_predictions(data["X_test"]) if use_individual else {}

        meta_labels = self.create_meta_labels(meta_preds, data["y_meta"])

        X_meta_enhanced = self.enhance_features(data["X_meta"], meta_preds, meta_individual)
        X_test_enhanced = self.enhance_features(data["X_test"], test_preds, test_individual)

        self.train_manager(X_meta_enhanced, meta_labels)
        test_manager_probs = self.manager.predict_proba(X_test_enhanced)[:, 1]

        return {
            "y_test": data["y_test"],
            "test_analyst_preds": test_preds,
            "test_manager_probs": test_manager_probs,
            "test_timestamps": test_df.index,
            "X_test_enhanced": X_test_enhanced,
        }
