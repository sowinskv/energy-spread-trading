from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from src.ml.ensemble import EnsembleAnalyst, EnsembleClassifier, MultiHorizonEnsemble
from src.ml.features import EnergyFeatureEngineer, TimeSeriesImputer
from src.trading.metrics import asymmetric_trading_loss

AnalystModel = EnsembleAnalyst | MultiHorizonEnsemble | xgb.XGBRegressor


class FoldTrainer:
    def __init__(self, config: DictConfig, bool_cols: list[str], numeric_cols: list[str]) -> None:
        self.config = config
        self.bool_cols = bool_cols
        self.numeric_cols = numeric_cols
        self.preprocessor: Pipeline | None = None
        self.analyst: AnalystModel | None = None
        self.classifier: EnsembleClassifier | None = None

    def create_preprocessor(self) -> Pipeline:
        peak_hours = tuple(self.config.get("features", {}).get("peak_hours", [6, 7, 8, 9, 16, 17, 18, 19, 20]))
        self.preprocessor = Pipeline(
            [
                ("imputer", TimeSeriesImputer(bool_cols=self.bool_cols, numeric_cols=self.numeric_cols)),
                ("feature_engineer", EnergyFeatureEngineer(peak_hours=peak_hours)),
            ]
        )
        return self.preprocessor

    def prepare_fold_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        drop_cols: list[str] | None = None,
        add_back_cols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        self.create_preprocessor()
        leakage_cols = self.config.data.leakage_cols

        X_train = self.preprocessor.fit_transform(train_df.drop(columns=leakage_cols))
        y_train = train_df[self.config.data.target_col]

        X_test = self.preprocessor.transform(test_df.drop(columns=leakage_cols))
        y_test = test_df[self.config.data.target_col]

        # ablation: add back columns that were dropped (e.g. raw integers)
        if add_back_cols:
            for col in add_back_cols:
                if col == "hour":
                    X_train[col] = X_train.index.hour
                    X_test[col] = X_test.index.hour
                elif col == "month":
                    X_train[col] = X_train.index.month
                    X_test[col] = X_test.index.month
                elif col == "day_of_week":
                    X_train[col] = X_train.index.dayofweek
                    X_test[col] = X_test.index.dayofweek

        # ablation: drop specific feature columns
        if drop_cols:
            existing = [c for c in drop_cols if c in X_train.columns]
            X_train = X_train.drop(columns=existing)
            X_test = X_test.drop(columns=existing)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    def train_analyst(
        self,
        X_prim: pd.DataFrame,
        y_prim: pd.Series,
        primary_index: pd.DatetimeIndex | None = None,
        analyst_config: DictConfig | None = None,
        *,
        verbose: bool = True,
    ) -> AnalystModel:
        cfg = analyst_config or self.config

        if cfg.ensemble.enable and cfg.ensemble.multi_horizon:
            horizons = cfg.ensemble.get("horizons", [1, 4, 12, 24])
            self.analyst = MultiHorizonEnsemble(cfg, horizons=horizons)
            self.analyst.fit(X_prim, y_prim, primary_index, verbose=verbose)
        elif cfg.ensemble.enable:
            self.analyst = EnsembleAnalyst(cfg)
            self.analyst.fit(X_prim, y_prim, verbose=verbose)
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

    def train_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        analyst_config: DictConfig | None = None,
        *,
        verbose: bool = True,
    ) -> EnsembleClassifier:
        """Train classification ensemble for directional prediction."""
        cfg = analyst_config or self.config
        self.classifier = EnsembleClassifier(cfg)
        self.classifier.fit(X, y, verbose=verbose)
        return self.classifier

    def _two_stage_predict(
        self, X: pd.DataFrame, timestamps: pd.DatetimeIndex | None = None
    ) -> NDArray[np.floating]:
        """Combine classifier direction/confidence with regressor magnitude.

        synthetic_pred = (2*prob - 1) * |regression_pred|
        - Sign from classifier
        - Magnitude ∝ classifier_confidence × regression_magnitude
        - Works naturally with existing conviction metrics
        """
        prob = self.classifier.predict_proba(X)
        reg_pred = self.predict_analyst(X, timestamps)

        confidence_signed = 2 * prob - 1  # [-1, 1]: sign=direction, |.|=confidence
        synthetic = confidence_signed * np.abs(reg_pred)
        return synthetic

    def predict_final(
        self, X: pd.DataFrame, timestamps: pd.DatetimeIndex | None = None
    ) -> NDArray[np.floating]:
        """Final predictions — two-stage if classifier is trained, else regression."""
        if self.classifier is not None:
            return self._two_stage_predict(X, timestamps)
        return self.predict_analyst(X, timestamps)

    def run_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        analyst_config: DictConfig | None = None,
        drop_cols: list[str] | None = None,
        add_back_cols: list[str] | None = None,
    ) -> dict[str, pd.Series | NDArray[np.floating] | pd.DatetimeIndex | dict]:
        data = self.prepare_fold_data(train_df, test_df, drop_cols=drop_cols, add_back_cols=add_back_cols)

        self.train_analyst(
            data["X_train"],
            data["y_train"],
            primary_index=train_df.index,
            analyst_config=analyst_config,
        )

        # two-stage: train classifier on binary target for directional accuracy
        two_stage = self.config.ensemble.get("two_stage", False)
        if two_stage:
            self.train_classifier(
                data["X_train"],
                data["y_train"],
                analyst_config=analyst_config,
            )

        # --- train-set metrics (for overfitting detection) ---
        train_preds = self.predict_final(data["X_train"], train_df.index)
        y_train_np = np.array(data["y_train"])
        train_residuals = y_train_np - train_preds
        ss_res = np.sum(train_residuals**2)
        ss_tot = np.sum((y_train_np - np.mean(y_train_np)) ** 2)
        train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        train_mse = float(np.mean(train_residuals**2))

        test_preds = self.predict_final(data["X_test"], test_df.index)

        # --- test-set metrics ---
        y_test_np = np.array(data["y_test"])
        test_residuals = y_test_np - test_preds
        ss_res_t = np.sum(test_residuals**2)
        ss_tot_t = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
        test_r2 = 1 - (ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0.0
        test_mse = float(np.mean(test_residuals**2))

        return {
            "y_test": data["y_test"],
            "test_preds": test_preds,
            "test_timestamps": test_df.index,
            "fit_metrics": {
                "train_r2": train_r2,
                "train_mse": train_mse,
                "test_r2": test_r2,
                "test_mse": test_mse,
            },
        }
