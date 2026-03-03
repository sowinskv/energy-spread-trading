from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.ml.trainer import FoldTrainer


@pytest.fixture
def config():
    return OmegaConf.create(
        {
            "data": {
                "target_col": "spread",
                "leakage_cols": ["spread"],
                "bool_cols": [],
            },
            "cv": {"meta_train_days": 2},
            "model": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "random_state": 42},
            "meta_model": {
                "n_estimators": 10,
                "max_depth": 2,
                "learning_rate": 0.1,
                "random_state": 42,
                "confidence_threshold": 0.5,
            },
            "ensemble": {"enable": False},
        }
    )


class TestMetaLabels:
    def test_profitable_trades_labeled_positive(self):
        preds = np.array([1.0, -1.0, 1.0])
        y_true = pd.Series([2.0, -2.0, 2.0])
        labels = FoldTrainer.create_meta_labels(preds, y_true, cost_per_mwh=0.5)
        assert (labels == 1).all()

    def test_unprofitable_trades_labeled_zero(self):
        preds = np.array([1.0, -1.0, 1.0])
        y_true = pd.Series([-2.0, 2.0, -2.0])
        labels = FoldTrainer.create_meta_labels(preds, y_true, cost_per_mwh=0.5)
        assert (labels == 0).all()

    def test_cost_matters(self):
        preds = np.array([1.0])
        y_true = pd.Series([0.4])
        labels_low = FoldTrainer.create_meta_labels(preds, y_true, cost_per_mwh=0.1)
        labels_high = FoldTrainer.create_meta_labels(preds, y_true, cost_per_mwh=0.5)
        assert labels_low[0] == 1
        assert labels_high[0] == 0


class TestEnhanceFeatures:
    def test_adds_prediction_column(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = np.array([0.1, 0.2, 0.3])
        enhanced = FoldTrainer.enhance_features(X, preds, pred_col="my_pred")
        assert "my_pred" in enhanced.columns
        np.testing.assert_array_equal(enhanced["my_pred"].values, preds)

    def test_individual_preds_add_consensus(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = np.array([0.1, 0.2, 0.3])
        individual = {
            "model_a": np.array([1.0, -1.0, 1.0]),
            "model_b": np.array([1.0, 1.0, -1.0]),
        }
        enhanced = FoldTrainer.enhance_features(X, preds, individual_preds=individual)
        assert "prediction_variance" in enhanced.columns
        assert "prediction_range" in enhanced.columns
        assert "model_consensus" in enhanced.columns

    def test_original_dataframe_unchanged(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = np.array([0.1, 0.2, 0.3])
        FoldTrainer.enhance_features(X, preds)
        assert list(X.columns) == ["a"]
