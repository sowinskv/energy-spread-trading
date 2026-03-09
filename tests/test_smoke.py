from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

# ── constants ────────────────────────────────────────────────────────────────
DATA_FILE = "data/energy_data.csv"

TRAIN_ROWS = 500   # ~21 days — enough for all 168-h rolling feature windows
PURGE_ROWS = 8     # small gap between train and test
TEST_ROWS = 48     # 2 days of test predictions

LEAKAGE_COLS = [
    "spread_SDAC_IDA1_PL", "IDA1_DE", "IDA1_PL", "IDA1_SK",
    "IDA2_DE", "IDA2_PL", "IDA2_SK", "IDA3_DE", "IDA3_PL", "IDA3_SK",
]
BOOL_COLS = ["IS_ACTIVE_DOWN_SDAC_PL", "IS_ACTIVE_UP_SDAC_PL"]

EXPECTED_RESULT_KEYS = {"y_test", "test_preds", "test_timestamps", "conformal_mask", "fit_metrics"}
EXPECTED_FIT_METRIC_KEYS = {"train_r2", "train_mse", "test_r2", "test_mse"}


# ── shared fixture ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def fold_data():
    """load a small slice of real data -- skipped if the file is absent."""
    try:
        from src.data.loader import load_and_format_raw_data
        df = load_and_format_raw_data(DATA_FILE)
    except FileNotFoundError:
        pytest.skip(f"data file not found: {DATA_FILE}")

    X_full = df.drop(columns=LEAKAGE_COLS)
    numeric_cols = [c for c in X_full.columns if c not in BOOL_COLS]

    train_df = df.iloc[:TRAIN_ROWS]
    test_df = df.iloc[TRAIN_ROWS + PURGE_ROWS : TRAIN_ROWS + PURGE_ROWS + TEST_ROWS]

    return {
        "train_df": train_df,
        "test_df": test_df,
        "bool_cols": BOOL_COLS,
        "numeric_cols": numeric_cols,
    }


# ── config helpers ────────────────────────────────────────────────────────────
def _make_config(*, two_stage: bool = False, conformal: bool = False) -> OmegaConf:
    return OmegaConf.create({
        "data": {
            "target_col": "spread_SDAC_IDA1_PL",
            "leakage_cols": LEAKAGE_COLS,
            "winsor_pct": 0.01,
        },
        "model": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "random_state": 42,
        },
        # model_params required by EnsembleClassifier when two_stage=True
        "model_params": {
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "min_child_weight": 1,
            },
        },
        "ensemble": {
            "enable": False,
            "two_stage": two_stage,
            # single fast model keeps the two-stage test under a few seconds
            "models": ["xgboost"] if two_stage else [],
        },
        "conformal": {
            "enable": conformal,
            "alpha": 0.50,   # wide alpha so mask isn't all-True during test
            "cal_days": 3,   # 72 h cal window — fits comfortably in 500-row train
        },
    })


def _make_trainer(fold_data, **cfg_kwargs):
    from src.ml.trainer import FoldTrainer
    cfg = _make_config(**cfg_kwargs)
    return FoldTrainer(cfg, fold_data["bool_cols"], fold_data["numeric_cols"])


# ── tests ─────────────────────────────────────────────────────────────────────
class TestRunFoldSmoke:

    def test_result_has_expected_keys(self, fold_data):
        """run_fold must return all expected top-level keys."""
        trainer = _make_trainer(fold_data)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert set(result.keys()) == EXPECTED_RESULT_KEYS

    def test_prediction_length_matches_test_set(self, fold_data):
        """test_preds and y_test must have the same length as the test slice."""
        trainer = _make_trainer(fold_data)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert len(result["test_preds"]) == TEST_ROWS
        assert len(result["y_test"]) == TEST_ROWS

    def test_timestamps_match_test_set(self, fold_data):
        """test_timestamps must cover every row of the test slice."""
        trainer = _make_trainer(fold_data)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert len(result["test_timestamps"]) == TEST_ROWS

    def test_fit_metrics_keys(self, fold_data):
        """fit_metrics must contain all four expected keys."""
        trainer = _make_trainer(fold_data)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert set(result["fit_metrics"].keys()) == EXPECTED_FIT_METRIC_KEYS

    def test_fit_metrics_are_finite(self, fold_data):
        """ALL fit_metrics values must be finite (catches silent NaN/Inf)."""
        trainer = _make_trainer(fold_data)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        for key, val in result["fit_metrics"].items():
            assert np.isfinite(val), f"fit_metric '{key}' is not finite: {val}"

    def test_conformal_mask_none_when_disabled(self, fold_data):
        """conformal_mask must be None when conformal.enable is False."""
        trainer = _make_trainer(fold_data, conformal=False)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert result["conformal_mask"] is None

    def test_conformal_mask_bool_array_when_enabled(self, fold_data):
        """conformal_mask must be a bool array of correct length when conformal is enabled."""
        trainer = _make_trainer(fold_data, conformal=True)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        mask = result["conformal_mask"]
        assert mask is not None, "conformal_mask is None even though conformal.enable=True"
        assert np.issubdtype(mask.dtype, np.bool_), f"unexpected mask dtype: {mask.dtype}"
        assert len(mask) == TEST_ROWS

    def test_two_stage_prediction_shape(self, fold_data):
        """2-stage (classifier + regressor) predictions must match test set length."""
        trainer = _make_trainer(fold_data, two_stage=True)
        result = trainer.run_fold(fold_data["train_df"], fold_data["test_df"])
        assert len(result["test_preds"]) == TEST_ROWS
