from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.loader import (
    get_expanding_walk_forward_splits,
    get_purged_walk_forward_splits,
    validate_config,
    validate_dataframe,
)


class TestValidateDataframe:
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)

    def test_high_nan_ratio_warns(self, capsys):
        df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, np.nan]})
        validate_dataframe(df, stage="test")
        assert "high NaN ratio" in capsys.readouterr().out

    def test_clean_dataframe_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        validate_dataframe(df)

    def test_duplicate_timestamps_warns(self, capsys):
        idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        validate_dataframe(df, stage="test")
        assert "duplicate timestamps" in capsys.readouterr().out


class TestValidateConfig:
    def test_valid_config(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "file_path": "data.csv",
                    "target_col": "spread",
                    "leakage_cols": ["a"],
                },
                "cv": {"train_days": 180, "test_days": 30, "purge_days": 7, "n_splits": 3},
                "model": {"n_estimators": 100},
            }
        )
        validate_config(cfg)

    def test_missing_key_raises(self):
        cfg = OmegaConf.create({"data": {"file_path": "x"}})
        with pytest.raises(KeyError, match="missing required config key"):
            validate_config(cfg)


class TestCVSplits:
    def test_expanding_split_count(self):
        splits = get_expanding_walk_forward_splits(
            df_length=10000, initial_train_days=90, test_days=30, purge_days=7, n_splits=3
        )
        assert len(splits) == 3

    def test_expanding_no_overlap(self):
        splits = get_expanding_walk_forward_splits(
            df_length=10000, initial_train_days=90, test_days=30, purge_days=7, n_splits=3
        )
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0], "train/test indices must not overlap"

    def test_expanding_train_grows(self):
        splits = get_expanding_walk_forward_splits(
            df_length=10000, initial_train_days=90, test_days=30, purge_days=7, n_splits=3
        )
        train_sizes = [len(s[0]) for s in splits]
        assert train_sizes == sorted(train_sizes), "expanding window train size must grow"

    def test_purged_split_count(self):
        splits = get_purged_walk_forward_splits(df_length=10000, train_days=90, test_days=30, purge_days=7, n_splits=3)
        assert len(splits) == 3

    def test_purged_no_overlap(self):
        splits = get_purged_walk_forward_splits(df_length=10000, train_days=90, test_days=30, purge_days=7, n_splits=3)
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0]

    def test_purged_gap_equals_purge_days(self):
        purge_days = 7
        splits = get_purged_walk_forward_splits(
            df_length=10000, train_days=90, test_days=30, purge_days=purge_days, n_splits=2
        )
        for train_idx, test_idx in splits:
            gap = test_idx[0] - train_idx[-1] - 1
            assert gap == purge_days * 24, f"gap should be {purge_days * 24}h, got {gap}"

    def test_too_small_dataset_raises(self):
        with pytest.raises(ValueError):
            get_purged_walk_forward_splits(df_length=100, train_days=90, test_days=30, purge_days=7, n_splits=3)
