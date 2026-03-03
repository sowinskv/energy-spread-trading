from __future__ import annotations

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
            "model": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "random_state": 42},
            "ensemble": {"enable": False},
        }
    )


class TestFoldTrainer:
    def test_create_preprocessor(self, config):
        trainer = FoldTrainer(config, bool_cols=[], numeric_cols=["a", "b"])
        pipe = trainer.create_preprocessor()
        assert pipe is not None
        assert len(pipe.steps) == 2
