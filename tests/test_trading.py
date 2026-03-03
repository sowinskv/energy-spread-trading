from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.trading.exit_strategies import (
    confidence_based_position_sizing,
    detect_market_volatility_regime,
    multi_horizon_consensus,
)
from src.trading.metrics import asymmetric_trading_loss, calculate_enhanced_meta_trading_metrics_with_exits


@pytest.fixture
def base_config():
    return OmegaConf.create(
        {
            "meta_model": {"confidence_threshold": 0.5},
            "trading": {"cost_per_mwh": 0.5, "position_size_mwh": 1.0, "currency": "EUR"},
            "exit_rules": {
                "enable": False,
                "max_concurrent_positions": 5,
                "trailing_stop": {"enable": False, "initial_stop": 8.0, "trail_amount": 3.0},
                "time_based": {
                    "profit_take_hours": 48,
                    "profit_take_threshold": 2.0,
                    "max_hold_hours": 168,
                    "initial_stop_loss": -8.0,
                    "stop_loss_decay": -0.02,
                },
                "position_management": {"enable_consensus_exits": False, "confidence_exit_threshold": 0.1},
            },
        }
    )


class TestTradingMetrics:
    def test_perfect_prediction_positive_pnl(self, base_config):
        y_true = pd.Series([1.0, -1.0, 2.0, -2.0] * 50)
        y_pred = np.array([1.0, -1.0, 2.0, -2.0] * 50)
        meta_probs = np.ones(200) * 0.9

        metrics = calculate_enhanced_meta_trading_metrics_with_exits(
            y_true,
            y_pred,
            meta_probs,
            base_config,
            use_exit_rules=False,
        )
        assert metrics["total_pnl"] > 0
        assert metrics["hit_rate"] > 50

    def test_no_trades_when_confidence_zero(self, base_config):
        y_true = pd.Series([1.0, -1.0, 2.0])
        y_pred = np.array([1.0, -1.0, 2.0])
        meta_probs = np.zeros(3)

        metrics = calculate_enhanced_meta_trading_metrics_with_exits(
            y_true,
            y_pred,
            meta_probs,
            base_config,
            use_exit_rules=False,
        )
        assert metrics["total_trades"] == 0

    def test_metrics_keys_present(self, base_config):
        y_true = pd.Series([1.0, -1.0] * 100)
        y_pred = np.array([1.0, -1.0] * 100)
        meta_probs = np.ones(200) * 0.7

        metrics = calculate_enhanced_meta_trading_metrics_with_exits(
            y_true,
            y_pred,
            meta_probs,
            base_config,
            use_exit_rules=False,
        )
        expected_keys = {
            "total_pnl",
            "sharpe_ratio",
            "sortino_ratio",
            "hit_rate",
            "max_drawdown",
            "percent_traded",
            "avg_position_size",
            "total_trades",
            "consensus_trades",
        }
        assert expected_keys == set(metrics.keys())


class TestAsymmetricLoss:
    def test_returns_grad_hess(self):
        y_true = np.array([1.0, -1.0, 0.5])
        y_pred = np.array([0.5, 0.5, -0.5])
        grad, hess = asymmetric_trading_loss(y_true, y_pred)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape

    def test_false_positive_penalized_more(self):
        y_true = np.array([-2.0])
        y_pred = np.array([1.0])
        grad_fp, _ = asymmetric_trading_loss(y_true, y_pred)

        y_true_fn = np.array([2.0])
        y_pred_fn = np.array([-1.0])
        grad_fn, _ = asymmetric_trading_loss(y_true_fn, y_pred_fn)

        assert abs(grad_fp[0]) > abs(grad_fn[0])


class TestPositionSizing:
    def test_high_confidence_larger_position(self):
        probs = np.array([0.9, 0.5, 0.6])
        sizes = confidence_based_position_sizing(probs, base_position=1.0)
        assert sizes[0] > sizes[1]

    def test_below_min_confidence_gets_base_size(self):
        probs = np.array([0.3])
        sizes = confidence_based_position_sizing(probs, base_position=1.0, min_confidence=0.5)
        assert sizes[0] == pytest.approx(1.0)


class TestVolatilityRegime:
    def test_short_series_returns_normal(self):
        returns = pd.Series([0.1] * 10)
        assert detect_market_volatility_regime(returns) == "normal"

    def test_stable_series_returns_low_or_normal(self):
        returns = pd.Series(np.random.RandomState(42).randn(500) * 0.01)
        regime = detect_market_volatility_regime(returns)
        assert regime in ("low_vol", "normal")


class TestMultiHorizonConsensus:
    def test_agreement_required(self):
        preds_1h = np.array([1.0, -1.0, 1.0])
        preds_4h = np.array([1.0, 1.0, 1.0])
        consensus = multi_horizon_consensus(preds_1h, preds_4h)
        assert not consensus[1]

    def test_both_zero_no_consensus(self):
        preds_1h = np.array([0.0])
        preds_4h = np.array([0.0])
        consensus = multi_horizon_consensus(preds_1h, preds_4h)
        assert not consensus[0]
