from __future__ import annotations

import numpy as np
import pandas as pd

from src.trading.metrics import asymmetric_trading_loss, calculate_conviction_metrics


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


class TestConvictionMetrics:
    def test_perfect_predictions_positive_pnl(self):
        y_true = pd.Series([1.0, -1.0, 2.0, -2.0] * 50)
        preds = np.array([1.0, -1.0, 2.0, -2.0] * 50)
        metrics = calculate_conviction_metrics(y_true, preds, cost_per_mwh=0.1)
        assert metrics["total_pnl"] > 0
        assert metrics["hit_rate"] > 50

    def test_returns_expected_keys(self):
        y_true = pd.Series([1.0, -1.0] * 100)
        preds = np.array([0.5, -0.5] * 100)
        metrics = calculate_conviction_metrics(y_true, preds)
        expected = {"total_pnl", "sharpe_ratio", "sortino_ratio", "hit_rate",
                    "max_drawdown", "total_trades", "percent_traded",
                    "avg_position_size", "net_pnls"}
        assert expected == set(metrics.keys())

    def test_larger_predictions_larger_positions(self):
        rng = np.random.RandomState(42)
        y_true = pd.Series(rng.randn(200) + 0.5)
        small_preds = rng.randn(200) * 0.1
        large_preds = rng.randn(200) * 5.0
        m_small = calculate_conviction_metrics(y_true, small_preds, cost_per_mwh=0.0, max_position=10.0)
        m_large = calculate_conviction_metrics(y_true, large_preds, cost_per_mwh=0.0, max_position=10.0)
        assert m_large["avg_position_size"] > m_small["avg_position_size"]

    def test_max_position_clips(self):
        y_true = pd.Series([1.0] * 200)
        huge_preds = np.array([100.0] * 200)
        metrics = calculate_conviction_metrics(y_true, huge_preds, max_position=1.5)
        assert metrics["avg_position_size"] <= 1.5 + 1e-6
