from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def calculate_conviction_metrics(
    y_true: pd.Series | NDArray[np.floating],
    predictions: NDArray[np.floating],
    cost_per_mwh: float = 0.5,
    max_position: float = 2.0,
    vol_lookback: int = 168,
    min_conviction: float = 0.0,
) -> dict[str, float | int | NDArray]:
    """Conviction-based trading metrics — no binary gate, continuous sizing.

    Position size = clip(|pred| / rolling_std(pred), 0, max_position).
    Predictions with conviction < min_conviction are skipped (size=0).
    PnL per hour = sign(pred) * size * actual_spread - cost * size.
    """
    y = np.asarray(y_true, dtype=float)
    preds = np.asarray(predictions, dtype=float)
    n = len(y)

    # rolling prediction volatility for normalization
    pred_series = pd.Series(preds)
    rolling_vol = pred_series.rolling(vol_lookback, min_periods=24).std().bfill()
    rolling_vol = rolling_vol.clip(lower=1e-6).values

    # conviction = |prediction| / recent prediction volatility
    conviction = np.abs(preds) / rolling_vol
    position_sizes = np.clip(conviction, 0, max_position)

    # skip low-conviction predictions — adaptive to prediction volatility regime
    low_conviction = conviction < min_conviction
    position_sizes[low_conviction] = 0.0

    # direction from sign of prediction
    direction = np.sign(preds)

    # PnL calculation
    gross_pnl = direction * position_sizes * y
    costs = position_sizes * cost_per_mwh
    net_pnl = gross_pnl - costs

    equity = np.cumsum(net_pnl)

    # metrics
    total_pnl = float(equity[-1]) if n > 0 else 0.0

    std = np.std(net_pnl)
    sharpe = float((np.mean(net_pnl) / std) * np.sqrt(8760)) if std > 0 else 0.0

    downside = net_pnl[net_pnl < 0]
    ds_std = np.std(downside)
    sortino = float((np.mean(net_pnl) / ds_std) * np.sqrt(8760)) if len(downside) > 0 and ds_std > 0 else sharpe

    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - running_max))

    traded_mask = position_sizes > 0
    n_trades = int(np.sum(traded_mask))
    hit_rate = float(np.mean(net_pnl[traded_mask] > 0) * 100) if n_trades > 0 else 0.0
    pct_traded = float(np.mean(traded_mask) * 100)
    avg_size = float(np.mean(position_sizes[traded_mask])) if n_trades > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "total_trades": n_trades,
        "percent_traded": pct_traded,
        "avg_position_size": avg_size,
        "net_pnls": net_pnl,
    }


def asymmetric_trading_loss(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Custom loss function for trading that penalizes false positives heavily"""
    residual = y_pred - y_true
    grad = residual.copy()
    hess = np.ones_like(y_pred)

    fp_mask = (y_true < 0) & (y_pred > 0)
    fn_mask = (y_true > 0) & (y_pred < 0)

    grad[fp_mask] *= 5.0
    hess[fp_mask] *= 5.0
    grad[fn_mask] *= 2.0
    hess[fn_mask] *= 2.0

    magnitude_weight = 1.0 + (np.abs(y_true) / 10.0)
    grad = grad * magnitude_weight
    hess = hess * magnitude_weight
    return grad, hess
