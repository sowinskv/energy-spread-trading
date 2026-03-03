from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from .position_manager import TrailingStopManager
from .exit_strategies import (
    calculate_dynamic_exits, consensus_exit_rules, confidence_based_position_sizing,
    detect_market_volatility_regime, multi_horizon_consensus, optimal_trading_hours
)


def calculate_enhanced_meta_trading_metrics_with_exits(
    y_true: pd.Series | NDArray[np.floating],
    y_pred: pd.Series | NDArray[np.floating],
    meta_probs: NDArray[np.floating],
    config: DictConfig,
    confidence_threshold: float = 0.5,
    cost_per_mwh: float = 0.5,
    position_size_mwh: float = 1.0,
    use_dynamic_thresholds: bool = True,
    use_confidence_sizing: bool = True,
    timestamps: pd.DatetimeIndex | None = None,
    use_exit_rules: bool = True,
) -> dict[str, float | int]:
    """Calculate comprehensive trading metrics with exit rule simulation"""
    
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    meta_probs_np = np.array(meta_probs)
    
    if use_dynamic_thresholds and len(y_true_np) > 100:
        returns = pd.Series(y_true_np)
        regime = detect_market_volatility_regime(returns)
        regime_adjustments = {'low_vol': 0.8, 'normal': 1.0, 'high_vol': 1.3}
        dynamic_threshold = confidence_threshold * regime_adjustments[regime]
    else:
        dynamic_threshold = confidence_threshold
    
    smooth_window = 4
    if len(y_pred_np) >= smooth_window:
        y_pred_4h = pd.Series(y_pred_np).rolling(smooth_window, min_periods=1).mean().values
        consensus_mask = multi_horizon_consensus(y_pred_np, y_pred_4h)
    else:
        consensus_mask = np.ones_like(y_pred_np, dtype=bool)
    
    if timestamps is not None:
        active_hours, peak_hours = optimal_trading_hours(timestamps)
        timing_multiplier = np.where(peak_hours, 1.2, np.where(active_hours, 1.0, 0.3))
    else:
        timing_multiplier = np.ones_like(y_pred_np)
    
    base_trade_mask = (meta_probs_np > dynamic_threshold).astype(int)
    consensus_trade_mask = base_trade_mask * consensus_mask.astype(int)
    
    if use_confidence_sizing:
        position_sizes = confidence_based_position_sizing(
            meta_probs_np, base_position=position_size_mwh, max_multiplier=2.0
        )
        position_sizes = position_sizes * timing_multiplier
    else:
        position_sizes = np.full_like(y_pred_np, position_size_mwh)
    
    if not use_exit_rules:
        intended_position = np.sign(y_pred_np)
        actual_position = intended_position * consensus_trade_mask * position_sizes
        raw_pnl = actual_position * y_true_np
        fees = consensus_trade_mask * cost_per_mwh * np.abs(actual_position)
        net_pnl = pd.Series(raw_pnl - fees)
    else:
        net_pnl = []
        active_positions = {}
        position_history = []
        trailing_stop = TrailingStopManager(config) if config.exit_rules.trailing_stop.enable else None
        max_positions = config.exit_rules.max_concurrent_positions
        
        for i in range(len(y_true_np)):
            current_price = y_true_np[i]
            current_time = timestamps[i] if timestamps is not None else i
            
            position_ids = list(active_positions.keys())
            if position_ids and use_exit_rules:
                entry_prices = [pos['entry_price'] for pos in active_positions.values()]
                entry_times = [pos['entry_time'] for pos in active_positions.values()]
                positions_array = [pos['size'] * pos['direction'] for pos in active_positions.values()]
                
                if timestamps is not None:
                    exits_time = calculate_dynamic_exits(
                        np.array(positions_array), 
                        np.full(len(positions_array), current_price),
                        pd.Series([current_time] * len(positions_array)),
                        np.array(entry_prices),
                        pd.Series(entry_times),
                        config
                    )
                else:
                    exits_time = np.zeros(len(positions_array))
                
                if (config.exit_rules.position_management.enable_consensus_exits and 
                    i >= smooth_window):
                    exits_consensus = consensus_exit_rules(
                        np.array(positions_array),
                        np.array([y_pred_np[i]] * len(positions_array)),
                        np.array([y_pred_4h[i]] * len(positions_array)),
                        config
                    )
                else:
                    exits_consensus = np.zeros(len(positions_array))
                
                trailing_exits = []
                if trailing_stop:
                    current_profits = [(current_price - pos['entry_price']) * pos['direction'] * pos['size'] - cost_per_mwh * pos['size'] 
                                     for pos in active_positions.values()]
                    trailing_exits = trailing_stop.update_stops(position_ids, current_profits)
                
                exit_mask = (exits_time == 1) | (exits_consensus == 1)
                for j, pos_id in enumerate(position_ids):
                    if exit_mask[j] or pos_id in trailing_exits:
                        pos = active_positions[pos_id]
                        exit_pnl = (current_price - pos['entry_price']) * pos['direction'] * pos['size'] - cost_per_mwh * pos['size']
                        position_history.append({
                            'entry_time': pos['entry_time'],
                            'exit_time': current_time,
                            'pnl': exit_pnl,
                            'size': pos['size'],
                            'direction': pos['direction'],
                            'exit_reason': 'dynamic_rule'
                        })
                        net_pnl.append(exit_pnl)
                        del active_positions[pos_id]
            
            if consensus_trade_mask[i] == 1 and len(active_positions) < max_positions:
                direction = np.sign(y_pred_np[i])
                size = position_sizes[i]
                active_positions[f"{current_time}_{i}"] = {
                    'entry_price': current_price,
                    'direction': direction,
                    'size': size,
                    'entry_time': current_time
                }
        
        final_price = y_true_np[-1] if len(y_true_np) > 0 else 0
        for pos in active_positions.values():
            exit_pnl = (final_price - pos['entry_price']) * pos['direction'] * pos['size'] - cost_per_mwh * pos['size']
            position_history.append({
                'entry_time': pos['entry_time'],
                'exit_time': current_time if 'current_time' in locals() else 'end',
                'pnl': exit_pnl,
                'size': pos['size'],
                'direction': pos['direction'],
                'exit_reason': 'end_of_period'
            })
            net_pnl.append(exit_pnl)
        
        net_pnl = pd.Series(net_pnl if net_pnl else [0])
    
    equity_curve = net_pnl.cumsum()
    
    if net_pnl.std() != 0:
        sharpe = (net_pnl.mean() / net_pnl.std()) * np.sqrt(8760)
    else:
        sharpe = 0
        
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    
    if use_exit_rules:
        pct_traded = len([p for p in position_history]) / len(y_true_np) * 100 if len(y_true_np) > 0 else 0
        hit_rate = (net_pnl > 0).mean() * 100 if len(net_pnl) > 0 else 0
        total_trades = len(position_history)
        avg_position_size = np.mean([abs(pos['size']) for pos in position_history]) if position_history else 0
    else:
        pct_traded = np.mean(consensus_trade_mask) * 100
        executed_trades = net_pnl[consensus_trade_mask == 1] if len(net_pnl) == len(consensus_trade_mask) else net_pnl
        hit_rate = (executed_trades > 0).mean() * 100 if len(executed_trades) > 0 else 0
        total_trades = int(np.sum(consensus_trade_mask))
        avg_position_size = np.mean(position_sizes[consensus_trade_mask == 1]) if np.sum(consensus_trade_mask) > 0 else 0
    
    if len(net_pnl) > 0:
        downside_returns = net_pnl[net_pnl < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino = (net_pnl.mean() / downside_returns.std()) * np.sqrt(8760)
        else:
            sortino = sharpe if sharpe != 0 else 0
    else:
        sortino = 0
    
    return {
        "total_pnl": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "percent_traded": pct_traded,
        "avg_position_size": avg_position_size,
        "total_trades": total_trades,
        "consensus_trades": total_trades if use_exit_rules else int(np.sum(consensus_mask & (base_trade_mask == 1)))
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