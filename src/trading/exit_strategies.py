from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig


def calculate_dynamic_exits(
    positions: NDArray[np.floating],
    prices: NDArray[np.floating],
    timestamps: pd.Series,
    entry_prices: NDArray[np.floating],
    entry_times: pd.Series,
    config: DictConfig,
) -> NDArray[np.floating]:
    """Calculate position exits based on time-based rules"""
    exits = np.zeros_like(positions)
    current_pnl = (prices - entry_prices) * np.sign(positions)
    position_age_hours = (timestamps - entry_times).dt.total_seconds() / 3600

    for i in range(len(positions)):
        if positions[i] != 0:
            if abs(entry_prices[i]) > 0:
                profit_pct = (current_pnl[i] / abs(entry_prices[i])) * 100
            else:
                profit_pct = 0
            age = position_age_hours[i]

            if (age >= config.exit_rules.time_based.profit_take_hours and 
                profit_pct >= config.exit_rules.time_based.profit_take_threshold):
                exits[i] = 1
                continue

            stop_loss_pct = (config.exit_rules.time_based.initial_stop_loss + 
                           age * config.exit_rules.time_based.stop_loss_decay)
            if profit_pct <= stop_loss_pct:
                exits[i] = 1
                continue

            if age >= config.exit_rules.time_based.max_hold_hours:
                exits[i] = 1
                continue
    return exits


def consensus_exit_rules(
    current_positions: NDArray[np.floating],
    predictions_1h: NDArray[np.floating],
    predictions_4h: NDArray[np.floating],
    config: DictConfig,
) -> NDArray[np.floating]:
    """More conservative consensus exit rules"""
    exits = np.zeros_like(current_positions)
    direction_1h = np.sign(predictions_1h)
    direction_4h = np.sign(predictions_4h)
    
    confidence_threshold = config.exit_rules.position_management.confidence_exit_threshold
    avg_confidence = (np.abs(predictions_1h) + np.abs(predictions_4h)) / 2
    very_low_confidence = avg_confidence < (confidence_threshold * 0.5)

    for i in range(len(current_positions)):
        if current_positions[i] != 0:
            position_direction = np.sign(current_positions[i])

            if (very_low_confidence[i] and 
                direction_1h[i] == -position_direction and 
                direction_4h[i] == -position_direction):
                exits[i] = 1
    return exits


def confidence_based_position_sizing(
    meta_probs: NDArray[np.floating],
    base_position: float = 1.0,
    min_confidence: float = 0.5,
    max_multiplier: float = 2.0,
) -> NDArray[np.floating]:
    """Calculate position sizes based on model confidence"""
    confidence_excess = np.maximum(0, meta_probs - min_confidence)
    confidence_normalized = confidence_excess / (1 - min_confidence)
    
    multipliers = 1.0 + (max_multiplier - 1.0) * confidence_normalized
    positions = base_position * multipliers
    
    return np.clip(positions, 0.1, max_multiplier * base_position)


def detect_market_volatility_regime(returns: pd.Series, window: int = 72) -> str:
    """Detect current market volatility regime"""
    if len(returns) < window:
        return 'normal'
        
    recent_vol = returns.rolling(window).std().iloc[-1]
    historical_vol = returns.rolling(window*3).std().mean()
    
    vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
    
    if vol_ratio > 1.5:
        return 'high_vol'  
    elif vol_ratio < 0.7:
        return 'low_vol'   
    else:
        return 'normal'


def multi_horizon_consensus(
    predictions_1h: NDArray[np.floating],
    predictions_4h: NDArray[np.floating],
    consensus_threshold: float = 0.75,
) -> NDArray[np.bool_]:
    """Check for multi-horizon prediction consensus"""
    direction_1h = np.sign(predictions_1h)
    direction_4h = np.sign(predictions_4h)
    
    agreement = (direction_1h == direction_4h) & (direction_1h != 0)
    
    consensus_strength = (np.abs(predictions_1h) + np.abs(predictions_4h)) / 2
    strong_consensus = consensus_strength > np.percentile(consensus_strength, 75)
    
    return agreement & strong_consensus


def optimal_trading_hours(
    timestamp_index: pd.DatetimeIndex,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Determine optimal trading hours based on timestamp"""
    hours = timestamp_index.hour
    weekday = timestamp_index.dayofweek
    
    active_hours = (hours >= 6) & (hours <= 22)
    active_days = weekday < 5  
    
    peak_morning = (hours >= 8) & (hours <= 10)
    peak_evening = (hours >= 18) & (hours <= 20)
    peak_hours = peak_morning | peak_evening
    
    return active_hours & active_days, peak_hours & active_days