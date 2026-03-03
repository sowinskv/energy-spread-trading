from __future__ import annotations

from typing import List, Dict, Any
from omegaconf import DictConfig


class TrailingStopManager:
    """Manages trailing stop losses for active positions"""
    
    def __init__(self, config: DictConfig) -> None:
        self.initial_stop_pct = config.exit_rules.trailing_stop.initial_stop
        self.trail_pct = config.exit_rules.trailing_stop.trail_amount
        self.highest_profits = {}

    def update_stops(self, position_ids: List[str], current_profits: List[float]) -> List[str]:
        """Update trailing stops and return list of positions that should be exited"""
        exits = []
        for pos_id, profit in zip(position_ids, current_profits):
            if pos_id not in self.highest_profits:
                self.highest_profits[pos_id] = profit

            if profit > self.highest_profits[pos_id]:
                self.highest_profits[pos_id] = profit

            if self.highest_profits[pos_id] > self.initial_stop_pct:
                stop_level = self.highest_profits[pos_id] - self.trail_pct
                if profit <= stop_level:
                    exits.append(pos_id)
                    del self.highest_profits[pos_id]
        return exits