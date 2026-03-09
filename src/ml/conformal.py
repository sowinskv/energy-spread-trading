from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ConformalRegressor:
    """Split conformal prediction intervals for regression.

    Provides a marginal coverage guarantee:
        P(y ∈ [pred − q, pred + q]) ≥ 1 − alpha

    Calibrated on the last `cal_days` hours of training data (in-sample,
    so coverage is conservative — intervals are slightly too wide — but
    requires no second training pass).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.quantile_: float | None = None

    def calibrate(self, residuals: NDArray[np.floating]) -> ConformalRegressor:
        n = len(residuals)
        if n < 10:
            logger.warning("conformal: only %d calibration points — intervals may be unreliable", n)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile_ = float(np.quantile(np.abs(residuals), level))
        logger.debug("conformal quantile (alpha=%.2f): %.4f", self.alpha, self.quantile_)
        return self

    def predict_interval(
        self, predictions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        if self.quantile_ is None:
            raise ValueError("call calibrate() before predict_interval()")
        preds = np.asarray(predictions, dtype=float)
        return preds - self.quantile_, preds + self.quantile_

    def uncertain_mask(self, predictions: NDArray[np.floating]) -> NDArray[np.bool_]:
        """True where the prediction interval straddles zero — skip these trades."""
        lo, hi = self.predict_interval(np.asarray(predictions, dtype=float))
        return (lo < 0) & (hi > 0)

    @property
    def is_calibrated(self) -> bool:
        return self.quantile_ is not None
