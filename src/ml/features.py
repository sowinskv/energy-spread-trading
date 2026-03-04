from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """handles multi-tier imputation"""

    def __init__(self, bool_cols: list[str], numeric_cols: list[str]) -> None:
        self.bool_cols = bool_cols
        self.numeric_cols = numeric_cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TimeSeriesImputer:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.bool_cols] = X[self.bool_cols].ffill()
        for col in self.bool_cols:
            X[col] = X[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0}).fillna(0).astype(int)

        X[self.numeric_cols] = X[self.numeric_cols].interpolate(method="linear", limit=3, limit_direction="forward")

        for col in self.numeric_cols:
            X[col] = X[col].fillna(X[col].shift(168))
            X[col] = X[col].fillna(X[col].shift(24))
            X[col] = X[col].bfill()

        return X


DEFAULT_PEAK_HOURS = (6, 7, 8, 9, 16, 17, 18, 19, 20)


class EnergyFeatureEngineer(BaseEstimator, TransformerMixin):
    """handles temporal, spatial, regime, and fundamental features."""

    def __init__(self, peak_hours: tuple[int, ...] | None = None) -> None:
        self.peak_hours = peak_hours or DEFAULT_PEAK_HOURS

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> EnergyFeatureEngineer:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["hour"] = X.index.hour
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

        X["month"] = X.index.month
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

        X["day_of_week"] = X.index.dayofweek
        X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)
        X["is_peak"] = X.index.hour.isin(self.peak_hours).astype(int)

        # residual load is demand minus renewables
        X["residual_load"] = X["grid_demand_fcst"] - (X["fcst_pv_tot_gen"] + X["fcst_wi_tot_gen"])

        X["pv_gradient"] = X["fcst_pv_tot_gen"].diff()
        X["wind_gradient"] = X["fcst_wi_tot_gen"].diff()
        X["demand_gradient"] = X["grid_demand_fcst"].diff()
        X["residual_gradient"] = X["residual_load"].diff()

        X["spread_pl_de_sdac"] = X["SDAC_PL"] - X["SDAC_DE"]
        X["spread_pl_sk_sdac"] = X["SDAC_PL"] - X["SDAC_SK"]
        X["total_cross_border_flow"] = X["SE_SDAC_PL_LT"] + X["SE_SDAC_PL_SE"]

        # if this is highly positive, the grid is desperate for power
        X["system_imbalance"] = X["AC_UP_SDAC_PL"] - X["AC_DOWN_SDAC_PL"]
        X["net_position_momentum"] = X["NP_PL_GLOBAL_SDAC_PL"].diff()

        # --- stationarity: diffs capture direction of change, not regime-dependent levels ---
        X["spread_pl_de_change"] = X["spread_pl_de_sdac"].diff()
        X["spread_pl_sk_change"] = X["spread_pl_sk_sdac"].diff()
        X["system_imbalance_change"] = X["system_imbalance"].diff()
        X["cross_border_flow_change"] = X["total_cross_border_flow"].diff()
        X["sdac_return_24h"] = (X["SDAC_PL"] - X["SDAC_PL"].shift(24)) / (
            X["SDAC_PL"].shift(24).abs() + 1.0
        )

        # volatility & regime filters (strictly shifted by 24h to prevent leakage!)
        lags = [24, 48, 168]
        for lag in lags:
            X[f"sdac_pl_lag_{lag}h"] = X["SDAC_PL"].shift(lag)

        X["sdac_rolling_mean_24h"] = X["SDAC_PL"].shift(24).rolling(window=24).mean()
        X["sdac_rolling_std_24h"] = X["SDAC_PL"].shift(24).rolling(window=24).std()
        X["sdac_rolling_mean_168h"] = X["SDAC_PL"].shift(24).rolling(window=168).mean()

        sdac_shifted = X["SDAC_PL"].shift(24)

        # bollinger band width proxy (volatility indicator)
        X["sdac_bb_width"] = (X["sdac_rolling_std_24h"] * 2) / (X["sdac_rolling_mean_24h"].abs() + 1e-5)

        # ewma (exponentially weighted moving average) for short-term trend
        X["sdac_ewma_24h"] = sdac_shifted.ewm(span=24).mean()

        # renewable share of total generation (higher = more volatile prices)
        total_gen = X["fcst_pv_tot_gen"] + X["fcst_wi_tot_gen"]
        X["renewable_share"] = total_gen / (X["grid_demand_fcst"] + 1e-5)

        # residual load rolling stats (shifted to prevent leakage)
        res_shifted = X["residual_load"].shift(24)
        X["residual_rolling_mean_24h"] = res_shifted.rolling(window=24).mean()
        X["residual_rolling_std_24h"] = res_shifted.rolling(window=24).std()
        residual_rolling_mean_168h = res_shifted.rolling(window=168).mean()
        residual_rolling_std_168h = res_shifted.rolling(window=168).std()

        # stationarity: z-score contextualizes current value vs recent history
        X["residual_load_zscore"] = (X["residual_load"] - residual_rolling_mean_168h) / (
            residual_rolling_std_168h + 1e-5
        )

        # --- spread-specific features (target's own dynamics) ---
        # momentum: is the spread trending up or down?
        X["spread_momentum_24h"] = X["target_lag_24h"] - X["target_lag_48h"]
        X["spread_momentum_168h"] = X["target_lag_24h"] - X["target_lag_168h"]

        # mean-reversion z-score: how far is the spread from its rolling mean?
        X["spread_zscore"] = (X["target_lag_24h"] - X["target_rolling_mean_168h"]) / (
            X["target_rolling_std_48h"] + 1e-5
        )

        # acceleration: is momentum speeding up or slowing down?
        X["spread_acceleration"] = X["spread_momentum_24h"].diff(24)

        # clean up nans introduced by diff(), shift(), and rolling()
        # backfill catches the start of the dataset, fillna(0) acts as a final safety net
        X = X.bfill().fillna(0)

        return X
