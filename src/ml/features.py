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

        X[self.numeric_cols] = X[self.numeric_cols].interpolate(method="linear", limit=3)

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

        # --- production constraint: current-day SDAC prices are NOT known ---
        # trading decisions (offers) are made before SDAC clears, so the most
        # recent available SDAC data is from yesterday.  shift ALL SDAC-related
        # columns by 24h up-front so every downstream feature is leak-free.
        sdac_cols = [c for c in X.columns if "SDAC" in c]
        for col in sdac_cols:
            X[col] = X[col].shift(24)

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

        X["residual_load"] = X["grid_demand_fcst"] - (X["fcst_pv_tot_gen"] + X["fcst_wi_tot_gen"])

        X["pv_gradient"] = X["fcst_pv_tot_gen"].diff()
        X["wind_gradient"] = X["fcst_wi_tot_gen"].diff()
        X["demand_gradient"] = X["grid_demand_fcst"].diff()
        X["residual_gradient"] = X["residual_load"].diff()

        X["spread_pl_de_sdac"] = X["SDAC_PL"] - X["SDAC_DE"]
        X["spread_pl_sk_sdac"] = X["SDAC_PL"] - X["SDAC_SK"]
        X["total_cross_border_flow"] = X["SE_SDAC_PL_LT"] + X["SE_SDAC_PL_SE"]

        X["system_imbalance"] = X["AC_UP_SDAC_PL"] - X["AC_DOWN_SDAC_PL"]
        X["net_position_momentum"] = X["NP_PL_GLOBAL_SDAC_PL"].diff()

        X["spread_pl_de_change"] = X["spread_pl_de_sdac"].diff()
        X["spread_pl_sk_change"] = X["spread_pl_sk_sdac"].diff()
        X["system_imbalance_change"] = X["system_imbalance"].diff()
        X["cross_border_flow_change"] = X["total_cross_border_flow"].diff()
        X["sdac_return_24h"] = (X["SDAC_PL"] - X["SDAC_PL"].shift(24)) / (
            X["SDAC_PL"].shift(24).abs() + 1.0
        )

        # SDAC_PL is already shifted 24h (base shift above)
        X["sdac_pl_lag_24h"] = X["SDAC_PL"]
        X["sdac_pl_lag_48h"] = X["SDAC_PL"].shift(24)
        X["sdac_pl_lag_168h"] = X["SDAC_PL"].shift(144)

        X["sdac_rolling_mean_24h"] = X["SDAC_PL"].rolling(window=24).mean()
        X["sdac_rolling_std_24h"] = X["SDAC_PL"].rolling(window=24).std()
        X["sdac_rolling_mean_168h"] = X["SDAC_PL"].rolling(window=168).mean()

        X["sdac_bb_width"] = (X["sdac_rolling_std_24h"] * 2) / (X["sdac_rolling_mean_24h"].abs() + 1e-5)
        X["sdac_ewma_24h"] = X["SDAC_PL"].ewm(span=24).mean()

        total_gen = X["fcst_pv_tot_gen"] + X["fcst_wi_tot_gen"]
        X["renewable_share"] = total_gen / (X["grid_demand_fcst"] + 1e-5)

        res_shifted = X["residual_load"].shift(24)
        X["residual_rolling_mean_24h"] = res_shifted.rolling(window=24).mean()
        X["residual_rolling_std_24h"] = res_shifted.rolling(window=24).std()
        residual_rolling_mean_168h = res_shifted.rolling(window=168).mean()
        residual_rolling_std_168h = res_shifted.rolling(window=168).std()

        X["residual_load_zscore"] = (X["residual_load"] - residual_rolling_mean_168h) / (
            residual_rolling_std_168h + 1e-5
        )

        X["spread_momentum_24h"] = X["target_lag_24h"] - X["target_lag_48h"]
        X["spread_momentum_168h"] = X["target_lag_24h"] - X["target_lag_168h"]

        X["spread_zscore"] = (X["target_lag_24h"] - X["target_rolling_mean_168h"]) / (
            X["target_rolling_std_48h"] + 1e-5
        )

        X["spread_acceleration"] = X["spread_momentum_24h"].diff(24)

        X = X.bfill()
        X = X.fillna(0)

        return X
