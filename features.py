import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Handles multi-tier imputation as defined in prepare_data.py"""
    def __init__(self, bool_cols, numeric_cols):
        self.bool_cols = bool_cols
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # tier 1: booleans
        X[self.bool_cols] = X[self.bool_cols].ffill()
        for col in self.bool_cols:
            X[col] = X[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
            
        # tier 2: short gaps
        X[self.numeric_cols] = X[self.numeric_cols].interpolate(method='linear', limit=3, limit_direction='forward')
        
        # tier 3: long gaps (fallback to 1 week ago)
        for col in self.numeric_cols:
            X[col] = X[col].fillna(X[col].shift(168))
            X[col] = X[col].fillna(X[col].shift(24))
            X[col] = X[col].bfill()
            
        return X

class EnergyFeatureEngineer(BaseEstimator, TransformerMixin):
    """handles temporal, spatial, regime, and fundamental features."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. temporal
        X['hour'] = X.index.hour
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['day_of_week'] = X.index.dayofweek
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        
        # 2. fundamental & gradients (the velocity)
        # residual load is demand minus renewables
        X['residual_load'] = X['grid_demand_fcst'] - (X['fcst_pv_tot_gen'] + X['fcst_wi_tot_gen'])
        
        # how fast are the fundamentals changing?
        X['pv_gradient'] = X['fcst_pv_tot_gen'].diff()
        X['wind_gradient'] = X['fcst_wi_tot_gen'].diff()
        X['demand_gradient'] = X['grid_demand_fcst'].diff()
        X['residual_gradient'] = X['residual_load'].diff()
        
        # 3. spatial & interconnector flow
        X['spread_pl_de_sdac'] = X['SDAC_PL'] - X['SDAC_DE']
        X['spread_pl_sk_sdac'] = X['SDAC_PL'] - X['SDAC_SK']
        X['total_cross_border_flow'] = X['SE_SDAC_PL_LT'] + X['SE_SDAC_PL_SE']
        
        # 4. order imbalance & balancing pressure
        # if this is highly positive, the grid is desperate for power
        X['system_imbalance'] = X['AC_UP_SDAC_PL'] - X['AC_DOWN_SDAC_PL']
        X['net_position_momentum'] = X['NP_PL_GLOBAL_SDAC_PL'].diff()
        
        # 5. volatility & regime filters (strictly shifted by 24h to prevent leakage!)
        lags = [24, 48, 168]
        for lag in lags:
            X[f'sdac_pl_lag_{lag}h'] = X['SDAC_PL'].shift(lag)
            
        # rolling stats on the day-ahead price
        X['sdac_rolling_mean_24h'] = X['SDAC_PL'].shift(24).rolling(window=24).mean()
        X['sdac_rolling_std_24h'] = X['SDAC_PL'].shift(24).rolling(window=24).std()
        
        # bollinger band width proxy (volatility indicator)
        # adding 1e-5 to prevent division by zero when prices are exactly 0
        X['sdac_bb_width'] = (X['sdac_rolling_std_24h'] * 2) / (X['sdac_rolling_mean_24h'].abs() + 1e-5)
        
        # ewma (exponentially weighted moving average) for short-term trend
        X['sdac_ewma_24h'] = X['SDAC_PL'].shift(24).ewm(span=24).mean()
        
        # clean up nans introduced by diff(), shift(), and rolling()
        # backfill catches the start of the dataset, fillna(0) acts as a final safety net
        X = X.bfill().fillna(0) 
        
        return X