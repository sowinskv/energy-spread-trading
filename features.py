# features.py
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
    """Handles temporal, spatial, and rolling features."""
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
        
        # 2. spatial
        X['spread_pl_de_sdac'] = X['SDAC_PL'] - X['SDAC_DE']
        X['spread_pl_sk_sdac'] = X['SDAC_PL'] - X['SDAC_SK']
        
        # 3. lags & rolling (assuming target lag is pre-calculated or using SDAC lags)
        lags = [24, 48, 168]
        for lag in lags:
            X[f'sdac_pl_lag_{lag}h'] = X['SDAC_PL'].shift(lag)
            
        X['sdac_rolling_mean_24h'] = X['SDAC_PL'].shift(24).rolling(window=24).mean()
        
        # drop rows with NaNs caused by lagging
        X = X.fillna(0) 
        return X