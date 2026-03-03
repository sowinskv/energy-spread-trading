from __future__ import annotations

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig


def load_and_format_raw_data(filepath: str) -> pd.DataFrame:
    """Load energy data CSV and format for pipeline consumption"""
    print("loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    
    cols_to_exclude = ['date_cet', 'IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
    numeric_cols = [col for col in df.columns if col not in cols_to_exclude]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
    for col in ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
        
    df['date_cet'] = pd.to_datetime(df['date_cet'])
    df.set_index('date_cet', inplace=True)
    if df.index.duplicated().any():
        df = df.groupby(df.index).first()
    df = df.asfreq('h')
    
    df['spread_SDAC_IDA1_PL'] = df['SDAC_PL'] - df['IDA1_PL']
    df['spread_SDAC_IDA1_PL'] = df['spread_SDAC_IDA1_PL'].interpolate(method='linear', limit_direction='both')
    
    df['target_lag_24h'] = df['spread_SDAC_IDA1_PL'].shift(24)
    df['target_lag_48h'] = df['spread_SDAC_IDA1_PL'].shift(48)
    df['target_lag_168h'] = df['spread_SDAC_IDA1_PL'].shift(168)
    
    df['target_rolling_mean_24h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=24).mean()
    df['target_rolling_std_24h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=24).std()
    df['target_rolling_mean_168h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=168).mean()
    
    return df


def prepare_dataset(config: DictConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = load_and_format_raw_data(config.data.file_path)
    bool_cols = list(config.data.bool_cols)
    X_full = df.drop(columns=config.data.leakage_cols)
    numeric_cols = [c for c in X_full.columns if c not in bool_cols]
    return df, bool_cols, numeric_cols


def get_expanding_walk_forward_splits(
    df_length: int, initial_train_days: int, test_days: int, purge_days: int, n_splits: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Generate expanding window walk-forward CV splits"""
    initial_train_steps = initial_train_days * 24
    test_steps = test_days * 24
    purge_steps = purge_days * 24
    
    splits = []
    for i in range(n_splits):
        test_start = initial_train_steps + purge_steps + (i * (test_steps + purge_steps))
        test_end = test_start + test_steps
        
        if test_end > df_length:
            break
            
        train_start = 0 
        train_end = test_start - purge_steps
        
        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    
    return splits


def get_purged_walk_forward_splits(
    df_length: int, train_days: int, test_days: int, purge_days: int, n_splits: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Generate fixed-window purged walk-forward CV splits"""
    train_steps = train_days * 24
    test_steps = test_days * 24
    purge_steps = purge_days * 24
    step_size = test_steps 
    splits = []
    end_idx = df_length
    for i in range(n_splits):
        test_end = end_idx - (i * step_size)
        test_start = test_end - test_steps
        purge_start = test_start - purge_steps
        train_end = purge_start
        train_start = train_end - train_steps
        if train_start < 0:
            raise ValueError("dataset is too tiny for this.")
        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return splits[::-1]