'''
1) tree-based models dont understand time (that 00 comes after 23), so we need to convert cyclical features to sin/cos pairs
2) residual load = load - renewable generation (wind + solar) is a key driver of price dynamics, so we create this feature to capture that relationship
3) spatial / cross-border spread features can capture market coupling effects, so we create features like SDAC_DE - SDAC_PL to represent the price difference between Germany and Poland
4) autoregressive lag features can help capture temporal dependencies, so we create lagged versions of the target variable (spread_SDAC_IDA1_PL) at various time lags (1 hour, 24 hours, 168 hours) to capture short-term and weekly patterns
(a safe min. lag for european power models is 24 hours)
5) rolling windows (capturing volatility) -- lets give it a sense of "short-term momentum" -- calculate the moving averages and standard deviations over recent time windows
'''

import pandas as pd
import numpy as np
import os

print("loading clean data...")
df = pd.read_csv('data/energy_data_clean.csv', index_col='date_cet', parse_dates=True)

print("1. building temporal features")
# basic time components
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# cyclic encoding for hour and month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("2. building fundamental features")
# residual load (demand minus renewables)
df['residual_load'] = df['grid_demand_fcst'] - (df['fcst_pv_tot_gen'] + df['fcst_wi_tot_gen'])

# hour-over-hour gradients (deltas)
df['pv_gradient'] = df['fcst_pv_tot_gen'].diff()
df['wind_gradient'] = df['fcst_wi_tot_gen'].diff()
df['demand_gradient'] = df['grid_demand_fcst'].diff()

print("3. building spatial features")
# cross-border day-ahead spreads
df['spread_pl_de_sdac'] = df['SDAC_PL'] - df['SDAC_DE']
df['spread_pl_sk_sdac'] = df['SDAC_PL'] - df['SDAC_SK']

print("4. building lagged features")
target = 'spread_SDAC_IDA1_PL'

# we use 24h as the absolute minimum lag to avoid data leakage in day-ahead markets
lags = [24, 48, 168] # 1 day, 2 days, 1 week
for lag in lags:
    df[f'target_lag_{lag}h'] = df[target].shift(lag)
    df[f'sdac_pl_lag_{lag}h'] = df['SDAC_PL'].shift(lag)

print("5. building rolling window features")

# rolling statistics on the 24h-lagged target (strictly backward-looking from the 24h point)
# we shift by 24h first, THEN calculate the rolling mean to ensure zero leakage
df['target_rolling_mean_24h'] = df[target].shift(24).rolling(window=24).mean()
df['target_rolling_std_24h'] = df[target].shift(24).rolling(window=24).std()
df['target_rolling_mean_168h'] = df[target].shift(24).rolling(window=168).mean()

print("cleaning up and saving...")
# dropping the first 192 rows (168 + 24) because the lags and rolling windows introduce NaNs at the very beginning
initial_shape = df.shape
df.dropna(inplace=True)
print(f"dropped {initial_shape[0] - df.shape[0]} rows due to lagging nans.")

# save the final modeling dataset
os.makedirs('data', exist_ok=True)
df.to_csv('data/model_input.csv')

print(f"feature engineering complete! final shape: {df.shape}")
print("data saved to data/model_input.csv")