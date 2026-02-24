import pandas as pd
import numpy as np

print("loading and cleaning raw data...")
df = pd.read_csv('data/energy_data.csv', low_memory=False)

# numeric conversion
cols_to_exclude = ['date_cet', 'IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
numeric_cols = [col for col in df.columns if col not in cols_to_exclude]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# datetime indexing and duplicate handling
df['date_cet'] = pd.to_datetime(df['date_cet'])
df.set_index('date_cet', inplace=True)

if df.index.duplicated().any():
    df = df.groupby(df.index).first() 

df = df.asfreq('h')

# target creation
df['spread_SDAC_IDA1_PL'] = df['SDAC_PL'] - df['IDA1_PL']

print(f"missing values before imputation:\n{df.isna().sum().head(5)}")

# --- multi-tier imputation strategy ---
print("starting multi-tier imputation...")

# tier 1: booleans (forward fill)
# we do this before mapping to 1/0 to catch the raw string NaNs
bool_cols = ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
df[bool_cols] = df[bool_cols].ffill()
for col in bool_cols:
    df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)

# tier 2: short gaps (linear interpolation, max 3 hours)
# strictly forward-facing to prevent data leakage
for col in numeric_cols + ['spread_SDAC_IDA1_PL']:
    df[col] = df[col].interpolate(method='linear', limit=3, limit_direction='forward')

# tier 3: long gaps (historical seasonal averages)
# we use the value from exactly 1 week ago (168 hours). 
# if that is also missing, we fallback to 24 hours ago.
for col in numeric_cols + ['spread_SDAC_IDA1_PL']:
    # fill with 1-week lag
    df[col] = df[col].fillna(df[col].shift(168))
    # fallback to 24-hour lag for anything remaining
    df[col] = df[col].fillna(df[col].shift(24))
    # final fallback to forward fill just in case the gap is at the very beginning of the dataset
    df[col] = df[col].ffill().bfill() 

print(f"missing values after imputation:\n{df.isna().sum().head(5)}")

# save clean dataset
print("saving clean data to data/energy_data_clean.csv")
df.to_csv('data/energy_data_clean.csv')

print("pipeline complete!")