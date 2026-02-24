import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# setup plot directory
os.makedirs('plots', exist_ok=True)
sns.set_theme(style="darkgrid")

# load data
df = pd.read_csv('data/energy_data.csv', low_memory=False)

# numeric conversion
cols_to_exclude = ['date_cet', 'IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
numeric_cols = [col for col in df.columns if col not in cols_to_exclude]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# boolean mapping
for col in ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']:
    df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)

# datetime indexing
df['date_cet'] = pd.to_datetime(df['date_cet'])
df.set_index('date_cet', inplace=True)

# handle duplicates
if df.index.duplicated().any():
    print(f"found {df.index.duplicated().sum()} duplicates. averaging...")
    df = df.groupby(df.index).first() 

# frequency enforcement
df = df.asfreq('h') 

# target creation
df['spread_SDAC_IDA1_PL'] = df['SDAC_PL'] - df['IDA1_PL']

print("preprocessing complete")
print(f"dataset shape: {df.shape}")

# price landscape plot
print("generating price landscape")
cutoff_date = df.index.max() - pd.Timedelta(days=14)
subset = df[df.index >= cutoff_date]

plt.figure(figsize=(15, 6))
plt.plot(subset.index, subset['SDAC_PL'], label='Day-Ahead (PL)', alpha=0.7)
plt.plot(subset.index, subset['IDA1_PL'], label='Intraday (PL)', alpha=0.7)
plt.fill_between(subset.index, subset['SDAC_PL'], subset['IDA1_PL'], 
                 label='Spread Area', color='gray', alpha=0.3)
plt.title('market dynamics: last 14 days')
plt.ylabel('price (eur/mwh)')
plt.legend()
plt.savefig('plots/eda_timeseries.png')
plt.close()

# hourly seasonality
print("generating seasonality plots")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='hour', y='spread_SDAC_IDA1_PL')
plt.title('spread distribution by hour')
plt.savefig('plots/eda_hourly_box.png')
plt.close()

# heatmap seasonality
pivot_df = df.copy()
pivot_df['day'] = pivot_df.index.date
grid = pivot_df.pivot_table(index='hour', columns='day', values='spread_SDAC_IDA1_PL')

plt.figure(figsize=(12, 8))
sns.heatmap(grid, cmap='RdYlGn', center=0)
plt.title('hourly spread heatmap')
plt.savefig('plots/eda_heatmap.png')
plt.close()

# distribution and outliers
print("generating distribution plots")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['spread_SDAC_IDA1_PL'].dropna(), bins=100, kde=True, ax=axes[0])
axes[0].set_title('target spread histogram')
sns.boxplot(y=df['spread_SDAC_IDA1_PL'], ax=axes[1])
axes[1].set_title('target spread boxplot')
plt.savefig('plots/eda_distribution.png')
plt.close()

# correlation matrix
print("generating correlation matrix")
core_cols = ['SDAC_PL', 'SDAC_DE', 'SDAC_SK', 'IDA1_PL', 'grid_demand_fcst', 'fcst_pv_tot_gen', 'spread_SDAC_IDA1_PL']
plt.figure(figsize=(10, 8))
sns.heatmap(df[core_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('feature correlation with spread')
plt.savefig('plots/eda_correlation.png')
plt.close()

# autocorrelation
print("generating acf/pacf plots")
spread_clean = df['spread_SDAC_IDA1_PL'].dropna()
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(spread_clean, lags=48, ax=ax[0])
plot_pacf(spread_clean, lags=48, ax=ax[1])
plt.savefig('plots/eda_acf_pacf.png')
plt.close()

# cleanup temporary columns
df.drop(['hour', 'day_of_week'], axis=1, inplace=True)

print("eda complete. files saved in plots/")