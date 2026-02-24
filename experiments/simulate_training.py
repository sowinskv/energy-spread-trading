import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

print("loading modeling data...")
df = pd.read_csv('data/model_input.csv', index_col='date_cet', parse_dates=True)

# 1. define target and features
target_col = 'spread_SDAC_IDA1_PL'
leakage_cols = [
    target_col,
    'IDA1_DE', 'IDA1_PL', 'IDA1_SK', 
    'IDA2_DE', 'IDA2_PL', 'IDA2_SK', 
    'IDA3_DE', 'IDA3_PL', 'IDA3_SK'
]

X = df.drop(columns=leakage_cols)
y_raw = df[target_col]

# 2. train/test split (chronological)
test_days = 30
split_idx = len(df) - (test_days * 24)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_raw_train, y_raw_test = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

# 3. transform to binary using the 90th percentile threshold
threshold_val = y_raw_train.abs().quantile(0.90)
y_train = (y_raw_train.abs() >= threshold_val).astype(int)
y_test = (y_raw_test.abs() >= threshold_val).astype(int)

# calculate imbalance weight
imbalance_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# 4. train the best model (using the exact parameters we found earlier)
print("training the optimized classifier...")
best_model = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=300,
    scale_pos_weight=imbalance_weight,
    subsample=0.8,
    max_depth=4,
    learning_rate=0.01,
    gamma=0,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

# 5. get probability predictions
proba = best_model.predict_proba(X_test)[:, 1]

# 6. financial simulation

print("\n--- running financial simulation ---")
transaction_cost = 2.00 # assumed cost in eur/mwh per trade

thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
results = []

for t in thresholds:
    # generate trade signals based on the current threshold
    trade_signals = (proba >= t).astype(int)
    
    total_trades = trade_signals.sum()
    true_spikes_caught = (trade_signals & y_test).sum()
    
    # calculate precision (avoid division by zero)
    precision = true_spikes_caught / total_trades if total_trades > 0 else 0
    
    # calculate profit: absolute raw spread minus transaction cost for every trade executed
    # (assuming we correctly position ourselves to capture the absolute divergence)
    profit_per_trade = (y_raw_test.abs() * trade_signals) - (transaction_cost * trade_signals)
    total_profit = profit_per_trade.sum()
    
    results.append({
        'threshold': t,
        'trades_executed': total_trades,
        'precision': precision,
        'total_profit_eur': total_profit
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 7. visualize the profit curve
os.makedirs('plots', exist_ok=True)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:green'
ax1.set_xlabel('model confidence threshold')
ax1.set_ylabel('total profit (eur)', color=color)
ax1.plot(results_df['threshold'], results_df['total_profit_eur'], marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('trades executed', color=color)  
ax2.plot(results_df['threshold'], results_df['trades_executed'], marker='x', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('trading simulation: profit vs. confidence threshold')
fig.tight_layout()  
plt.savefig('plots/trading_simulation.png')
plt.close()

print("\nsimulation complete. check plots/trading_simulation.png to find the optimal cutoff!")