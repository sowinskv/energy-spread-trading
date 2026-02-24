import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import os

print("loading modeling data...")
df = pd.read_csv('data/model_input.csv', index_col='date_cet', parse_dates=True)

# 1. define target and prevent data leakage
target_col = 'spread_SDAC_IDA1_PL'
leakage_cols = [
    target_col,
    'IDA1_DE', 'IDA1_PL', 'IDA1_SK', 
    'IDA2_DE', 'IDA2_PL', 'IDA2_SK', 
    'IDA3_DE', 'IDA3_PL', 'IDA3_SK'
]

X = df.drop(columns=leakage_cols)
y = df[target_col]

print(f"training with {len(X.columns)} features.")

# 2. train/test split (chronological)
test_days = 30
split_idx = len(df) - (test_days * 24)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"train set: {X_train.index.min()} to {X_train.index.max()}")
print(f"test set: {X_test.index.min()} to {X_test.index.max()}")

# 3. time-series cross validation setup

print("setting up time-series cross validation...")
# we use 3 splits. it trains on a growing window of the past to validate the near future.
tscv = TimeSeriesSplit(n_splits=3)

# 4. hyperparameter grid
# we allow much deeper trees (up to 12) to capture the complex logic behind price spikes
param_grid = {
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# base model using absolute error to stop it from aggressively reverting to the mean
xgb_base = xgb.XGBRegressor(
    objective='reg:absoluteerror', 
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# 5. randomized search

print("starting randomized search (this might take a minute)...")
# n_iter=10 tests 10 random combinations from the grid to save time. 
# you can increase this to 30+ for even better results later.
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=15, 
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n--- best parameters found ---")
for key, value in random_search.best_params_.items():
    print(f"{key}: {value}")

# 6. train final model with best parameters
print("\nevaluating best model on test set...")
best_model = random_search.best_estimator_

preds = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"final mean absolute error (mae): {mae:.2f} eur/mwh")
print(f"final root mean squared error (rmse): {rmse:.2f} eur/mwh")

# 7. visualize predictions vs actuals
os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(15, 6))
plot_idx = -24 * 7 # last 7 days
plt.plot(y_test.index[plot_idx:], y_test.values[plot_idx:], label='actual spread', alpha=0.8)
plt.plot(y_test.index[plot_idx:], preds[plot_idx:], label='tuned xgboost', alpha=0.9, linestyle='--', color='orange')
plt.title('tuned model: actual vs predicted spread (last 7 days)')
plt.ylabel('spread (eur/mwh)')
plt.legend()
plt.savefig('plots/model_predictions_tuned.png')
plt.close()

# 8. feature importance (best model)
print("generating tuned feature importance plot...")
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)

plt.figure(figsize=(10, 8))
top_features.sort_values().plot(kind='barh', color='darkseagreen')
plt.title('top 15 features (tuned model)')
plt.xlabel('f-score')
plt.tight_layout()
plt.savefig('plots/model_feature_importance_tuned.png')
plt.close()

print("tuning complete. check the plots directory!")