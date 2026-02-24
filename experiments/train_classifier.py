import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, average_precision_score
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
y_raw = df[target_col]

print(f"training with {len(X.columns)} features.")

# 2. train/test split (chronological)
test_days = 30
split_idx = len(df) - (test_days * 24)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_raw_train, y_raw_test = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

print(f"train set: {X_train.index.min()} to {X_train.index.max()}")
print(f"test set: {X_test.index.min()} to {X_test.index.max()}")

# 3. transform to binary classification problem
print("\ntransforming target into extreme event categories...")
# define "extreme" as the top 10% of absolute spreads in the TRAIN set only
threshold = y_raw_train.abs().quantile(0.90)
print(f"spike threshold determined at: +/- {threshold:.2f} eur/mwh")

# 1 if it's an extreme spike, 0 if it's a normal hour
y_train = (y_raw_train.abs() >= threshold).astype(int)
y_test = (y_raw_test.abs() >= threshold).astype(int)

print(f"train set imbalance: {y_train.sum()} spikes out of {len(y_train)} hours")
print(f"test set imbalance: {y_test.sum()} spikes out of {len(y_test)} hours")

# 4. time-series cross validation setup
tscv = TimeSeriesSplit(n_splits=3)

# 5. hyperparameter grid (adjusted for classification)
param_grid = {
    'max_depth': [4, 6, 8], # trees don't need to be as deep for classification
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5] # introduces pruning to prevent overfitting on noisy data
}

# calculate scale_pos_weight to handle the 90/10 class imbalance
# formula: sum(negative instances) / sum(positive instances)
imbalance_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# base model using binary:logistic for probability outputs
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=300,
    scale_pos_weight=imbalance_weight,
    random_state=42,
    n_jobs=-1
)

# 6. randomized search
print("\nstarting randomized search (optimizing for precision/recall)...")
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=15, 
    scoring='average_precision', # PR-AUC is best for imbalanced data
    cv=tscv,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n--- best parameters found ---")
for key, value in random_search.best_params_.items():
    print(f"{key}: {value}")

# 7. evaluate best model
print("\nevaluating best model on test set...")
best_model = random_search.best_estimator_

# predict classes and probabilities
preds = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1] # get probability of class 1

print("\n--- classification report ---")
print(classification_report(y_test, preds))

# 8. visualize results
os.makedirs('plots', exist_ok=True)


# A. confusion matrix
print("generating confusion matrix plot...")
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'spike'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title(f'confusion matrix (threshold: {threshold:.2f} eur)')
plt.savefig('plots/classifier_confusion_matrix.png')
plt.close()

# B. probability vs actuals timeline
print("generating timeline plot...")
plt.figure(figsize=(15, 6))
plot_idx = -24 * 7 # last 7 days
plt.plot(y_test.index[plot_idx:], y_raw_test.values[plot_idx:], label='actual raw spread (eur)', color='gray', alpha=0.5)

# scatter the actual spikes in red
spike_indices = y_test.index[y_test == 1]
plt.scatter(spike_indices[spike_indices >= y_test.index[plot_idx]], 
            y_raw_test.loc[spike_indices[spike_indices >= y_test.index[plot_idx]]], 
            color='red', label='actual spikes', zorder=5)

# plot the model's confidence probability on a secondary axis
ax2 = plt.gca().twinx()
ax2.plot(y_test.index[plot_idx:], proba[plot_idx:], label='model probability signal', color='blue', alpha=0.7, linestyle='--')
ax2.set_ylabel('probability of spike')
ax2.set_ylim(0, 1)

plt.title('actual spreads vs. model spike probability (last 7 days)')
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('plots/classifier_timeline.png')
plt.close()

# C. feature importance
print("generating feature importance plot...")
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)

plt.figure(figsize=(10, 8))
top_features.sort_values().plot(kind='barh', color='mediumpurple')
plt.title('top 15 features for predicting spikes')
plt.xlabel('f-score')
plt.tight_layout()
plt.savefig('plots/classifier_feature_importance.png')
plt.close()

print("classification training complete. check the plots directory!")