import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

print("loading modeling data...")
df = pd.read_csv('data/model_input.csv', index_col='date_cet', parse_dates=True)

target_col = 'spread_SDAC_IDA1_PL'
leakage_cols = [
    target_col, 'IDA1_DE', 'IDA1_PL', 'IDA1_SK', 
    'IDA2_DE', 'IDA2_PL', 'IDA2_SK', 'IDA3_DE', 'IDA3_PL', 'IDA3_SK'
]

X = df.drop(columns=leakage_cols)
y_raw = df[target_col]

# 1. train/test split (chronological)
test_days = 30
split_idx = len(df) - (test_days * 24)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_raw_train, y_raw_test = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

# 2. binarize target (90th percentile)
threshold_val = y_raw_train.abs().quantile(0.90)
y_train = (y_raw_train.abs() >= threshold_val).astype(int)
y_test = (y_raw_test.abs() >= threshold_val).astype(int)

print(f"spike threshold: +/- {threshold_val:.2f} eur/mwh")

# 3. scale the features
print("scaling features...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. create 3d sequences for lstm
print("generating time-series sequences...")
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24 # look back 24 hours to predict the next hour
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

print(f"training tensor shape: {X_train_seq.shape}")

# 5. handle class imbalance
weight_for_0 = 1.0
weight_for_1 = (len(y_train_seq) - y_train_seq.sum()) / y_train_seq.sum()
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"class weights: {class_weight}")

# 6. build lstm model
print("building lstm architecture...")
model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False),
    Dropout(0.3), # dropout helps prevent the network from memorizing the training data
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # sigmoid outputs a probability between 0 and 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# early stopping prevents overfitting if the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 7. train the network
print("training lstm (this might take a few minutes)...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# 8. evaluate model
print("\nevaluating lstm on test set...")
proba = model.predict(X_test_seq).flatten()
# use the same 0.50 threshold we started with for xgboost
preds = (proba >= 0.50).astype(int) 

print("\n--- lstm classification report ---")
print(classification_report(y_test_seq, preds))

# 9. plot training history
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('lstm training history')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('plots/lstm_training_history.png')
plt.close()

print("lstm evaluation complete. check the classification report!")