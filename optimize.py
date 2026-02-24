# optimize.py
import pandas as pd
import numpy as np
import optuna
from omegaconf import OmegaConf
import xgboost as xgb
from sklearn.pipeline import Pipeline
from features import TimeSeriesImputer, EnergyFeatureEngineer
import warnings

# so we can read only the optuna logs
warnings.filterwarnings("ignore")

# 1. grab our custom math and logic
def asymmetric_trading_loss(y_true, y_pred):
    residual = y_pred - y_true
    grad = residual.copy()
    hess = np.ones_like(y_pred)
    fp_mask = (y_true < 0) & (y_pred > 0)
    fn_mask = (y_true > 0) & (y_pred < 0)
    grad[fp_mask] *= 5.0
    hess[fp_mask] *= 5.0
    grad[fn_mask] *= 2.0
    hess[fn_mask] *= 2.0
    magnitude_weight = 1.0 + (np.abs(y_true) / 10.0) 
    grad = grad * magnitude_weight
    hess = hess * magnitude_weight
    return grad, hess

def calculate_meta_trading_metrics(y_true, y_pred, meta_probs, confidence_threshold=0.5, cost_per_mwh=0.5):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    meta_probs_np = np.array(meta_probs)
    
    intended_position = np.sign(y_pred_np)
    trade_mask = (meta_probs_np > confidence_threshold).astype(int)
    actual_position = intended_position * trade_mask
    
    raw_pnl = actual_position * y_true_np
    fees = trade_mask * cost_per_mwh
    net_pnl = pd.Series(raw_pnl - fees)
    
    if net_pnl.std() != 0:
        sharpe = (net_pnl.mean() / net_pnl.std()) * np.sqrt(8760)
    else:
        sharpe = 0.0
    return sharpe

def get_purged_walk_forward_splits(df_length, train_days, test_days, purge_days, n_splits):
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
        if train_start < 0: raise ValueError("dataset is too tiny for this.")
        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return splits[::-1]

# 2. load data once up top so we don't waste time reloading every trial
config = OmegaConf.load("config.yaml")
df = pd.read_csv(config.data.file_path, index_col='date_cet', parse_dates=True)
X_full = df.drop(columns=config.data.leakage_cols)
bool_cols = ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
numeric_cols = [c for c in X_full.columns if c not in bool_cols]

splits = get_purged_walk_forward_splits(
    df_length=len(df),
    train_days=config.cv.train_days,
    test_days=config.cv.test_days,
    purge_days=config.cv.purge_days,
    n_splits=config.cv.n_splits
)

# 3. the optuna playground
def objective(trial):
    # a. tweak the analyst's brain
    analyst_params = {
        'n_estimators': trial.suggest_int('analyst_n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('analyst_max_depth', 4, 9),
        'learning_rate': trial.suggest_float('analyst_lr', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('analyst_subsample', 0.6, 1.0),
        'random_state': 42
    }
    
    # b. tweak the manager's brain (keeping it shallow so it doesn't overthink)
    manager_params = {
        'n_estimators': trial.suggest_int('manager_n_estimators', 50, 200, step=50),
        'max_depth': trial.suggest_int('manager_max_depth', 2, 5), 
        'learning_rate': trial.suggest_float('manager_lr', 0.01, 0.1, log=True),
        'random_state': 42
    }
    
    # c. figure out the sweet spot for the confidence threshold
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.35, 0.75)
    
    fold_sharpes = []

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        split_meta_idx = len(train_df) - (config.cv.meta_train_days * 24)
        primary_df = train_df.iloc[:split_meta_idx]
        meta_df = train_df.iloc[split_meta_idx:]
        
        preprocessor = Pipeline([
            ('imputer', TimeSeriesImputer(bool_cols=bool_cols, numeric_cols=numeric_cols)),
            ('feature_engineer', EnergyFeatureEngineer())
        ])
        
        X_prim_prep = preprocessor.fit_transform(primary_df.drop(columns=config.data.leakage_cols))
        y_prim = primary_df[config.data.target_col]
        
        X_meta_prep = preprocessor.transform(meta_df.drop(columns=config.data.leakage_cols))
        y_meta = meta_df[config.data.target_col]
        
        X_test_prep = preprocessor.transform(test_df.drop(columns=config.data.leakage_cols))
        y_test = test_df[config.data.target_col]

        # let the analyst learn
        analyst = xgb.XGBRegressor(**analyst_params, objective=asymmetric_trading_loss)
        analyst.fit(X_prim_prep, y_prim)
        
        # grade the analyst to create training data for the manager
        meta_analyst_preds = analyst.predict(X_meta_prep)
        meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
        meta_labels = (meta_raw_pnl > 0).astype(int)
        
        # let the manager learn from the analyst's mistakes
        X_meta_enhanced = X_meta_prep.copy()
        X_meta_enhanced['analyst_pred'] = meta_analyst_preds
        manager = xgb.XGBClassifier(**manager_params)
        manager.fit(X_meta_enhanced, meta_labels)
        
        # put them both to the test
        test_analyst_preds = analyst.predict(X_test_prep)
        X_test_enhanced = X_test_prep.copy()
        X_test_enhanced['analyst_pred'] = test_analyst_preds
        test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
        
        # how much money did we actually make?
        sharpe = calculate_meta_trading_metrics(y_test, test_analyst_preds, test_manager_probs, confidence_threshold)
        fold_sharpes.append(sharpe)
        
    # the whole point is to pump up that sharpe ratio
    return np.mean(fold_sharpes)

if __name__ == "__main__":
    print("running the optuna script...")
    
    # we want the highest sharpe possible
    study = optuna.create_study(direction='maximize')
    
    # run 30 trials (bump this up if you're letting it run overnight)
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*40)
    print("aaand we're done. here's the alpha:")
    print(f"best cv sharpe ratio: {study.best_value:.2f}")
    print("\nthe winning combo:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*40)
    print("go drop these into your config.yaml and you're golden.")