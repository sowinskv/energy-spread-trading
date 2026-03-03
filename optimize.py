import pandas as pd
import numpy as np
import optuna
from omegaconf import OmegaConf
import xgboost as xgb
from sklearn.pipeline import Pipeline
from features import TimeSeriesImputer, EnergyFeatureEngineer
from ensemble_models import EnsembleAnalyst, MultiHorizonEnsemble
from src.trading.metrics import calculate_enhanced_meta_trading_metrics_with_exits, asymmetric_trading_loss
from src.core.data.loader import load_and_format_raw_data, get_purged_walk_forward_splits
import warnings
import sqlite3

warnings.filterwarnings("ignore")

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

config = OmegaConf.load("config.yaml")
df = load_and_format_raw_data(config.data.file_path)
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

def objective(trial):
    use_ensemble = trial.suggest_categorical('use_ensemble', [True, False])
    
    if use_ensemble:
        ensemble_params = {
            'reweight_frequency': trial.suggest_categorical('reweight_frequency', [168, 336, 720, 1440]),
            'performance_window': trial.suggest_categorical('performance_window', [48, 168, 336, 720]),
            'multi_horizon': trial.suggest_categorical('multi_horizon', [True, False])
        }
        
        if ensemble_params['multi_horizon']:
            horizon_choice = trial.suggest_categorical('horizons', ['short', 'medium', 'long', 'extended'])
            horizon_map = {
                'short': [1, 4],
                'medium': [1, 4, 12], 
                'long': [1, 4, 12, 24],
                'extended': [1, 2, 6, 12, 24]
            }
            ensemble_params['horizons'] = horizon_map[horizon_choice]
        
        analyst_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200, step=25),
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 10),
            'learning_rate': trial.suggest_float('xgb_lr', 0.005, 0.05, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.7, 1.0)
        }
        
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('rf_max_depth', 10, 25),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9])
        }
        
        et_params = {
            'n_estimators': trial.suggest_int('et_n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('et_max_depth', 10, 25),
            'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9])
        }
    else:
        analyst_params = {
            'n_estimators': trial.suggest_int('analyst_n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('analyst_max_depth', 4, 9),
            'learning_rate': trial.suggest_float('analyst_lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('analyst_subsample', 0.6, 1.0),
            'random_state': 42
        }
    
    manager_params = {
        'n_estimators': trial.suggest_int('manager_n_estimators', 50, 200, step=50),
        'max_depth': trial.suggest_int('manager_max_depth', 2, 5), 
        'learning_rate': trial.suggest_float('manager_lr', 0.01, 0.1, log=True),
        'random_state': 42
    }
    
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.35, 0.75)
    
    fold_hit_rates = []

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

        if use_ensemble:
            temp_config = OmegaConf.create({
                'ensemble': ensemble_params,
                'model': analyst_params,
                'model_params': {
                    'xgboost': {
                        'n_estimators': analyst_params['n_estimators'],
                        'max_depth': analyst_params['max_depth'], 
                        'learning_rate': analyst_params['learning_rate'],
                        'subsample': analyst_params['subsample'],
                        'colsample_bytree': analyst_params['colsample_bytree']
                    },
                    'random_forest': {
                        'n_estimators': rf_params['n_estimators'],
                        'max_depth': rf_params['max_depth'],
                        'min_samples_split': rf_params['min_samples_split'],
                        'min_samples_leaf': rf_params['min_samples_leaf'],
                        'max_features': rf_params['max_features']
                    },
                    'extra_trees': {
                        'n_estimators': et_params['n_estimators'],
                        'max_depth': et_params['max_depth'],
                        'min_samples_split': et_params['min_samples_split'],
                        'min_samples_leaf': et_params['min_samples_leaf'],
                        'max_features': et_params['max_features']
                    },
                    'ridge': {
                        'alpha': 1.0
                    }
                }
            })
            
            if ensemble_params['multi_horizon']:
                analyst = MultiHorizonEnsemble(temp_config)
                analyst.fit(X_prim_prep, y_prim, primary_df.index)
            else:
                analyst = EnsembleAnalyst(temp_config)
                analyst.fit(X_prim_prep, y_prim)
            
            if ensemble_params['multi_horizon']:
                meta_analyst_preds = analyst.predict(X_meta_prep, meta_df.index)
            else:
                meta_analyst_preds = analyst.predict(X_meta_prep)
        else:
            analyst = xgb.XGBRegressor(**analyst_params, objective=asymmetric_trading_loss)
            analyst.fit(X_prim_prep, y_prim)
            meta_analyst_preds = analyst.predict(X_meta_prep)
        
        meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
        meta_labels = (meta_raw_pnl > 0).astype(int)
        
        X_meta_enhanced = X_meta_prep.copy()
        X_meta_enhanced['analyst_pred'] = meta_analyst_preds
        manager = xgb.XGBClassifier(**manager_params)
        manager.fit(X_meta_enhanced, meta_labels)
        
        if use_ensemble and ensemble_params.get('multi_horizon'):
            test_analyst_preds = analyst.predict(X_test_prep, test_df.index)
        else:
            test_analyst_preds = analyst.predict(X_test_prep)
        X_test_enhanced = X_test_prep.copy()
        X_test_enhanced['analyst_pred'] = test_analyst_preds
        test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
        
        try:
            temp_config_full = OmegaConf.create(OmegaConf.to_yaml(config))
            temp_config_full.meta_model.confidence_threshold = confidence_threshold
            
            metrics = calculate_enhanced_meta_trading_metrics_with_exits(
                y_test, test_analyst_preds, test_manager_probs, temp_config_full, use_exit_rules=False
            )
            hit_rate = metrics['hit_rate'] if not np.isnan(metrics['hit_rate']) else 0
        except:
            y_true_np = np.array(y_test)
            y_pred_np = np.array(test_analyst_preds)
            meta_probs_np = np.array(test_manager_probs)
            
            intended_position = np.sign(y_pred_np)
            trade_mask = (meta_probs_np > confidence_threshold).astype(int)
            actual_position = intended_position * trade_mask
            
            raw_pnl = actual_position * y_true_np
            fees = trade_mask * 0.5
            net_pnl = raw_pnl - fees
            
            executed_trades = net_pnl[trade_mask == 1]
            if len(executed_trades) > 0:
                hit_rate = (executed_trades > 0).mean() * 100
            else:
                hit_rate = 0
        
        fold_hit_rates.append(hit_rate)
        
    return np.mean(fold_hit_rates)

if __name__ == "__main__":
    print("running optuna optimization for maximum hit rate...")
    print("targeting high-performing models: XGBoost, Random Forest, Extra Trees (89% of ensemble)")
    
    study = optuna.create_study(
    study_name="energy_hit_rate_optimization",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("\n" + "="*50)
    print("optimization complete. best results:")
    print(f"best cv hit rate: {study.best_value:.1f}%")
    print("\noptimal parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*50)
    print("update your config.yaml with these parameters.")