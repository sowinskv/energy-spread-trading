import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import TimeSeriesImputer, EnergyFeatureEngineer

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
    """
    Calculates PnL but ONLY takes the trade if the Meta-Model is confident.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    meta_probs_np = np.array(meta_probs)
    
    # the analyst's intended position
    intended_position = np.sign(y_pred_np)
    
    # the manager's veto power (1 = trade, 0 = sit out)
    trade_mask = (meta_probs_np > confidence_threshold).astype(int)
    
    # actual executed position
    actual_position = intended_position * trade_mask
    
    # we only pay fees when we actually trade
    raw_pnl = actual_position * y_true_np
    fees = trade_mask * cost_per_mwh
    net_pnl = pd.Series(raw_pnl - fees)
    
    equity_curve = net_pnl.cumsum()
    
    if net_pnl.std() != 0:
        sharpe = (net_pnl.mean() / net_pnl.std()) * np.sqrt(8760)
    else:
        sharpe = 0
        
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    
    # Trades executed %
    pct_traded = np.mean(trade_mask) * 100
    
    return {
        "total_pnl": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "percent_traded": pct_traded
    }

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
        if train_start < 0: raise ValueError("Dataset too small.")
        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return splits[::-1]

def main():
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="xgboost_meta_labeling"):
        
        print("Loading data...")
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

        fold_pnls, fold_sharpes, fold_dds, fold_traded = [], [], [], []

        print(f"Starting Meta-Labeling CV Pipeline...")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold} ---")
            
            # 1. SPLIT THE DATA
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # we split the train_df into Primary (Analyst) and Meta (Manager)
            split_meta_idx = len(train_df) - (config.cv.meta_train_days * 24)
            primary_df = train_df.iloc[:split_meta_idx]
            meta_df = train_df.iloc[split_meta_idx:]
            
            # 2. PREPROCESSING PIPELINE (Train on Primary, apply to all)
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

            # 3. TRAIN PRIMARY MODEL (The Analyst)
            analyst = xgb.XGBRegressor(**config.model, objective=asymmetric_trading_loss)
            analyst.fit(X_prim_prep, y_prim)
            
            # 4. GENERATE META LABELS ON THE MANAGER'S DATA
            meta_analyst_preds = analyst.predict(X_meta_prep)
            
            # did the analyst make a profit after the 0.50 fee? 
            # 1 if profit > 0, else 0
            meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
            meta_labels = (meta_raw_pnl > 0).astype(int)

            
            
            # 5. TRAIN META MODEL (The Manager)
            # give the manager the market features AND the analyst's prediction
            X_meta_enhanced = X_meta_prep.copy()
            X_meta_enhanced['analyst_pred'] = meta_analyst_preds
            
            meta_params = dict(config.meta_model)
            meta_params.pop('confidence_threshold', None)
            
            manager = xgb.XGBClassifier(**meta_params)
            manager.fit(X_meta_enhanced, meta_labels)
            
            # 6. EVALUATE ON TEST SET
            test_analyst_preds = analyst.predict(X_test_prep)
            
            X_test_enhanced = X_test_prep.copy()
            X_test_enhanced['analyst_pred'] = test_analyst_preds
            
            # the manager outputs probability of profit
            test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
            
            # calculate final metrics using the Meta-Model filter
            metrics = calculate_meta_trading_metrics(
                y_test, 
                test_analyst_preds, 
                test_manager_probs, 
                confidence_threshold=config.meta_model.confidence_threshold
            )
            
            fold_pnls.append(metrics['total_pnl'])
            fold_sharpes.append(metrics['sharpe_ratio'])
            fold_dds.append(metrics['max_drawdown'])
            fold_traded.append(metrics['percent_traded'])
            
            mlflow.log_metric(f"fold_{fold}_PnL", metrics['total_pnl'])
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics['max_drawdown'])
            
            print(f"Trading-> PnL: €{metrics['total_pnl']:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"Risk   -> Max Drawdown: €{metrics['max_drawdown']:.2f} | Traded: {metrics['percent_traded']:.1f}% of hours")

        # AVERAGE METRICS
        print(f"\n=========================================")
        print(f"META-LABELING AVERAGE TRADING:")
        print(f"PnL: €{np.mean(fold_pnls):.2f} | Sharpe: {np.mean(fold_sharpes):.2f}")
        print(f"Max Drawdown: €{np.mean(fold_dds):.2f} | Avg Hours Traded: {np.mean(fold_traded):.1f}%")
        print(f"=========================================")

        print("Run complete. Compare the Drawdown to your previous baseline!")

if __name__ == "__main__":
    main()