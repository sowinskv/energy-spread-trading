import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
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

def calculate_meta_trading_metrics(y_true, y_pred, meta_probs, confidence_threshold=0.5, cost_per_mwh=0.5, position_size_mwh=1):
    """
    Calculates PnL but ONLY takes the trade if the Meta-Model is confident.
    
    IMPORTANT: Assumes prices are in EUR/MWh (standard for EU Market Coupling).
    All P&L calculations return EUR values based on SDAC/IDA price spreads.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    meta_probs_np = np.array(meta_probs)
    
    intended_position = np.sign(y_pred_np)
    trade_mask = (meta_probs_np > confidence_threshold).astype(int)
    actual_position = intended_position * trade_mask * position_size_mwh
    
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
    
    pct_traded = np.mean(trade_mask) * 100
    
    executed_trades = net_pnl[trade_mask == 1]
    if len(executed_trades) > 0:
        hit_rate = (executed_trades > 0).mean() * 100
    else:
        hit_rate = 0
    
    if len(net_pnl) > 0:
        downside_returns = net_pnl[net_pnl < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino = (net_pnl.mean() / downside_returns.std()) * np.sqrt(8760)
        else:
            sortino = sharpe if sharpe != 0 else 0
    else:
        sortino = 0
    
    return {
        "total_pnl": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "percent_traded": pct_traded
    }

def load_and_format_raw_data(filepath):
    print("loading raw data and formatting...")
    df = pd.read_csv(filepath, low_memory=False)
    
    cols_to_exclude = ['date_cet', 'IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
    numeric_cols = [col for col in df.columns if col not in cols_to_exclude]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
    for col in ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0).astype(int)
        
    df['date_cet'] = pd.to_datetime(df['date_cet'])
    df.set_index('date_cet', inplace=True)
    if df.index.duplicated().any():
        df = df.groupby(df.index).first()
    df = df.asfreq('h')
    df['spread_SDAC_IDA1_PL'] = df['SDAC_PL'] - df['IDA1_PL']
    df['spread_SDAC_IDA1_PL'] = df['spread_SDAC_IDA1_PL'].interpolate(method='linear', limit_direction='both')
    
    df['target_lag_24h'] = df['spread_SDAC_IDA1_PL'].shift(24)
    df['target_lag_48h'] = df['spread_SDAC_IDA1_PL'].shift(48)
    df['target_lag_168h'] = df['spread_SDAC_IDA1_PL'].shift(168)
    
    df['target_rolling_mean_24h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=24).mean()
    df['target_rolling_std_24h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=24).std()
    df['target_rolling_mean_168h'] = df['spread_SDAC_IDA1_PL'].shift(24).rolling(window=168).mean()
    
    return df

def get_expanding_walk_forward_splits(df_length, initial_train_days, test_days, purge_days, n_splits):
    """
    Expanding window: each fold gets progressively more training data.
    More realistic for production where you accumulate historical data daily.
    """
    initial_train_steps = initial_train_days * 24
    test_steps = test_days * 24
    purge_steps = purge_days * 24
    
    splits = []
    for i in range(n_splits):
        test_start = initial_train_steps + purge_steps + (i * (test_steps + purge_steps))
        test_end = test_start + test_steps
        
        if test_end > df_length:
            break
            
        train_start = 0 
        train_end = test_start - purge_steps
        
        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    
    return splits

def main():
    """main training pipeline with meta-labeling approach.
    
    CURRENCY ASSUMPTION: all price data assumed to be in EUR/MWh as per
    European Market Coupling standards (SDAC uses EUR for settlement).
    """
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="xgboost_meta_labeling"):
        
        print("loading data...")
        df = load_and_format_raw_data(config.data.file_path)
        
        X_full = df.drop(columns=config.data.leakage_cols)
        bool_cols = ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
        numeric_cols = [c for c in X_full.columns if c not in bool_cols]

        splits = get_expanding_walk_forward_splits(
            df_length=len(df),
            initial_train_days=270,  
            test_days=config.cv.test_days,
            purge_days=config.cv.purge_days,
            n_splits=config.cv.n_splits
        )

        fold_pnls, fold_sharpes, fold_dds, fold_traded = [], [], [], []
        fold_hit_rates, fold_sortinos = [], []

        print("starting cross-validation pipeline...")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- fold {fold} ---")
            
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

            analyst = xgb.XGBRegressor(**config.model, objective=asymmetric_trading_loss)
            analyst.fit(X_prim_prep, y_prim)
            
            meta_analyst_preds = analyst.predict(X_meta_prep)
            meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
            meta_labels = (meta_raw_pnl > 0).astype(int)
            X_meta_enhanced = X_meta_prep.copy()
            X_meta_enhanced['analyst_pred'] = meta_analyst_preds
            
            meta_params = dict(config.meta_model)
            meta_params.pop('confidence_threshold', None)

            manager = xgb.XGBClassifier(**meta_params)
            manager.fit(X_meta_enhanced, meta_labels)
            
            test_analyst_preds = analyst.predict(X_test_prep)
            
            X_test_enhanced = X_test_prep.copy()
            X_test_enhanced['analyst_pred'] = test_analyst_preds
            
            test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
            metrics = calculate_meta_trading_metrics(
                y_test, 
                test_analyst_preds, 
                test_manager_probs, 
                confidence_threshold=config.meta_model.confidence_threshold,
                cost_per_mwh=config.trading.cost_per_mwh,
                position_size_mwh=config.trading.position_size_mwh
            )
            
            fold_pnls.append(metrics['total_pnl'])
            fold_sharpes.append(metrics['sharpe_ratio'])
            fold_dds.append(metrics['max_drawdown'])
            fold_traded.append(metrics['percent_traded'])
            fold_hit_rates.append(metrics['hit_rate'])
            fold_sortinos.append(metrics['sortino_ratio'])
            
            mlflow.log_metric(f"fold_{fold}_PnL", metrics['total_pnl'])
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics['max_drawdown'])
            mlflow.log_metric(f"fold_{fold}_HitRate", metrics['hit_rate'])
            mlflow.log_metric(f"fold_{fold}_Sortino", metrics['sortino_ratio'])
            
            print(f"pnl: {config.trading.currency}{metrics['total_pnl']:.2f} | sharpe: {metrics['sharpe_ratio']:.2f} | sortino: {metrics['sortino_ratio']:.2f}")
            print(f"drawdown: {config.trading.currency}{metrics['max_drawdown']:.2f} | hit rate: {metrics['hit_rate']:.1f}% | traded: {metrics['percent_traded']:.1f}%")

        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_sortino = np.mean(fold_sortinos)
        avg_hit_rate = np.mean(fold_hit_rates)
        avg_dd = np.mean(fold_dds)
        avg_traded = np.mean(fold_traded)
        
        print("\n" + "="*50)
        print("average results across folds:")
        print(f"pnl: {config.trading.currency}{avg_pnl:.2f} | sharpe: {avg_sharpe:.2f} | sortino: {avg_sortino:.2f}")
        print(f"drawdown: {config.trading.currency}{avg_dd:.2f} | hit rate: {avg_hit_rate:.1f}% | hours traded: {avg_traded:.1f}%")
        print("="*50)
        
        mlflow.log_metric("avg_PnL", avg_pnl)
        mlflow.log_metric("avg_Sharpe", avg_sharpe)
        mlflow.log_metric("avg_Sortino", avg_sortino)
        mlflow.log_metric("avg_HitRate", avg_hit_rate)
        mlflow.log_metric("avg_MaxDD", avg_dd)
        mlflow.log_metric("avg_PercentTraded", avg_traded)

        print("\ntraining complete. check mlflow for detailed results.")

if __name__ == "__main__":
    main()