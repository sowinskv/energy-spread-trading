import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from features import TimeSeriesImputer, EnergyFeatureEngineer
from ensemble_models import EnsembleAnalyst, MultiHorizonEnsemble

from src.trading.position_manager import TrailingStopManager
from src.trading.exit_strategies import (
    calculate_dynamic_exits, consensus_exit_rules, confidence_based_position_sizing,
    detect_market_volatility_regime, multi_horizon_consensus, optimal_trading_hours
)
from src.trading.metrics import calculate_enhanced_meta_trading_metrics_with_exits, asymmetric_trading_loss


























def load_and_format_raw_data(filepath):
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
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="xgboost_meta_labeling"):
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
        fold_position_sizes, fold_total_trades, fold_consensus_trades = [], [], []

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\nFOLD {fold}")
            
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
            test_timestamps = test_df.index if hasattr(test_df, 'index') else None

            print("training...")
            if config.ensemble.enable and config.ensemble.multi_horizon:
                ensemble_analyst = MultiHorizonEnsemble(config, horizons=config.ensemble.horizons)
                ensemble_analyst.fit(X_prim_prep, y_prim, primary_df.index)
                
                meta_analyst_preds = ensemble_analyst.predict(X_meta_prep, meta_df.index)
                test_analyst_preds = ensemble_analyst.predict(X_test_prep, test_timestamps)
                
                meta_individual_preds = {}
                test_individual_preds = {}
                
            elif config.ensemble.enable:
                ensemble_analyst = EnsembleAnalyst(config)
                ensemble_analyst.fit(X_prim_prep, y_prim)
                
                meta_analyst_preds = ensemble_analyst.predict(X_meta_prep)
                test_analyst_preds = ensemble_analyst.predict(X_test_prep)
                
                meta_individual_preds = ensemble_analyst.get_individual_predictions(X_meta_prep)
                test_individual_preds = ensemble_analyst.get_individual_predictions(X_test_prep)
                
                print(f"✓ ensemble: {len(ensemble_analyst.models)} models")
                print(f"  weights: {ensemble_analyst.weights}")
                
            else:
                print("fallback: single xgboost")
                analyst = xgb.XGBRegressor(**config.model, objective=asymmetric_trading_loss)
                analyst.fit(X_prim_prep, y_prim)
                
                meta_analyst_preds = analyst.predict(X_meta_prep)
                test_analyst_preds = analyst.predict(X_test_prep)
                
                meta_individual_preds = {}
                test_individual_preds = {}
            
            meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
            meta_labels = (meta_raw_pnl > 0).astype(int)
            
            X_meta_enhanced = X_meta_prep.copy()
            X_meta_enhanced['analyst_pred_ensemble'] = meta_analyst_preds
            
            if config.ensemble.enable and not config.ensemble.multi_horizon:
                for model_name, preds in meta_individual_preds.items():
                    X_meta_enhanced[f'analyst_pred_{model_name}'] = preds
                
                pred_values = list(meta_individual_preds.values())
                X_meta_enhanced['prediction_variance'] = np.var(pred_values, axis=0)
                X_meta_enhanced['prediction_range'] = np.max(pred_values, axis=0) - np.min(pred_values, axis=0)
                X_meta_enhanced['model_consensus'] = np.mean([
                    np.sign(pred) for pred in pred_values
                ], axis=0)

            meta_params = dict(config.meta_model)
            meta_params.pop('confidence_threshold', None)

            manager = xgb.XGBClassifier(**meta_params)
            manager.fit(X_meta_enhanced, meta_labels)
            
            X_test_enhanced = X_test_prep.copy()
            X_test_enhanced['analyst_pred_ensemble'] = test_analyst_preds
            
            if config.ensemble.enable and not config.ensemble.multi_horizon:
                for model_name, preds in test_individual_preds.items():
                    X_test_enhanced[f'analyst_pred_{model_name}'] = preds
                
                pred_values = list(test_individual_preds.values())
                X_test_enhanced['prediction_variance'] = np.var(pred_values, axis=0)
                X_test_enhanced['prediction_range'] = np.max(pred_values, axis=0) - np.min(pred_values, axis=0)
                X_test_enhanced['model_consensus'] = np.mean([
                    np.sign(pred) for pred in pred_values
                ], axis=0)
            
            test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
            
            metrics_no_exits = calculate_enhanced_meta_trading_metrics_with_exits(
                y_test, 
                test_analyst_preds, 
                test_manager_probs, 
                config,
                confidence_threshold=config.meta_model.confidence_threshold,
                cost_per_mwh=config.trading.cost_per_mwh,
                position_size_mwh=config.trading.position_size_mwh,
                use_dynamic_thresholds=True,
                use_confidence_sizing=True,
                timestamps=test_timestamps,
                use_exit_rules=False
            )
            
            metrics = calculate_enhanced_meta_trading_metrics_with_exits(
                y_test, 
                test_analyst_preds, 
                test_manager_probs, 
                config,
                confidence_threshold=config.meta_model.confidence_threshold,
                cost_per_mwh=config.trading.cost_per_mwh,
                position_size_mwh=config.trading.position_size_mwh,
                use_dynamic_thresholds=True,
                use_confidence_sizing=True,
                timestamps=test_timestamps,
                use_exit_rules=config.exit_rules.enable
            )
            
            fold_pnls.append(metrics['total_pnl'])
            fold_sharpes.append(metrics['sharpe_ratio'])
            fold_dds.append(metrics['max_drawdown'])
            fold_traded.append(metrics['percent_traded'])
            fold_hit_rates.append(metrics['hit_rate'])
            fold_sortinos.append(metrics['sortino_ratio'])
            fold_position_sizes.append(metrics['avg_position_size'])
            fold_total_trades.append(metrics['total_trades'])
            fold_consensus_trades.append(metrics['consensus_trades'])
            
            mlflow.log_metric(f"fold_{fold}_PnL", metrics['total_pnl'])
            mlflow.log_metric(f"fold_{fold}_PnL_NoExits", metrics_no_exits['total_pnl'])
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics['max_drawdown'])
            mlflow.log_metric(f"fold_{fold}_HitRate", metrics['hit_rate'])
            mlflow.log_metric(f"fold_{fold}_HitRate_NoExits", metrics_no_exits['hit_rate'])
            mlflow.log_metric(f"fold_{fold}_Sortino", metrics['sortino_ratio'])
            mlflow.log_metric(f"fold_{fold}_AvgPositionSize", metrics['avg_position_size'])
            mlflow.log_metric(f"fold_{fold}_TotalTrades", metrics['total_trades'])
            mlflow.log_metric(f"fold_{fold}_ConsensusRate", metrics['consensus_trades'] / max(1, metrics['total_trades']))
            
            print(f"\n{'FOLD RESULTS':^40}")
            print("-" * 40)
            print(f"{'PnL (exits)':<15} {config.trading.currency}{metrics['total_pnl']:>8.2f}")
            print(f"{'PnL (no exits)':<15} {config.trading.currency}{metrics_no_exits['total_pnl']:>8.2f}")  
            print(f"{'Hit Rate':<15} {metrics['hit_rate']:>7.1f}%")
            print(f"{'Trades':<15} {metrics['total_trades']:>8}")
            print(f"{'Sharpe':<15} {metrics['sharpe_ratio']:>8.2f}")
            print(f"{'Sortino':<15} {metrics['sortino_ratio']:>8.2f}")
            print(f"{'Drawdown':<15} {config.trading.currency}{metrics['max_drawdown']:>8.2f}")


        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_sortino = np.mean(fold_sortinos)
        avg_hit_rate = np.mean(fold_hit_rates)
        avg_dd = np.mean(fold_dds)
        avg_traded = np.mean(fold_traded)
        avg_position_size = np.mean(fold_position_sizes)
        avg_total_trades = np.mean(fold_total_trades)
        avg_consensus_rate = np.mean([c/max(1,t) for c,t in zip(fold_consensus_trades, fold_total_trades)])
        
        exit_status = "Enabled" if config.exit_rules.enable else "Disabled"
        print("\n")
        print(f"BACKTEST RESULTS".center(50))
        print("-"*50)
        print(f"{'PnL':<20} {config.trading.currency} {avg_pnl:.2f}")
        print(f"{'Max Drawdown':<20} {config.trading.currency} {avg_dd:.2f}")
        print(f"{'Sharpe Ratio':<20} {avg_sharpe:.2f}")
        print(f"{'Sortino Ratio':<20} {avg_sortino:.2f}")
        print(f"{'Hit Rate':<20} {avg_hit_rate:.1f}%")
        print(f"{'Consensus Rate':<20} {avg_consensus_rate:.1%}")
        print(f"{'Hours Traded':<20} {avg_traded:.1f}%")
        print(f"{'Avg Trades / Fold':<20} {avg_total_trades:.0f}")
        print(f"{'Avg Position Size':<20} {avg_position_size:.2f}")
        print(" "*50)
        
        mlflow.log_metric("avg_PnL", avg_pnl)
        mlflow.log_metric("avg_Sharpe", avg_sharpe)
        mlflow.log_metric("avg_Sortino", avg_sortino)
        mlflow.log_metric("avg_HitRate", avg_hit_rate)
        mlflow.log_metric("avg_MaxDD", avg_dd)
        mlflow.log_metric("avg_PercentTraded", avg_traded)
        mlflow.log_metric("avg_PositionSize", avg_position_size)
        mlflow.log_metric("avg_TotalTrades", avg_total_trades)
        mlflow.log_metric("avg_ConsensusRate", avg_consensus_rate)

        print("\ntraining complete.")

if __name__ == "__main__":
    main()