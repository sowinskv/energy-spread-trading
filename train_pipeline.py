import numpy as np
import mlflow
from omegaconf import OmegaConf

from src.trading.metrics import calculate_enhanced_meta_trading_metrics_with_exits
from src.core.data.loader import load_and_format_raw_data, get_expanding_walk_forward_splits
from src.ml.trainer import FoldTrainer


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

            print("training...")
            trainer = FoldTrainer(config, bool_cols, numeric_cols)
            result = trainer.run_fold(train_df, test_df)
            
            y_test = result['y_test']
            test_analyst_preds = result['test_analyst_preds']
            test_manager_probs = result['test_manager_probs']
            test_timestamps = result['test_timestamps']
            
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