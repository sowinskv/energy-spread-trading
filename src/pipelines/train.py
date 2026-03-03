import logging

import mlflow
import numpy as np
from omegaconf import OmegaConf

from src.data.loader import get_expanding_walk_forward_splits, prepare_dataset
from src.ml.trainer import FoldTrainer
from src.trading.metrics import calculate_enhanced_meta_trading_metrics_with_exits

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="xgboost_meta_labeling"):
        df, bool_cols, numeric_cols = prepare_dataset(config)

        splits = get_expanding_walk_forward_splits(
            df_length=len(df),
            initial_train_days=config.cv.initial_train_days,
            test_days=config.cv.test_days,
            purge_days=config.cv.purge_days,
            n_splits=config.cv.n_splits,
        )

        fold_pnls, fold_sharpes, fold_dds, fold_traded = [], [], [], []
        fold_hit_rates, fold_sortinos = [], []
        fold_position_sizes, fold_total_trades, fold_consensus_trades = [], [], []

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            logger.info("\nFOLD %d", fold)

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            logger.info("training...")
            trainer = FoldTrainer(config, bool_cols, numeric_cols)
            result = trainer.run_fold(train_df, test_df)

            y_test = result["y_test"]
            test_analyst_preds = result["test_analyst_preds"]
            test_manager_probs = result["test_manager_probs"]
            test_timestamps = result["test_timestamps"]

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
                use_exit_rules=False,
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
                use_exit_rules=config.exit_rules.enable,
            )

            fold_pnls.append(metrics["total_pnl"])
            fold_sharpes.append(metrics["sharpe_ratio"])
            fold_dds.append(metrics["max_drawdown"])
            fold_traded.append(metrics["percent_traded"])
            fold_hit_rates.append(metrics["hit_rate"])
            fold_sortinos.append(metrics["sortino_ratio"])
            fold_position_sizes.append(metrics["avg_position_size"])
            fold_total_trades.append(metrics["total_trades"])
            fold_consensus_trades.append(metrics["consensus_trades"])

            mlflow.log_metric(f"fold_{fold}_PnL", metrics["total_pnl"])
            mlflow.log_metric(f"fold_{fold}_PnL_NoExits", metrics_no_exits["total_pnl"])
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics["max_drawdown"])
            mlflow.log_metric(f"fold_{fold}_HitRate", metrics["hit_rate"])
            mlflow.log_metric(f"fold_{fold}_HitRate_NoExits", metrics_no_exits["hit_rate"])
            mlflow.log_metric(f"fold_{fold}_Sortino", metrics["sortino_ratio"])
            mlflow.log_metric(f"fold_{fold}_AvgPositionSize", metrics["avg_position_size"])
            mlflow.log_metric(f"fold_{fold}_TotalTrades", metrics["total_trades"])
            mlflow.log_metric(
                f"fold_{fold}_ConsensusRate", metrics["consensus_trades"] / max(1, metrics["total_trades"])
            )

            fold_report = [
                f"\n{'FOLD RESULTS':^40}",
                "-" * 40,
                f"{'PnL (exits)':<15} {config.trading.currency}{metrics['total_pnl']:>8.2f}",
                f"{'PnL (no exits)':<15} {config.trading.currency}{metrics_no_exits['total_pnl']:>8.2f}",
                f"{'Hit Rate':<15} {metrics['hit_rate']:>7.1f}%",
                f"{'Trades':<15} {metrics['total_trades']:>8}",
                f"{'Sharpe':<15} {metrics['sharpe_ratio']:>8.2f}",
                f"{'Sortino':<15} {metrics['sortino_ratio']:>8.2f}",
                f"{'Drawdown':<15} {config.trading.currency}{metrics['max_drawdown']:>8.2f}",
            ]
            logger.info("\n".join(fold_report))

        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_sortino = np.mean(fold_sortinos)
        avg_hit_rate = np.mean(fold_hit_rates)
        avg_dd = np.mean(fold_dds)
        avg_traded = np.mean(fold_traded)
        avg_position_size = np.mean(fold_position_sizes)
        avg_total_trades = np.mean(fold_total_trades)
        avg_consensus_rate = np.mean([c / max(1, t) for c, t in zip(fold_consensus_trades, fold_total_trades)])

        backtest_report = [
            "\n",
            f"{'BACKTEST RESULTS':^50}",
            "-" * 50,
            f"{'PnL':<20} {config.trading.currency} {avg_pnl:.2f}",
            f"{'Max Drawdown':<20} {config.trading.currency} {avg_dd:.2f}",
            f"{'Sharpe Ratio':<20} {avg_sharpe:.2f}",
            f"{'Sortino Ratio':<20} {avg_sortino:.2f}",
            f"{'Hit Rate':<20} {avg_hit_rate:.1f}%",
            f"{'Consensus Rate':<20} {avg_consensus_rate:.1%}",
            f"{'Hours Traded':<20} {avg_traded:.1f}%",
            f"{'Avg Trades / Fold':<20} {avg_total_trades:.0f}",
            f"{'Avg Position Size':<20} {avg_position_size:.2f}",
            " " * 50,
        ]
        logger.info("\n".join(backtest_report))

        mlflow.log_metric("avg_PnL", avg_pnl)
        mlflow.log_metric("avg_Sharpe", avg_sharpe)
        mlflow.log_metric("avg_Sortino", avg_sortino)
        mlflow.log_metric("avg_HitRate", avg_hit_rate)
        mlflow.log_metric("avg_MaxDD", avg_dd)
        mlflow.log_metric("avg_PercentTraded", avg_traded)
        mlflow.log_metric("avg_PositionSize", avg_position_size)
        mlflow.log_metric("avg_TotalTrades", avg_total_trades)
        mlflow.log_metric("avg_ConsensusRate", avg_consensus_rate)


if __name__ == "__main__":
    main()
