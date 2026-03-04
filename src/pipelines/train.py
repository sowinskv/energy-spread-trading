import logging

import mlflow
import numpy as np
from omegaconf import OmegaConf

from src.data.loader import get_expanding_walk_forward_splits, prepare_dataset
from src.ml.trainer import FoldTrainer
from src.trading.metrics import calculate_conviction_metrics
from src.ui.display import backtest_summary, fold_header, fold_results, header, status

logger = logging.getLogger(__name__)


def _bootstrap_sharpe(
    pnls: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[tuple[float, float], float]:
    """Bootstrap Sharpe ratio confidence interval and p-value.

    Returns:
        (lower, upper) CI bounds and p-value (prob Sharpe <= 0).
    """
    rng = np.random.default_rng(42)
    n = len(pnls)
    if n < 10:
        return (0.0, 0.0), 1.0

    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(pnls, size=n, replace=True)
        std = sample.std()
        sharpes[i] = (sample.mean() / std) * np.sqrt(8760) if std > 0 else 0.0

    alpha = (1 - ci) / 2
    lower = float(np.percentile(sharpes, alpha * 100))
    upper = float(np.percentile(sharpes, (1 - alpha) * 100))
    p_value = float(np.mean(sharpes <= 0))
    return (lower, upper), p_value


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="conviction_sizing"):
        header("ENERGY")
        df, bool_cols, numeric_cols = prepare_dataset(config)

        splits = get_expanding_walk_forward_splits(
            df_length=len(df),
            initial_train_days=config.cv.initial_train_days,
            test_days=config.cv.test_days,
            purge_days=config.cv.purge_days,
            n_splits=config.cv.n_splits,
            embargo_days=config.cv.get("embargo_days", 0),
        )

        fold_pnls, fold_sharpes, fold_dds, fold_traded = [], [], [], []
        fold_hit_rates, fold_sortinos = [], []
        fold_total_trades, fold_avg_sizes = [], []
        all_net_pnls: list[np.ndarray] = []

        cost = config.trading.cost_per_mwh
        max_pos = config.trading.get("max_position", 2.0)
        min_pred = config.trading.get("min_prediction_threshold", 0.0)

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            fold_header(fold, len(train_idx), len(test_idx))
            status("training...")
            trainer = FoldTrainer(config, bool_cols, numeric_cols)
            result = trainer.run_fold(train_df, test_df)

            y_test = result["y_test"]
            test_preds = result["test_preds"]
            fit_metrics = result["fit_metrics"]

            metrics = calculate_conviction_metrics(
                y_test,
                test_preds,
                cost_per_mwh=cost,
                max_position=max_pos,
                min_prediction_threshold=min_pred,
            )

            fold_pnls.append(metrics["total_pnl"])
            fold_sharpes.append(metrics["sharpe_ratio"])
            fold_dds.append(metrics["max_drawdown"])
            fold_traded.append(metrics["percent_traded"])
            fold_hit_rates.append(metrics["hit_rate"])
            fold_sortinos.append(metrics["sortino_ratio"])
            fold_total_trades.append(metrics["total_trades"])
            fold_avg_sizes.append(metrics["avg_position_size"])

            mlflow.log_metric(f"fold_{fold}_PnL", metrics["total_pnl"])
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics["max_drawdown"])
            mlflow.log_metric(f"fold_{fold}_HitRate", metrics["hit_rate"])
            mlflow.log_metric(f"fold_{fold}_Sortino", metrics["sortino_ratio"])
            mlflow.log_metric(f"fold_{fold}_TotalTrades", metrics["total_trades"])
            mlflow.log_metric(f"fold_{fold}_AvgSize", metrics["avg_position_size"])

            fold_results(metrics, config.trading.currency, fit_metrics=fit_metrics)

            all_net_pnls.append(metrics["net_pnls"])

        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_sortino = np.mean(fold_sortinos)
        avg_hit_rate = np.mean(fold_hit_rates)
        avg_dd = np.mean(fold_dds)
        avg_traded = np.mean(fold_traded)
        avg_total_trades = np.mean(fold_total_trades)
        avg_position = np.mean(fold_avg_sizes)

        # bootstrap sharpe CI
        combined_pnls = np.concatenate(all_net_pnls) if all_net_pnls else np.array([0.0])
        bootstrap_ci, bootstrap_p = _bootstrap_sharpe(combined_pnls)

        backtest_summary(
            avg_pnl=avg_pnl,
            avg_dd=avg_dd,
            avg_sharpe=avg_sharpe,
            avg_sortino=avg_sortino,
            avg_hit_rate=avg_hit_rate,
            avg_traded=avg_traded,
            avg_trades=avg_total_trades,
            avg_position=avg_position,
            currency=config.trading.currency,
            n_folds=len(splits),
            sharpe_ci=bootstrap_ci,
            sharpe_p=bootstrap_p,
        )

        mlflow.log_metric("avg_PnL", avg_pnl)
        mlflow.log_metric("avg_Sharpe", avg_sharpe)
        mlflow.log_metric("avg_Sortino", avg_sortino)
        mlflow.log_metric("avg_HitRate", avg_hit_rate)
        mlflow.log_metric("avg_MaxDD", avg_dd)
        mlflow.log_metric("avg_PercentTraded", avg_traded)
        mlflow.log_metric("avg_TotalTrades", avg_total_trades)
        mlflow.log_metric("avg_AvgSize", avg_position)


if __name__ == "__main__":
    main()
