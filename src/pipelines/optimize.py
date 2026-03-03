import logging
import warnings

import numpy as np
import optuna
from omegaconf import OmegaConf

from src.data.loader import get_purged_walk_forward_splits, prepare_dataset
from src.ml.trainer import FoldTrainer
from src.trading.metrics import calculate_enhanced_meta_trading_metrics_with_exits

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _load_data():
    config = OmegaConf.load("config.yaml")
    df, bool_cols, numeric_cols = prepare_dataset(config)
    splits = get_purged_walk_forward_splits(
        df_length=len(df),
        train_days=config.cv.train_days,
        test_days=config.cv.test_days,
        purge_days=config.cv.purge_days,
        n_splits=config.cv.n_splits,
        embargo_days=config.cv.get("embargo_days", 0),
    )
    return config, df, bool_cols, numeric_cols, splits


def objective(trial, config, df, bool_cols, numeric_cols, splits):
    # --- analyst models (regularized ranges) ---
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 150, step=10),
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 5),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 10, 50),
    }

    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("rf_max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 10, 40),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 5, 20),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5, 0.7]),
    }

    et_params = {
        "n_estimators": trial.suggest_int("et_n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("et_max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("et_min_samples_split", 10, 40),
        "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 5, 20),
        "max_features": trial.suggest_categorical("et_max_features", ["sqrt", "log2", 0.5, 0.7]),
    }

    ridge_alpha = trial.suggest_float("ridge_alpha", 0.1, 50.0, log=True)

    # --- fixed architecture: ensemble, no multi-horizon ---
    ensemble_params = {
        "enable": True,
        "models": ["xgboost", "random_forest", "extra_trees", "ridge"],
        "reweight_frequency": trial.suggest_categorical("reweight_frequency", [168, 336, 720]),
        "performance_window": trial.suggest_categorical("performance_window", [24, 168, 336]),
        "multi_horizon": False,
    }

    # --- meta-model (regularized) ---
    manager_params = {
        "n_estimators": trial.suggest_int("manager_n_estimators", 30, 150, step=10),
        "max_depth": trial.suggest_int("manager_max_depth", 2, 4),
        "learning_rate": trial.suggest_float("manager_lr", 0.01, 0.1, log=True),
        "random_state": 42,
    }

    confidence_threshold = trial.suggest_float("confidence_threshold", 0.40, 0.65)

    # --- cross-validation ---
    fold_pnls = []
    fold_hit_rates = []
    fold_trade_counts = []

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        trainer = FoldTrainer(config, bool_cols, numeric_cols)
        data = trainer.prepare_fold_data(train_df, test_df)

        temp_config = OmegaConf.create(
            {
                "ensemble": ensemble_params,
                "model": xgb_params,
                "model_params": {
                    "xgboost": xgb_params,
                    "random_forest": rf_params,
                    "extra_trees": et_params,
                    "ridge": {"alpha": ridge_alpha},
                },
            }
        )
        trainer.train_analyst(
            data["X_prim"], data["y_prim"], data["primary_df"].index, analyst_config=temp_config, verbose=False
        )

        meta_analyst_preds = trainer.predict_analyst(data["X_meta"], data["meta_df"].index)
        meta_labels = trainer.create_meta_labels(meta_analyst_preds, data["y_meta"])

        X_meta_enhanced = trainer.enhance_features(data["X_meta"], meta_analyst_preds, pred_col="analyst_pred")
        trainer.train_manager(X_meta_enhanced, meta_labels, manager_params=manager_params)

        test_analyst_preds = trainer.predict_analyst(data["X_test"], test_df.index)
        X_test_enhanced = trainer.enhance_features(data["X_test"], test_analyst_preds, pred_col="analyst_pred")
        test_manager_probs = trainer.manager.predict_proba(X_test_enhanced)[:, 1]

        try:
            temp_config_full = OmegaConf.create(OmegaConf.to_yaml(config))
            temp_config_full.meta_model.confidence_threshold = confidence_threshold

            metrics = calculate_enhanced_meta_trading_metrics_with_exits(
                data["y_test"], test_analyst_preds, test_manager_probs, temp_config_full, use_exit_rules=False
            )
            pnl = metrics["total_pnl"] if not np.isnan(metrics["total_pnl"]) else 0
            hit_rate = metrics["hit_rate"] if not np.isnan(metrics["hit_rate"]) else 0
            n_trades = metrics["total_trades"]
        except Exception:
            pnl = 0
            hit_rate = 0
            n_trades = 0

        fold_pnls.append(pnl)
        fold_hit_rates.append(hit_rate)
        fold_trade_counts.append(n_trades)

    avg_pnl = np.mean(fold_pnls)
    avg_hit_rate = np.mean(fold_hit_rates)
    avg_trades = np.mean(fold_trade_counts)

    # --- composite objective: hit rate x PnL, penalize low trade volume ---
    # gate: if fewer than 30 trades/fold on average, heavy penalty
    if avg_trades < 30:
        return 0.0

    # normalize: hit_rate is 0-100, pnl can be anything
    # score = hit_rate * log(1 + max(pnl, 0)) — rewards both, multiplicative
    pnl_component = np.log1p(max(avg_pnl, 0))
    score = avg_hit_rate * pnl_component

    return score


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    from rich.text import Text

    from src.ui.display import INDENT, _heading, _kv, _rule, console, header, status

    header("ENERGY")

    config, df, bool_cols, numeric_cols, splits = _load_data()

    _heading("01", "OPTIMIZE", "optuna / hit rate x pnl composite")
    status("ensemble: xgboost + random forest + extra trees + ridge")
    status("regularized ranges / no multi-horizon")
    status("200 trials / purged walk-forward cv / min 30 trades gate")
    console.print()

    n_trials = 200

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n = trial.number + 1
        current = trial.value or 0.0
        best = study.best_value
        is_best = current >= best

        t = Text()
        t.append(f"{INDENT}T {n:03d} / {n_trials}   ", style="dim")
        t.append(f"{current:>8.1f}", style="bold" if is_best else "")
        t.append("   best ", style="dim")
        t.append(f"{best:>8.1f}", style="bold")
        if is_best:
            t.append("  ←", style="dim")
        console.print(t)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="energy_v3_regularized",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(
        lambda trial: objective(trial, config, df, bool_cols, numeric_cols, splits),
        n_trials=n_trials,
        callbacks=[_trial_callback],
    )

    _rule('"RESULTS"')
    console.print()
    _kv("best composite score", f"{study.best_value:.1f}", bold_value=True)
    console.print()

    _heading("02", "OPTIMAL PARAMETERS")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            _kv(key, f"{value:.4f}")
        else:
            _kv(key, str(value))
    console.print()
    _rule()
