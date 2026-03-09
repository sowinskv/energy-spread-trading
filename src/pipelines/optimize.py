import logging
import warnings

import numpy as np
import optuna
from omegaconf import OmegaConf

from src.data.loader import get_expanding_walk_forward_splits, prepare_dataset
from src.ml.trainer import FoldTrainer
from src.trading.metrics import calculate_conviction_metrics

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _load_data():
    config = OmegaConf.load("config.yaml")
    df, bool_cols, numeric_cols = prepare_dataset(config)
    splits = get_expanding_walk_forward_splits(
        df_length=len(df),
        initial_train_days=config.cv.initial_train_days,
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

    # --- trading ---
    min_conviction = trial.suggest_float("min_conviction", 0.3, 1.2, step=0.1)

    # --- fixed architecture: ensemble, no multi-horizon ---
    ensemble_params = {
        "enable": True,
        "models": ["xgboost", "random_forest", "extra_trees", "ridge"],
        "multi_horizon": False,
    }

    # --- cross-validation ---
    fold_pnls = []
    fold_hit_rates = []
    fold_trade_counts = []

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        temp_config = OmegaConf.create(
            {
                "data": OmegaConf.to_container(config.data),
                "conformal": OmegaConf.to_container(config.get("conformal", {"enable": False})),
                "ensemble": {**ensemble_params, "two_stage": config.ensemble.get("two_stage", False)},
                "model": xgb_params,
                "model_params": {
                    "xgboost": xgb_params,
                    "random_forest": rf_params,
                    "extra_trees": et_params,
                    "ridge": {"alpha": ridge_alpha},
                },
            }
        )

        trainer = FoldTrainer(temp_config, bool_cols, numeric_cols)
        result = trainer.run_fold(train_df, test_df, analyst_config=temp_config, verbose=False)

        test_preds = result["test_preds"]
        conformal_mask = result.get("conformal_mask")

        try:
            metrics = calculate_conviction_metrics(
                result["y_test"], test_preds,
                cost_per_mwh=config.trading.cost_per_mwh,
                max_position=config.trading.get("max_position", 2.0),
                min_conviction=min_conviction,
                timestamps=test_df.index,
                skip_hours=tuple(config.trading.get("skip_hours", [])),
                conformal_mask=conformal_mask,
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

    # --- constrained objective: max PnL subject to hit_rate >= 70% ---
    # gate: too few trades → unreliable metrics
    if avg_trades < 30:
        return 0.0

    # hit rate is 0-100 scale from metrics; normalize to 0-1
    hr = avg_hit_rate / 100.0 if avg_hit_rate > 1 else avg_hit_rate

    # hard penalty below 70%: linear ramp from 0 at 50% to 1 at 70%
    # above 70%: mild bonus (don't over-optimize hit rate at cost of PnL)
    if hr < 0.50:
        hr_weight = 0.0
    elif hr < 0.70:
        hr_weight = (hr - 0.50) / 0.20  # linear 0→1 over [50%, 70%]
    else:
        hr_weight = 1.0 + 0.5 * (hr - 0.70)  # mild bonus above 70%

    pnl_component = np.log1p(max(avg_pnl, 0))
    score = hr_weight * pnl_component

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

    _heading("01", "OPTIMIZE", "optuna / max PnL s.t. hit_rate >= 70%")
    status("ensemble: xgboost + random forest + extra trees + ridge")
    status("regularized ranges / no multi-horizon")
    status("200 trials / expanding walk-forward cv / min 30 trades gate")
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
        study_name="energy_v4_two_stage",
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
