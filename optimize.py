import logging
import warnings

import numpy as np
import optuna
import xgboost as xgb
from omegaconf import OmegaConf

from src.core.data.loader import get_purged_walk_forward_splits, prepare_dataset
from src.ml.trainer import FoldTrainer
from src.trading.metrics import asymmetric_trading_loss, calculate_enhanced_meta_trading_metrics_with_exits

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
    )
    return config, df, bool_cols, numeric_cols, splits


def objective(trial, config, df, bool_cols, numeric_cols, splits):
    use_ensemble = trial.suggest_categorical("use_ensemble", [True, False])

    if use_ensemble:
        ensemble_params = {
            "reweight_frequency": trial.suggest_categorical("reweight_frequency", [168, 336, 720, 1440]),
            "performance_window": trial.suggest_categorical("performance_window", [48, 168, 336, 720]),
            "multi_horizon": trial.suggest_categorical("multi_horizon", [True, False]),
        }

        if ensemble_params["multi_horizon"]:
            horizon_choice = trial.suggest_categorical("horizons", ["short", "medium", "long", "extended"])
            horizon_map = {"short": [1, 4], "medium": [1, 4, 12], "long": [1, 4, 12, 24], "extended": [1, 2, 6, 12, 24]}
            ensemble_params["horizons"] = horizon_map[horizon_choice]

        analyst_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
            "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.7, 1.0),
        }

        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 10, 25),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.7, 0.8, 0.9]),
        }

        et_params = {
            "n_estimators": trial.suggest_int("et_n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("et_max_depth", 10, 25),
            "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("et_max_features", ["sqrt", "log2", 0.7, 0.8, 0.9]),
        }
    else:
        analyst_params = {
            "n_estimators": trial.suggest_int("analyst_n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("analyst_max_depth", 4, 9),
            "learning_rate": trial.suggest_float("analyst_lr", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("analyst_subsample", 0.6, 1.0),
            "random_state": 42,
        }

    manager_params = {
        "n_estimators": trial.suggest_int("manager_n_estimators", 50, 200, step=50),
        "max_depth": trial.suggest_int("manager_max_depth", 2, 5),
        "learning_rate": trial.suggest_float("manager_lr", 0.01, 0.1, log=True),
        "random_state": 42,
    }

    confidence_threshold = trial.suggest_float("confidence_threshold", 0.35, 0.75)

    fold_hit_rates = []

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        trainer = FoldTrainer(config, bool_cols, numeric_cols)
        data = trainer.prepare_fold_data(train_df, test_df)

        if use_ensemble:
            temp_config = OmegaConf.create(
                {
                    "ensemble": ensemble_params,
                    "model": analyst_params,
                    "model_params": {
                        "xgboost": {
                            "n_estimators": analyst_params["n_estimators"],
                            "max_depth": analyst_params["max_depth"],
                            "learning_rate": analyst_params["learning_rate"],
                            "subsample": analyst_params["subsample"],
                            "colsample_bytree": analyst_params["colsample_bytree"],
                        },
                        "random_forest": {
                            "n_estimators": rf_params["n_estimators"],
                            "max_depth": rf_params["max_depth"],
                            "min_samples_split": rf_params["min_samples_split"],
                            "min_samples_leaf": rf_params["min_samples_leaf"],
                            "max_features": rf_params["max_features"],
                        },
                        "extra_trees": {
                            "n_estimators": et_params["n_estimators"],
                            "max_depth": et_params["max_depth"],
                            "min_samples_split": et_params["min_samples_split"],
                            "min_samples_leaf": et_params["min_samples_leaf"],
                            "max_features": et_params["max_features"],
                        },
                        "ridge": {"alpha": 1.0},
                    },
                }
            )
            trainer.train_analyst(data["X_prim"], data["y_prim"], data["primary_df"].index, analyst_config=temp_config)
        else:
            analyst = xgb.XGBRegressor(**analyst_params, objective=asymmetric_trading_loss)
            analyst.fit(data["X_prim"], data["y_prim"])
            trainer.analyst = analyst

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
            hit_rate = metrics["hit_rate"] if not np.isnan(metrics["hit_rate"]) else 0
        except Exception as e:
            logger.warning("metrics calculation failed, using fallback: %s", e)
            y_true_np = np.array(data["y_test"])
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    config, df, bool_cols, numeric_cols, splits = _load_data()

    logger.info("running optuna optimization for maximum hit rate...")
    logger.info("targeting high-performing models: XGBoost, Random Forest, Extra Trees (89%% of ensemble)")

    study = optuna.create_study(
        study_name="energy_hit_rate_optimization",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(
        lambda trial: objective(trial, config, df, bool_cols, numeric_cols, splits),
        n_trials=100,
    )

    results = [
        "\n" + "=" * 50,
        "optimization complete. best results:",
        f"best cv hit rate: {study.best_value:.1f}%",
        "\noptimal parameters:",
    ]
    for key, value in study.best_params.items():
        if isinstance(value, float):
            results.append(f"  {key}: {value:.4f}")
        else:
            results.append(f"  {key}: {value}")
    results.append("=" * 50)
    results.append("update your config.yaml with these parameters.")
    logger.info("\n".join(results))
