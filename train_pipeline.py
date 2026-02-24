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
    """
    Custom objective function for trading.
    Penalizes wrong-direction predictions heavily, scaled by spread magnitude.
    """
    # base gradient & hessian
    residual = y_pred - y_true
    grad = residual.copy()
    hess = np.ones_like(y_pred)
    
    # directional penalties (ad. 1. in notes)
    fp_mask = (y_true < 0) & (y_pred > 0)
    
    # FN (ad. 2. in notes)
    fn_mask = (y_true > 0) & (y_pred < 0)
    
    grad[fp_mask] *= 5.0
    hess[fp_mask] *= 5.0
    
    grad[fn_mask] *= 2.0
    hess[fn_mask] *= 2.0
    
    # profit-weighting (ad. 3. in notes)
    magnitude_weight = 1.0 + (np.abs(y_true) / 10.0) 
    
    grad = grad * magnitude_weight
    hess = hess * magnitude_weight
    
    return grad, hess

def get_purged_walk_forward_splits(df_length, train_days, test_days, purge_days, n_splits):
    """
    yirelds train and test indices for purged walk-forward CV, working backwards 
    from the most recent data to ensure the last fold tests on the latest month.
    """
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
        
        if train_start < 0:
            raise ValueError(f"Dataset too small for fold {i+1}. Need more historical data.")
            
        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)
        
        splits.append((train_indices, test_indices))
        
    # reverse so fold 1 is the oldest in time, and fold N is the most recent
    return splits[::-1]

def main():
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=config.mlflow.run_name):
        
        mlflow.log_params(config.model)
        mlflow.log_params(config.cv)

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

        fold_maes = []
        fold_rmses = []

        print(f"Starting {config.cv.n_splits}-Fold Purged Walk-Forward CV...")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"--- Fold {fold} ---")
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            y_train = train_df[config.data.target_col]
            X_train = train_df.drop(columns=config.data.leakage_cols)
            
            y_test = test_df[config.data.target_col]
            X_test = test_df.drop(columns=config.data.leakage_cols)

            pipeline = Pipeline([
                ('imputer', TimeSeriesImputer(bool_cols=bool_cols, numeric_cols=numeric_cols)),
                ('feature_engineer', EnergyFeatureEngineer()),
                ('regressor', xgb.XGBRegressor(**config.model,
                                               objective=asymmetric_trading_loss
                ))
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            
            fold_mae = mean_absolute_error(y_test, preds)
            fold_rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            fold_maes.append(fold_mae)
            fold_rmses.append(fold_rmse)
            
            mlflow.log_metric(f"fold_{fold}_MAE", fold_mae)
            
            print(f"Train Period: {train_df.index.min().date()} to {train_df.index.max().date()}")
            print(f"Purge Period: {train_df.index.max().date()} to {test_df.index.min().date()}")
            print(f"Test Period : {test_df.index.min().date()} to {test_df.index.max().date()}")
            print(f"Fold {fold} MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f}\n")

        avg_mae = np.mean(fold_maes)
        avg_rmse = np.mean(fold_rmses)
        mlflow.log_metric("CV_Avg_MAE", avg_mae)
        mlflow.log_metric("CV_Avg_RMSE", avg_rmse)
        
        print(f"=========================================")
        print(f"CV Average MAE: {avg_mae:.2f} | CV Average RMSE: {avg_rmse:.2f}")

        # plot the LAST fold for visual checking
        plt.figure(figsize=(15, 6))
        plot_idx = -24 * 7
        plt.plot(y_test.index[plot_idx:], y_test.values[plot_idx:], label='Actual', alpha=0.8)
        plt.plot(y_test.index[plot_idx:], preds[plot_idx:], label='Predicted', alpha=0.9, linestyle='--')
        plt.title(f'Pipeline Predictions (Last 7 Days of Fold {config.cv.n_splits})')
        plt.legend()

        os.makedirs("temp_plots", exist_ok=True)
        plot_path = "temp_plots/predictions_cv.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path, "evaluation_plots")

        # log the pipeline fitted on the FINAL fold
        mlflow.sklearn.log_model(pipeline, "xgboost_pipeline_latest_fold")
        print("Run complete. Check MLflow UI.")

if __name__ == "__main__":
    main()