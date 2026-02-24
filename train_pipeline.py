# train_pipeline.py
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

def main():
    config = OmegaConf.load("config.yaml")
    
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=config.mlflow.run_name):
        
        mlflow.log_params(config.model)
        mlflow.log_param("test_days", config.data.test_days)

        print("Loading data...")
        df = pd.read_csv(config.data.file_path, index_col='date_cet', parse_dates=True)
        
        split_idx = len(df) - (config.data.test_days * 24)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        y_train = train_df[config.data.target_col]
        X_train = train_df.drop(columns=config.data.leakage_cols)
        
        y_test = test_df[config.data.target_col]
        X_test = test_df.drop(columns=config.data.leakage_cols)

        bool_cols = ['IS_ACTIVE_DOWN_SDAC_PL', 'IS_ACTIVE_UP_SDAC_PL']
        numeric_cols = [c for c in X_train.columns if c not in bool_cols]

        pipeline = Pipeline([
            ('imputer', TimeSeriesImputer(bool_cols=bool_cols, numeric_cols=numeric_cols)),
            ('feature_engineer', EnergyFeatureEngineer()),
            ('regressor', xgb.XGBRegressor(**config.model))
        ])

        print("training pipeline...")
        pipeline.fit(X_train, y_train)

        print("evaluating...")
        preds = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        
        print(f"Test MAE: {mae:.2f} | Test RMSE: {rmse:.2f}")

        plt.figure(figsize=(15, 6))
        plot_idx = -24 * 7
        plt.plot(y_test.index[plot_idx:], y_test.values[plot_idx:], label='Actual', alpha=0.8)
        plt.plot(y_test.index[plot_idx:], preds[plot_idx:], label='Predicted', alpha=0.9, linestyle='--')
        plt.title('Pipeline Model Predictions (Last 7 Days)')
        plt.legend()

        os.makedirs("temp_plots", exist_ok=True)
        plot_path = "temp_plots/predictions.png"
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_artifact(plot_path, "evaluation_plots")

        # log the entire pipeline (including the preprocessing logic!)
        mlflow.sklearn.log_model(pipeline, "xgboost_pipeline")
        
        print(f"Run complete. Check MLflow UI.")

if __name__ == "__main__":
    main()