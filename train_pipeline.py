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

def calculate_trading_metrics(y_true, y_pred, cost_per_mwh=0.5):
    """
    Calculates pure financial metrics: PnL, Sharpe, Drawdown, and Hit Rate.
    """
    # convert to numpy to safely align arrays ignoring pandas index
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # logic: if we predict positive, we buy (+1). if negative, we sell (-1).
    # profit is our directional bet multiplied by the actual spread.
    raw_pnl = np.sign(y_pred_np) * y_true_np
    
    # subtract the broker fee for every single hour we trade
    net_pnl = pd.Series(raw_pnl - cost_per_mwh)
    
    equity_curve = net_pnl.cumsum()
    
    # annualized sharpe (8760 hours in a year)
    if net_pnl.std() != 0:
        sharpe = (net_pnl.mean() / net_pnl.std()) * np.sqrt(8760)
    else:
        sharpe = 0
        
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    
    dir_acc = np.mean(np.sign(y_pred_np) == np.sign(y_true_np))
    
    return {
        "total_pnl": equity_curve.iloc[-1],
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "directional_accuracy": dir_acc * 100 # as percentage
    }

def get_purged_walk_forward_splits(df_length, train_days, test_days, purge_days, n_splits):
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
        
    return splits[::-1]

def main():
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="xgboost_quant_backtest"):
        
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

        # lists to store metrics across folds
        fold_maes, fold_rmses = [], []
        fold_pnls, fold_sharpes, fold_dds, fold_accs = [], [], [], []

        print(f"Starting {config.cv.n_splits}-Fold CV with QUANT METRICS...")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold} ---")
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            y_train = train_df[config.data.target_col]
            X_train = train_df.drop(columns=config.data.leakage_cols)
            
            y_test = test_df[config.data.target_col]
            X_test = test_df.drop(columns=config.data.leakage_cols)

            pipeline = Pipeline([
                ('imputer', TimeSeriesImputer(bool_cols=bool_cols, numeric_cols=numeric_cols)),
                ('feature_engineer', EnergyFeatureEngineer()),
                ('regressor', xgb.XGBRegressor(**config.model, objective=asymmetric_trading_loss))
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            
            # 1. statistical metrics
            fold_mae = mean_absolute_error(y_test, preds)
            fold_rmse = np.sqrt(mean_squared_error(y_test, preds))
            fold_maes.append(fold_mae)
            fold_rmses.append(fold_rmse)
            
            # 2. financial metrics
            trade_metrics = calculate_trading_metrics(y_test, preds, cost_per_mwh=0.5)
            fold_pnls.append(trade_metrics['total_pnl'])
            fold_sharpes.append(trade_metrics['sharpe_ratio'])
            fold_dds.append(trade_metrics['max_drawdown'])
            fold_accs.append(trade_metrics['directional_accuracy'])
            
            # log to mlflow
            mlflow.log_metric(f"fold_{fold}_MAE", fold_mae)
            mlflow.log_metric(f"fold_{fold}_PnL", trade_metrics['total_pnl'])
            mlflow.log_metric(f"fold_{fold}_Sharpe", trade_metrics['sharpe_ratio'])
            
            print(f"Stats  -> MAE: {fold_mae:.2f} | RMSE: {fold_rmse:.2f}")
            print(f"Trading-> PnL: €{trade_metrics['total_pnl']:.2f} | Sharpe: {trade_metrics['sharpe_ratio']:.2f}")
            print(f"Risk   -> Max Drawdown: €{trade_metrics['max_drawdown']:.2f} | Hit Rate: {trade_metrics['directional_accuracy']:.1f}%")

        # average everything
        avg_mae = np.mean(fold_maes)
        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_dd = np.mean(fold_dds)
        avg_acc = np.mean(fold_accs)
        
        mlflow.log_metric("CV_Avg_MAE", avg_mae)
        mlflow.log_metric("CV_Avg_PnL", avg_pnl)
        mlflow.log_metric("CV_Avg_Sharpe", avg_sharpe)
        
        print(f"\n=========================================")
        print(f"CV AVERAGE STATS : MAE: {avg_mae:.2f}")
        print(f"CV AVERAGE TRADING: PnL: €{avg_pnl:.2f} | Sharpe: {avg_sharpe:.2f} | Hit Rate: {avg_acc:.1f}%")
        print(f"=========================================")

        # generate an equity curve plot for the last fold to visualize the money
        plt.figure(figsize=(12, 5))
        net_pnl = (np.sign(preds) * np.array(y_test)) - 0.5
        equity_curve = pd.Series(net_pnl).cumsum()
        plt.plot(y_test.index, equity_curve, color='green', label='Cumulative PnL (€)')
        plt.title(f'Simulated Equity Curve (Fold {config.cv.n_splits}) - Fees Included')
        plt.ylabel('Profit in Euros (1 MWh volume)')
        plt.fill_between(y_test.index, equity_curve, 0, where=(equity_curve > 0), color='green', alpha=0.1)
        plt.fill_between(y_test.index, equity_curve, 0, where=(equity_curve < 0), color='red', alpha=0.1)
        plt.legend()
        
        os.makedirs("temp_plots", exist_ok=True)
        plot_path = "temp_plots/equity_curve.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path, "evaluation_plots")

        mlflow.sklearn.log_model(pipeline, "xgboost_pipeline_quant")
        print("Run complete. Check MLflow UI for financial metrics!")

if __name__ == "__main__":
    main()