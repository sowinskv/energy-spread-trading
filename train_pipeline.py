import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from features import TimeSeriesImputer, EnergyFeatureEngineer


def confidence_based_position_sizing(meta_probs, base_position=1.0, min_confidence=0.5, max_multiplier=2.0):
    confidence_excess = np.maximum(0, meta_probs - min_confidence)
    confidence_normalized = confidence_excess / (1 - min_confidence)
    
    multipliers = 1.0 + (max_multiplier - 1.0) * confidence_normalized
    positions = base_position * multipliers
    
    return np.clip(positions, 0.1, max_multiplier * base_position)


def detect_market_volatility_regime(returns, window=72):
    if len(returns) < window:
        return 'normal'
        
    recent_vol = returns.rolling(window).std().iloc[-1]
    historical_vol = returns.rolling(window*3).std().mean()
    
    vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
    
    if vol_ratio > 1.5:
        return 'high_vol'  
    elif vol_ratio < 0.7:
        return 'low_vol'   
    else:
        return 'normal'


def multi_horizon_consensus(predictions_1h, predictions_4h, consensus_threshold=0.75):
    """
    Only trade when multiple time horizons agree on direction.
    This improves hit rate by filtering out noisy signals.
    """
    direction_1h = np.sign(predictions_1h)
    direction_4h = np.sign(predictions_4h)
    
    agreement = (direction_1h == direction_4h) & (direction_1h != 0)
    
    consensus_strength = (np.abs(predictions_1h) + np.abs(predictions_4h)) / 2
    strong_consensus = consensus_strength > np.percentile(consensus_strength, 75)
    
    return agreement & strong_consensus


def optimal_trading_hours(timestamp_index):
    """
    Identify optimal trading hours based on market activity.
    Avoid trading during low-liquidity periods.
    """
    hours = timestamp_index.hour
    weekday = timestamp_index.dayofweek
    
    active_hours = (hours >= 6) & (hours <= 22)
    active_days = weekday < 5  
    
    peak_morning = (hours >= 8) & (hours <= 10)
    peak_evening = (hours >= 18) & (hours <= 20)
    peak_hours = peak_morning | peak_evening
    
    return active_hours & active_days, peak_hours & active_days


def asymmetric_trading_loss(y_true, y_pred):
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

def calculate_enhanced_meta_trading_metrics(y_true, y_pred, meta_probs, confidence_threshold=0.5, 
                                          cost_per_mwh=0.5, position_size_mwh=1.0,
                                          use_dynamic_thresholds=True, use_confidence_sizing=True,
                                          timestamps=None):
    """
    Enhanced trading with multiple profit/hit rate optimization techniques.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    meta_probs_np = np.array(meta_probs)
    
    if use_dynamic_thresholds and len(y_true_np) > 100:
        returns = pd.Series(y_true_np)
        regime = detect_market_volatility_regime(returns)
        
        regime_adjustments = {
            'low_vol': 0.8,    
            'normal': 1.0,     
            'high_vol': 1.3   
        }
        dynamic_threshold = confidence_threshold * regime_adjustments[regime]
    else:
        dynamic_threshold = confidence_threshold
    
    if len(y_pred_np) >= smooth_window:
        y_pred_4h = pd.Series(y_pred_np).rolling(smooth_window, min_periods=1).mean().values
        consensus_mask = multi_horizon_consensus(y_pred_np, y_pred_4h)
    else:
        consensus_mask = np.ones_like(y_pred_np, dtype=bool)
    
    if timestamps is not None:
        active_hours, peak_hours = optimal_trading_hours(timestamps)
        timing_multiplier = np.where(peak_hours, 1.2, np.where(active_hours, 1.0, 0.3))
    else:
        timing_multiplier = np.ones_like(y_pred_np)
    
    base_trade_mask = (meta_probs_np > dynamic_threshold).astype(int)
    consensus_trade_mask = base_trade_mask * consensus_mask.astype(int)
    final_trade_mask = consensus_trade_mask
    
    intended_position = np.sign(y_pred_np)
    
    if use_confidence_sizing:
        position_sizes = confidence_based_position_sizing(
            meta_probs_np, 
            base_position=position_size_mwh,
            max_multiplier=2.0
        )
        position_sizes = position_sizes * timing_multiplier
    else:
        position_sizes = np.full_like(y_pred_np, position_size_mwh)
    
    actual_position = intended_position * final_trade_mask * position_sizes
    raw_pnl = actual_position * y_true_np
    fees = final_trade_mask * cost_per_mwh * np.abs(actual_position)
    net_pnl = pd.Series(raw_pnl - fees)
    
    equity_curve = net_pnl.cumsum()
    
    if net_pnl.std() != 0:
        sharpe = (net_pnl.mean() / net_pnl.std()) * np.sqrt(8760)
    else:
        sharpe = 0
        
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    
    pct_traded = np.mean(final_trade_mask) * 100
    avg_position_size = np.mean(position_sizes[final_trade_mask == 1]) if np.sum(final_trade_mask) > 0 else 0
    
    executed_trades = net_pnl[final_trade_mask == 1]
    if len(executed_trades) > 0:
        hit_rate = (executed_trades > 0).mean() * 100
    else:
        hit_rate = 0
    
    if len(net_pnl) > 0:
        downside_returns = net_pnl[net_pnl < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino = (net_pnl.mean() / downside_returns.std()) * np.sqrt(8760)
        else:
            sortino = sharpe if sharpe != 0 else 0
    else:
        sortino = 0
    
    return {
        "total_pnl": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "percent_traded": pct_traded,
        "avg_position_size": avg_position_size,
        "total_trades": int(np.sum(final_trade_mask)),
        "consensus_trades": int(np.sum(consensus_mask & (base_trade_mask == 1)))
    }

def load_and_format_raw_data(filepath):
    print("loading raw data and formatting...")
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
    """
    Expanding window: each fold gets progressively more training data.
    More realistic for production where you accumulate historical data daily.
    """
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
    """main training pipeline with meta-labeling approach.
    
    CURRENCY ASSUMPTION: all price data assumed to be in EUR/MWh as per
    European Market Coupling standards (SDAC uses EUR for settlement).
    """
    config = OmegaConf.load("config.yaml")
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="xgboost_meta_labeling"):
        
        print("loading data...")
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

        print("starting cross-validation pipeline...")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- fold {fold} ---")
            
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

            analyst = xgb.XGBRegressor(**config.model, objective=asymmetric_trading_loss)
            analyst.fit(X_prim_prep, y_prim)
            
            meta_analyst_preds = analyst.predict(X_meta_prep)
            meta_raw_pnl = np.sign(meta_analyst_preds) * y_meta - 0.5
            meta_labels = (meta_raw_pnl > 0).astype(int)
            X_meta_enhanced = X_meta_prep.copy()
            X_meta_enhanced['analyst_pred'] = meta_analyst_preds
            
            meta_params = dict(config.meta_model)
            meta_params.pop('confidence_threshold', None)

            manager = xgb.XGBClassifier(**meta_params)
            manager.fit(X_meta_enhanced, meta_labels)
            
            test_analyst_preds = analyst.predict(X_test_prep)
            
            X_test_enhanced = X_test_prep.copy()
            X_test_enhanced['analyst_pred'] = test_analyst_preds
            
            test_manager_probs = manager.predict_proba(X_test_enhanced)[:, 1]
            
            test_timestamps = test_df.index if hasattr(test_df, 'index') else None
            
            metrics = calculate_enhanced_meta_trading_metrics(
                y_test, 
                test_analyst_preds, 
                test_manager_probs, 
                confidence_threshold=config.meta_model.confidence_threshold,
                cost_per_mwh=config.trading.cost_per_mwh,
                position_size_mwh=config.trading.position_size_mwh,
                use_dynamic_thresholds=True,
                use_confidence_sizing=True,
                timestamps=test_timestamps
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
            mlflow.log_metric(f"fold_{fold}_MaxDD", metrics['max_drawdown'])
            mlflow.log_metric(f"fold_{fold}_HitRate", metrics['hit_rate'])
            mlflow.log_metric(f"fold_{fold}_Sortino", metrics['sortino_ratio'])
            mlflow.log_metric(f"fold_{fold}_AvgPositionSize", metrics['avg_position_size'])
            mlflow.log_metric(f"fold_{fold}_TotalTrades", metrics['total_trades'])
            mlflow.log_metric(f"fold_{fold}_ConsensusRate", metrics['consensus_trades'] / max(1, metrics['total_trades']))
            
            print(f"pnl: {config.trading.currency}{metrics['total_pnl']:.2f} | sharpe: {metrics['sharpe_ratio']:.2f} | sortino: {metrics['sortino_ratio']:.2f}")
            print(f"drawdown: {config.trading.currency}{metrics['max_drawdown']:.2f} | hit rate: {metrics['hit_rate']:.1f}% | traded: {metrics['percent_traded']:.1f}%")
            print(f"trades: {metrics['total_trades']} | avg size: {metrics['avg_position_size']:.2f} | consensus: {metrics['consensus_trades']}/{metrics['total_trades']}")

        avg_pnl = np.mean(fold_pnls)
        avg_sharpe = np.mean(fold_sharpes)
        avg_sortino = np.mean(fold_sortinos)
        avg_hit_rate = np.mean(fold_hit_rates)
        avg_dd = np.mean(fold_dds)
        avg_traded = np.mean(fold_traded)
        avg_position_size = np.mean(fold_position_sizes)
        avg_total_trades = np.mean(fold_total_trades)
        avg_consensus_rate = np.mean([c/max(1,t) for c,t in zip(fold_consensus_trades, fold_total_trades)])
        
        print("\n" + "="*50)
        print("enhanced trading results with profit optimization:")
        print(f"pnl: {config.trading.currency}{avg_pnl:.2f} | sharpe: {avg_sharpe:.2f} | sortino: {avg_sortino:.2f}")
        print(f"drawdown: {config.trading.currency}{avg_dd:.2f} | hit rate: {avg_hit_rate:.1f}% | hours traded: {avg_traded:.1f}%")
        print(f"avg trades per fold: {avg_total_trades:.0f} | avg size: {avg_position_size:.2f} | consensus rate: {avg_consensus_rate:.1%}")
        print("="*50)
        
        mlflow.log_metric("avg_PnL", avg_pnl)
        mlflow.log_metric("avg_Sharpe", avg_sharpe)
        mlflow.log_metric("avg_Sortino", avg_sortino)
        mlflow.log_metric("avg_HitRate", avg_hit_rate)
        mlflow.log_metric("avg_MaxDD", avg_dd)
        mlflow.log_metric("avg_PercentTraded", avg_traded)
        mlflow.log_metric("avg_PositionSize", avg_position_size)
        mlflow.log_metric("avg_TotalTrades", avg_total_trades)
        mlflow.log_metric("avg_ConsensusRate", avg_consensus_rate)

        print("\ntraining complete. check mlflow for detailed results.")

if __name__ == "__main__":
    main()