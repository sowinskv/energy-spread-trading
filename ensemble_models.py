import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import mlflow


class EnsembleAnalyst:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = None
        self.performance_history = []
        self.scaler = StandardScaler()
        self._init_models()
    
    def _init_models(self):
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=self.config.model_params.xgboost.n_estimators,
            max_depth=self.config.model_params.xgboost.max_depth,
            learning_rate=self.config.model_params.xgboost.learning_rate,
            subsample=self.config.model_params.xgboost.subsample,
            colsample_bytree=self.config.model_params.xgboost.colsample_bytree,
            random_state=42,
            objective='reg:squarederror'
        )
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=self.config.model_params.lightgbm.n_estimators,
            max_depth=self.config.model_params.lightgbm.max_depth,
            learning_rate=self.config.model_params.lightgbm.learning_rate,
            num_leaves=self.config.model_params.lightgbm.num_leaves,
            subsample=self.config.model_params.lightgbm.subsample,
            colsample_bytree=self.config.model_params.lightgbm.colsample_bytree,
            min_child_samples=self.config.model_params.lightgbm.min_child_samples,
            min_split_gain=self.config.model_params.lightgbm.min_split_gain,
            random_state=43,
            objective='regression',
            boosting_type='gbdt',
            verbosity=-1
        )
        
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=self.config.model_params.random_forest.n_estimators,
            max_depth=self.config.model_params.random_forest.max_depth,
            min_samples_split=self.config.model_params.random_forest.min_samples_split,
            min_samples_leaf=self.config.model_params.random_forest.min_samples_leaf,
            max_features=self.config.model_params.random_forest.max_features,
            random_state=44,
            n_jobs=-1
        )
        
        self.models['ridge'] = Ridge(
            alpha=self.config.model_params.ridge.alpha,
            random_state=45
        )
        
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=self.config.model_params.extra_trees.n_estimators,
            max_depth=self.config.model_params.extra_trees.max_depth,
            min_samples_split=self.config.model_params.extra_trees.min_samples_split,
            min_samples_leaf=self.config.model_params.extra_trees.min_samples_leaf,
            max_features=self.config.model_params.extra_trees.max_features,
            random_state=47,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        print(f"training ensemble with {len(self.models)} models")
        
        X_scaled = self.scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        individual_predictions = {}
        
        for name, model in self.models.items():
            print(f"training {name}")
            
            try:
                if name in ['ridge']:
                    model.fit(X_scaled, y)
                    individual_predictions[name] = model.predict(X_scaled)
                else:
                    model.fit(X, y)
                    individual_predictions[name] = model.predict(X)
                    
                print(f"✓ {name} trained")
                
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                del self.models[name]
        
        self.weights = self._calculate_performance_weights(X, y, individual_predictions)
        
        try:
            mlflow.log_params({
                f"ensemble_models_{len(self.models)}": list(self.models.keys()),
                f"ensemble_weights_{hash(str(self.weights))}": self.weights
            })
        except Exception as e:
            print(f"mlflow warning: {e}")
        
        return self
    
    def _calculate_performance_weights(self, X, y, predictions):
        if len(predictions) < 2:
            return {name: 1.0 for name in self.models.keys()}
        
        recent_size = max(100, int(len(y) * 0.3))
        y_recent = y.iloc[-recent_size:]
        
        errors = {}
        for name, pred in predictions.items():
            if len(pred) >= recent_size:
                recent_pred = pred[-recent_size:]
                mse = np.mean((y_recent - recent_pred) ** 2)
                errors[name] = 1.0 / (1.0 + mse)
        
        total_weight = sum(errors.values())
        weights = {name: weight / total_weight for name, weight in errors.items()}
        
        min_weight = 0.1 / len(weights)
        for name in weights:
            weights[name] = max(weights[name], min_weight)
        
        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def predict(self, X):
        if not self.models:
            raise ValueError("no trained models available")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            try:
                if name in ['ridge']:
                    predictions[name] = model.predict(X_scaled)
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                print(f"warning: {name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("no models could generate predictions")
        
        if self.weights is None:
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        else:
            weights = {name: self.weights.get(name, 0) for name in predictions.keys()}
        
        final_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            final_pred += weights[name] * pred
        
        return final_pred
    
    def get_individual_predictions(self, X):
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            try:
                if name in ['ridge']:
                    predictions[name] = model.predict(X_scaled)
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                print(f"warning: {name} prediction failed: {e}")
                predictions[name] = np.zeros(len(X))
        
        return predictions
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class MultiHorizonEnsemble:
    def __init__(self, config, horizons=[1, 4, 12, 24]):
        self.config = config
        self.horizons = horizons
        self.horizon_ensembles = {}
        self.horizon_weights = None
    
    def fit(self, X, y, timestamps):
        print(f"training multi-horizon ensemble for horizons: {self.horizons}")
        
        for horizon in self.horizons:
            print(f"training horizon {horizon}h")
            
            X_horizon = self._create_horizon_features(X, horizon)
            
            ensemble = EnsembleAnalyst(self.config)
            ensemble.fit(X_horizon, y)
            
            self.horizon_ensembles[horizon] = ensemble
        
        self.horizon_weights = self._calculate_horizon_weights(X, y, timestamps)
        
        return self
    
    def _create_horizon_features(self, X, horizon):
        X_horizon = X.copy()
        
        rolling_cols = [col for col in X.columns if 'rolling' in col or 'lag' in col]
        
        for col in rolling_cols:
            if horizon > 1:
                X_horizon[f"{col}_h{horizon}"] = X[col].shift(horizon)
        
        return X_horizon.ffill().fillna(0)
    
    def _calculate_horizon_weights(self, X, y, timestamps):
        returns = y.diff().abs()
        recent_vol = returns.rolling(24).std().iloc[-1] if len(returns) > 24 else returns.std()
        
        if recent_vol > returns.quantile(0.7):
            weights = {1: 0.4, 4: 0.3, 12: 0.2, 24: 0.1}
        elif recent_vol < returns.quantile(0.3):
            weights = {1: 0.1, 4: 0.2, 12: 0.3, 24: 0.4}
        else:
            weights = {1: 0.25, 4: 0.25, 12: 0.25, 24: 0.25}
        
        return {h: weights.get(h, 0.25) for h in self.horizons}
    
    def predict(self, X, timestamps):
        horizon_predictions = {}
        
        for horizon, ensemble in self.horizon_ensembles.items():
            X_horizon = self._create_horizon_features(X, horizon)
            horizon_predictions[horizon] = ensemble.predict(X_horizon)
        
        if self.horizon_weights is None:
            weights = {h: 1.0 / len(self.horizons) for h in self.horizons}
        else:
            weights = self.horizon_weights
        
        final_pred = np.zeros(len(X))
        for horizon, pred in horizon_predictions.items():
            final_pred += weights[horizon] * pred
        
        return final_pred