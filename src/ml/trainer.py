import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from features import TimeSeriesImputer, EnergyFeatureEngineer
from ensemble_models import EnsembleAnalyst, MultiHorizonEnsemble
from src.trading.metrics import asymmetric_trading_loss


class FoldTrainer:
    def __init__(self, config, bool_cols, numeric_cols):
        self.config = config
        self.bool_cols = bool_cols
        self.numeric_cols = numeric_cols
        self.preprocessor = None
        self.analyst = None
        self.manager = None
        self.individual_preds_meta = {}
        self.individual_preds_test = {}

    def create_preprocessor(self):
        self.preprocessor = Pipeline([
            ('imputer', TimeSeriesImputer(bool_cols=self.bool_cols, numeric_cols=self.numeric_cols)),
            ('feature_engineer', EnergyFeatureEngineer())
        ])
        return self.preprocessor

    def split_train_data(self, train_df):
        split_meta_idx = len(train_df) - (self.config.cv.meta_train_days * 24)
        primary_df = train_df.iloc[:split_meta_idx]
        meta_df = train_df.iloc[split_meta_idx:]
        return primary_df, meta_df

    def prepare_fold_data(self, train_df, test_df):
        primary_df, meta_df = self.split_train_data(train_df)
        self.create_preprocessor()

        leakage_cols = self.config.data.leakage_cols

        X_prim = self.preprocessor.fit_transform(primary_df.drop(columns=leakage_cols))
        y_prim = primary_df[self.config.data.target_col]

        X_meta = self.preprocessor.transform(meta_df.drop(columns=leakage_cols))
        y_meta = meta_df[self.config.data.target_col]

        X_test = self.preprocessor.transform(test_df.drop(columns=leakage_cols))
        y_test = test_df[self.config.data.target_col]

        return {
            'primary_df': primary_df,
            'meta_df': meta_df,
            'X_prim': X_prim, 'y_prim': y_prim,
            'X_meta': X_meta, 'y_meta': y_meta,
            'X_test': X_test, 'y_test': y_test,
        }

    def train_analyst(self, X_prim, y_prim, primary_index=None, analyst_config=None):
        cfg = analyst_config or self.config

        if cfg.ensemble.enable and cfg.ensemble.multi_horizon:
            horizons = cfg.ensemble.get('horizons', [1, 4, 12, 24])
            self.analyst = MultiHorizonEnsemble(cfg, horizons=horizons)
            self.analyst.fit(X_prim, y_prim, primary_index)
        elif cfg.ensemble.enable:
            self.analyst = EnsembleAnalyst(cfg)
            self.analyst.fit(X_prim, y_prim)
        else:
            self.analyst = xgb.XGBRegressor(**cfg.model, objective=asymmetric_trading_loss)
            self.analyst.fit(X_prim, y_prim)

        return self.analyst

    def predict_analyst(self, X, timestamps=None):
        if isinstance(self.analyst, MultiHorizonEnsemble):
            return self.analyst.predict(X, timestamps)
        return self.analyst.predict(X)

    def get_individual_predictions(self, X):
        if isinstance(self.analyst, EnsembleAnalyst):
            return self.analyst.get_individual_predictions(X)
        return {}

    @staticmethod
    def create_meta_labels(analyst_preds, y_true, cost_per_mwh=0.5):
        raw_pnl = np.sign(analyst_preds) * y_true - cost_per_mwh
        return (raw_pnl > 0).astype(int)

    @staticmethod
    def enhance_features(X, analyst_preds, individual_preds=None, pred_col='analyst_pred_ensemble'):
        X_enhanced = X.copy()
        X_enhanced[pred_col] = analyst_preds

        if individual_preds:
            for model_name, preds in individual_preds.items():
                X_enhanced[f'analyst_pred_{model_name}'] = preds

            pred_values = list(individual_preds.values())
            X_enhanced['prediction_variance'] = np.var(pred_values, axis=0)
            X_enhanced['prediction_range'] = np.max(pred_values, axis=0) - np.min(pred_values, axis=0)
            X_enhanced['model_consensus'] = np.mean([
                np.sign(pred) for pred in pred_values
            ], axis=0)

        return X_enhanced

    def train_manager(self, X_meta_enhanced, meta_labels, manager_params=None):
        if manager_params is None:
            params = dict(self.config.meta_model)
            params.pop('confidence_threshold', None)
        else:
            params = manager_params

        self.manager = xgb.XGBClassifier(**params)
        self.manager.fit(X_meta_enhanced, meta_labels)
        return self.manager

    def run_fold(self, train_df, test_df, analyst_config=None):
        data = self.prepare_fold_data(train_df, test_df)

        self.train_analyst(
            data['X_prim'], data['y_prim'],
            primary_index=data['primary_df'].index,
            analyst_config=analyst_config,
        )

        meta_preds = self.predict_analyst(data['X_meta'], data['meta_df'].index)
        test_preds = self.predict_analyst(data['X_test'], test_df.index)

        cfg = analyst_config or self.config
        use_individual = cfg.ensemble.enable and not cfg.ensemble.get('multi_horizon', False)

        meta_individual = self.get_individual_predictions(data['X_meta']) if use_individual else {}
        test_individual = self.get_individual_predictions(data['X_test']) if use_individual else {}

        meta_labels = self.create_meta_labels(meta_preds, data['y_meta'])

        X_meta_enhanced = self.enhance_features(data['X_meta'], meta_preds, meta_individual)
        X_test_enhanced = self.enhance_features(data['X_test'], test_preds, test_individual)

        self.train_manager(X_meta_enhanced, meta_labels)
        test_manager_probs = self.manager.predict_proba(X_test_enhanced)[:, 1]

        return {
            'y_test': data['y_test'],
            'test_analyst_preds': test_preds,
            'test_manager_probs': test_manager_probs,
            'test_timestamps': test_df.index,
            'X_test_enhanced': X_test_enhanced,
        }
