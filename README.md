# Energy

A spread forecasting and trading pipeline for the Polish energy market. Predicts the SDAC–IDA1 price spread using a multi-horizon ensemble of tree-based models, then filters trades through a meta-labeling layer that decides *whether* to act on each prediction.

## How it works

The system has two stages. First, an ensemble of XGBoost, Random Forest, Extra Trees and Ridge regression models forecasts the spread across multiple time horizons (1h, 4h, 12h, 24h). Their predictions are blended with adaptive, performance-based weights. Second, a meta-model (XGBoost classifier) learns which predictions are actually worth trading — a concept borrowed from de Prado's meta-labeling. Only trades where the meta-model is confident enough get executed.

Walk-forward cross-validation with purge gaps prevents lookahead bias. MLflow tracks every experiment.

## Structure

```
src/
├── data/          data loading, validation, CV splits
├── ml/            ensemble models, feature engineering, fold trainer
├── pipelines/     training and hyperparameter optimization entrypoints
└── trading/       metrics, exit strategies, position management
```

## Usage

```bash
make train          # run the training pipeline
make optimize       # run optuna hyperparameter search
make test           # run the test suite (30 tests)
make check          # lint + test
make format         # auto-format with ruff
```

Everything is configured through `config.yaml`. MLflow UI: `uv run mlflow ui`.

## Current performance

Backtest across 3 expanding walk-forward folds:

| Metric | Value |
|---|---|
| PnL | EUR 1783.24 |
| Sharpe | 14.00 |
| Sortino | 38.54 |
| Hit Rate | 71.5% |
| Hours Traded | 13.8% |

Exit rules exist but are disabled — the current version needs work.

## What's next

Feature selection, more data sources, making exit rules not terrible.
