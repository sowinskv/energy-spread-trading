# Energy

A spread forecasting and trading pipeline for the Polish energy market.

Predicts the SDAC–IDA1 price spread using a two-stage ensemble approach: a regressor estimates spread magnitude while a classifier predicts direction. Trades are filtered through conviction-based sizing that only acts when confidence is high enough.

## How it works

2 stages:

"1"
An ensemble of XGBoost, Random Forest, Extra Trees and Ridge regression models forecasts the spread magnitude. Their predictions are blended with adaptive, performance-based weights.

"2"
A classification ensemble (XGBoost, Random Forest, Extra Trees, Logistic Regression) predicts the probability that the spread is positive. The final prediction combines both: 

`(2 × P(up) − 1) × |regression prediction|`.

Trades are sized by conviction — the ratio of prediction magnitude to recent prediction volatility. Low-conviction hours (0–5) are skipped entirely. Walk-forward cross-validation with purge gaps and embargo periods prevents lookahead bias. MLflow tracks every experiment.

## Structure

```
src/
├── data/          data loading, validation, CV splits
├── ml/            ensemble models, feature engineering, fold trainer
├── pipelines/     training and hyperparameter optimization entrypoints
├── trading/       conviction-based metrics, position sizing
└── ui/            terminal display formatting
```

## Usage

```bash
make train          # run the training pipeline
make optimize       # run optuna hyperparameter search
make test           # run the test suite
make lint           # lint with ruff
make format         # auto-format with ruff
make check          # lint + test
make clean          # remove __pycache__ and .pytest_cache
```

Everything is configured through `config.yaml`. MLflow UI: `uv run mlflow ui`.

## Current performance

Backtest across 4 expanding walk-forward folds:

| Metric       | Value     |
| ------------ | --------- |
| PnL          | EUR 1,770 |
| Sharpe       | 20.35     |
| Sortino      | 28.58     |
| Hit Rate     | 72.1%     |
| Max Drawdown | EUR −164  |
| Hours Traded | 27.8%     |
