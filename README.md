# Energy

spread forecasting and trading system for the Polish day-ahead electricity market.

predicts the SDAC–IDA1 price spread. trades only when conviction is high.

---

## 00 — Contents

```
00    contents
01    architecture
02    structure
03    features
04    usage
05    configuration
06    validation
```

---

## 01 — Architecture

2 stages:

**01.1 — regression**
ensemble of XGBoost, Random Forest, Extra Trees, and Ridge.
estimates spread magnitude. models are weighted by recent performance.

**01.2 — classification**
ensemble of XGBoost, Random Forest, Extra Trees, and Logistic Regression.
predicts probability that the spread is positive.

**01.3 — synthesis**
final prediction combines both stages:

```
prediction = (2 × P(up) − 1) × |regression estimate|
```

sign comes from the classifier. magnitude from the regressor.
conviction determines position size. low-conviction hours are skipped.

---

## 02 — Structure

```
src/
    data/           loading, validation, walk-forward CV splits
    ml/             ensembles, feature engineering, fold trainer
    pipelines/      training and optimization entrypoints
    trading/        conviction metrics, position sizing
    ui/             terminal display

experiments/        ablation, diagnostics, frontiers
tests/              unit and integration tests
config.yaml         single source of truth
```

---

## 03 — Features

all SDAC-derived features are shifted 24h.
at decision time, current-day prices are unknown — we only see yesterday.

**03.1 — market state** (lagged)
cross-border flows, system imbalance, activation capacity,
inter-market spreads (PL–DE, PL–SK), net position momentum.

**03.2 — price dynamics** (lagged)
SDAC lags (24h, 48h, 168h), rolling mean/std, EWMA, Bollinger width,
24h return, spread momentum, mean-reversion z-score.

**03.3 — fundamentals** (forward-looking, available pre-delivery)
demand forecast, PV/wind generation forecasts, residual load,
renewable share, gradients.

**03.4 — calendar**
hour, day-of-week, month (sin/cos encoded), weekend flag, peak hours.

---

## 04 — Usage

```
make train              run backtest
make optimize           optuna hyperparameter search (continues existing study)
make fresh-optimize     delete old study, start from scratch
make test               run tests
make check              lint + test
make clean              remove caches
```

MLflow UI:

```
uv run mlflow ui
```

---

## 05 — Configuration

everything lives in `config.yaml`.

model hyperparameters, ensemble composition, CV strategy,
trading parameters (conviction threshold, position limits, skip hours),
and leakage column exclusions.

---

## 06 — Validation

backtest across 4 expanding walk-forward folds. no lookahead, no overfitting.

```
pnl                   EUR    1,673
hit rate                      72.4%
sharpe                        18.98
sortino                       29.35
max drawdown          EUR      −96
hours traded                  17.9%
sharpe 95% ci         [ 17.05, 21.00 ]
sharpe p-value                0.000
```

**06.1 — why this is credible**

no current-day data.
every SDAC column is shifted 24h. the model only sees yesterday's
cleared prices and today's published forecasts.

no overfitting.
train R² is 0.07–0.09, test R² is ~0. the model doesn't memorize —
it barely fits the training set. the edge is directional, not magnitude.

temporal isolation.
expanding walk-forward CV with a 7-day purge gap before each test fold
and a 2-day embargo between folds. no information crosses the boundary.

statistical significance.
bootstrap Sharpe p-value = 0.000. the 95% confidence interval lower
bound (17.05) is still excellent. all 4 folds are independently profitable.

selectivity.
conviction threshold of 1.2 means the model trades only 17.9% of hours.
it stays out when uncertain.

---
