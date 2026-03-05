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
```

---

## 01 — Architecture

two stages.

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
