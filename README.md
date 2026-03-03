# The 1st energy trading pipeline I ever worked on.

> Multi-horizon ensemble learning system for energy price forecasting and algorithmic trading.

## Features

- **Multi-Horizon Ensemble Models** - XGBoost, Random Forest, Extra Trees, Ridge regression
- **Adaptive Model Weighting** - Performance-based dynamic weight allocation
- **Cross-Validation Pipeline** - Time-series aware backtesting framework
- **MLflow Integration** - Experiment tracking and model versioning
- **Exit Rule Management** - Confidence-based position management (not active, current version svcks)

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Training Pipeline

```bash
uv run train_pipeline.py
```

### Configuration

Edit `config.yaml` for model parameters and trading settings.

### MLflow UI

```bash
uv run mlflow ui
```
