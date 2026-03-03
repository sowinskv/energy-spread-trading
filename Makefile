.PHONY: train optimize test lint format check clean

train:
	uv run train_pipeline.py

optimize:
	uv run optimize.py

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
