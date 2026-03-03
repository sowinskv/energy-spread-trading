.PHONY: train optimize test lint format check clean

train:
	uv run python -m src.pipelines.train

optimize:
	uv run python -m src.pipelines.optimize

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
