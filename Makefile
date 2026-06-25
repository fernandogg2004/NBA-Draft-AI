# Single-command entrypoints. On Windows without `make`, run the underlying commands
# directly (see README) or use `python scripts/reproduce.py`.

.PHONY: install dev test lint typecheck reproduce

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy

reproduce:
	python scripts/reproduce.py

pipeline:
	python scripts/run_pipeline.py

eda:
	python scripts/run_eda.py

evaluation:
	python scripts/run_evaluation.py
