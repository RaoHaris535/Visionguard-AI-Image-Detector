.DEFAULT_GOAL := help

.PHONY: help install audit split train evaluate tune test lint format web api compose-up compose-down evidence

help:
	@echo "VisionGuard development commands"
	@echo "  make install       Install the project and development tools"
	@echo "  make audit         Audit data/raw and generate EDA evidence"
	@echo "  make split         Create deterministic train/validation/test manifests"
	@echo "  make train         Train baseline and transfer-learning models"
	@echo "  make evaluate      Evaluate saved predictions once on the test set"
	@echo "  make tune          Run at least eight Keras Tuner trials"
	@echo "  make test          Run the automated tests"
	@echo "  make lint          Run Ruff checks"
	@echo "  make web           Start the local Flask UI"
	@echo "  make api           Start the inference API"
	@echo "  make compose-up    Start Flask and PostgreSQL"

install:
	uv sync --extra dev

audit:
	uv run python scripts/audit_dataset.py --data-dir data/raw --output-dir data/audit

split:
	uv run python scripts/create_splits.py --audit-csv data/audit/images.csv --output-dir data/splits

train:
	uv run python scripts/train_models.py --config configs/training.yaml

evaluate:
	uv run python scripts/evaluate_models.py --config configs/training.yaml

tune:
	uv run python scripts/tune_transfer.py --config configs/training.yaml --max-trials 8

test:
	uv run pytest --cov=visionguard --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

web:
	uv run visionguard-web

api:
	uv run visionguard-api

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

evidence:
	uv run python scripts/evidence_manifest.py
