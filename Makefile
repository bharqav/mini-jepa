.PHONY: help install install-gpu test test-real lint format format-check check security docs clean bench smoke jepa serve

PY ?= python
VENV := .venv
BIN := $(VENV)/Scripts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python ## Install with dev + serve extras (CPU torch)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

install-gpu: $(BIN)/python ## Install with CUDA-enabled torch
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
	$(BIN)/python -m pip install -e ".[dev]"

test: ## Run weight-free test suite
	$(BIN)/python -m pytest tests/ -q

test-real: ## Run full suite including real Qwen weights (downloads ~1.2GB)
	JEV_REAL_WEIGHTS=1 $(BIN)/python -m pytest tests/ -q

lint: ## Lint with ruff
	$(BIN)/python -m ruff check src/ tests/ scripts/

format: ## Auto-format and fix
	$(BIN)/python -m ruff format src/ tests/ scripts/
	$(BIN)/python -m ruff check --fix src/ tests/ scripts/

format-check: ## Verify formatting without changing files
	$(BIN)/python -m ruff format --check src/ tests/ scripts/

security: ## Scan for committed secrets
	$(BIN)/python scripts/security_scan.py

docs: ## Validate markdown links and mermaid blocks
	$(BIN)/python scripts/docs_check.py

check: format-check lint test security docs ## All offline quality gates

smoke: ## Tiny end-to-end training smoke (CPU, minutes)
	$(BIN)/python scripts/train.py configs/smoke.yaml --name smoke

jepa: ## Run the JEPA latent-prediction smoke experiment
	$(BIN)/python scripts/jepa_smoke.py

bench: ## Benchmark a checkpoint (pass CKPT=path)
	$(BIN)/python scripts/benchmark.py --checkpoint $(CKPT) --batch-sizes 1 4 8

serve: ## Serve the API locally (pass CKPT=path)
	$(BIN)/python -m uvicorn jev_qwen.api:app --host 127.0.0.1 --port 8000

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +