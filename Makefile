.PHONY: help install install-gpu test test-cov test-real lint format format-check type-check check security audit docs load-test keys docker-build clean bench smoke jepa serve serve-demo demo quickstart

PY ?= python
VENV := .venv
BIN := $(VENV)/Scripts
URL ?= http://127.0.0.1:8000/v1/decide
TOKEN ?=
CONCURRENCY ?= 8
DURATION ?= 30
KEY_ID ?= local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python ## Install with dev + serve extras (CPU torch, pinned)
	$(BIN)/python -m pip install --upgrade pip setuptools
	$(BIN)/python -m pip install -r requirements/torch-cpu.txt
	$(BIN)/python -m pip install -c requirements/constraints.txt -e ".[dev]"

install-gpu: $(BIN)/python ## Install with CUDA-enabled torch (cu128, pinned)
	$(BIN)/python -m pip install --upgrade pip setuptools
	$(BIN)/python -m pip install -r requirements/torch-cuda.txt
	$(BIN)/python -m pip install -c requirements/constraints.txt -e ".[dev]"

test: ## Run weight-free test suite
	$(BIN)/python -m pytest tests/ -q

test-cov: ## Run weight-free tests with the coverage gate
	$(BIN)/python -m pytest tests/ -q --cov=jev_qwen --cov-branch --cov-report=term-missing

test-real: ## Run full suite including real Qwen weights (downloads ~1.2GB)
	JEV_REAL_WEIGHTS=1 $(BIN)/python -m pytest tests/ -q

lint: ## Lint with ruff
	$(BIN)/python -m ruff check src/ tests/ scripts/

format: ## Auto-format and fix
	$(BIN)/python -m ruff format src/ tests/ scripts/
	$(BIN)/python -m ruff check --fix src/ tests/ scripts/

format-check: ## Verify formatting without changing files
	$(BIN)/python -m ruff format --check src/ tests/ scripts/

type-check: ## Type-check the production surface with mypy
	$(BIN)/python -m mypy

security: ## Scan for committed secrets
	$(BIN)/python scripts/security_scan.py

audit: ## Scan installed dependencies for known vulnerabilities
	$(BIN)/python -m pip install --upgrade pip-audit setuptools --quiet
	$(BIN)/python -m pip_audit --skip-editable

docs: ## Validate markdown links and mermaid blocks
	$(BIN)/python scripts/docs_check.py

load-test: ## Load/soak a running API (URL=... TOKEN=... CONCURRENCY=... DURATION=...)
	$(BIN)/python scripts/load_test.py --url $(URL) --token "$(TOKEN)" --concurrency $(CONCURRENCY) --duration $(DURATION)

keys: ## Generate an API key JSON document (KEY_ID=...)
	$(BIN)/python scripts/generate_api_keys.py --key-id $(KEY_ID)

docker-build: ## Build the CPU container image
	docker build -f deploy/Dockerfile -t jev-qwen:cpu .

check: format-check lint type-check test-cov security docs ## All offline quality gates

smoke: ## Tiny end-to-end training smoke (CPU, minutes)
	$(BIN)/python scripts/train.py configs/smoke.yaml --name smoke

jepa: ## Run the JEPA latent-prediction smoke experiment
	$(BIN)/python scripts/jepa_smoke.py

bench: ## Benchmark a checkpoint (pass CKPT=path)
	$(BIN)/python scripts/benchmark.py --checkpoint $(CKPT) --batch-sizes 1 4 8

serve: ## Serve the API locally (pass CKPT=path, defaults to dev mode)
	$(BIN)/python -m jev_qwen.api --dev

serve-demo: ## Train a tiny checkpoint and serve it with dev mode
	$(MAKE) smoke
	$(BIN)/python -m jev_qwen.api --dev

demo: ## End-to-end demo: train tiny model + run a decision + open docs
	$(MAKE) smoke
	@echo "=== Demo decision ==="
	$(BIN)/python scripts/decide.py --checkpoint runs/smoke*/checkpoints/best \
	  --state "A server is returning repeated HTTP 503 errors." \
	  --type choice --question "What should the operator do first?" \
	  --candidate "inspect:Inspect service health and dependencies." \
	  --candidate "delete:Delete the production database." \
	  --candidate "ignore:Ignore the incident."
	@echo "=== API docs will open on first request to /docs ==="
	$(BIN)/python -m jev_qwen.api --dev

quickstart: install demo ## One command: install + demo

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
