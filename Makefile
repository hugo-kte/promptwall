PY      ?= .venv/bin/python
POLICY  ?= configs/policy.yaml
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help install data evaluate serve demo test lint format docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install PromptWall with dev tools
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

data: ## Download the four public datasets (~20 MB)
	$(PY) -m promptwall download

evaluate: ## Train + evaluate the classifier, ship the model, write reports/
	$(PY) -m promptwall evaluate

serve: ## Run the firewall on http://127.0.0.1:8080/v1 (upstream from $(POLICY))
	$(PY) -m promptwall serve -p $(POLICY)

demo: ## End-to-end PII round trip through a local Ollama model
	$(PY) -m promptwall demo -p $(POLICY)

test: ## Run the test suite with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Lint and check formatting
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format: ## Auto-format the code
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

docker-build: ## Build the container image
	docker build -t promptwall:latest .

clean: ## Remove caches and runtime output (keeps data and models)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov runtime
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
