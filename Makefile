.PHONY: help install install-dev sample pipeline api dashboard test lint format docker-build docker-up clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install runtime + dev dependencies
	pip install -r requirements-dev.txt

sample: ## Regenerate the demo dataset
	python scripts/generate_sample_data.py

pipeline: ## Run the end-to-end pipeline on the sample data
	python scripts/run_pipeline.py

api: ## Serve the REST API on :8000
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Serve the Streamlit dashboard on :8501
	streamlit run app/streamlit_app.py

test: ## Run the test suite
	pytest

lint: ## Static checks
	ruff check src app tests scripts

format: ## Auto-format
	black src app tests scripts && ruff check --fix src app tests scripts

docker-build: ## Build the container image
	docker build -t retail-sales-forecasting .

docker-up: ## Start API + dashboard with compose
	docker compose up --build

clean: ## Remove caches and generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache logs/*.log
