.PHONY: help sync lock lint format clean build

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

sync: ## Create/update .venv from uv.lock
	@ uv sync

lock: ## Re-resolve uv.lock from pyproject.toml
	@ uv lock

lint: ## Run the ruff linter
	@ uv run --only-group dev ruff check src examples

format: ## Run the ruff formatter
	@ uv run --only-group dev ruff format src examples
	@ uv run --only-group dev ruff check --fix src examples

clean: ## Remove build artifacts
	@ rm -rf dist build
	@ rm -rf .ruff_cache

build: ## Build the sdist and wheel
	@ uv build
