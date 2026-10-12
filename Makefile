# Local, untracked settings (e.g. the publish index). See .env.example.
# Only the variables defined in .env are exported to recipes.
-include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env 2>/dev/null)

.PHONY: help sync lock lint format clean build publish

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

publish: ## Publish a tagged release to the index set by UV_PUBLISH_URL (see .env.example)
	@ test -n "$(UV_PUBLISH_URL)" || { echo "UV_PUBLISH_URL is not set; copy .env.example to .env and fill it in"; exit 1; }
	@ git describe --exact-match --tags HEAD >/dev/null 2>&1 || { echo "HEAD is not tagged; tag a release (e.g. v0.1.0) before publishing"; exit 1; }
	@ $(MAKE) clean build
	@ uv publish
