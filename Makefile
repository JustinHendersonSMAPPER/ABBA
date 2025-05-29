.PHONY: help install test lint format clean build docs

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install all dependencies
	poetry install --with dev,test

test:  ## Run all tests
	poetry run pytest

test-coverage:  ## Run tests with coverage report
	poetry run pytest --cov=abba --cov-report=html --cov-report=term

lint:  ## Run all linters
	poetry run flake8 src tests
	poetry run pylint src tests
	poetry run mypy src tests
	poetry run bandit -r src
	poetry run vulture src

format:  ## Format code with black and isort
	poetry run black src tests
	poetry run isort src tests

check:  ## Run all checks (lint + test)
	make lint
	make test

pre-commit:  ## Run pre-commit hooks
	poetry run pre-commit run --all-files

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	poetry build

docs:  ## Build documentation (future)
	@echo "Documentation building not yet implemented"