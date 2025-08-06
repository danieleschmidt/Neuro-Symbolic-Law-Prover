.PHONY: install test lint typecheck format clean docs build publish

# Installation
install:
	pip install -e ".[dev]"

install-minimal:
	pip install -e "."

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=neuro_symbolic_law --cov-report=term-missing --cov-report=html

test-minimal:
	python test_minimal.py

# Code quality
lint:
	flake8 src/
	flake8 tests/

typecheck:
	mypy src/neuro_symbolic_law

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Security
security:
	bandit -r src/
	safety check

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

docs-clean:
	rm -rf docs/_build/

# Build and publish
build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow
dev-setup:
	python -m venv venv
	source venv/bin/activate && pip install -e ".[dev]"

ci-test: lint typecheck test-minimal security
	@echo "All CI checks passed!"

# Example runs
example:
	python examples/basic_usage.py

cli-help:
	python -m neuro_symbolic_law.cli --help

# Performance testing
perf-test:
	python -m pytest tests/ -v -k "not slow" --benchmark-only

# All tests
test-all: test-minimal test test-cov lint typecheck security
	@echo "All tests completed successfully!"