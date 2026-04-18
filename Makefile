# Credit Risk Intelligence Platform
# Makefile for common development tasks

.PHONY: help install dev test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Credit Risk Intelligence Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Start development environment"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start all services"
	@echo "  make docker-down  - Stop all services"
	@echo "  make train        - Run model training"
	@echo "  make serve        - Start API server"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov pytest-asyncio black isort flake8 mypy

# Development
dev:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Development environment started"
	@echo "  API: http://localhost:8000"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Jupyter: http://localhost:8888"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

test-integration:
	pytest tests/integration/ -v --tb=short

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Cleaning
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf venv .venv
	rm -rf *.egg-info dist build

# Docker
docker-build:
	docker build -t credit-risk-intelligence:latest -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# ML Pipeline
train:
	python -m src.pipelines.training_pipeline

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Database setup
setup-snowflake:
	snowsql -f scripts/setup_snowflake.sql

# Notebook
notebook:
	jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --port=8888
