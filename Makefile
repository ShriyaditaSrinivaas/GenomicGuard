.PHONY: install test lint format data train audit dashboard clean all

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Generate synthetic data
data:
	python scripts/generate_data.py

# Train all models
train:
	python scripts/train_models.py

# Run fairness audit
audit:
	python scripts/run_fairness_audit.py

# Generate clinical reports
reports:
	python scripts/generate_reports.py

# Run full pipeline: data → train → audit
pipeline: data train audit

# Run tests
test:
	pytest tests/ -v --cov=genomicguard --cov-report=term-missing

# Run linting
lint:
	flake8 genomicguard/ tests/ --max-line-length=100 --ignore=E402,W503,E203
	isort --check-only genomicguard/ tests/

# Format code
format:
	black genomicguard/ tests/ scripts/ dashboard/ --line-length=100
	isort genomicguard/ tests/ scripts/ dashboard/

# Launch dashboard
dashboard:
	streamlit run dashboard/app.py

# Clean generated files
clean:
	rm -rf data/synthetic/*.csv
	rm -rf models/*.joblib
	rm -rf reports/
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Full setup: install → pipeline
all: install pipeline
