.PHONY: lint format test check install-hooks

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

check: lint
	ruff format --check .
	pytest -q

install-hooks:
	pip install -r requirements-dev.txt
	pre-commit install
