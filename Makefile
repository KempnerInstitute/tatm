install: install-core

install-core:
	poetry install

install-dev:
	poetry install --with dev

install-test:
	poetry install --with test

check:
	isort --check-only src/tatm tests scripts
	black --check src/tatm tests scripts
	flake8 src/tatm tests scripts
	pytest --cov=tatm  --cov-report term-missing --cov-fail-under=85 tests

test:
	pytest --cov=tatm  --cov-report term-missing --cov-fail-under=85 tests

lint:
	isort src/tatm tests scripts
	black src/tatm tests scripts
	flake8 src/tatm tests scripts

build-docs:
	cd docs && make html

check-doc-links:
	cd docs && make linkcheck