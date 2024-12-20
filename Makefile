install: install-core

install-core:
	poetry install

install-dev:
	poetry install --with dev

install-test:
	poetry install --with test

check:
<<<<<<< HEAD
	isort --check-only src/tatm tests scripts
	black --check src/tatm tests scripts
	flake8 src/tatm tests scripts
=======
	isort --check-only src/tatm tests
	black --check src/tatm tests
	flake8 src/tatm tests
>>>>>>> main
	pytest --cov=tatm  --cov-report term-missing --cov-fail-under=85 tests

test:
	pytest --cov=tatm  --cov-report term-missing --cov-fail-under=85 tests

lint:
<<<<<<< HEAD
	isort src/tatm tests scripts
	black src/tatm tests scripts
	flake8 src/tatm tests scripts
=======
	isort src/tatm tests
	black src/tatm tests
	flake8 src/tatm tests
>>>>>>> main

build-docs:
	cd docs && make html

check-doc-links:
	cd docs && make linkcheck