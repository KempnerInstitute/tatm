install: install-core

install-core:
	poetry install

install-dev:
	poetry install --with dev

install-test:
	poetry install --with test

test:
	pytest --cov=tatm  --cov-report term-missing --cov-fail-under=85 tests

lint:
	black src/tatm tests
	flake8 src/tatm tests

docs:
	cd docs && make html