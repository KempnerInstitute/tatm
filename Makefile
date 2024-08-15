install: install-core

install-core:
	poetry install

install-dev:
	poetry install --with dev

install-test:
	poetry install --with test

test:
	pytest --cov=tatm tests

build-docs:
	cd docs && make html