# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=skchange
DOC_DIR=./docs
BUILD_TOOLS=./build_tools
TEST_DIR=testdir

.PHONY: help release install test full_test tests clean dist build doc docs

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

release: ## Make a release
	python $(BUILD_TOOLS)/make_release.py

install: ## Install for the current user using the default python command
	python setup.py build_ext --inplace && python setup.py install --user

test: ## Run unit tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest

full_test: ## Run all tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest --only_changed_modules False

tests: test

clean: ## Clean build dist and egg directories left after install
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./.pytest_cache
	rm -rf ./.ruff_cache
	rm -rf ./htmlcov
	rm -rf ./junit
	rm -rf ./$(PACKAGE).egg-info
	rm -rf coverage.xml
	rm -f MANIFEST
	rm -rf ./wheelhouse/*
	find . -type f -iname "*.so" -delete
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

dist: ## Make Python source distribution
	$(MAKE) build

build:
	python -m build --sdist --wheel

doc: ## Build documentation with Sphinx
	$(MAKE) -C $(DOC_DIR) html

docs: doc
