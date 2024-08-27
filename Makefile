MAKEFLAGS += --warn-undefined-variables
.SHELLFLAGS := -eu -o pipefail -c

all: help
.PHONY: all help setup_env ci black lint pre-commit-full test clean

# Use bash for inline if-statements
SHELL:=bash
REQUIREMENTS = requirements.txt

VENV_NAME ?= python_env

PYTHON = $(VENV_NAME)/bin/python
PYTEST = $(VENV_NAME)/bin/pytest
FLAKE8 = $(VENV_NAME)/bin/flake8
BLACK = $(VENV_NAME)/bin/black
PYCLEAN = $(VENV_NAME)/bin/pyclean
PRECOMMIT = $(VENV_NAME)/bin/pre-commit

##@ Helpers
help: ## display this help
	@echo "Trading environment"
	@echo "======================="
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m\033[0m"} /^[a-zA-Z0-9\(\)\$$_%\.\-\\]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

	@printf "\n"

##@ Preparation
setup_env: $(PYTHON) $(PYTEST) $(FLAKE8) $(BLACK) $(PYCLEAN) $(PRECOMMIT)  ## install local dev environment

$(PYTHON) $(PYTEST) $(FLAKE8) $(BLACK) $(PYCLEAN) $(PRECOMMIT): $(REQUIREMENTS)
	python3 -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r $<

verify_install: setup_env ## check environment installation/command version
	@echo "Testing python env..."
	@echo -e "\n***************************\npython version:"&&\
	$(PYTHON) --version &&\
	echo -e "\n***************************\npytest version:"&&\
	$(PYTEST) --version &&\
	echo -e "\n***************************\nflake8 version:"&&\
	$(FLAKE8) --version &&\
	echo -e "\n***************************\nblack version:"&&\
	$(BLACK) --version &&\
	echo -e "\n***************************\npyclean version:"&&\
	$(PYCLEAN) --version &&\
	echo -e "\n***************************\npre-commit version:"&&\
	$(PRECOMMIT) --version

##@ CI Functions
ci: pre-commit-full test ## run whole CI part

black: ## format your code using black
	$(BLACK) --version
	$(BLACK) --check .

lint: ## run flake8 linter
	$(FLAKE8) --version
	$(FLAKE8)

pre-commit-full: ## run pre-commit hooks
	$(PRECOMMIT) --version
	$(PRECOMMIT) run --all-files

test: logs/test_results.log ## run all unit tests

logs/test_results.log: src/*.py pytest.ini ## run all tests and log them
	$(PYTEST)

##@ Tear-down
clean: ## clean up temp files
	rm -rf .pytest_cache
	rm logs/*.log
	$(PYCLEAN) .
