VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3

setup:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

build:
	PYTHON_SYS_EXECUTABLE=$(PYTHON) cargo build --manifest-path ml-runner/Cargo.toml
