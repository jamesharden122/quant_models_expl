VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

setup:
python -m venv $(VENV)
$(PYTHON) -m pip install --upgrade pip
$(PYTHON) -m pip install -r requirements.txt

build:
PYTHON_SYS_EXECUTABLE=$(PYTHON) cargo build --manifest-path ml-runner/Cargo.toml

train-demo:
PYTHONPATH=py $(PYTHON) - <<'PY'
import json, polars as pl, trainer
cfg = json.dumps({"target": "y", "epochs": 1, "batch": 2, "outdir": "models"})
df = pl.DataFrame({"x": [1,2,3,4], "y": [2,4,6,8]})
print(trainer.train_from_polars(df, cfg))
PY

export-onnx:
@echo "Specify MODEL=<saved_model_dir>"
PYTHONPATH=py $(PYTHON) - <<'PY'
import os, onnx_export
model_dir = os.environ.get("MODEL")
if not model_dir:
    raise SystemExit("MODEL env var required")
print(len(onnx_export.savedmodel_to_onnx_bytes(model_dir)))
PY
