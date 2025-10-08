VENV ?= ml-project/.venv
PYTHON ?= $(VENV)/bin/python3

.PHONY: setup build train-demo export-onnx
.PHONY: export-onnx-keras

# Create the repo Python venv and install deps
setup:
	$(MAKE) -C ml-project setup

# Build Rust with the repo venv's Python for PyO3
build:
	PYTHON_SYS_EXECUTABLE=$(PYTHON) cargo build --manifest-path ml-runner/Cargo.toml

# Train a tiny demo model and print the SavedModel path
train-demo:
	$(PYTHON) ml-project/py/train_demo.py

# Convert a SavedModel to ONNX and print byte length
# Usage: make export-onnx MODEL=path/to/saved_model
export-onnx:
	@if [ -z "$(MODEL)" ]; then echo "MODEL is required (path to SavedModel)"; exit 1; fi
	$(PYTHON) -c "import os,sys; sys.path.append('ml-project/py'); import onnx_export as m; print(len(m.savedmodel_to_onnx_bytes(os.environ['MODEL'])))" 

# Convert a Keras .keras file to ONNX and print byte length
# Usage: make export-onnx-keras KERAS=path/to/final_model.keras
export-onnx-keras:
	@if [ -z "$(KERAS)" ]; then echo "KERAS is required (path to .keras file)"; exit 1; fi
	$(PYTHON) -c "import os,sys; sys.path.append('ml-project/py'); import onnx_export as m; print(len(m.keras_to_onnx_bytes(os.environ['KERAS'])))" 
