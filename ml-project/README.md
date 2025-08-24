# ML Runner Project

This repository bridges Rust and Python for machine learning workflows:

- Query data from SurrealDB into a Rust `polars::DataFrame`.
- Pass the DataFrame into Python via PyO3 and train a TensorFlow model.
- Export the resulting SavedModel to ONNX and store the bytes back in SurrealDB.

Both Rust and Python use **Polars 0.50.0** to ensure compatibility.

## Setup

```
cd ml-project
make setup
```

## Build

```
make build
```

## Demo

Train a tiny model from an in-memory DataFrame and print the SavedModel path:

```
make train-demo
```

Convert a SavedModel to ONNX bytes:

```
make export-onnx MODEL=path/to/saved_model
```

## Example Flow

1. Start with a `polars::DataFrame` (e.g., queried from SurrealDB).
2. `train_via_pyo3` forwards the DataFrame to `py/trainer.py` which trains and saves a TensorFlow model.
3. `tf_savedmodel_to_onnx_bytes` converts the model to ONNX bytes.
4. `store_onnx_in_surreal` upserts the ONNX model back into SurrealDB.

Requires Python **3.11.9** (see `.python-version`) and a running SurrealDB instance for database operations.

## Models

Custom model trainers live in the `models/` directory. Each trainer implements a
standard `BaseTrainer` interface that returns the saved model location and
evaluation metrics. The initial example, `mls_lstm_trainer.py`, trains the
Multi‑Layer Sequential LSTM from [this paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494622008791)
on stock closing prices and reports R², MAPE, and normalized RMSE for train and
test splits. Run it via `train_mls_lstm.py`.
