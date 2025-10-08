# Quant Models Exploration

End-to-end ML playground combining a Rust runner (Dioxus UI + server) with
Python training utilities. Rust calls into Python via PyO3; data interchange is
through Polars and TFRecord. Utilities are included for training a tiny demo
model and exporting to ONNX.

## Repository Structure

- `ml-project/`: Python ML code and tooling
  - `py/`: TFRecord read/write helpers, ONNX export
  - `tests/`: Python-side smoke/integration tests
  - `Makefile`, `requirements.txt`: venv + deps management
- `ml-runner/`: Rust app (Dioxus UI + feature‑gated server)
  - `src/pyexec/*`: PyO3 bridge to Python
  - `src/bin/server.rs`: server‑only CLI entry
  - `assets/`: web assets for the UI
  - `Cargo.toml`: features `web`, `server`
- `tmp_data/`: Local TFRecord workspace (ignored from VCS)

## Prerequisites

- Python 3.11.9
- Rust (stable toolchain)
- Optional for UI: Dioxus CLI (`dx`). Install with `cargo install dioxus-cli` if missing.

## Quickstart

1) Set up the repo Python environment and install dependencies

```
make setup
```

2) Build the Rust runner using the repo venv (sets `PYTHON_SYS_EXECUTABLE` for PyO3)

```
make build
```

3) Run smoke demos from the repo root

- Train a tiny demo model and print the SavedModel path:

```
make train-demo
```

- Convert a SavedModel to ONNX bytes and print the byte length:

```
make export-onnx MODEL=path/to/saved_model
```

- Convert a Keras `.keras` file to ONNX bytes:

```
make export-onnx-keras KERAS=path/to/final_model.keras
```

## Running the App

You can run the full-stack UI or a server-only CLI.

- Full-stack UI (web or desktop):

```
cd ml-runner
dx serve                 # web by default
dx serve --platform desktop
```

- Server-only CLI:

```
cargo run --features server --bin server stream   # run streaming pipeline
cargo run --features server --bin server train    # run streaming + training
```

If calling the server from another machine, set `SERVER_URL` (default
`http://127.0.0.1:8080`).

## Testing

- Python:
  - Fast: `pytest -q`
  - Coverage (target ≥70%):

```
pytest --cov=models --cov=py --cov-report=term-missing -q
```

  - Mark long runs `@pytest.mark.slow` and exclude by default with:

```
pytest -m "not slow"
```

- Rust:

```
cargo test
cargo test --features server   # server code and PyO3-backed paths
```

PyO3 bridge tests use tiny DataFrame fixtures and write TFRecords under
`tmp_data/`. Tests clean up temporary files where applicable.

## Notes & Conventions

- Use the repo venv for builds so PyO3 points at the correct Python:
  run `make build` from the repo root (it sets `PYTHON_SYS_EXECUTABLE`).
- Don’t commit large data/models; use `tmp_data/` for local TFRecords.
- Style:
  - Rust: `rustfmt`, `clippy` (`ml-runner/clippy.toml`), `anyhow::Result`, feature gates (`server`, `web`).
  - Python: PEP 8, type hints, snake_case files/functions.

## Troubleshooting

- `dx` not found: install Dioxus CLI with `cargo install dioxus-cli`.
- Build links to the wrong Python: ensure you ran `make setup` and then
  `make build` from the repo root to set `PYTHON_SYS_EXECUTABLE`.
