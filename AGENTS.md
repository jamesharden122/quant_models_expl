# Repository Guidelines

## Project Structure & Module Organization
- `ml-project/`: Python ML code and tooling. `models/` (trainers implementing `BaseTrainer` → `TrainingResult`), `py/` utilities (TFRecord read/write, ONNX export), `Makefile`, `requirements.txt`.
- `ml-runner/`: Rust app (Dioxus UI + feature‑gated server). `src/pyexec/*` (PyO3 bridge), `src/bin/server.rs` (server‑only CLI), `assets/` (web assets), `Cargo.toml` (features: `web`, `server`).
- `tmp_data/`: Local TFRecord workspace produced by pipelines (ignored from VCS).

## Build, Test, and Development Commands
- Setup Python: `cd ml-project && make setup` (creates venv, installs deps).
- Build Rust with repo venv: `make build` (sets `PYTHON_SYS_EXECUTABLE`).
- Smoke tests: `make train-demo` (tiny model; prints SavedModel path). ONNX: `make export-onnx MODEL=path/to/saved_model` (prints byte length).
- UI dev: `cd ml-runner && dx serve` or `dx serve --platform desktop`.
- Server‑only: `cargo run --features server --bin server stream` or `... train`. For remote hosts, set `SERVER_URL` (default `http://127.0.0.1:8080`).

## Coding Style & Naming Conventions
- Rust: format with `rustfmt`; lint with `clippy` (`ml-runner/clippy.toml`). Use snake_case modules, `anyhow::Result` for errors, feature gates (`server`, `web`).
- Python: PEP 8 (4‑space indents), type hints, snake_case files/functions. Place trainers under `ml-project/models/`; persist under `models/saved/` (e.g., `mls_lstm_YYYYMMDD_HHMMSS`).
- Don’t commit large data/models; use `tmp_data/` for local TFRecords.

## Testing Guidelines
- Python: `pytest -q`; coverage target ≥70%: `pytest --cov=models --cov=py --cov-report=term-missing -q`. Mark long runs `@pytest.mark.slow`; default exclude with `pytest -m "not slow"`.
- Rust: `cargo test` (UI) and `cargo test --features server` (server code). Prefer small `polars::DataFrame` fixtures; mock boundaries.
- PyO3 bridge: test minimal round‑trips (tiny DataFrame → TFRecord in `tmp_data/`); clean up temp files; use toy shapes/epochs.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat(server): add CLI mode`, `fix(pyexec): handle gzip`).
- PRs: include clear description, linked issues, reproduction steps, and local run commands. Add screenshots for UI changes (`dx serve` training form).

## Security & Configuration Tips
- Use Python 3.11.9; Polars 0.50.0 across Rust/Python. Ensure builds use the repo venv via `PYTHON_SYS_EXECUTABLE`. Configure `SERVER_URL` when calling server endpoints from another machine.

