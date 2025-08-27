# Development

Your new bare-bones project includes minimal organization with a single `main.rs` file and a few assets.

```
project/
├─ assets/ # Any assets that are used by the app should be placed here
├─ src/
│  ├─ main.rs # main.rs is the entry point to your application and currently contains all components for the app
├─ Cargo.toml # The Cargo.toml file defines the dependencies and feature flags for your project
```

### Serving Your App

Run the following command in the root of your project to start developing with the default platform:

```bash
dx serve
```

To run for a different platform, use the `--platform platform` flag. E.g.

```bash
dx serve --platform desktop
```

## Deployments

This crate can now be run in two different modes:

* **Full stack** – launch the Dioxus front-end together with the backend:

  ```bash
  dx serve
  ```

* **Server only** – start just the backend without any UI:
  From another terminal session connect to the host and trigger pipelines via
  the CLI helper:

  ```bash
  cargo run --bin server stream      # run streaming_pipe
  cargo run --bin server train       # run streaming_pipe then training_pipe
  ```

  Set `SERVER_URL` to point at a remote address if the server is not running on
  `127.0.0.1`.

## Web training form

When running the full-stack application, navigating to the root route renders a
training form. The form lets you specify the trainer Python file, the TFRecord
data file, and the callable to invoke, then executes the streaming and training
pipelines with those parameters.

  From another terminal session you can then issue commands directly against
  the library, for example running the streaming and training pipelines:
  cargo run --bin server --features server train

  Replace `train` with the desired command

