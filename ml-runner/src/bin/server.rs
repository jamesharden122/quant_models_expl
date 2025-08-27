use serde_json::json;

#[cfg(feature = "server")]
use dioxus::prelude::*;

#[cfg(feature = "server")]
#[tokio::main]
async fn main() {
    dioxus::logger::initialize_default();

    // Bind to the same socket selection logic as the front-end to expose the
    // `streaming_pipe` and `training_pipe` server functions without a UI.
    let socket_addr = dioxus_cli_config::fullstack_address_or_localhost();

    let router = axum::Router::new()
        .serve_dioxus_service(ServeConfigBuilder::new())
        .into_make_service();

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}

#[cfg(not(feature = "server"))]
#[cfg(feature = "server")]
use ml_backend::polars_ops;
#[cfg(feature = "server")]
use ml_runner::{streaming_pipe, training_pipe};

#[cfg(feature = "server")]
#[tokio::main]
async fn main() {
    let mut args = std::env::args();
    args.next();
    // Allow overriding the server URL (useful when connecting over SSH).
    let base_url =
        std::env::var("SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

    match args.next().as_deref() {
        Some("stream") => {
            let payload = json!({
                "column_set": ["Ret"],
                "bin_size": "1d",
                "where_cond": {},
                "writer_path": null,
            });

            if let Err(err) = reqwest::Client::new()
                .post(format!("{}/streaming_pipe", base_url))
                .json(&payload)
                .send()
                .await
                .and_then(|r| r.error_for_status())
            {
                eprintln!("streaming failed: {err}");
            }
        }
        Some("train") => {
            let stream_payload = json!({
                "column_set": ["Ret"],
                "bin_size": "1d",
                "where_cond": {},
                "writer_path": null,
            });

            if let Err(err) = reqwest::Client::new()
                .post(format!("{}/streaming_pipe", &base_url))
                .json(&stream_payload)
                .send()
                .await
                .and_then(|r| r.error_for_status())
            {
                eprintln!("streaming failed: {err}");
                return;
            }

            let train_payload = json!({
                "reader_py_path": null,
                "trainer_py_path": "../../ml-project/py/mls_lstm_trainer.py",
                "tfrecord_paths": ["../../tmp_data/data.tfrecord"],
                "callable_name": "train",
            });

            if let Err(err) = reqwest::Client::new()
                .post(format!("{}/training_pipe", base_url))
                .json(&train_payload)
                .send()
                .await
                .and_then(|r| r.error_for_status())
            {
                eprintln!("training failed: {err}");
            }
        }
        _ => {
            eprintln!("usage: server [stream|train]");
        }
    }
}

    match args.next().as_deref() {
        Some("train") => {
            let columns = vec!["Ret"];
            let where_cond = polars_ops::PartEqSurr::default();

            if let Err(e) = streaming_pipe(columns, "1d", where_cond, None).await {
                eprintln!("streaming failed: {e}");
                return;
            }

            if let Err(e) = training_pipe(
                None,
                "../../ml-project/py/mls_lstm_trainer.py".to_string(),
                vec!["../../tmp_data/data.tfrecord".to_string()],
                "train".to_string(),
            )
            .await
            {
                eprintln!("training failed: {e}");
            }
        }
        _ => eprintln!("usage: server train"),
    }
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("enable the `server` feature to run this binary");
}

