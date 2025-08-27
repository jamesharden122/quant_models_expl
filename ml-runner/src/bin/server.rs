#[cfg(feature = "server")]
use ml_backend::polars_ops;
#[cfg(feature = "server")]
use ml_runner::{streaming_pipe, training_pipe};

#[cfg(feature = "server")]
#[tokio::main]
async fn main() {
    let mut args = std::env::args();
    args.next();
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
