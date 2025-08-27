pub mod pyexec;
pub mod surr_queries;

use dioxus::prelude::*;
use dioxus_router::prelude::*;
#[cfg(feature = "server")]
use ml_backend::{polars_ops, surreal_queries};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize, Serialize, Clone, Debug)]
struct Model {
    name: String,
}

#[server]
#[cfg(feature = "server")]
pub async fn streaming_pipe<'a>(
    column_set: Vec<&'a str>,
    bin_size: &'a str,
    where_cond: polars_ops::PartEqSurr,
    writer_path: Option<String>,
) -> Result<(), ServerFnError> {
    let db = surreal_queries::make_db()?;
    let df: DataFrame = surr_queries::query_feature_bin_demo(db, column_set, bin_size, where_cond)?;

    let writer = writer_path
        .unwrap_or_else(|| "../../ml-project/py/pl2tfrecord_writer.py".to_string());

    pyexec::write_tfrecord_from_polars(
        &df,
        Path::new(&writer),
        "../../tmp_data/",
        None,
        true,
    );
    Ok(())
}

#[server]
pub async fn training_pipe(
    reader_py_path: Option<String>,
    trainer_py_path: String,
    tfrecord_paths: Vec<String>,
    callable_name: String,
) -> Result<(), ServerFnError> {
    let feature_spec = BTreeMap::new();
    let df = pyexec::load_tf_record_dataset(
        Path::new(&reader_py_path
            .unwrap_or_else(|| "../../ml-project/py/pl2tfrecord_writer.py".to_string())),
        tfrecord_paths,
        &feature_spec,
        Some("Ret"),
        512,
        true,
        true,
    );

    let fit_kwargs = BTreeMap::new();
    let _out = pyexec::train::run_mls_lstm_training(
        Path::new(&trainer_py_path),
        callable_name.as_str(),
        df,
        None,
        None,
        fit_kwargs,
    );
    Ok(())
}
#[component]
fn TrainingForm() -> Element {
    let trainer_py = use_signal(|| "../../ml-project/py/mls_lstm_trainer.py".to_string());
    let tfrecord_path = use_signal(|| "../../tmp_data/data.tfrecord".to_string());
    let callable = use_signal(|| "train".to_string());

    let run_training = move |evt: FormEvent| {
        evt.prevent_default();
        let trainer_py_val = trainer_py();
        let tfrecord_val = tfrecord_path();
        let callable_val = callable();
        spawn(async move {
            #[cfg(feature = "server")]
            {
                let columns = vec!["Ret"];
                let where_cond = polars_ops::PartEqSurr::default();
                let _ = streaming_pipe(columns, "1d", where_cond, None).await;
                let _ = training_pipe(
                    None,
                    trainer_py_val,
                    vec![tfrecord_val],
                    callable_val,
                )
                .await;
            }
        });
    };

    rsx! {
        form { onsubmit: run_training,
            label { "Trainer Python file" }
            input {
                value: trainer_py(),
                oninput: move |evt| trainer_py.set(evt.value())
            }
            label { "TFRecord file" }
            input {
                value: tfrecord_path(),
                oninput: move |evt| tfrecord_path.set(evt.value())
            }
            label { "Callable name" }
            input {
                value: callable(),
                oninput: move |evt| callable.set(evt.value())
            }
            button { r#type: "submit", "Run Training" }
        }
    }
}

#[component]
pub fn app() -> Element {
    rsx! {
        Router {
            Route { to: "/", TrainingForm {} }
        }
    }
}
