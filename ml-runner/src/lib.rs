pub mod pyexec;
pub mod surr_queries;

use dioxus::prelude::*;
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
async fn streaming_pipe<'a>(
    column_set: vec<&'a str>,
    bin_size: &'a str,
    where_cond: polars_ops::PartEqSurr,
    writer_path: Option<String>,
) -> Result<(), ServerFnError> {
    let db = surreal_queries::make_db?;
    let df: DataFrame = surr_queries::query_feature_bin_demo(db, column_set, bin_size, where_cond)?;
    pyexec::write_tfrecord_from_polars(
        &df,
        &Path::new(writer_path.unwrap_or("../../ml-project/py/pl2tfrecord_writer.py")),
        "../../tmp_data/",
        None,
        true,
    );
    Ok(())
}

#[server]
async fn training_pipe(
    reader_py_path: Option<String>,
    trainer_py_path: String,
    tfrecord_paths: Vec<String>,
    callable_name: String,
) -> Result<(), ServerFnError> {
    let feature_spec = BTreeMap::new();
    let df = pyexec::load_tf_record_dataset(
        Path::new(reader_py_path.unwrap_or("../../ml-project/py/pl2tfrecord_writer.py")),
        tfrecord_paths,
        &feature_spec,
        Some("Ret"),
        512,
        true,
        true,
    );
    let out = pyexec::train::run_mls_lstm_training(
        Path::new(trainer_py_path),
        callable_name.as_str(),
        df,
        None,
        None,
        fit_kwargs,
    );
}
#[component]
pub fn app() -> Element {
    let _search = use_signal(String::new);
    let _selected = use_signal(|| None as Option<String>);
    /*let models = use_future(cx, search, |search| {
        let term = search.current().clone();
        async move { fetch_models(&term).await.unwrap_or_default() }
    });*/

    rsx! {
        /*div { class: "model-trainer",
            input {
                value: "{search}",
                placeholder: "Search models...",
                oninput: move |e| search.set(e.value.clone()),
            }
            ul {
                models.value().unwrap_or(&vec![]).iter().map(|name| {
                    let name_clone = name.clone();
                    rsx! {
                        li {
                            onclick: move |_| selected.set(Some(name_clone.clone())),
                            "{name}"
                        }
                    }
                })
            }
            button {
                onclick: move |_| {
                    if let Some(name) = selected.get().clone() {
                        tokio::spawn(async move {
                            if let Err(e) = train_model(name).await {
                                eprintln!("training failed: {e}");
                            }
                        });
                    }
                },
                "Train"
            }
        }*/
    }
}
