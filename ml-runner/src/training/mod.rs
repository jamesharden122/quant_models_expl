use crate::pyexec;
use dioxus::prelude::*;
#[cfg(feature = "server")]
use polars::prelude::*;
#[cfg(feature = "server")]
use pyexec::pydictstructs::{MlsLstmTrain, TseriesTfRecLoad};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

#[cfg(feature = "server")]
pub async fn training_pipe(reader_struct: TseriesTfRecLoad, train_struct: MlsLstmTrain) -> Result<serde_json::Value, ServerFnError> {
    let ds = pyexec::load_tfrecord_dataset(reader_struct).map_err(|e| ServerFnError::new(e.to_string()))?;
    println!("The Data was loaded");
    //define the inputs to be passed to the python training function
    let kwargs = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kw: Bound<'_, PyDict> = train_struct.to_pydict(py)?; // Bound<PyDict>, not PyAny
        kw.set_item("full_ds", ds)?;
        // set other items...
        Ok(kw.unbind()) // back to Py<PyDict>
    })?;
    let res = pyexec::train::run_mls_lstm_training(Path::new(train_struct.trainer_path.as_str()), train_struct.class.as_str(), train_struct.attr.as_str(), kwargs)?;
    println!("Model saved at: {}", res.model_dir);
    println!("Metrics: {}", res.metrics);
    Ok(res.metrics)
}
