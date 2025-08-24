use std::path::PathBuf;

use anyhow::Result;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_polars::dataframe::PyDataFrame;
use serde_json::Value as JsonValue;
use surrealdb::{engine::any::{Any, connect}, sql::{Bytes, Value as DBValue}, Surreal};

/// Convert Rust DF -> Python Polars DF via PyO3 and call py.trainer.train_from_polars.
/// cfg_json includes {"target": "...", "epochs": 10, "batch": 256, "outdir": "models", ...}
pub fn train_via_pyo3(
    df: &DataFrame,
    trainer_module: &str,
    trainer_fn: &str,
    cfg_json: &str,
) -> Result<String> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let py_path = base.join("../py");
    let models_path = base.join("../models");
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (py_path.to_str().unwrap(),))?;
        path.call_method1("append", (models_path.to_str().unwrap(),))?;
    let py_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../py");
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        sys.getattr("path")?
            .call_method1("append", (py_path.to_str().unwrap(),))?;
        let module = PyModule::import(py, trainer_module)?;
        let func = module.getattr(trainer_fn)?;
        let py_df = PyDataFrame(df.clone()).into_py(py);
        let result = func.call1((py_df, cfg_json))?;
        Ok(result.extract::<String>()?)
    })
}

/// Use PyO3 to call py.onnx_export.savedmodel_to_onnx_bytes and return the bytes.
pub fn tf_savedmodel_to_onnx_bytes(saved_model_dir: &str) -> Result<Vec<u8>> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let py_path = base.join("../py");
    let models_path = base.join("../models");
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (py_path.to_str().unwrap(),))?;
        path.call_method1("append", (models_path.to_str().unwrap(),))?;
    let py_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../py");
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        sys.getattr("path")?
            .call_method1("append", (py_path.to_str().unwrap(),))?;
        let module = PyModule::import(py, "onnx_export")?;
        let func = module.getattr("savedmodel_to_onnx_bytes")?;
        let bytes = func.call1((saved_model_dir,))?.extract::<Vec<u8>>()?;
        Ok(bytes)
    })
}

/// Upsert ONNX bytes + metadata into SurrealDB (e.g., record id models:onnx/<model_id>).
pub async fn store_onnx_in_surreal(
    db: &Surreal<Any>,
    model_id: &str,
    onnx_bytes: &[u8],
    meta_json: &str,
) -> Result<()> {
    let meta: JsonValue = serde_json::from_str(meta_json)?;
    let sql = "UPSERT type::thing('models:onnx', $id) CONTENT {
        model_id: $id,
        framework: 'onnx',
        bytes: $bytes,
        meta: $meta,
        created_at: time::now()
    }";
    db.query(sql)
        .bind(("id", model_id))
        .bind(("bytes", Bytes::from(onnx_bytes.to_vec())))
        .bind(("meta", meta))
        .await?;
    Ok(())
}

fn vec_to_polars(_rows: Vec<DBValue>) -> Result<DataFrame> {
    // Placeholder: transform query rows into a Polars DataFrame
    Ok(DataFrame::default())
}

#[pyfunction]
fn make_df(url: &str, sql: &str) -> PyResult<PyDataFrame> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let df = rt
        .block_on(async {
            let db = connect(url).await?;
            let mut resp = db.query(sql).await?;
            let rows: Vec<DBValue> = resp.take(0)?;
            vec_to_polars(rows)
        })
        .map_err(|e: anyhow::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(df))
}

#[pyfunction]
fn add_sum_column(py_df: PyDataFrame) -> PyResult<PyDataFrame> {
    let mut df = py_df.0;
    let mut sum = &df.column("a")? + &df.column("b")?;
    sum.rename("sum");
    df.with_column(sum)?;
    Ok(PyDataFrame(df))
}

#[pymodule]
fn ml_runner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_df, m)?)?;
    m.add_function(wrap_pyfunction!(add_sum_column, m)?)?;
    Ok(())
}
