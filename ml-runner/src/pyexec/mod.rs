use crate::surr_queries;
use anyhow::Result;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::ffi::c_str;
use pyo3_polars::types::{PyDataFrame};
use serde_json::Value;
use std::path::PathBuf;
use pyo3::types::{IntoPyDict, PyDict, PyList,PyModule,PyAny};
use std::collections::BTreeMap;
use std::path::Path;
use std::ffi::CString;
#[cfg(feature="server")]
use surrealdb::{
    engine::any, 
    sql::Bytes,
    prelude::*,
};

/// Load & execute a Python module from a file path, returning the live module.
/// Registers the module in `sys.modules[name]` for subsequent imports.
pub fn import_module_from_path<'py>(
    py: Python<'py>,
    name: &str,
    path: &str,
) -> PyResult<Bound<'py, PyModule>> {
    // Read the source file (binary read is fine; we'll pass &[u8] to Python)
    let code: String = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {e}")))?;
 // Convert to C-compatible strings (no interior NULs allowed).
    let code_c = CString::new(code)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Python source contains NUL byte"))?;
    let file_c = CString::new(path)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Path contains NUL byte"))?;
    let name_c = CString::new(name)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Module name contains NUL byte"))?;
    // Compile & execute
    let module = PyModule::from_code(py, code_c.as_c_str(), file_c.as_c_str(), name_c.as_c_str())?;
    // (Optional but useful) register in sys.modules so `import name` works later
    let sys = py.import("sys")?;
    sys.getattr("modules")?.set_item(name, &module)?;
    Ok(module)
}

/// Convert a Rust Polars DataFrame to Python polars.DataFrame and write TFRecord file.
pub fn write_tfrecord_from_polars(
    df: &DataFrame,
    writer_py_path: &Path, // e.g., "py/pl2tfrecord_writer.py"
    out_path: &str,        // e.g., "/tmp/demo.tfrecord.gz"
    label: Option<&str>,
    compress: bool,
) -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        // Convert to Python polars.DataFrame
        let py_df = PyDataFrame(df.clone());

        // Import writer module from file
        let name = "pl2tfrecord_writer";
        let module = import_module_from_path(py, name, writer_py_path.to_str().unwrap())?;
        let func = module.getattr("write_tfrecord_from_polars")?;

        // Call: write_tfrecord_from_polars(py_df, out_path, label, compress)
        let kwargs = PyDict::new(py);
        kwargs.set_item("py_df", py_df)?;                 // already a Py object
        kwargs.set_item("path", out_path)?;               // &str is fine
        match label {
            Some(s) => kwargs.set_item("label", s)?,
            None => kwargs.set_item("label", py.None())?, // real Python None
        }
        kwargs.set_item("compress", compress)?;
        func.call((), Some(&kwargs))?;
        Ok(())
    })
}

/// Load TFRecord(s) into a tf.data.Dataset (returned as PyObject).
pub fn load_tfrecord_dataset(
    reader_py_path: &Path,                        // e.g., "py/pl2tfrecord_reader.py"
    tfrecord_paths: &[&str],                      // one or many
    feature_spec: &BTreeMap<String, String>,      // {"x1":"float32","x2":"int64",...}
    label: Option<&str>,
    batch_size: usize,
    shuffle: bool,
    gzip: bool,
) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        // Import reader module
        let name = "pl2tfrecord_reader";
        let module = import_module_from_path(py, name, reader_py_path.to_str().unwrap())?;
        let func = module.getattr("load_tfrecord_dataset")?;
        // Build Python args
        let py_paths = PyList::new(py, tfrecord_paths)?;
        let py_spec = feature_spec.into_py_dict(py)?;
        let ds: Py<PyAny> = func
            .call1((
                py_paths,
                py_spec,
                label,
                batch_size as i64,
                shuffle,
                gzip,
            ))?
            .into();
        Ok(ds)
    })
}


/// Use PyO3 to call py.onnx_export.savedmodel_to_onnx_bytes and return the bytes.
pub fn tf_savedmodel_to_onnx_bytes(saved_model_dir: &str) -> Result<Vec<u8>> {
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
#[cfg(feature="server")]
pub async fn store_onnx_in_surreal(
    db: &Surreal<any::Any>,
    model_id: &str,
    onnx_bytes: &[u8],
    meta_json: &str,
) -> Result<()> {
    let meta: Value = serde_json::from_str(meta_json)?;
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