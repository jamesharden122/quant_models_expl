pub mod pydictstructs;
pub mod train;
use crate::surr_queries;
use anyhow::Result;
use polars::prelude::*;
#[cfg(feature = "server")]
use pydictstructs::{MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict, PyList, PyModule, PyTuple};
use pyo3_polars::types::PyDataFrame;
use serde_json::Value;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::path::Path;
#[cfg(feature = "server")]
use surrealdb::{engine::any, sql::Bytes, Surreal};
#[cfg(feature = "server")]
use surrealml_core::storage::{
    header::normalisers::{linear_scaling::LinearScaling, wrapper::NormaliserType},
    header::Header,
    surml_file::SurMlFile,
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
    writer_py_path: &Path, // e.g., "py/pl2tfrecord_writer.py"
    attr: &str,
    kwargs: Py<PyDict>,
) -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        // Convert to Python polars.DataFrame
        // Import writer module from file
        let name = "pl2tfrecord_writer";
        let module = import_module_from_path(py, name, writer_py_path.to_str().unwrap())?;
        let func = module.getattr(attr)?;
        println!("we got here");
        // Call: write_tfrecord_from_polars(py_df, out_path, label, compress)
        func.call((), Some(kwargs.bind(py)))?;
        Ok(())
    })
}

/// Load TFRecord(s) into a tf.data.Dataset (returned as PyObject).
#[cfg(feature = "server")]
pub fn load_tfrecord_dataset(input_struct: TseriesTfRecLoad) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        // 1) Import reader module from the given path
        let module_name = "pl2tfrecord_reader";
        let reader_path = input_struct.reader_py_path.as_str();
        let module = import_module_from_path(py, module_name, reader_path)?;

        // 2) Get the function/attr to call (e.g., "load_tfrecord_dataset")
        let func = module.getattr(input_struct.attr.clone())?;

        // 3) Build kwargs from the struct
        let kwargs: Bound<'_, PyDict> = input_struct.to_pydict(py)?;

        // 4) Call the Python function with only kwargs (no positional args)
        let ds: Py<PyAny> = func.call(PyTuple::empty(py), Some(&kwargs))?.into();

        // 5) Optional: peek first batch to help debug shape/dtypes (best-effort)
        let first = ds
            .call_method0(py, "as_numpy_iterator")?
            .call_method0(py, "__next__")?;
        println!(
            "First batch: {:?}",
            Py::clone_ref(&first, py).as_any().to_string()
        );
        Ok(ds)
    })
}

/// Build .surml in Rust from an ONNX file, then call Python SurMlFile.upload(...)
#[cfg(feature = "server")]
pub async fn package_and_upload_surml(
    onnx_path: &str,
    surml_out: &str,
    url: &str,
    chunk_size: usize,
    namespace: &str,
    database: &str,
    username: &str,
    password: &str,
) -> Result<()> {
    // 1) Read ONNX bytes
    let model_bytes = fs::read(onnx_path)?;

    // 2) Build header (example — swap in your real feature/output names + normalisers)
    let mut header = Header::fresh();
    header.add_column("squarefoot".to_string());
    header.add_column("num_floors".to_string());
    header.add_output("house_price".to_string(), None);

    header.add_normaliser(
        "squarefoot".to_string(),
        NormaliserType::LinearScaling(LinearScaling { min: 0.0, max: 1.0 }),
    );
    header.add_normaliser(
        "num_floors".to_string(),
        NormaliserType::LinearScaling(LinearScaling { min: 0.0, max: 1.0 }),
    );

    // 3) Create .surml and write to disk
    let surml = SurMlFile::new(header, model_bytes);
    surml.write(surml_out)?;

    // 4) Call Python client: SurMlFile.upload(path=..., url=..., ...)
    upload_surml_via_python(
        surml_out, url, chunk_size, namespace, database, username, password,
    )
    .await?;

    Ok(())
}

/// PyO3 bridge that calls the official Python uploader.
#[cfg(feature = "server")]
async fn upload_surml_via_python(
    path: &str,
    url: &str,
    chunk_size: usize,
    namespace: &str,
    database: &str,
    username: &str,
    password: &str,
) -> Result<()> {
    let path = path.to_string();
    let url = url.to_string();
    let ns = namespace.to_string();
    let db = database.to_string();
    let user = username.to_string();
    let pass = password.to_string();

    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        Python::with_gil(|py| -> anyhow::Result<()> {
            // Import whichever module exposes SurMlFile in your env
            let m = py
                .import("surrealml")
                .or_else(|_| py.import("surrealml_core"))?;

            let surml_cls = m.getattr("SurMlFile")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("path", &path)?;
            kwargs.set_item("url", &url)?;
            kwargs.set_item("chunk_size", chunk_size)?;
            kwargs.set_item("namespace", &ns)?;
            kwargs.set_item("database", &db)?;
            kwargs.set_item("username", &user)?;
            kwargs.set_item("password", &pass)?;

            surml_cls.getattr("upload")?.call((), Some(&kwargs))?;
            Ok(())
        })
    })
    .await?;
    Ok(())
}

/*Load TFRecord(s) into a tf.data.Dataset (returned as PyObject).
pub fn load_tfrecord_dataset(
    //reader_py_path: &Path,   // e.g., "py/pl2tfrecord_reader.py"
    //tfrecord_paths: &[&str], // one or many
    //attr: &str,
    //feature_spec: &BTreeMap<String, String>, // {"x1":"float32","x2":"int64",...}
    //label: Option<&str>,
    //label_dtype: &str,
    //batch_size: usize,
    //shuffle: bool,
    //gzip: bool,
    input_struct: TseriesTfRecLoad,
) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        // Import reader module
        let name = "pl2tfrecord_reader";
        let module = import_module_from_path(py, name, &input_struct.reader_py_path.as_str())?;
        let func = module.getattr(attr)?;
        // Build Python args
        let py_paths = PyList::new(py, tfrecord_paths)?;
        let py_spec = feature_spec.into_py_dict(py)?;
        let ds: Py<PyAny> = func
            .call1((
                py_paths,
                py_spec,
                label,
                label_dtype,
                batch_size as i64,
                shuffle,
                gzip,
            ))?
            .into();
        // You can also take one element to inspect
        let first = ds
            .call_method0(py, "as_numpy_iterator")?
            .call_method0(py, "__next__")?;
        println!(
            "First batch: {:?}",
            Py::clone_ref(&first, py).as_any().to_string()
        );
        Ok(ds)
    })
}*/
