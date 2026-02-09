pub mod pydictstructs;
pub mod train;
use crate::error::{msg, Result};
use crate::surr_queries;
use polars::prelude::*;
#[cfg(feature = "server")]
use pydictstructs::{MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};
use pyo3_polars::types::PyDataFrame;
use serde_json::Value;
use std::ffi::CString;
use std::path::Path;
#[cfg(feature = "server")]
use surrealdb::{engine::any, sql::Bytes, Surreal};

/// Load & execute a Python module from a file path, returning the live module.
/// Registers the module in `sys.modules[name]` for subsequent imports.
pub fn import_module_from_path<'py>(py: Python<'py>, name: &str, path: &str) -> PyResult<Bound<'py, PyModule>> {
    // Read the source file (binary read is fine; we'll pass &[u8] to Python)
    let code: String = std::fs::read_to_string(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {e}")))?;
    // Convert to C-compatible strings (no interior NULs allowed).
    let code_c = CString::new(code).map_err(|_| pyo3::exceptions::PyValueError::new_err("Python source contains NUL byte"))?;
    let file_c = CString::new(path).map_err(|_| pyo3::exceptions::PyValueError::new_err("Path contains NUL byte"))?;
    let name_c = CString::new(name).map_err(|_| pyo3::exceptions::PyValueError::new_err("Module name contains NUL byte"))?;
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
        let module = import_module_from_path(py, name, writer_py_path.to_str().unwrap()).map_err(|e| msg(e.to_string()))?;
        let func = module.getattr(attr).map_err(|e| msg(e.to_string()))?;
        println!("we got here");
        // Call: write_tfrecord_from_polars(py_df, out_path, label, compress)
        func.call((), Some(kwargs.bind(py))).map_err(|e| msg(e.to_string()))?;
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
        let first = ds.call_method0(py, "as_numpy_iterator")?.call_method0(py, "__next__")?;
        println!("First batch: {:?}", Py::clone_ref(&first, py).as_any().to_string());
        Ok(ds)
    })
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
