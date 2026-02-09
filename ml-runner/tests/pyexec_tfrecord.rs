//! Integration tests for PyO3 TFRecord helpers.
//! These are ignored by default because they require the repo Python venv
//! (with TensorFlow, polars) and `PYTHON_SYS_EXECUTABLE` set for pyo3.

use ml_runner::pyexec;
#[cfg(feature = "server")]
use ml_runner::pyexec::pydictstructs::{MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad};
use polars::prelude::*;
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString, PyTuple},
};
use pyo3_polars::types::PyDataFrame;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn writer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ml-project/py/pl2tfrecord_writer.py")
}

fn reader_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ml-project/py/pl2tfrecord_reader.py")
}

fn tmp_file(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join("pl2tf_smoke").join(name);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create tmp parent dir");
    }
    p
}

#[test]
fn write_tfrecord_from_polars_smoke() {
    // Build a tiny DataFrame
    let df = df!(
        "x" => &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 12.0, 3.0, 4.0, 31.0, 23.0, 4.0, 2.0, 2.0],
        "y" => &[1i64, 0, 1, 1, 2, 3, 4, 5, 3, 2, 12, 3, 4, 31, 23, 4, 2, 2],
        "label" => &[0i64, 1, 0, 1, 2, 3, 4, 5, 3, 2, 12, 3, 4, 31, 23, 4, 2, 2]
    )
    .expect("df build");
    let py_df = PyDataFrame(df.clone());
    pyo3::prepare_freethreaded_python();
    let out_path = "./../tmp_data/demo.tfrecord.gz";
    /*let kwargs: Py<PyDict> = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("py_df", py_df)?; // already a Py object
        kwargs.set_item("path", out_path)?; // &str is fine
        kwargs.set_item("label", "label")?;
        kwargs.set_item("compress", true)?;
        Ok(kwargs.unbind())
    })
    .unwrap();
    */
    let kwargs: Py<PyDict> = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("df", py_df)?; // already a Py object
        kwargs.set_item("path", PyString::new(py, out_path))?; // &str is fine
        kwargs.set_item("feature_cols", PyList::new(py, ["x"])?)?;
        kwargs.set_item("target_col", PyString::new(py, "y"))?;
        kwargs.set_item("time_col", py.None())?;
        kwargs.set_item("sequence_length", PyInt::new(py, 5))?;
        kwargs.set_item("horizon", PyInt::new(py, 1))?;
        kwargs.set_item("stride", PyInt::new(py, 1))?;
        kwargs.set_item("group_col", py.None())?;
        kwargs.set_item("compress", true)?;
        Ok(kwargs.unbind())
    })
    .unwrap();

    // Call your writer with a FILE path
    pyexec::write_tfrecord_from_polars(writer_path().as_path(), "write_timeseries_tfrecord_from_polars", kwargs).expect("write tfrecord");

    // Optional: assert the file exists and is non-empty
    let meta = fs::metadata(&out_path).expect("stat output file");
    assert!(meta.is_file(), "output path should be a file");
    assert!(meta.len() > 0, "output file should not be empty");
}

#[test]
#[cfg(feature = "server")]
fn load_tfrecord_dataset_smoke() {
    // Prepare a TFRecord written by the previous test (or write inline if missing)
    pyo3::prepare_freethreaded_python();
    let record_path = tmp_file("test_read.tfrecord");
    // Create a minimal TFRecord if missing
    let df = df!(
    "x" => &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.7, 4.0, 5.0, 3.0, 2.0, 12.7, 3.0, 4.0, 31.0, 23.0, 4.0, 2.0, 2.0],
    "z" => [1i64, 0, 1, 8, 2, 30, 4, 50, 3, 2, 12, 3, 4, 31, 23, 4, 2, 2],
    "y" => &[1i64, 0, 1, 1, 2, 3, 4, 5, 3, 2, 12, 3, 4, 31, 23, 4, 2, 2],
    "label" => &[0i64, 1, 0, 1, 2, 3, 4, 5, 3, 2, 12, 3, 4, 31, 23, 4, 2, 2]
    )
    .expect("df build");
    let py_df = PyDataFrame(df.clone());
    /*
         let kwargs: Py<PyDict> = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("py_df", py_df)?; // already a Py object
            kwargs.set_item("path", out_path)?; // &str is fine
            kwargs.set_item("label", "label")?;
            kwargs.set_item("compress", true)?;
            Ok(kwargs.unbind())
        })
    .unwrap();
    */
    //let out_path = "./../tmp_data/demo.tfrecord.gz";
    let kwargs: Py<PyDict> = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("df", py_df)?; // already a Py object
        kwargs.set_item("path", PyString::new(py, record_path.to_str().unwrap()))?; // &str is fine
        kwargs.set_item("feature_cols", PyList::new(py, ["x", "z"])?)?;
        kwargs.set_item("target_col", PyString::new(py, "y"))?;
        kwargs.set_item("time_col", py.None())?;
        kwargs.set_item("sequence_length", PyInt::new(py, 5))?;
        kwargs.set_item("horizon", PyInt::new(py, 1))?;
        kwargs.set_item("stride", PyInt::new(py, 1))?;
        kwargs.set_item("group_col", py.None())?;
        kwargs.set_item("compress", true)?;
        Ok(kwargs.unbind())
    })
    .unwrap();
    // Call your writer with a FILE path
    pyexec::write_tfrecord_from_polars(writer_path().as_path(), "write_timeseries_tfrecord_from_polars".into(), kwargs).expect("write tfrecord");
    // Feature spec matching the written DF
    let mut spec = BTreeMap::new();
    spec.insert("x".to_string(), "float32".to_string());
    spec.insert("z".to_string(), "float32".to_string());
    let record_str = record_path.to_str().unwrap().to_string();
    let paths: Vec<&str> = vec![record_str.as_str()];
    let inp_param = TseriesTfRecLoad {
        reader_py_path: "../ml-project/py/pl2tfrecord_reader.py".to_string(),
        tfrecord_paths: vec!["../tmp_data/mom_data.tfrecord".to_string()],
        attr: "load_time_series_tfrecord_dataset".to_string(),
        feature_spec: spec, // {"x1":"float32","x2":"int64",...}
        label: Some("label".to_string()),
        label_dtype: Some("float32".to_string()),
        batch_size: 5,
        shuffle: false,
        gzip: true,
        include_cost: None,
    };
    let ds = pyexec::load_tfrecord_dataset(inp_param).expect("load_tfrecord_dataset should return a PyObject");

    // We only assert that a Python object is returned without exceptions.
    println!("{:?}", ds);
}
