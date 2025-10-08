//! Minimal integration test for Python import bridge.
//! Ignored by default; run with: `cargo test --tests -- --ignored`.

use pyo3::types::PyAnyMethods;
use std::env;

#[test]
fn import_module_from_temp_py_file() -> Result<(), Box<dyn std::error::Error>> {
    // Write a tiny Python module to a temp file
    let mut path = env::current_dir()?;
    path.push("../ml-project/tests/pyo3_run.py");
    // Import and call via the bridge
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let module =
            ml_runner::pyexec::import_module_from_path(py, "pyo3_run", path.to_str().unwrap())
                .expect("import temp module");
        let func = module.getattr("ping").expect("get attr");
        let out: String = func
            .call0()
            .expect("call ping")
            .extract()
            .expect("extract str");
        assert_eq!(out, "pong");
    });
    Ok(())
}
