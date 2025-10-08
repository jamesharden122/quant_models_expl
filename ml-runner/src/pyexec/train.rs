use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct TrainingResultRust {
    pub model_dir: String,
    pub metrics: Value,
}

/// Runner for MLSLSTMTrainer class:
/// - Always treats `callable_name` as a CLASS.
/// - Instantiates with **fit_kwargs** (mapped to Python kwargs).
/// - Calls `.train()` on the instance and returns {"model_dir", "metrics"}.
pub fn run_mls_lstm_training(
    trainer_py_path: &Path,
    callable_name: &str, // e.g. "MLSLSTMTrainer"
    attr_func: &str,
    fit_kwargs: Py<PyDict>,
) -> Result<TrainingResultRust, PyErr> {
    Python::with_gil(|py| -> PyResult<TrainingResultRust> {
        // 1) Ensure containing folder is importable
        {
            let sys = py.import("sys")?;
            let parent = trainer_py_path.parent().unwrap();
            let p = parent.to_str().unwrap();
            let sys_path = sys.getattr("path")?;
            sys_path.call_method1("insert", (0, p))?;
        }

        // 2) Import module
        let module = super::import_module_from_path(
            py,
            "trainer_module",
            trainer_py_path.to_str().unwrap(),
        )?;

        // 3) Get class symbol
        let attr = module.getattr(callable_name)?;
        // 4) Instantiate class with kwargs
        let inst = attr.call((), Some(fit_kwargs.bind(py)))?;
        // 5) Call .train()
        let train_meth = inst.getattr(attr_func)?;
        let out_any = train_meth.call0()?;

        // 6) Validate & extract dict
        let out_dict = out_any.downcast::<PyDict>()?;
        let model_dir_py = out_dict.get_item("model_dir")?;
        let metrics_py = out_dict.get_item("metrics")?;

        let model_dir: String = model_dir_py.unwrap().extract()?;
        // convert metrics to JSON
        let json_mod = py.import("json")?;
        let dumps = json_mod.getattr("dumps")?;
        let metrics_json: String = dumps.call1((metrics_py,))?.extract()?;
        let metrics: Value = serde_json::from_str(&metrics_json).unwrap();

        Ok(TrainingResultRust { model_dir, metrics })
    })
}
