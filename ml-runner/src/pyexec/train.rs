use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct TrainingResultRust {
    pub model_dir: String,
    pub metrics: Value,
}

/// Generic runner:
/// - If `callable_name` is a **function**: it is invoked as
///     `fn(ds, val_ds=None, test_ds=None, **kwargs) -> {"model_dir": str, "metrics": mapping}`.
/// - If `callable_name` is a **class**: it is instantiated with **kwargs only** (no datasets),
///     and then its `.train()` method is called, expected to return the same dict.
///
/// Notes:
/// - This lets you keep your MLSLSTMTrainer class with TFRecord-based pipelines, while also
///   supporting function-style trainers that want pre-built datasets.
///
/// - In class mode, `ds_*` arguments are ignored (your trainer builds its own datasets).
pub fn run_mls_lstm_training(
    trainer_py_path: &Path,
    callable_name: &str,
    ds_train: PyObject,
    ds_val: Option<PyObject>,
    ds_test: Option<PyObject>,
    fit_kwargs: Option<BTreeMap<String, PyObject>>,
) -> Result<TrainingResultRust> {
    Python::with_gil(|py| -> Result<TrainingResultRust> {
        // 1) Ensure the Python file's folder is importable (avoid duplicate inserts)
        {
            let sys = py.import("sys")?;
            let parent = trainer_py_path
                .parent()
                .ok_or_else(|| anyhow!("trainer path has no parent"))?;
            let p = parent.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
            let sys_path = sys.getattr("path")?;
            let contains: bool = sys_path.call_method1("count", (p,))?.extract()?;
            if !contains {
                sys_path.call_method1("insert", (0, p))?;
            }
        }

        // 2) Import the module using parent's util
        let module = super::import_module_from_path(py, "trainer_module", trainer_py_path)?;

        // 3) Resolve target
        let attr = module.getattr(callable_name).map_err(|_| {
            anyhow!(
                "callable '{}' not found in {:?}",
                callable_name,
                trainer_py_path
            )
        })?;

        // Decide function vs class using inspect.isfunction(...)
        let inspect = py.import("inspect")?;
        let is_function: bool = inspect.getattr("isfunction")?.call1((attr,))?.extract()?;

        // 4) Execute in the appropriate mode
        let out = if is_function {
            // -------- Function mode --------
            // Build kwargs: { "ds": ..., "val_ds": ..., "test_ds": ..., **fit_kwargs }
            let kwargs = PyDict::new(py);
            kwargs.set_item("ds", ds_train.as_ref(py))?;
            match ds_val {
                Some(v) => kwargs.set_item("val_ds", v.as_ref(py))?,
                None => kwargs.set_item("val_ds", py.None())?,
            }
            match ds_test {
                Some(v) => kwargs.set_item("test_ds", v.as_ref(py))?,
                None => kwargs.set_item("test_ds", py.None())?,
            }
            if let Some(extra) = fit_kwargs {
                for (k, v) in extra {
                    kwargs.set_item(k, v.as_ref(py))?;
                }
            }
            attr.call((), Some(kwargs)).map_err(|e| {
                anyhow!(
                    "Python function '{}' raised: {}\n(Hint: signature should be fn(ds, val_ds=None, test_ds=None, **kwargs))",
                    callable_name, e
                )
            })?
        } else {
            // -------- Class mode --------
            // Instantiate the class with **fit_kwargs only** (no datasets)
            let ctor_kwargs = PyDict::new(py);
            if let Some(extra) = fit_kwargs {
                for (k, v) in extra {
                    ctor_kwargs.set_item(k, v.as_ref(py))?;
                }
            }
            let inst = attr
                .call((), Some(ctor_kwargs))
                .map_err(|e| anyhow!("Instantiating class '{}' failed: {}", callable_name, e))?;

            // Call .train() (no args)
            let train_meth = inst
                .getattr("train")
                .map_err(|_| anyhow!("Class '{}' has no .train() method", callable_name))?;
            train_meth.call0().map_err(|e| {
                anyhow!(
                    "Calling '{}.train()' raised: {}\n(Hint: ensure train() returns a dict with keys 'model_dir' and 'metrics')",
                    callable_name, e
                )
            })?
        };

        // 5) Validate & extract output
        let out_dict: &PyDict = out.downcast::<PyDict>().map_err(|_| {
            anyhow!("training output must be a dict with keys 'model_dir' and 'metrics'")
        })?;

        let model_dir_py = out_dict
            .get_item("model_dir")
            .ok_or_else(|| anyhow!("training output missing key 'model_dir'"))?;
        let metrics_py = out_dict
            .get_item("metrics")
            .ok_or_else(|| anyhow!("training output missing key 'metrics'"))?;

        let model_dir: String = model_dir_py
            .extract()
            .map_err(|_| anyhow!("'model_dir' must be a string"))?;

        // Convert Python mapping -> serde_json::Value
        // If you enable pyo3 feature "serde", you can replace this roundtrip with metrics_py.extract()
        let json_mod = py.import("json")?;
        let dumps = json_mod.getattr("dumps")?;
        let metrics_json: String = dumps.call1((metrics_py,))?.extract()?;
        let metrics: Value = serde_json::from_str(&metrics_json)?;

        Ok(TrainingResultRust { model_dir, metrics })
    })
}
