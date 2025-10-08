#![cfg(feature = "server")]

use chrono::NaiveDate;
use ml_backend::featscreate::FeatList;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TseriesTfRecBento {
    pub column_set: Vec<String>, // Columns to fetch from DB
    pub srt: Option<Vec<String>>,
    pub exclude_cols: Vec<String>,
    pub out_path: Option<String>,
    pub query_params: (String, Vec<i64>),
    pub attr: String,
    pub writer_path: Option<String>, // Python writer
    pub feature_names: Option<Vec<FeatList>>,
    pub target_col: Option<String>,
    pub time_col: Option<String>,
    pub sequence_length: i64,
    pub horizon: i64,
    pub stride: i64,
    pub group_col: Option<String>,
    pub compress: bool,
    pub return_col: Option<String>,
    pub sigma_col: Option<String>,
    pub cost_col: Option<String>,
}

impl TseriesTfRecBento {
    pub fn to_pydict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let kw = PyDict::new(py);

        kw.set_item("sequence_length", self.sequence_length)?;
        kw.set_item("horizon", self.horizon)?;
        kw.set_item("stride", self.stride)?;
        kw.set_item("compress", self.compress)?;
        if let Some(t) = &self.target_col {
            kw.set_item("target_col", t)?;
        }
        kw.set_item("time_col", py.None())?; // default None
        if let Some(g) = &self.group_col {
            kw.set_item("group_col", g)?;
        }
        if let Some(r) = &self.return_col {
            kw.set_item("return_col", r)?;
        }
        if let Some(s) = &self.sigma_col {
            kw.set_item("sigma_col", s)?;
        }
        if let Some(c) = &self.cost_col {
            kw.set_item("cost_col", c)?;
        }
        Ok(kw)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TseriesTfRecWrdsGlobalInd {
    pub data_path: String,
    pub column_set: Vec<String>, // Columns to fetch from DB
    pub srt: Option<Vec<String>>,
    pub exclude_cols: Vec<String>,
    pub out_path: Option<String>,
    pub query_params: Option<(Vec<String>, NaiveDate, NaiveDate)>,
    pub attr: String,
    pub writer_path: Option<String>, // Python writer
    pub feature_names: Option<Vec<FeatList>>,
    pub target_col: Option<String>,
    pub time_col: Option<String>,
    pub sequence_length: i64,
    pub horizon: i64,
    pub stride: i64,
    pub group_col: Option<String>,
    pub compress: bool,
    pub return_col: Option<String>,
    pub cost_col: Option<String>,
    pub sigma_col: Option<String>,
}

impl TseriesTfRecWrdsGlobalInd {
    pub fn to_pydict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let kw = PyDict::new(py);

        kw.set_item("sequence_length", self.sequence_length)?;
        kw.set_item("horizon", self.horizon)?;
        kw.set_item("stride", self.stride)?;
        kw.set_item("compress", self.compress)?;
        if let Some(t) = &self.target_col {
            kw.set_item("target_col", t)?;
        }
        kw.set_item("time_col", py.None())?; // default None
        if let Some(g) = &self.group_col {
            kw.set_item("group_col", g)?;
        }
        if let Some(r) = &self.return_col {
            kw.set_item("return_col", r)?;
        }
        if let Some(s) = &self.sigma_col {
            kw.set_item("sigma_col", s)?;
        }
        if let Some(c) = &self.cost_col {
            kw.set_item("cost_col", c)?;
        }
        Ok(kw)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TseriesTfRecLoad {
    pub reader_py_path: String,      // e.g., "py/pl2tfrecord_reader.py"
    pub tfrecord_paths: Vec<String>, // one or many
    pub attr: String,
    pub feature_spec: BTreeMap<String, String>, // {"x1":"float32","x2":"int64",...}
    pub label: Option<String>,
    pub label_dtype: Option<String>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub gzip: bool,
    pub include_cost: Option<bool>,
}

impl TseriesTfRecLoad {
    pub fn to_pydict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let kw = PyDict::new(py);
        // Paths (list of str)
        let py_paths = PyList::new(py, self.tfrecord_paths.clone())?;
        kw.set_item("paths", py_paths)?;
        // Feature spec (dict[str,str])
        let py_feat = PyDict::new(py);
        for (k, v) in self.feature_spec.clone() {
            py_feat.set_item(k, v)?;
        }
        kw.set_item("feature_spec", py_feat)?;
        // Label (optional str)
        if let Some(lbl) = self.label.clone() {
            kw.set_item("label", lbl)?;
        }
        // Label dtype (str)
        if let Some(dtyp) = &self.label_dtype {
            kw.set_item("label_dtype", dtyp)?;
        }
        // Training params
        kw.set_item("batch_size", self.batch_size)?;
        kw.set_item("shuffle", self.shuffle)?;
        kw.set_item("gzip", self.gzip)?;
        if let Some(cst) = &self.include_cost {
            kw.set_item("include_cost", cst)?
        }
        Ok(kw)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlsLstmTrain {
    pub trainer_path: String,
    pub class: String,
    pub attr: String,
    pub time_steps: Option<u32>,
    pub input_dim: Option<u32>,
    pub val_split: f32,
    pub test_split: f32,
    pub epochs: usize,
    pub batch_size: Option<usize>,
    pub verbose: usize,
    pub shuffle_before_split: bool,
    pub seed: Option<usize>,
    pub save_every_epoch: bool,
    pub save_weights_only: bool,
    pub monitor: String,
    pub save_best_only: bool,
    pub run_name: Option<String>,
}

impl MlsLstmTrain {
    pub fn to_pydict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let kwargs = PyDict::new(py);
        // Optional shape hints (None -> infer from dataset element_spec)
        if let Some(time_steps) = &self.time_steps {
            kwargs.set_item("time_steps", time_steps)?;
        } else {
            kwargs.set_item("time_steps", py.None())?;
        }
        if let Some(input_dim) = &self.input_dim {
            kwargs.set_item("input_dim", input_dim)?
        } else {
            kwargs.set_item("input_dim", py.None())?
        }
        kwargs.set_item("val_split", self.val_split)?;
        kwargs.set_item("test_split", self.test_split)?; // 0.0 = no test carved out                                           // Provide an explicit test dataset ONLY if you have one; otherwise None or omit
        kwargs.set_item("test_ds", py.None())?;
        // Training params
        kwargs.set_item("epochs", self.epochs)?;
        // Leave None if your dataset is ALREADY batched; otherwise set a batch size (int)
        if let Some(batch_size) = &self.batch_size {
            kwargs.set_item("batch_size", batch_size)?;
        } else {
            kwargs.set_item("batch_size", py.None())?;
        }
        kwargs.set_item("verbose", PyInt::new(py, self.verbose))?;
        kwargs.set_item("shuffle_before_split", self.shuffle_before_split)?;
        if let Some(seed) = &self.seed {
            kwargs.set_item("seed", seed)?;
        } else {
            kwargs.set_item("seed", py.None())?;
        }
        kwargs.set_item("save_every_epoch", self.save_every_epoch)?;
        kwargs.set_item("save_weights_only", self.save_weights_only)?; // false = full .keras each epoch
        kwargs.set_item("monitor", &self.monitor)?;
        kwargs.set_item("save_best_only", self.save_best_only)?;
        if let Some(run_name) = &self.run_name {
            kwargs.set_item("run_name", run_name)?;
        } else {
            kwargs.set_item("run_name", py.None())?;
        }
        Ok(kwargs)
    }
}
