#![cfg(feature = "server")]

use ml_backend::featscreate::FeatList;
use ml_backend::surreal_queries::DbParams;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBacktestRequest {
    // data query
    pub db: DbParams,
    pub query: DataQuery,
    // feature engineering
    pub features: FeatureStage,
    // inference
    pub inference: InferenceCfg,
    // columns
    pub cols: ColumnCfg,
    // backtest parameters
    pub params: BacktestParams,
    // output
    #[serde(default)]
    pub output: OutputCfg,
    pub time: Option<TimeConst>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConst {
    pub time_col: String,   // e.g. Some("bin"), Some("t0"), Some("t1")
    pub time_start: String, // e.g. "2025-07-15T10:25:00Z" (RFC3339) or "7/15/2025, 10:25:00 AM"
    pub time_end: String,   // same format as start
}
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BacktestKind {
    WalkForward,
    Historical,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BacktestParams {
    pub kind: BacktestKind,
    pub time_steps: usize,
    pub stride: usize,
    pub horizon: usize,               // steps ahead for r_{t,t+1}
    pub sigma_target: f32,            // sigma_tgt
    pub cost_lin: f32,                // c_lin
    pub cost_imp: Option<(f32, f32)>, // (c_imp, alpha)
    pub eps: f32,
    pub ann_factor: f64, // annualization (e.g., 252.0)
}

impl Default for BacktestParams {
    fn default() -> Self {
        Self {
            kind: BacktestKind::Historical,
            time_steps: 64,
            stride: 1,
            horizon: 1,
            sigma_target: 0.15,
            cost_lin: 0.0,
            cost_imp: None,
            eps: 1e-8,
            ann_factor: 252.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BacktestOutput {
    pub rows: DataFrame, // per-step flattened outputs
    pub sharpe: f64,
    pub sortino: f64,
    pub mdd: f64,
    pub t_stat: f64,
    pub information_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuery {
    pub column_set: Vec<String>,
    pub bin_size: String,
    pub inst_id: Vec<i64>,
    pub table: String,
    #[serde(default)]
    pub sort: Option<Vec<String>>, // formerly `srt`
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureStage {
    #[serde(default)]
    pub feature_transformer_names: Option<Vec<FeatList>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCfg {
    pub onnx_model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnCfg {
    pub feature_cols: Vec<String>,
    pub return_col: String,
    pub sigma_col: String,
    #[serde(default)]
    pub time_col: Option<String>,
    #[serde(default)]
    pub bench_col: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCfg {
    /// CSV file path for per-step rows
    #[serde(default = "default_out_path")]
    pub csv_path: String,
}
fn default_out_path() -> String {
    "../tmp_data/backtest_output.csv".to_string()
}
impl Default for OutputCfg {
    fn default() -> Self {
        Self {
            csv_path: default_out_path(),
        }
    }
}

/// Build sliding windows [N, T, F] and collect end indices in the original df.
pub fn build_windows(
    df: &DataFrame,
    feature_cols: &[String],
    time_steps: usize,
    stride: usize,
) -> PolarsResult<(ndarray::Array3<f32>, Vec<usize>)> {
    // Select & cast once
    let fdf = df
        .select(feature_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?
        .lazy()
        .cast_all(DataType::Float64, false)
        .collect()?;

    let n = fdf.height();
    let f = fdf.width();

    if n < time_steps {
        return Ok((ndarray::Array3::<f32>::zeros((0, time_steps, f)), vec![]));
    }

    // Typed handles (no extra cast)
    let cols: Vec<&Float64Chunked> = fdf.get_columns().iter().map(|s| s.f64().unwrap()).collect();

    // exact window count (full windows only)
    let win_count = 1 + (n - time_steps) / stride;

    // pre-alloc
    let mut batches: Vec<f32> = Vec::with_capacity(win_count * time_steps * f);
    let mut end_idx: Vec<usize> = Vec::with_capacity(win_count);

    // Build windows
    let mut i = 0usize;
    for _ in 0..win_count {
        let end = i + time_steps - 1;

        // Temporary buffer for one window in row-major [T, F]
        let mut window_buf = vec![0f32; time_steps * f];

        // Pull each column's window with a zero-copy view, then materialize once
        for c in 0..f {
            let wcol = cols[c].slice(i as i64, time_steps); // view
            let v: Vec<Option<f64>> = wcol.to_vec(); // materialize once

            // place into row-major buffer at [t, c]
            for (t, opt) in v.into_iter().enumerate() {
                // your original policy: NULL -> 0.0
                window_buf[t * f + c] = opt.unwrap_or(0.0) as f32;
            }
        }

        // append the whole window at once
        batches.extend_from_slice(&window_buf);

        end_idx.push(end);
        i += stride;
    }

    let arr = ndarray::Array::from_shape_vec((win_count, time_steps, f), batches)
        .map_err(|_| PolarsError::ComputeError("shape mismatch".into()))?;
    Ok((arr, end_idx))
}
