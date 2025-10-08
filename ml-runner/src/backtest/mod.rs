#![allow(dead_code)]
pub mod helpers;
mod metric;
use dioxus::prelude::*;
use ndarray::prelude::*;
use polars::prelude::*;

#[cfg(feature = "server")]
use crate::inference::onnx_infer_candle;
#[cfg(feature = "server")]
use crate::surr_queries;
#[cfg(feature = "server")]
use ml_backend::{featscreate::apply_by_names, surreal_queries::DbParams};

#[cfg(feature = "server")]
use helpers::*;
use metric::*;

// Re-export common types for easier external use
pub use helpers::{BacktestKind, BacktestParams, RunBacktestRequest};
#[cfg(feature = "server")]
trait Backtest {
    fn run(
        &self,
        df: &DataFrame,
        feature_cols: &[String],
        return_col: &str,
        sigma_col: &str,
        time_col: Option<&str>,
        onnx_model_path: &str,
        params: &BacktestParams,
        bench_col: Option<&str>,
    ) -> PolarsResult<BacktestOutput>;
}

struct HistoricalBt;

#[cfg(feature = "server")]
impl HistoricalBt {
    fn run_core(
        &self,
        df: &DataFrame,
        feature_cols: &[String],
        return_col: &str,
        sigma_col: &str,
        time_col: Option<&str>,
        onnx_model_path: &str,
        params: &BacktestParams,
        bench_col: Option<&str>,
    ) -> PolarsResult<BacktestOutput> {
        // 1) Build sliding windows over feature columns
        let (x_windows, end_idx): (ndarray::Array3<f32>, Vec<usize>) =
            helpers::build_windows(df, feature_cols, params.time_steps, params.stride)?;

        if end_idx.is_empty() {
            return Ok(BacktestOutput {
                rows: DataFrame::empty(),
                sharpe: 0.0,
                sortino: 0.0,
                mdd: 0.0,
                t_stat: 0.0,
                information_ratio: None,
            });
        }
        println!("x_windows slice: {:?}", x_windows.slice(s![0, .., ..]));
        //println!("x_windows: {:?}", x_windows);
        // 2) Inference: ONNX model expects [batch, T, F]
        let preds: Vec<f32> = onnx_infer_candle(onnx_model_path, x_windows.into_dyn())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

        // 3) Align predictions w_t at window end indices
        let mut t_end_vals: Vec<i64> = Vec::with_capacity(end_idx.len());
        let mut w_t: Vec<f64> = Vec::with_capacity(end_idx.len());
        let mut r_next: Vec<f64> = Vec::with_capacity(end_idx.len());
        let mut sigma_t: Vec<f64> = Vec::with_capacity(end_idx.len());
        let mut turnover: Vec<f64> = Vec::with_capacity(end_idx.len());
        let mut r_gross: Vec<f64> = Vec::with_capacity(end_idx.len());
        let mut r_net: Vec<f64> = Vec::with_capacity(end_idx.len());
        let binding = df.column(return_col)?.cast(&DataType::Float64)?;
        let ret_s = binding.f64().unwrap();
        let binding = df.column(sigma_col)?.cast(&DataType::Float64)?;
        let sig_s = binding.f64().unwrap();
        let time_i64: Option<Int64Chunked> = if let Some(tc) = time_col {
            Some(
                df.column(tc)?
                    .cast(&DataType::Int64)?
                    .i64()
                    .unwrap()
                    .clone(),
            )
        } else {
            None
        };

        let mut prev_w = None::<f64>;
        for (k, &end) in end_idx.iter().enumerate() {
            // index for decision point/end of window
            let t_end = end as i64;
            let w = preds.get(k).copied().unwrap_or(0.0) as f64;
            let sig = sig_s.get(end).unwrap_or(0.0);
            let r = ret_s.get(end + params.horizon - 1).unwrap_or(0.0);

            let turn = prev_w.map(|pw| (w - pw).abs()).unwrap_or(0.0);
            prev_w = Some(w);

            let r_g = if sig.abs() > (params.eps as f64) {
                w * ((params.sigma_target as f64) / sig) * r
            } else {
                0.0
            };
            let mut cost = (params.cost_lin as f64) * turn;
            if let Some((c_imp, alpha)) = params.cost_imp {
                let denom = sig + params.eps as f64;
                cost += (c_imp as f64) * ((turn / denom).powf(alpha as f64));
            }
            let r_n = r_g - cost;

            w_t.push(w);
            r_next.push(r);
            sigma_t.push(sig);
            turnover.push(turn);
            r_gross.push(r_g);
            r_net.push(r_n);
            t_end_vals.push(if let Some(ref ti) = time_i64 {
                ti.get(end).unwrap_or(t_end)
            } else {
                t_end
            });
        }

        let rows = DataFrame::new(vec![
            Column::new("t_end".into(), t_end_vals),
            Column::new("w_t".into(), w_t),
            Column::new("r_next".into(), r_next),
            Column::new("sigma_t".into(), sigma_t),
            Column::new("turnover".into(), turnover),
            Column::new("R_gross".into(), r_gross),
            Column::new("R_net".into(), r_net.clone()),
        ])?;

        let r_net_s = rows.column("R_net")?.as_materialized_series();
        // metrics from Series input
        let sharpe = sharpe_ratio(r_net_s, None, params.ann_factor)?;
        let sortino = sortino_ratio(r_net_s, None, params.ann_factor)?;
        let mdd = max_drawdown(r_net_s)?;
        let t_stat = t_statistic(r_net_s)?;
        let information_ratio = if let Some(bc) = bench_col {
            let bench = df.column(bc)?.as_materialized_series();
            Some(information_ratio(r_net_s, bench, params.ann_factor)?)
        } else {
            None
        };

        Ok(BacktestOutput {
            rows,
            sharpe,
            sortino,
            mdd,
            t_stat,
            information_ratio,
        })
    }
}
#[cfg(feature = "server")]
impl Backtest for HistoricalBt {
    fn run(
        &self,
        df: &DataFrame,
        feature_cols: &[String],
        return_col: &str,
        sigma_col: &str,
        time_col: Option<&str>,
        onnx_model_path: &str,
        params: &BacktestParams,
        bench_col: Option<&str>,
    ) -> PolarsResult<BacktestOutput> {
        self.run_core(
            df,
            feature_cols,
            return_col,
            sigma_col,
            time_col,
            onnx_model_path,
            params,
            bench_col,
        )
    }
}
#[cfg(feature = "server")]
fn make_backtest(kind: BacktestKind) -> Box<dyn Backtest + Send + Sync> {
    match kind {
        BacktestKind::Historical => Box::new(HistoricalBt),
        BacktestKind::WalkForward => Box::new(HistoricalBt), // placeholder; same core for now
    }
}

#[cfg(feature = "server")]
pub async fn run_backtest_series(
    req: RunBacktestRequest,
) -> Result<(serde_json::Value), ServerFnError> {
    let db = ml_backend::surreal_queries::make_db(
        req.db.url.as_str(),
        req.db.user.as_str(),
        req.db.pass.as_str(),
        req.db.ns.as_str(),
        req.db.dbname.as_str(),
    )
    .await?;

    println!("logged in");

    // Query core table
    let mut df: DataFrame = surr_queries::query_feature_bin_demo(
        &db,
        req.query.column_set.iter().map(|s| s.as_str()).collect(),
        req.query.bin_size.clone(),
        req.query.inst_id.clone(),
        req.query.sort.clone(),
    )
    .await?;
    println!("DataFrame: {:?}", df);
    println!("{:?}", req.features.feature_transformer_names);
    // Optional feature engineering
    if let Some(names) = req.features.feature_transformer_names.clone() {
        println!("{:?}", "got here");
        df = apply_by_names(df, names).await?;
        println!("Transformed DataFrame: {:?}", df);
    }

    let bt = make_backtest(req.params.kind);
    let out = bt.run(
        &df,
        &req.cols.feature_cols,
        &req.cols.return_col,
        &req.cols.sigma_col,
        req.cols.time_col.as_deref(),
        &req.inference.onnx_model_path,
        &req.params,
        req.cols.bench_col.as_deref(),
    )?;

    // Write the per-step DataFrame
    let mut file = std::fs::File::create(&req.output.csv_path)
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    polars::prelude::CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut out.rows.clone()) // keep your current behavior
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    // Write JSON summary next to it
    let summary_path = format!(
        "{}{}",
        req.output.csv_path.trim_end_matches(".csv"),
        "_summary.json"
    );
    let summary = serde_json::json!({
        "sharpe": out.sharpe,
        "sortino": out.sortino,
        "mdd": out.mdd,
        "t_stat": out.t_stat,
        "information_ratio": out.information_ratio,
        "rows": out.rows.height(),
        "cols": out.rows.width(),
        "path": req.output.csv_path,
    });
    std::fs::write(&summary_path, serde_json::to_vec_pretty(&summary).unwrap())
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    println!("{:?}", summary);
    Ok((summary))
}
