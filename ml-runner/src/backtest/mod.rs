#![allow(dead_code)]
pub mod helpers;
pub mod historical;
mod metric;
use dioxus::prelude::*;
use ndarray::prelude::*;
use polars::prelude::*;

#[cfg(feature = "server")]
use crate::inference::onnx_infer_candle;
#[cfg(feature = "server")]
use crate::surr_queries;
#[cfg(feature = "server")]
use helpers::*;
pub use helpers::{BacktestKind, BacktestParams, RunBacktestRequest};
#[cfg(feature = "server")]
use historical::*;
use metric::*;
#[cfg(feature = "server")]
use ml_backend::{featscreate::apply_by_names, surreal_queries::DbParams};

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
