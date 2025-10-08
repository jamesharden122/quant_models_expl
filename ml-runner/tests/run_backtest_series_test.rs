#![cfg(feature = "server")]
use ml_runner::backtest::{helpers::*,run_backtest_series,};
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{FeatList, apply_by_names, MomFactor},
    surreal_queries::DbParams,
};
use dioxus::prelude::ServerFnError;
#[tokio::test]
#[cfg(feature = "server")]
async fn run_backtest_series_fails_with_invalid_db() -> Result<(), ServerFnError>{
    // Use an invalid scheme so SurrealDB initialization fails immediately.
    let run_backtest_request = RunBacktestRequest {
        db: DbParams {
            url: "https://quant-platform-06cb0tpcrpsspao10de28go15s.aws-use1.surreal.cloud/rpc"
                .to_string(),
            user: "root".to_string(), pass: "root".to_string(),
            ns: "equities".to_string(), dbname: "historical".to_string(),
        },
        query: DataQuery {
            column_set: vec![
                "instrument_id".to_string(),
                "bin".to_string(),
                "t0".to_string(),
                "t1".to_string(),
                "mean_price".to_string(),
                "ret".to_string(),
            ],
            table: String::from("equities_returns"),
            bin_size: "5m".to_string(),
            inst_id: vec![8147, 11667],
            sort: Some(vec!["instrument_id".to_string(), "bin".to_string()]),
        },
        features: FeatureStage {
            feature_transformer_names: Some(vec![FeatList::MomFactor(MomFactor)]),
        },
        inference: InferenceCfg {
            onnx_model_path: "./../ml-project/models/saved/test/final_model.onnx".to_string(),
        },
        cols: ColumnCfg {
            feature_cols: vec![ 
                "ret_sma20".to_string(), "ret_sma50".to_string(), 
                "ret_ema_small_pt1".to_string(), "ret_ema_large_pt6".to_string(), 
                "ret_var10".to_string() ,"ret_macd1s6l".to_string()
            ],
            return_col: "ret".to_string(),
            sigma_col: "sigma".to_string(),
            time_col: None,
            bench_col: None,
        },
        params: BacktestParams {
            kind: BacktestKind::Historical,
            time_steps: 5,
            stride: 1,
            horizon: 1,
            sigma_target: 0.15,
            cost_lin: 0.0,
            cost_imp: None,
            eps: 1e-8,
            ann_factor: 252.0,
        },
        output: OutputCfg {
            csv_path: String::from("../tmp_data/ml_runner_bt_invalid_db.csv"),
        },
        time: None,
        /*Some(TimeCost { 
            time_col: "bin", 
            time_star: String::from("2025-07-15T10:25:00Z"), 
            time_end: String::From("2025-07-15T1l:25:00Z")
        }),*/
    };
    run_backtest_series(
        run_backtest_request,
    )
    .await?;
    Ok(())
}

