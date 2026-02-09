#![cfg(feature = "server")]

use axum::{
    extract::Query,
    http::Method,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json,
};
use dioxus::prelude::*;
use ml_backend::{
    featscreate::{globalindexes::GlobalIndexes, momindexes::MomFactor, FeatList},
    surreal_queries::DbParams,
};
use ml_runner::backtest::helpers::{BacktestKind, BacktestParams, ColumnCfg, DataQuery, FeatureStage, InferenceCfg, OutputCfg, RunBacktestRequest};
use ml_runner::pipelines::tsmomnn::back_test_time_series_momentum_lstm;
use ml_runner::pipelines::tsmomnn::run_time_series_momentum_lstm;
use ml_runner::pyexec::pydictstructs::{MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad};
use serde::Deserialize;
use serde_json::json;
use tower_http::cors::{Any, CorsLayer};

#[cfg(feature = "server")]
/// Run the headless servers:
///   cargo run --manifest-path ml-runner/Cargo.toml --no-default-features --features server --bin server
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    dioxus::logger::initialize_default();
    // Allow switching to the ort-candle backend via environment.
    if std::env::var("ORT_BACKEND").ok().as_deref() == Some("candle") {
        ml_runner::inference::init_candle_backend();
    }

    // Bind to the same socket selection logic as the front-end to expose routes.
    let socket_addr = dioxus_cli_config::fullstack_address_or_localhost();
    println!("🚀 Server listening on http://{}", socket_addr);
    // Base router with headless endpoints
    #[cfg(feature = "server")]
    let base = axum::Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/tsmomnn/backtest", get(tsmomnn_backtest_get))
        .route("/tsmomnn/backtest", post(tsmomnn_backtest_post))
        .route("/tsmomnn/train", get(tsmomnn_train_get).post(tsmomnn_train_post))
        .layer(
            CorsLayer::new()
                .allow_origin(Any) // in prod, list your exact origins
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any), // or specific: [CONTENT_TYPE, AUTHORIZATION, ...]
        );

    let router = base.into_make_service();

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}

#[cfg(feature = "server")]
#[derive(Debug, Deserialize, Clone)]
struct BtQuery {
    // Full JSON override (URL-encoded JSON string) for RunBacktestRequest
    req_json: Option<String>,
    // DB overrides
    db_url: Option<String>,
    db_user: Option<String>,
    db_pass: Option<String>,
    db_ns: Option<String>,
    db_db: Option<String>,
    // Query stage
    table: Option<String>,
    column_set: Option<String>, // csv
    sort: Option<String>,       // csv
    bin_size: Option<String>,
    inst_ids: Option<String>, // csv of i64
    // Feature stage
    feats: Option<String>, // csv of feat names (e.g., MomFactor)
    // Inference
    model_path: Option<String>,
    // Cols
    feature_cols: Option<String>, // csv
    return_col: Option<String>,
    sigma_col: Option<String>,
    time_col: Option<String>,
    bench_col: Option<String>,
    // Params
    kind: Option<String>,
    time_steps: Option<usize>,
    stride: Option<usize>,
    horizon: Option<usize>,
    sigma_target: Option<f32>,
    cost_lin: Option<f32>,
    cost_imp: Option<String>, // "c_imp,alpha"
    eps: Option<f32>,
    ann_factor: Option<f64>,
    // Output
    out_csv: Option<String>,
    // Optional time constraint
    time_col_name: Option<String>,
    time_start: Option<String>,
    time_end: Option<String>,
}

#[cfg(feature = "server")]
/// GET /tsmomnn/backtest
///
/// Triggers a time-series momentum LSTM backtest. Supply either:
/// - req_json: URL-encoded JSON for RunBacktestRequest, or
/// - flat query params (e.g., inst_ids, bin_size, model_path, out_csv, etc.).
///
/// Example (defaults + overrides):
///   cargo run --manifest-path ml-runner/Cargo.toml --no-default-features --features server --bin server
///   curl "http://127.0.0.1:8080/tsmomnn/backtest?inst_ids=8147,11667&bin_size=5m&model_path=../ml-project/models/saved/test/final_model.onnx&out_csv=../tmp_data/my_bt.csv"
///
/// Example (full JSON via req_json):
///   curl --get --data-urlencode 'req_json={...RunBacktestRequest JSON...}' http://127.0.0.1:8080/tsmomnn/backtest
#[axum::debug_handler]
async fn tsmomnn_backtest_get(Query(q): Query<BtQuery>) -> impl IntoResponse {
    // Helper parsers
    fn csv_list(s: &str) -> Vec<String> {
        s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
    }
    fn csv_i64(s: &str) -> Vec<i64> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    }
    fn parse_kind(s: &str) -> BacktestKind {
        match s.to_ascii_lowercase().as_str() {
            "walkforward" | "walk_forward" => BacktestKind::WalkForward,
            _ => BacktestKind::Historical,
        }
    }
    fn parse_feats(s: &str) -> Vec<FeatList> {
        csv_list(s)
            .into_iter()
            .filter_map(|name| match name.to_ascii_lowercase().as_str() {
                "mom" | "momfactor" | "mom_factor" => Some(FeatList::MomFactor(MomFactor)),
                _ => None,
            })
            .collect()
    }
    fn parse_cost_imp(s: &str) -> Option<(f32, f32)> {
        let mut it = s.split(',').filter_map(|x| x.trim().parse::<f32>().ok());
        match (it.next(), it.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }
    println!("{:?}", q);
    // If full JSON is provided, use it directly
    if let Some(req_json) = &q.req_json {
        match serde_json::from_str::<RunBacktestRequest>(req_json) {
            Ok(req) => {
                println!("Cloned Request: {:?}", req.clone());
                return match back_test_time_series_momentum_lstm(req).await {
                    Ok((res)) => (StatusCode::OK, Json(res)),
                    Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status":"error","message": e.to_string()}))),
                };
            }

            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"status":"error","message": format!("invalid req_json: {}", e)}))),
        }
    }

    // Defaults with overrides
    let db = DbParams {
        url: q
            .db_url
            .clone()
            .unwrap_or_else(|| std::env::var("SUR_URL").unwrap_or_else(|_| "https://quant-platform-06cb0tpcrpsspao10de28go15s.aws-use1.surreal.cloud/rpc".to_string())),
        user: q.db_user.clone().unwrap_or_else(|| std::env::var("SUR_USER").unwrap_or_else(|_| "root".to_string())),
        pass: q.db_pass.clone().unwrap_or_else(|| std::env::var("SUR_PASS").unwrap_or_else(|_| "root".to_string())),
        ns: q.db_ns.clone().unwrap_or_else(|| std::env::var("SUR_NS").unwrap_or_else(|_| "equities".to_string())),
        dbname: q.db_db.clone().unwrap_or_else(|| std::env::var("SUR_DB").unwrap_or_else(|_| "historical".to_string())),
    };

    let inst_id = q.inst_ids.as_deref().map(csv_i64).unwrap_or_else(|| vec![8147, 11667]);
    let column_set = q
        .column_set
        .as_deref()
        .map(csv_list)
        .unwrap_or_else(|| vec!["instrument_id".into(), "bin".into(), "t0".into(), "t1".into(), "mean_price".into(), "ret".into(), "sigma".into()]);
    let sort = q.sort.as_deref().map(csv_list);
    let bin_size = q.bin_size.clone().unwrap_or_else(|| "5m".into());
    let table = q.table.clone().unwrap_or_else(|| "equities_returns".into());

    let feats = q.feats.as_deref().map(parse_feats);
    println!("feats {:?}", feats);

    let feature_cols = q.feature_cols.as_deref().map(csv_list).unwrap_or_else(|| {
        vec![
            "ret_sma20".into(),
            "ret_sma50".into(),
            "ret_ema_small_pt1".into(),
            "ret_ema_large_pt6".into(),
            "ret_var10".into(),
            "ret_macd1s6l".into(),
        ]
    });
    let return_col = q.return_col.clone().unwrap_or_else(|| "ret".into());
    let sigma_col = q.sigma_col.clone().unwrap_or_else(|| "sigma".into());

    let kind = q.kind.as_deref().map(parse_kind).unwrap_or(BacktestKind::Historical);
    let params = BacktestParams {
        kind,
        time_steps: q.time_steps.unwrap_or(5),
        stride: q.stride.unwrap_or(1),
        horizon: q.horizon.unwrap_or(1),
        sigma_target: q.sigma_target.unwrap_or(0.15),
        cost_lin: q.cost_lin.unwrap_or(0.0),
        cost_imp: q.cost_imp.as_deref().and_then(parse_cost_imp),
        eps: q.eps.unwrap_or(1e-8),
        ann_factor: q.ann_factor.unwrap_or(252.0),
    };

    let out_csv = q.out_csv.clone().unwrap_or_else(|| "../tmp_data/backtest_output.csv".into());
    let onnx_model_path = q.model_path.clone().unwrap_or_else(|| "./../ml-project/models/saved/test/final_model.onnx".into());

    let time = match (&q.time_col_name, &q.time_start, &q.time_end) {
        (Some(time_col), Some(time_start), Some(time_end)) => Some(ml_runner::backtest::helpers::TimeConst {
            time_col: time_col.clone(),
            time_start: time_start.clone(),
            time_end: time_end.clone(),
        }),
        _ => None,
    };

    let req = RunBacktestRequest {
        db,
        query: DataQuery {
            column_set,
            bin_size,
            inst_id,
            table,
            sort,
        },
        features: FeatureStage { feature_transformer_names: feats },
        inference: InferenceCfg { onnx_model_path },
        cols: ColumnCfg {
            feature_cols,
            return_col,
            sigma_col,
            time_col: q.time_col.clone(),
            bench_col: q.bench_col.clone(),
        },
        params,
        output: OutputCfg { csv_path: out_csv },
        time,
    };
    println!("{:?}", req.clone());
    match back_test_time_series_momentum_lstm(req.clone()).await {
        Ok(res) => (StatusCode::OK, Json(res)),
        Err(e) => {
            let res = json!({ "status": "error", "message": e.to_string() });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(res))
        }
    }
}

#[cfg(feature = "server")]
/// POST /tsmomnn/backtest
/// Body: JSON RunBacktestRequest
/// Example:
///   cargo run --manifest-path ml-runner/Cargo.toml --no-default-features --features server --bin server
///   curl -X POST http://127.0.0.1:8080/tsmomnn/backtest \
///        -H 'content-type: application/json' \
///        -d '{"db": {"url":"http://127.0.0.1:8000/rpc","user":"root","pass":"root","ns":"equities","dbname":"historical"},
///             "query": {"column_set":["instrument_id","bin","t0","t1","mean_price","ret","sigma"],"bin_size":"5m","inst_id":[8147,11667],"table":"equities_returns","sort":["instrument_id","bin"]},
///             "features": {"feature_transformer_names":["MomFactor"]},
///             "inference": {"onnx_model_path":"../ml-project/models/saved/test/final_model.onnx"},
///             "cols": {"feature_cols":["ret_sma20","ret_sma50","ret_ema_small_pt1","ret_ema_large_pt6","ret_var10","ret_macd1s6l"],"return_col":"ret","sigma_col":"sigma"},
///             "params": {"kind":"Historical","time_steps":5,"stride":1,"horizon":1,"sigma_target":0.15,"cost_lin":0.0,"eps":1e-8,"ann_factor":252.0},
///             "output": {"csv_path":"../tmp_data/my_bt.csv"}}'
#[axum::debug_handler]
async fn tsmomnn_backtest_post(Json(req): Json<RunBacktestRequest>) -> impl IntoResponse {
    match back_test_time_series_momentum_lstm(req).await {
        Ok((res)) => (StatusCode::OK, Json(json!({"status":"ok"}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status":"error","message": e.to_string()}))),
    }
}

#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
struct TrainQuery {
    req_json: Option<String>,
}

#[cfg(feature = "server")]
#[derive(Debug, Deserialize, Clone)]
struct TrainReq {
    db: DbParams,
    write: TseriesTfRecBento,
    load: TseriesTfRecLoad,
    train: MlsLstmTrain,
}

#[cfg(feature = "server")]
/// GET /tsmomnn/train
///
/// Runs the streaming + training pipeline.
/// Supply req_json as URL-encoded JSON mapping to:
///   { "db": DbParams, "write": TseriesTfRec, "load": TseriesTfRecLoad, "train": MlsLstmTrain }
///
/// Example:
///   cargo run --manifest-path ml-runner/Cargo.toml --no-default-features --features server --bin server
///   curl --get --data-urlencode 'req_json={"db": {"url":"http://127.0.0.1:8000/rpc","user":"root","pass":"root","ns":"equities","dbname":"historical"},
///                                 "write": {"column_set":["instrument_id","bin","t0","t1","mean_price","ret"],"srt":["instrument_id","bin"],"exclude_cols":["instrument_id","bin","t0","t1"],"out_path":"../tmp_data/mom_data.tfrecord","attr":"write_timeseries_tfrecord_from_polars","writer_path":"../ml-project/py/pl2tfrecord_writer.py","feature_names":["MomFactor"],"target_col":"ret","time_col":null,"sequence_length":5,"horizon":1,"stride":1,"group_col":"instrument_id","compress":true,"return_col":null,"sigma_col":null,"cost_col":null},
///                                 "load": {"reader_py_path":"../ml-project/py/pl2tfrecord_reader.py","tfrecord_paths":["../tmp_data/mom_data.tfrecord"],"attr":"load_time_series_tfrecord_dataset","feature_spec":{"mean_price":"float32","ret_sma20":"float32","ret_sma50":"float32","ret_ema_small_pt1":"float32","ret_ema_large_pt6":"float32","ret_var10":"float32","ret_macd1s6l":"float32"},"label":"label","label_dtype":"float32","batch_size":5,"shuffle":false,"gzip":true,"include_cost":false},
///                                 "train": {"trainer_path":"../ml-project/models/mls_lstm_trainer.py","class":"MLSLSTMTrainer","attr":"train","time_steps":5,"input_dim":7,"val_split":0.1,"test_split":0.1,"epochs":10,"batch_size":null,"verbose":1,"shuffle_before_split":false,"seed":42,"save_every_epoch":false,"save_weights_only":false,"monitor":"val_loss","save_best_only":true,"run_name":"test"}}
///                               ' http://127.0.0.1:8080/tsmomnn/train
async fn tsmomnn_train_get(Query(q): Query<TrainQuery>) -> impl IntoResponse {
    println!("{:?}", "running the train routine for time series momentum lstm network!");
    let Some(req_json) = &q.req_json else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "status":"error",
                "message":"provide req_json mapping to {db,write,load,train}"
            })),
        );
    };
    pyo3::prepare_freethreaded_python();
    match serde_json::from_str::<TrainReq>(req_json) {
        Ok(req) => match run_time_series_momentum_lstm(req.db, req.write, req.load, req.train).await {
            Ok((val)) => (StatusCode::OK, Json(json!(val))),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status":"error","message": e.to_string()}))),
        },
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"status":"error","message": format!("invalid req_json: {}", e)}))),
    }
}

#[cfg(feature = "server")]
/// POST /tsmomnn/train
/// Body: JSON { db: DbParams, write: TseriesTfRec, load: TseriesTfRecLoad, train: MlsLstmTrain }
/// Example:
///   cargo run --manifest-path ml-runner/Cargo.toml --no-default-features --features server --bin server
///   curl -X POST http://127.0.0.1:8080/tsmomnn/train \
///        -H 'content-type: application/json' \
///        -d '{"db": {"url":"http://127.0.0.1:8000/rpc","user":"root","pass":"root","ns":"equities","dbname":"historical"},
///             "write": {"column_set":["instrument_id","bin","t0","t1","mean_price","ret"],"srt":["instrument_id","bin"],"exclude_cols":["instrument_id","bin","t0","t1"],"out_path":"../tmp_data/mom_data.tfrecord","attr":"write_timeseries_tfrecord_from_polars","writer_path":"../ml-project/py/pl2tfrecord_writer.py","feature_names":["MomFactor"],"target_col":"ret","time_col":null,"sequence_length":5,"horizon":1,"stride":1,"group_col":"instrument_id","compress":true,"return_col":null,"sigma_col":null,"cost_col":null},
///             "load": {"reader_py_path":"../ml-project/py/pl2tfrecord_reader.py","tfrecord_paths":["../tmp_data/mom_data.tfrecord"],"attr":"load_time_series_tfrecord_dataset","feature_spec":{"mean_price":"float32","ret_sma20":"float32","ret_sma50":"float32","ret_ema_small_pt1":"float32","ret_ema_large_pt6":"float32","ret_var10":"float32","ret_macd1s6l":"float32"},"label":"label","label_dtype":"float32","batch_size":5,"shuffle":false,"gzip":true,"include_cost":false},
///             "train": {"trainer_path":"../ml-project/models/mls_lstm_trainer.py","class":"MLSLSTMTrainer","attr":"train","time_steps":5,"input_dim":7,"val_split":0.1,"test_split":0.1,"epochs":10,"batch_size":null,"verbose":1,"shuffle_before_split":false,"seed":42,"save_every_epoch":false,"save_weights_only":false,"monitor":"val_loss","save_best_only":true,"run_name":"test"}}'
async fn tsmomnn_train_post(Json(req): Json<TrainReq>) -> impl IntoResponse {
    pyo3::prepare_freethreaded_python();
    match run_time_series_momentum_lstm(req.db, req.write, req.load, req.train).await {
        Ok((val)) => (StatusCode::OK, Json(json!(val))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status":"error","message": e.to_string()}))),
    }
}

// No non-server CLI mode here; this binary focuses on serving web requests only.
