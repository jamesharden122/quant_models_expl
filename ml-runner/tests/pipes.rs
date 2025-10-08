use chrono::NaiveDate;
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{FeatList, GlobalIndexes, MomFactor},
    polars_ops,
    surreal_queries::DbParams,
};
#[cfg(feature = "server")]
use ml_runner::pyexec::pydictstructs::{
    MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad, TseriesTfRecWrdsGlobalInd,
};
#[cfg(feature = "server")]
use ml_runner::{streaming_pipe, streaming_pipe_wrds_global_index, training_pipe};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString, PyTuple};
use std::collections::BTreeMap;
#[cfg(feature = "server")]
use wrds_io::finance_data_structs::crsp::finance_tickers;

#[tokio::test]
#[cfg(feature = "server")]
async fn streaming_pipe_smoke() {
    pyo3::prepare_freethreaded_python();
    let db_params = DbParams {
        url: "https://quant-platform-06cb0tpcrpsspao10de28go15s.aws-use1.surreal.cloud/rpc"
            .to_string(),
        user: "root".to_string(),
        pass: "root".to_string(),
        ns: "equities".to_string(),
        dbname: "historical".to_string(),
    };
    // Build kwargs for the Python writer (we'll augment with df/path/feature_cols inside streaming_pipe)
    let kwargs = TseriesTfRecBento {
        column_set: vec![
            "instrument_id".to_string(),
            "bin".to_string(),
            "t0".to_string(),
            "t1".to_string(),
            "mean_price".to_string(),
            "ret".to_string(),
        ],
        srt: Some(vec!["instrument_id".to_string(), "bin".to_string()]),
        exclude_cols: vec![
            "bin".to_string(),
            "t0".to_string(),
            "t1".to_string(),
            "ret".to_string(),
        ],
        out_path: Some("../tmp_data/mom_data.tfrecord".to_string()),
        query_params: (String::from("5m"), vec![8147, 11667]),
        writer_path: None,
        feature_names: Some(vec![FeatList::MomFactor(MomFactor)]),
        target_col: Some("ret".into()),
        return_col: None,
        time_col: None,
        attr: String::from("write_timeseries_tfrecord_from_polars"),
        sequence_length: 5,
        horizon: 1,
        stride: 1,
        group_col: Some("instrument_id".into()),
        compress: true,
        sigma_col: None,
        cost_col: None,
    };

    let res = streaming_pipe(db_params, kwargs).await;
    assert!(
        res.is_ok(),
        "streaming_pipe should complete without error: {res:?}"
    );
}

#[tokio::test]
#[cfg(feature = "server")]
async fn streaming_pipe_mls_sharpe_smoke() {
    pyo3::prepare_freethreaded_python();
    let db_params = DbParams {
        url: "https://quant-platform-06cb0tpcrpsspao10de28go15s.aws-use1.surreal.cloud/rpc"
            .to_string(),
        user: "root".to_string(),
        pass: "root".to_string(),
        ns: "equities".to_string(),
        dbname: "historical".to_string(),
    };
    // Build kwargs for the Python writer (we'll augment with df/path/feature_cols inside streaming_pipe)
    let kwargs = TseriesTfRecBento {
        column_set: vec![
            "instrument_id".to_string(),
            "bin".to_string(),
            "t0".to_string(),
            "t1".to_string(),
            "mean_price".to_string(),
            "ret".to_string(),
        ],
        srt: Some(vec!["instrument_id".to_string(), "bin".to_string()]),
        exclude_cols: vec![
            "bin".to_string(),
            "t0".to_string(),
            "t1".to_string(),
            "ret".to_string(),
            "sigma".to_string(),
            "cost".to_string(),
        ],
        out_path: Some("../tmp_data/mom_data_sharpe.tfrecord".to_string()),
        query_params: (String::from("5m"), vec![8147, 11667]),
        writer_path: None,
        feature_names: Some(vec![FeatList::MomFactor(MomFactor)]),
        target_col: None,
        return_col: Some("ret".into()),
        time_col: None,
        attr: String::from("write_timeseries_tfrecord_for_sharpe_from_polars"),
        sequence_length: 5,
        horizon: 1,
        stride: 1,
        group_col: Some("instrument_id".into()),
        compress: true,
        sigma_col: Some("sigma".into()),
        cost_col: Some("cost".into()),
    };

    let res = streaming_pipe(db_params, kwargs).await;
    assert!(
        res.is_ok(),
        "streaming_pipe should complete without error: {res:?}"
    );
}

// RAYON_NUM_THREADS=14 POLARS_MAX_THREADS=14 cargo test  --features server streaming_pipe_global_comp_smoke -- --no-capture --test-threads=1
#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_pipe_global_comp_smoke() {
    pyo3::prepare_freethreaded_python();
    let kwargs = TseriesTfRecWrdsGlobalInd {
        data_path: "../../data/raw_files/global_indexes_daily_25.parquet".to_string(),
        column_set: vec![
            "tic".to_string(),
            "datadate".to_string(),
            "prccd".to_string(),
            "indexid".to_string(),
            "dvpsxd".to_string(),
        ],
        srt: None,
        exclude_cols: vec![],
        out_path: Some("../tmp_data/mom_data.tfrecord".to_string()),
        query_params: Some((
            finance_tickers().unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
        )),
        attr: String::from("write_timeseries_tfrecord_for_sharpe_from_polars"),
        writer_path: None,
        feature_names: Some(vec![FeatList::GlobalIndexes(GlobalIndexes)]),
        target_col: Some("ret".into()),
        time_col: Some("datadate".into()),
        sequence_length: 5,
        horizon: 1,
        stride: 1,
        group_col: Some("tic".into()),
        compress: true,
        return_col: None,
        sigma_col: None,
        cost_col: None,
    };

    let res = streaming_pipe_wrds_global_index(kwargs).await;
    assert!(
        res.is_ok(),
        "streaming_pipe should complete without error: {res:?}"
    );
}

#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn training_pipe_smoke() {
    pyo3::prepare_freethreaded_python();

    let feature_spec: BTreeMap<String, String> = BTreeMap::from([
        //("instrument_id".to_string(), "int64".to_string()),
        //("bin".to_string(), "string".to_string()),
        //("t0".to_string(), "string".to_string()),
        //("t1".to_string(), "string".to_string()),
        ("mean_price".to_string(), "float32".to_string()),
        ("ret_sma20".to_string(), "float32".to_string()),
        ("ret_sma50".to_string(), "float32".to_string()),
        ("ret_ema_small_pt1".to_string(), "float32".to_string()),
        ("ret_ema_large_pt6".to_string(), "float32".to_string()),
        ("ret_var10".to_string(), "float32".to_string()),
        ("ret_macd1s6l".to_string(), "float32".to_string()),
        //("ret".to_string(), "float32".to_string()),
    ]);
    let inp_param = TseriesTfRecLoad {
        reader_py_path: "../ml-project/py/pl2tfrecord_reader.py".to_string(),
        tfrecord_paths: vec!["../tmp_data/mom_data_sharpe.tfrecord".to_string()],
        attr: "load_sharpe_timeseries_dataset".to_string(),
        feature_spec, // {"x1":"float32","x2":"int64",...}
        label: Some("label".to_string()),
        label_dtype: Some("float32".to_string()),
        batch_size: 5,
        shuffle: false,
        gzip: true,
        include_cost: None,
    };

    let trn_param = MlsLstmTrain {
        trainer_path: "../ml-project/models/mls_lstm_trainer.py".to_string(),
        class: "MLSLSTMTrainer".to_string(),
        attr: "train".to_string(),
        time_steps: Some(5),
        input_dim: Some(7),
        val_split: 0.1,
        test_split: 0.1,
        epochs: 10,
        batch_size: None,
        verbose: 1,
        shuffle_before_split: false,
        seed: Some(42),
        save_every_epoch: false,
        save_weights_only: false,
        monitor: "val_loss".to_string(),
        save_best_only: true,
        run_name: Some("test".to_string()),
    };

    let res = training_pipe(inp_param, trn_param).await.unwrap();
    println!("{:?}", res);
}

#[tokio::test]
#[cfg(feature = "server")]
async fn training_pipe_mls_sharpe_smoke() {
    pyo3::prepare_freethreaded_python();

    let feature_spec: BTreeMap<String, String> = BTreeMap::from([
        //("instrument_id".to_string(), "int64".to_string()),
        //("bin".to_string(), "string".to_string()),
        //("t0".to_string(), "string".to_string()),
        //("t1".to_string(), "string".to_string()),
        ("mean_price".to_string(), "float32".to_string()),
        ("ret_sma20".to_string(), "float32".to_string()),
        ("ret_sma50".to_string(), "float32".to_string()),
        ("ret_ema_small_pt1".to_string(), "float32".to_string()),
        ("ret_ema_large_pt6".to_string(), "float32".to_string()),
        //("ret_var10".to_string(), "float32".to_string()),
        ("ret_macd1s6l".to_string(), "float32".to_string()),
        //("ret".to_string(), "float32".to_string()),
    ]);
    let inp_param = TseriesTfRecLoad {
        reader_py_path: "../ml-project/py/pl2tfrecord_reader.py".to_string(),
        tfrecord_paths: vec!["../tmp_data/mom_data_sharpe.tfrecord".to_string()],
        attr: "load_sharpe_timeseries_dataset".to_string(),
        feature_spec, // {"x1":"float32","x2":"int64",...}
        label: None,
        label_dtype: None,
        batch_size: 5,
        shuffle: false,
        gzip: true,
        include_cost: Some(true),
    };

    let trn_param = MlsLstmTrain {
        trainer_path: "../ml-project/models/mls_lstm_trainer.py".to_string(),
        class: "MLSLSTMTrainer".to_string(),
        attr: "train".to_string(),
        time_steps: Some(5),
        input_dim: Some(6),
        val_split: 0.1,
        test_split: 0.1,
        epochs: 10,
        batch_size: None,
        verbose: 1,
        shuffle_before_split: false,
        seed: Some(42),
        save_every_epoch: false,
        save_weights_only: false,
        monitor: "val_loss".to_string(),
        save_best_only: true,
        run_name: Some("test".to_string()),
    };

    let res = training_pipe(inp_param, trn_param).await.unwrap();
    println!("{:?}", res);
}
