use chrono::NaiveDate;
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{
        apply_by_names,
        corrmatrix::{CorrFactor, Method},
        globalindexes::GlobalIndexes,
        momindexes::MomFactor,
        FeatList,
    },
    polars_ops,
    surreal_queries::DbParams,
};
#[cfg(feature = "server")]
use ml_runner::pyexec::pydictstructs::{
    MlsLstmTrain, TsPolWrdsMarket, TseriesTfRecBento, TseriesTfRecLoad, TseriesTfRecWrdsMarket,
};
#[cfg(feature = "server")]
use ml_runner::{
    streaming::streaming_pipe, streaming::streaming_pipe_wrds_duck,
    streaming::strmpolars::streaming_pipe_wrds_duck_polars, training::training_pipe,
};
use polars::prelude::*;
use std::collections::BTreeMap;
#[cfg(feature = "server")]
use wrds_io::{
    finance_data_structs::{
        crsp::{finance_tickers, GlobalDailyIndex},
        usindexes::UsMarketIndex,
        world_indices::GlobalRets,
        ToPolars,
    },
    instantiatedb::duckdbinst::DbType,
};

#[tokio::test]
#[cfg(feature = "server")]
async fn streaming_pipe_smoke() {
    pyo3::prepare_freethreaded_python();
    let db_params = DbParams {
        url: "".to_string(),
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
        url: "".to_string(),
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
    let kwargs = TseriesTfRecWrdsMarket {
        data_path: "../../data/raw_files/parqueut/country_returns_wide.parquet".to_string(),
        column_set: vec![
            "tic".to_string(),
            "datadate".to_string(),
            "prccd".to_string(),
            "indexid".to_string(),
            "dvpsxd".to_string(),
        ],
        srt: None,
        exclude_cols: vec![
            "conm".to_string(),
            "indextype".to_string(),
            "gvkeyx".to_string(),
            "indexid".to_string(),
            "prccddiv".to_string(),
            "prccddivn".to_string(),
            "prchd".to_string(),
            "prcld".to_string(),
            "prccd_lag1".to_string(),
            "newnum".to_string(),
            "oldnum".to_string(),
        ],
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
        polars_schema: GlobalDailyIndex::schema(),
    };
    let res = streaming_pipe_wrds_duck(
        kwargs,
        DbType::GlobalDailyIndex,
        true,
        Some("../../data/raw_files/parqueut/global_indexes_daily_25.parquet"),
        "global_indexes_daily",
        None,
    )
    .await;
    assert!(
        res.is_ok(),
        "streaming_pipe should complete without error: {res:?}"
    );
}

#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_pipe_world_indices_smoke() {
    use wrds_io::finance_data_structs::crsp::GlobalDailyIndex;

    pyo3::prepare_freethreaded_python();
    let kwargs = TseriesTfRecWrdsMarket {
        data_path: "../../data/raw_files/parqueut/country_returns_wide.parquet".to_string(),
        column_set: vec![],
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
        feature_names: Some(vec![FeatList::CorrFactor(CorrFactor {
            method: Method::Pearson,
        })]),
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
        polars_schema: GlobalRets::schema(),
    };
    let res = streaming_pipe_wrds_duck(
        kwargs,
        DbType::GlobalRets,
        true,
        Some("../../data/raw_files/parqueut/country_returns_wide.parquet"),
        "global_sec_indexes_daily",
        Some(vec!["date"]),
    )
    .await;
    assert!(
        res.is_ok(),
        "streaming_pipe should complete without error: {res:?}"
    );
}

#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_pipe_wrldind_mrk_ind_merge_smoke() {
    pyo3::prepare_freethreaded_python();
    let kwargs_gind = TsPolWrdsMarket {
        data_path: "../../data/raw_files/parqueut/country_returns_wide.parquet".to_string(),
        column_set: vec![],
        srt: None,
        query_params: Some((
            finance_tickers().unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
        )),
        feature_names: None,
        polars_schema: GlobalRets::schema(),
    };

    let kwargs_mind = TsPolWrdsMarket {
        data_path: "../../data/raw_files/parqueut/crsp_ciz_sample/market_index/market_indexes_daily.parquet".to_string(),
        column_set: vec![],
        srt: None,
        query_params: Some((
            finance_tickers().unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
        )),
        feature_names:None,
        polars_schema: UsMarketIndex::schema(),
    };

    let res_global = streaming_pipe_wrds_duck_polars(
        kwargs_gind,
        DbType::GlobalRets,
        true,
        "global_sec_indexes_daily",
        None,
    )
    .await;
    let res_market = streaming_pipe_wrds_duck_polars(
        kwargs_mind,
        DbType::UsMarket,
        true,
        "us_market_indexes_daily",
        None,
    )
    .await;
    let mut dfm = res_market.unwrap();
    let mut dfg = res_global.unwrap();
    let dfg_drop_cols: Vec<&str> = vec![];
    let dfm_drop_cols: Vec<&str> = vec!["spindx", "totcnt", "totval", "usdcnt", "usdval"];
    dfg = polars_ops::utils::drop_columns(dfg, dfg_drop_cols).unwrap();
    dfm = polars_ops::utils::drop_columns(dfm, dfm_drop_cols).unwrap();
    println!("market df: {:?}", dfg.head(Some(30)));
    println!("global df: {:?}", dfm.head(Some(30)));
    println!("market column names {:?}", dfm.get_column_names());
    let mut out = dfg.left_join(&dfm, ["date"], ["date"]).unwrap();
    out = polars_ops::utils::drop_columns(out, vec!["date"]).unwrap();
    out = apply_by_names(
        out,
        vec![FeatList::CorrFactor(CorrFactor {
            method: Method::Pearson,
        })],
    )
    .await
    .unwrap();

    println!("global df: {:?}", out.head(Some(30)));

    // Write merged DataFrame to CSV at crate root
    use polars::prelude::CsvWriter;
    use std::fs::File;
    let mut f = File::create("./world_market_merged.csv").expect("create csv file at repo root");
    let mut out_to_write = out.clone();
    CsvWriter::new(&mut f)
        .finish(&mut out_to_write)
        .expect("write merged csv");
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
