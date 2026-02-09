use chrono::NaiveDate;
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{
        apply_by_names,
        corrmatrix::{CorrFactor, Method},
        est_returns::Returns,
        globalindexes::GlobalIndexes,
        mean_by_groups::MeanByGroups,
        momindexes::MomFactor,
        prod_by_groups::ProdByGroups,
        sharpes::{RarRatios, RatioTypes},
        DynamicGroupCfg, FeatList, GroupingCfg, PrcDataSource, TimeFrame,
    },
    listtobin, polars_ops,
    surreal_queries::DbParams,
};
#[cfg(feature = "server")]
use ml_runner::pyexec::pydictstructs::{MlsLstmTrain, TsPolWrdsMarket, TseriesTfRecBento, TseriesTfRecLoad, TseriesTfRecWrdsMarket};
#[cfg(feature = "server")]
use ml_runner::{
    pipelines::bank_pipeline::data_pipeline::{create_daily_dataset, create_rar},
    streaming::streaming_pipe,
    streaming::streaming_pipe_wrds_duck,
    streaming::strmpolars::{streaming_pipe_merge_wrds_duck_polars, streaming_pipe_wrds_duck_polars},
    training::training_pipe,
};
use polars::prelude::*;
use std::{collections::BTreeMap, fs::File, time::Instant};

#[cfg(feature = "server")]
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use wrds_io::{
    createdatasets::{
        usbanks::{full_bank_ds, BankCrspDly, BankCrspPaths},
        CreateDuckFls, MergeDuckFls,
    },
    finance_data_structs::{
        crsp::{finance_tickers, GlobalDailyIndex},
        usindexes::UsMarketIndex,
        world_indices::GlobalRets,
        DuckCrudModel, ToPolars,
    },
    instantiatedb::duckdbinst::DbType,
};

#[cfg(feature = "server")]
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    if path == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}
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
        exclude_cols: vec!["bin".to_string(), "t0".to_string(), "t1".to_string(), "ret".to_string()],
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
    assert!(res.is_ok(), "streaming_pipe should complete without error: {res:?}");
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
        exclude_cols: vec!["bin".to_string(), "t0".to_string(), "t1".to_string(), "ret".to_string(), "sigma".to_string(), "cost".to_string()],
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
    assert!(res.is_ok(), "streaming_pipe should complete without error: {res:?}");
}

#[tokio::test]
#[cfg(feature = "server")]
async fn full_bank_ds_from_parquet_smoke() {
    let full_data_dir = expand_tilde("~/Dropbox/Desktop/banking/data/full_data");
    let parquet_path = full_data_dir.join("gai_bhc_yr.parquet");
    assert!(parquet_path.exists(), "missing parquet file at `{}`", parquet_path.display());

    let cache_file = full_data_dir.join("full_bank_data_cache.plrs");

    let df = full_bank_ds(
        Some(parquet_path.to_str().expect("parquet_path is not valid utf-8")),
        Some(full_data_dir.to_str().expect("db_dir is not valid utf-8")),
        &cache_file,
        None,
        None,
    )
    .await
    .expect("full_bank_ds should load from parquet or db");
    assert!(df.height() > 0, "expected non-empty DataFrame");
    assert!(cache_file.exists(), "expected cache file to be written");

    let df2 = full_bank_ds(None, Some(full_data_dir.to_str().expect("db_dir is not valid utf-8")), &cache_file, None, None)
        .await
        .expect("full_bank_ds should load from cache");
    println!("shape: {:?}, column names: {:?}", df.shape(), df2.get_column_names());
}

// RAYON_NUM_THREADS=14 POLARS_MAX_THREADS=14 cargo test  --features server streaming_pipe_global_comp_smhttps://docs.rs/ort-candle/0.1.0+0.8/ort_candle/oke -- --no-capture --test-threads=1
#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_pipe_global_comp_smoke() {
    pyo3::prepare_freethreaded_python();
    let kwargs = TseriesTfRecWrdsMarket {
        data_path: "../../data/raw_files/parqueut/country_returns_wide.parquet".to_string(),
        column_set: vec!["tic".to_string(), "datadate".to_string(), "prccd".to_string(), "indexid".to_string(), "dvpsxd".to_string()],
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
        query_params: Some((finance_tickers().unwrap(), NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2025, 10, 2).unwrap())),
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
    assert!(res.is_ok(), "streaming_pipe should complete without error: {res:?}");
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
        query_params: Some((finance_tickers().unwrap(), NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2025, 10, 2).unwrap())),
        attr: String::from("write_timeseries_tfrecord_for_sharpe_from_polars"),
        writer_path: None,
        feature_names: Some(vec![FeatList::CorrFactor(CorrFactor { method: Method::Pearson })]),
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
    assert!(res.is_ok(), "streaming_pipe should complete without error: {res:?}");
}

#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn streaming_pipe_wrldind_mrk_ind_merge_smoke() {
    pyo3::prepare_freethreaded_python();
    let kwargs_gind = TsPolWrdsMarket {
        data_path: "/home/yakaman/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/country_indexes/monthly/country_returns_wide.parquet".to_string(),
        column_set: vec![],
        srt: None,
        query_params: Some((finance_tickers().unwrap(), NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2025, 10, 2).unwrap())),
        feature_names: None,
        polars_schema: GlobalRets::schema(),
    };

    let kwargs_mind = TsPolWrdsMarket {
        data_path: "/home/yakaman/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/crsp_ciz_sample/market_index/monthly/market_indexes_monthly.parquet".to_string(),
        column_set: vec![],
        srt: None,
        query_params: Some((finance_tickers().unwrap(), NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2025, 10, 2).unwrap())),
        feature_names: None,
        polars_schema: UsMarketIndex::schema(),
    };

    let res_global = streaming_pipe_wrds_duck_polars(kwargs_gind, DbType::GlobalRets, true, "global_sec_indexes_daily", None, false).await;
    let res_market = streaming_pipe_wrds_duck_polars(kwargs_mind, DbType::UsMarket, true, "us_market_indexes_daily", None, false).await;
    let mut dfm = res_market.unwrap();
    dfm = dfm.lazy().with_columns([col("date").cast(DataType::Date).dt().truncate(lit("1mo")).alias("date")]).collect().unwrap();
    let mut dfg = res_global.unwrap();
    dfg = dfg.lazy().with_columns([col("date").cast(DataType::Date).dt().truncate(lit("1mo")).alias("date")]).collect().unwrap();

    let dfg_drop_cols: Vec<&str> = vec![];
    let dfm_drop_cols: Vec<&str> = vec!["spindx", "totcnt", "totval", "usdcnt", "usdval"];
    dfg = polars_ops::utils::drop_columns(dfg, dfg_drop_cols).unwrap();
    dfm = polars_ops::utils::drop_columns(dfm, dfm_drop_cols).unwrap();
    println!("market df: {:?}", dfg.head(Some(30)));
    println!("global df: {:?}", dfm.head(Some(30)));
    println!("market column names {:?}", dfm.get_column_names());
    let mut out = dfg.left_join(&dfm, ["date"], ["date"]).unwrap();
    println!("shape: {:?}", out.shape());
    //listtobin::iterate_and_match_polars_df("../tmp_data", out.clone()).await;
    out = polars_ops::utils::drop_columns(out, vec!["date"]).unwrap();
    out = apply_by_names(out, vec![FeatList::CorrFactor(CorrFactor { method: Method::Pearson })]).await.unwrap();

    println!("global df: {:?}", out.head(Some(30)));

    // Write merged DataFrame to CSV at crate root

    let mut f = File::create("./world_market_merged_monthly.csv").expect("create csv file at repo root");
    let mut out_to_write = out.clone();
    CsvWriter::new(&mut f).finish(&mut out_to_write).expect("write merged csv");
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

#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn merge_full_ds_possibly_overwrite() {
    use wrds_io::instantiatedb::polars_utils::save_cache;

    let full_data_dir = expand_tilde("~/Dropbox/Desktop/banking/data/full_data");
    let parquet_path = full_data_dir.join("gai_bhc_yr.parquet");
    let cache_file = full_data_dir.join("full_bank_data_cache.plrs");

    assert!(
        cache_file.exists() || parquet_path.exists(),
        "missing full-bank inputs: cache `{}` or parquet `{}`",
        cache_file.display(),
        parquet_path.display()
    );
    let fds = full_bank_ds(parquet_path.to_str(), Some(full_data_dir.to_str().expect("db_dir is not valid utf-8")), &cache_file, None, None)
        .await
        .expect("full_bank_ds should load from cache");

    let bhck_crsp = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/bank_regulatory/bhck_crsp_link/bhck_crsp_link_file.parquet");
    let crsp_mthly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/crsp/us_bhc/monthly_usbank_crsp.parquet");
    let ff_mthly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/factors/us/monthly_ff_factors.parquet");
    let crsp_dly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/crsp/us_bhc/daily_usbank_crsp.parquet");
    let ff_dly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/factors/us/daily_ff_factors.parquet");

    for p in [&bhck_crsp, &crsp_mthly, &ff_mthly, &crsp_dly, &ff_dly] {
        assert!(p.exists(), "missing parquet input: {}", p.display());
    }

    let base_dir = Path::new("/home/yakaman/Dropbox/Desktop/banking/data/equities");
    let duckdb_dir = base_dir.join("duckdb");
    let polars_cache_dir = base_dir.join("polars_cache");
    let duckdb_source_dir = duckdb_dir.join("source");
    let duckdb_merge_dir = duckdb_dir.join("merged");

    std::fs::create_dir_all(&duckdb_source_dir).expect("create duckdb source dir");
    std::fs::create_dir_all(&duckdb_merge_dir).expect("create duckdb merged dir");
    std::fs::create_dir_all(&polars_cache_dir).expect("create polars cache dir");

    let cache_file = polars_cache_dir.join("bank_securities_dly.plrs");

    let create = BankCrspPaths {
        bhck_crsp: Some(vec![bhck_crsp.to_string_lossy().to_string()]),
        crsp_mthly: Some(vec![crsp_mthly.to_string_lossy().to_string()]),
        ff_mthly: Some(vec![ff_mthly.to_string_lossy().to_string()]),
        crsp_dly: Some(vec![crsp_dly.to_string_lossy().to_string()]),
        ff_dly: Some(vec![ff_dly.to_string_lossy().to_string()]),
    };

    let bhck_crsp_duck = duckdb_source_dir.join("bhck_crsp_link.duckdb");
    let us_crsp_dly_duck = duckdb_source_dir.join("us_crsp_dly.duckdb");
    let ff_dly_duck = duckdb_source_dir.join("fama_french_daily.duckdb");

    let merge = BankCrspDly {
        bhck_crsp: bhck_crsp_duck.to_string_lossy().to_string(),
        crsp_dly: us_crsp_dly_duck.to_string_lossy().to_string(),
        ff_dly: ff_dly_duck.to_string_lossy().to_string(),
    };

    let weekly_cfg = TimeFrame::Weekly.get_dyn_grp_cfg(PrcDataSource::Crsp);
    let annual_cfg = DynamicGroupCfg {
        index_column: "date".into(),
        every: String::from("1y"),
        period: String::from("1y"),
        offset: String::from("0d"),
        label: String::from("dp"),
        include_boundaries: false,
        closed_window: String::from("right"),
        start_by: String::from("wb"),
    };

    let [_df_daily, df_ret]: [DataFrame; 2] = create_daily_dataset(Some(create), Some(merge), weekly_cfg, "20GB", duckdb_source_dir, cache_file.clone(), duckdb_merge_dir)
        .await
        .expect("create_daily_data should succeed");

    let df_rar = create_rar(df_ret, annual_cfg).await.expect("create_rar should succeed");

    let fds_typed = fds
        .lazy()
        .with_columns([col("entity").cast(DataType::Int64), col("year").cast(DataType::Int64)])
        .filter(col("entity").is_not_null().and(col("year").is_not_null()))
        .collect()
        .expect("cast fds entity/year to int64");
    println!("shape: {:?}", fds_typed.clone().shape());
    let rar_typed = df_rar
        .lazy()
        .with_columns([col("rssd9001").cast(DataType::Int64).alias("entity"), col("year").cast(DataType::Int64)])
        .filter(col("entity").is_not_null().and(col("year").is_not_null()))
        .collect()
        .expect("cast rar entity/year to int64");

    let merged = fds_typed
        .lazy()
        .join(rar_typed.lazy(), [col("year"), col("entity")], [col("year"), col("entity")], JoinArgs::new(JoinType::Left))
        .drop(by_name(["year_right", "entity_right"], false))
        .collect()
        .expect("merge fds with rar output");

    println!("shape: {:?}", merged.clone().shape());
    save_cache(&merged, &cache_file.as_path());
}

// Mirrors `words_db/tests/createdatasets_enum_usbank_crsp_dly_polars.rs:create_and_merge_usbank_crsp_dly_enum_to_polars_df`,
// but runs through `ml_runner::streaming::strmpolars::streaming_pipe_merge_wrds_duck_polars`.
#[cfg(feature = "server")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn strm_pipe_merge_wrds_duck_plrs_smoke() {
    use ml_backend::{featscreate::sharpes::RarRatios, polars_ops::factor_est::ApplyExprs};

    let bhck_crsp = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/bank_regulatory/bhck_crsp_link/bhck_crsp_link_file.parquet");
    let crsp_mthly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/crsp/us_bhc/monthly_usbank_crsp.parquet");
    let ff_mthly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/factors/us/monthly_ff_factors.parquet");
    let crsp_dly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/crsp/us_bhc/daily_usbank_crsp.parquet");
    let ff_dly = expand_tilde("~/Dropbox/Desktop/tesero-sol/software_development/trading/data/raw_files/parquet/factors/us/daily_ff_factors.parquet");

    for p in [&bhck_crsp, &crsp_mthly, &ff_mthly, &crsp_dly, &ff_dly] {
        assert!(p.exists(), "missing parquet input: {}", p.display());
    }
    let tm = Instant::now();

    let base_dir = Path::new("/home/yakaman/Dropbox/Desktop/banking/data/equities");
    let duckdb_dir = base_dir.join("duckdb");
    let polars_cache_dir = base_dir.join("polars_cache");
    let duckdb_source_dir = duckdb_dir.join("source");
    let duckdb_merge_dir = duckdb_dir.join("merged");

    std::fs::create_dir_all(&duckdb_source_dir).expect("create duckdb source dir");
    std::fs::create_dir_all(&duckdb_merge_dir).expect("create duckdb merged dir");
    std::fs::create_dir_all(&polars_cache_dir).expect("create polars cache dir");

    let cache_file = polars_cache_dir.join("bank_securities_dly.plrs");

    let create = CreateDuckFls::UsBankCrsp(BankCrspPaths {
        bhck_crsp: Some(vec![bhck_crsp.to_string_lossy().to_string()]),
        crsp_mthly: Some(vec![crsp_mthly.to_string_lossy().to_string()]),
        ff_mthly: Some(vec![ff_mthly.to_string_lossy().to_string()]),
        crsp_dly: Some(vec![crsp_dly.to_string_lossy().to_string()]),
        ff_dly: Some(vec![ff_dly.to_string_lossy().to_string()]),
    });

    let bhck_crsp_duck = duckdb_source_dir.join("bhck_crsp_link.duckdb");
    let us_crsp_dly_duck = duckdb_source_dir.join("us_crsp_dly.duckdb");
    let ff_dly_duck = duckdb_source_dir.join("fama_french_daily.duckdb");

    let merge = MergeDuckFls::UsBankCrspDly(BankCrspDly {
        bhck_crsp: bhck_crsp_duck.to_string_lossy().to_string(),
        crsp_dly: us_crsp_dly_duck.to_string_lossy().to_string(),
        ff_dly: ff_dly_duck.to_string_lossy().to_string(),
    });
    println!("Create Duck Files Time: {:?}", tm.elapsed());
    let opts = DynamicGroupCfg {
        index_column: "date".into(),
        every: String::from("1y"),
        period: String::from("1y"),
        offset: String::from("0d"),
        label: String::from("dp"),
        include_boundaries: false,
        closed_window: String::from("right"),
        start_by: String::from("wb"),
    };

    let mut df = streaming_pipe_merge_wrds_duck_polars(
        Some((create, &duckdb_source_dir)),
        Some(merge),
        Some(cache_file.clone()),
        None,
        "20GB",
        10,
        &duckdb_merge_dir,
        "bank_securities_dly",
        None,
    )
    .await
    .expect("merge+polars should succeed");
    println!("{:?}", df.get_column_names());
    df = polars_ops::utils::add_year_month_week_cols(
        df.lazy().select([
            col("rssd9001").cast(DataType::Int64),
            col("permno").cast(DataType::Int64),
            col("permco").cast(DataType::Int64),
            col("date"),
            col("ret"),
            col("mktrf"),
            col("rf"),
            col("prc"),
        ]),
        col("date"),
    )
    .with_columns([col("prc").abs()])
    .collect()
    .unwrap();

    /*
    let mbg_spec = vec![FeatList::MeanByGrp(MeanByGroups {
        grpby: Some(vec![String::from("permco"), String::from("permno")]),
        dynamic_group: Some(TimeFrame::Weekly.get_dyn_grp_cfg(PrcDataSource::Crsp)),
        columns: vec!["mktrf".to_string(), "rf".to_string()],
    })];
    */
    let pbg_spec = vec![FeatList::ProdByGrp(ProdByGroups {
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        dynamic_group: Some(TimeFrame::Weekly.get_dyn_grp_cfg(PrcDataSource::Crsp)),
        columns: vec!["mktrf".to_string(), "rf".to_string()],
    })];

    let ret_spec = vec![FeatList::Returns(Returns {
        source: PrcDataSource::Crsp,
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        grouping: GroupingCfg::Dynamic(TimeFrame::Weekly.get_dyn_grp_cfg(PrcDataSource::Crsp)),
    })];

    let rar_spec = vec![FeatList::Sharpe(RarRatios {
        ratios: [Some(RatioTypes::Sharpe), None, None], //Some(RatioTypes::Sortino), None],
        source: PrcDataSource::Crsp,
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        grouping: Some(GroupingCfg::Dynamic(opts)),
    })];

    let mut df_ret = apply_by_names(df.clone(), ret_spec).await.unwrap();
    let df_pbg = apply_by_names(df.clone(), pbg_spec).await.unwrap();
    df_ret = df_ret
        .lazy()
        .join(
            df_pbg.lazy(),
            [col("rssd9001"), col("permco"), col("permno"), col("date")],
            [col("rssd9001"), col("permco"), col("permno"), col("date")],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([col("mktrf_cr").alias("mktrf"), col("rf_cr").alias("rf")])
        .drop(by_name(["mktrf_cr", "rf_cr", "rssd9001_right", "permno_right", "permco_right"], false))
        .collect()
        .unwrap();
    let df_rar = apply_by_names(df_ret.clone(), rar_spec)
        .await
        .unwrap()
        .lazy()
        .with_columns([col("date").dt().year().alias("year")])
        .collect()
        .unwrap();

    //let full_df = df.clone();
    //let df = df.drop("ret")
    println!("Read Cache DataFrame Time: {:?}", tm.elapsed());
    println!("Main Dataframe Shape{:?}", df.shape());
    println!("Returns Dataframe Shape{:?}", df_ret.shape());
    println!("Risk Adjusted returns Dataframe shape{:?}", df_rar.shape());
    println!("{:?}", df.head(Some(30)));
    println!("{:?}", df_ret.head(Some(30)));
    println!("{:?}", df_rar.head(Some(30)));
}
