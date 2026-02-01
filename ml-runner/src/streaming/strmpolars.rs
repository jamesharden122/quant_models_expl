#[cfg(feature = "server")]
use crate::pyexec::pydictstructs::TsPolWrdsMarket;
use chrono::{Datelike, NaiveDate};
use dioxus::prelude::*;
#[cfg(feature = "server")]
use ml_backend::{featscreate::apply_by_names, listtobin::iterate_and_match_polars_df, polars_ops};
#[cfg(feature = "server")]
use polars::prelude::*;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
#[cfg(feature = "server")]
use wrds_io::{
    createdatasets::{CreateDuckFls, MergeDuckFls},
    finance_data_structs::{
        crsp, equity_factors, get_polars_df_from_sql, global_equities, usindexes, wdi,
        world_indices, DuckCrudModel,
    },
    instantiatedb::duckdbinst::{open_duck_db_from_file, start_duck_db, DbType},
};

#[cfg(feature = "server")]
fn df_to_cache_bytes(mut df: DataFrame) -> PolarsResult<Vec<u8>> {
    df.serialize_to_bytes()
}

#[cfg(feature = "server")]
fn df_from_cache_bytes(bytes: Vec<u8>) -> PolarsResult<DataFrame> {
    let mut cur = std::io::Cursor::new(bytes); // Cursor implements Read + Seek
    DataFrame::deserialize_from_reader(&mut cur)
}

#[cfg(feature = "server")]
fn save_cache(df: &DataFrame, path: &Path) -> PolarsResult<()> {
    let mut w = std::io::BufWriter::new(File::create(path)?);
    let mut df = df.clone();
    df.serialize_into_writer(&mut w)?;
    Ok(())
}

#[cfg(feature = "server")]
fn load_cache(path: &Path) -> PolarsResult<DataFrame> {
    let mut r = std::io::BufReader::new(File::open(path)?);
    DataFrame::deserialize_from_reader(&mut r)
}
//mdf {
//  MergeDuckFls,
//  duckdb table name to output to polars dataframe,
//  size of duckdb instance,
//  path of the duckdb database file,
//}
#[cfg(feature = "server")]
pub async fn streaming_pipe_merge_wrds_duck_polars(
    cdf: Option<(CreateDuckFls, &Path)>,
    mdf: Option<MergeDuckFls>,
    plrs_cache_file: Option<impl AsRef<Path>>,
    plrs_cache: Option<Vec<u8>>,
    max_mem: &str,
    thread_count: i64,
    df_path: &Path,
    table: &str,
) -> Result<DataFrame, ServerFnError> {
    let cache_path = plrs_cache_file.as_ref().map(|p| p.as_ref().to_path_buf());

    if let Some(c) = plrs_cache {
        return df_from_cache_bytes(c).map_err(|e| ServerFnError::new(format!("{e:?}")));
    }

    if let Some(cf) = cache_path.as_ref() {
        if cf.exists() {
            return load_cache(cf).map_err(|e| ServerFnError::new(format!("{e:?}")));
        }
    }

    if let Some((create, out_dir)) = cdf {
        create
            .create_db_files(out_dir)
            .await
            .map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    }

    let merged_db_path = match mdf {
        Some(merge) => merge
            .merge_db_files(df_path, max_mem, thread_count)
            .await
            .map_err(|e| ServerFnError::new(format!("{e:?}")))?,
        None => df_path.to_path_buf(),
    };

    let conn = open_duck_db_from_file(
        merged_db_path
            .to_str()
            .ok_or_else(|| ServerFnError::new("merged duckdb path is not valid utf-8"))?,
        max_mem,
        thread_count,
    )
    .await
    .map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    let sql = format!("SELECT * FROM {}", table);
    let mut chunks = get_polars_df_from_sql(&conn, sql.as_str())
        .await
        .map_err(|e| ServerFnError::new(format!("{e:?}")))?;

    if chunks.is_empty() {
        return Err(ServerFnError::new("empty query result from merged duckdb"));
    }

    let mut df = chunks.remove(0);

    for c in chunks {
        df.vstack_mut(&c)
            .map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    }

    if let Some(cf) = cache_path.as_ref() {
        save_cache(&df, cf).map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    }

    Ok(df)
}
#[cfg(feature = "server")]
pub async fn streaming_pipe_wrds_duck_polars(
    kwargs_struct: TsPolWrdsMarket,
    dbt: DbType,
    load_frm_parq: bool,
    tbl: &str,
    pre_features_filter: Option<Vec<&str>>,
    to_bin: bool,
) -> Result<DataFrame, ServerFnError> {
    //************************************************//
    //Read data from parquet file to the duck database//
    //************************************************//
    let conn = Arc::new(Mutex::new(if load_frm_parq {
        start_duck_db("4GB", 14)
            .await
            .expect("duckdb in-memory should start")
    } else {
        open_duck_db_from_file(kwargs_struct.data_path.as_str(), "4GB", 14)
            .await
            .expect("duckdb file should open")
    }));

    if load_frm_parq {
        dbt.ingest(conn.clone(), kwargs_struct.data_path.as_str())
            .await
            .unwrap();
    }

    let sql = format!("DESCRIBE {}", tbl);
    let conn_cloned = conn.clone();
    let guard = conn_cloned.lock().unwrap();
    let mut stmt = guard.prepare(&sql).unwrap();
    let _rows = stmt.query([]).unwrap();
    //*************************************************//
    //Filter data base)d on the tickers and a date tuple and convert to dataframe//
    //*************************************************//
    let tvec = kwargs_struct.query_params.clone().unwrap().0;
    let (d1, d2) = kwargs_struct
        .query_params
        .as_ref()
        .map(|(_, d1, d2)| (*d1, *d2))
        .unwrap();
    let data: Vec<polars::frame::row::Row> = match dbt {
        DbType::GlobalDailyIndex => {
            crsp::GlobalDailyIndex::read_gdi_batch(conn, "tic".to_string(), tvec, (d1, d2))
                .await
                .unwrap()
        }
        DbType::GlobalRets => world_indices::GlobalRets::read_range(conn, (d1, d2))
            .await
            .unwrap(),
        DbType::UsMarket => usindexes::UsMarketIndex::read_range(conn, (d1, d2))
            .await
            .unwrap(),
        DbType::GlobalEquities => global_equities::GlobalEquities::read_range(conn, (d1, d2))
            .await
            .unwrap(),
        DbType::GlobalEquitiesMonthly => {
            global_equities::GlobalEquitiesMonthly::read_range(conn.clone(), (d1, d2))
                .await
                .unwrap()
        }
        DbType::WdiWide => {
            let countries: Vec<&str> = tvec.iter().map(|s| s.as_str()).collect();
            wdi::WdiWide::read_indicator_countries(
                conn.clone(),
                "wdi_wide",
                "NY.GDP.MKTP.CD",
                (d1.year(), d2.year()),
                &countries,
            )
            .await
            .unwrap()
        }
        DbType::EquityFactorsMonthly => {
            equity_factors::EquityFactorsMonthly::read_range(conn.clone(), (d1, d2))
                .await
                .unwrap()
        }
        DbType::GlobalFundQtrly => {
            return Err(ServerFnError::new(
                "DbType::GlobalFundQtrly is not supported by streaming_pipe_wrds_duck_polars",
            ))
        }
        other => {
            return Err(ServerFnError::new(format!(
                "DbType::{other:?} is not supported by streaming_pipe_wrds_duck_polars"
            )))
        }
    };
    let mut df = DataFrame::from_rows_and_schema(&data, &kwargs_struct.polars_schema).unwrap();
    df = match pre_features_filter {
        Some(cols) => polars_ops::utils::drop_columns(df, cols).unwrap(),
        _ => df,
    };
    //**************************************************//
    //Apply feature engineering functions by there names//
    //**************************************************//
    if let Some(names) = kwargs_struct.feature_names {
        df = apply_by_names(df, names)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;
    }
    println!("streamed shape: {:?}", df.shape());
    println!("{:?}", df);
    if to_bin == true {
        iterate_and_match_polars_df("../../../tmp_data", df.clone()).await;
    }
    Ok(df)
}
