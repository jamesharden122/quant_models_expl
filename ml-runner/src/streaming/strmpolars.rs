#[cfg(feature = "server")]
use crate::pyexec::pydictstructs::TsPolWrdsMarket;
use dioxus::prelude::*;
#[cfg(feature = "server")]
use ml_backend::{featscreate::apply_by_names, polars_ops};
#[cfg(feature = "server")]
use polars::prelude::*;
use std::sync::Arc;
#[cfg(feature = "server")]
use wrds_io::{
    finance_data_structs::crsp,
    finance_data_structs::usindexes,
    finance_data_structs::world_indices,
    instantiatedb::duckdbinst::{open_duck_db_from_file, start_duck_db, DbType},
};
#[cfg(feature = "server")]
pub async fn streaming_pipe_wrds_duck_polars(
    kwargs_struct: TsPolWrdsMarket,
    dbt: DbType,
    load_frm_parq: bool,
    tbl: &str,
    pre_features_filter: Option<Vec<&str>>,
) -> Result<DataFrame, ServerFnError> {
    //************************************************//
    //Read data from parquet file to the duck database//
    //************************************************//
    let conn = if load_frm_parq {
        let conn = Arc::new(
            start_duck_db("4GB", 14)
                .await
                .expect("duckdb in-memory should start"),
        );
        dbt.ingest(conn.clone(), kwargs_struct.data_path.as_str())
            .await
            .unwrap();
        conn
    } else {
        Arc::new(
            open_duck_db_from_file(kwargs_struct.data_path.as_str(), "4GB", 14)
                .await
                .expect("duckdb in-memory should start"),
        )
    };
    let mut stmt = conn.prepare(format!("DESCRIBE  {}", tbl).as_str()).unwrap(); //
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
            crsp::GlobalDailyIndex::read_gdi_batch(conn.clone(), "tic".to_string(), tvec, (d1, d2))
                .await
                .unwrap()
        }
        DbType::GlobalRets => world_indices::GlobalRets::read_range(conn.clone(), (d1, d2))
            .await
            .unwrap(),
        DbType::UsMarket => usindexes::UsMarketIndex::read_range(conn.clone(), (d1, d2))
            .await
            .unwrap(),
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
    Ok(df)
}
