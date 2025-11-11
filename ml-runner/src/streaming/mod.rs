pub mod strmpolars;
use crate::pyexec;
#[cfg(feature = "server")]
use crate::pyexec::pydictstructs::{TseriesTfRecBento, TseriesTfRecWrdsMarket};
use crate::surr_queries;
use dioxus::prelude::*;
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::apply_by_names, polars_ops, surreal_queries, surreal_queries::DbParams,
};
#[cfg(feature = "server")]
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::types::PyDataFrame;
use std::{path::Path, sync::Arc};
#[cfg(feature = "server")]
use wrds_io::{
    finance_data_structs::crsp,
    finance_data_structs::usindexes,
    finance_data_structs::world_indices,
    instantiatedb::duckdbinst::{open_duck_db_from_file, start_duck_db, DbType},
};

#[cfg(feature = "server")]
pub async fn streaming_pipe(
    db_params: DbParams,
    kwargs_struct: TseriesTfRecBento, // Kwargs passed to the Python function (we will augment with df/feature_cols/path)
) -> Result<(), ServerFnError> {
    let kw_temp = kwargs_struct.clone();
    let db = surreal_queries::make_db(
        db_params.url.as_str(),
        db_params.user.as_str(),
        db_params.pass.as_str(),
        db_params.ns.as_str(),
        db_params.dbname.as_str(),
    )
    .await?;
    let mut df: DataFrame = surr_queries::query_feature_bin_demo(
        &db,
        kwargs_struct
            .column_set
            .iter()
            .map(|s| s.as_str())
            .collect(),
        kwargs_struct.query_params.0, //bin_size
        kwargs_struct.query_params.1, //ints_ids
        kwargs_struct.srt,
    )
    .await?;
    // Apply feature engineers by name (implemented in bento_queries::featscreate)
    if let Some(names) = kwargs_struct.feature_names {
        df = apply_by_names(df, names)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;
    }
    println!("streamed shape: {:?}", df.shape());
    println!("{:?}", df);
    // Compute feature columns by excluding reserved columns from the fetched set
    let feature_cols: Vec<String> = df
        .get_column_names_str()
        .iter()
        .cloned()
        .filter(|c| !kwargs_struct.exclude_cols.iter().any(|ex| ex == c))
        .map(|c| c.to_string())
        .collect();
    //Set Python objects
    let py_df = PyDataFrame(df);
    let writer = kwargs_struct
        .writer_path
        .unwrap_or_else(|| "../ml-project/py/pl2tfrecord_writer.py".to_string());
    let out_path = kwargs_struct
        .out_path
        .unwrap_or_else(|| "../tmp_data/data.tfrecord".to_string());
    // Augment provided kwargs with df/path/feature_cols
    let kwargs = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kw = kw_temp.to_pydict(py)?;
        kw.set_item("df", py_df)?;
        // Only set path if caller didn't provide one
        if !kw.contains("path")? {
            kw.set_item("path", out_path)?;
        }
        if !kw.contains("feature_cols")? {
            kw.set_item("feature_cols", feature_cols)?;
        }
        Ok(kw.unbind())
    })
    .map_err(|e| ServerFnError::new(e.to_string()))?;
    println!("{:?}", "Dict Sucessfully created");
    pyexec::write_tfrecord_from_polars(Path::new(&writer), kwargs_struct.attr.as_str(), kwargs)
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    Ok(())
}

#[cfg(feature = "server")]
pub async fn streaming_pipe_wrds_duck(
    kwargs_struct: TseriesTfRecWrdsMarket,
    dbt: DbType,
    load_frm_parq: bool,
    pth: Option<&str>,
    tbl: &str,
    pre_features_filter: Option<Vec<&str>>,
) -> Result<(), ServerFnError> {
    let kw_temp = kwargs_struct.clone();
    //************************************************//
    //Read data from parquet file to the duck database//
    //************************************************//
    let conn = if load_frm_parq {
        let conn = Arc::new(
            start_duck_db("4GB", 14)
                .await
                .expect("duckdb in-memory should start"),
        );
        dbt.ingest(conn.clone(), pth.unwrap()).await.unwrap();
        conn
    } else {
        Arc::new(
            open_duck_db_from_file(pth.unwrap(), "4GB", 14)
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
    //*****************************************************//
    //filter columns and convert to python polars dataframe//
    //*****************************************************//
    let feature_cols: Vec<String> = df
        .get_column_names_str()
        .iter()
        .cloned()
        .filter(|c| !kwargs_struct.exclude_cols.iter().any(|ex| ex == c))
        .map(|c| c.to_string())
        .collect();
    println!(
        "Dataframe Columns: {:?} /n Potential Feature Columns: {:?}",
        df.get_column_names_str(),
        feature_cols
    );
    let py_df = PyDataFrame(df);
    let _writer = kwargs_struct
        .writer_path
        .unwrap_or_else(|| "../ml-project/py/pl2tfrecord_writer.py".to_string());
    let out_path = kwargs_struct
        .out_path
        .unwrap_or_else(|| "../tmp_data/data.tfrecord".to_string());

    let _kwargs = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let kw = kw_temp.to_pydict(py)?;
        kw.set_item("df", py_df)?;
        if !kw.contains("path")? {
            kw.set_item("path", out_path)?;
        }
        if !kw.contains("feature_cols")? {
            kw.set_item("feature_cols", feature_cols)?;
        }
        Ok(kw.unbind())
    })
    .map_err(|e| ServerFnError::new(e.to_string()))?;
    /*
    println!("{:?}", "Dict Sucessfully created");
    pyexec::write_tfrecord_from_polars(Path::new(&writer), kwargs_struct.attr.as_str(), kwargs)
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    */
    Ok(())
}
