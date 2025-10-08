#[cfg(feature = "server")]
pub mod backtest;
#[cfg(feature = "server")]
pub mod inference;
pub mod pipelines;
pub mod pyexec;
pub mod surr_queries;
use dioxus::prelude::*;
#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{apply_by_names, FeatList, MomFactor},
    polars_ops, surreal_queries,
    surreal_queries::DbParams,
};
#[cfg(feature = "server")]
use polars::prelude::*;
#[cfg(feature = "server")]
use pyexec::pydictstructs::{
    MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad, TseriesTfRecWrdsGlobalInd,
};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use pyo3_polars::types::PyDataFrame;
use std::{path::Path, sync::Arc};
#[cfg(feature = "server")]
use wrds_io::finance_data_structs::crsp;

#[cfg(feature = "server")]
pub async fn streaming_pipe(
    db_params: DbParams,
    kwargs_struct: TseriesTfRecBento, // Kwargs passed to the Python function (we will augment with df/feature_cols/path)
                                      //forc_span: i64,
) -> Result<(), ServerFnError> {
    let mut kw_temp = kwargs_struct.clone();
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
    //let mut i = 0;
    //let mut forc_span_temp = forc_span;
    if let Some(names) = kwargs_struct.feature_names {
        //if i > 0 {forc_span_temp = None}
        df = apply_by_names(df, names)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;
        //i += 1
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
    println!("Dataframe Columns: {:?}", df.get_column_names_str());
    println!("Potential Feature Columns: {:?}", feature_cols);
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
pub async fn streaming_pipe_wrds_global_index(
    kwargs_struct: TseriesTfRecWrdsGlobalInd,
) -> Result<(), ServerFnError> {
    let mut kw_temp = kwargs_struct.clone();

    //Instantiate in memory duck database
    let conn = wrds_io::start_duck_db("4GB", 14)
        .await
        .expect("duckdb in-memory should start");
    let conn = Arc::new(conn);
    //************************************************//
    //Read data from parquet file to the duck database//
    //************************************************//
    let processed =
        crsp::GlobalDailyIndex::duck_from_parquet(conn.clone(), kwargs_struct.data_path)
            .await
            .expect("upsert from parquet should succeed");
    let mut stmt = conn.prepare("DESCRIBE  global_indexes_daily").unwrap();
    let mut rows = stmt.query([]).unwrap();
    //*************************************************//
    //Filter data based on the tickers and a date tuple and convert to dataframe//
    //*************************************************//
    let tic_vec = kwargs_struct.query_params.clone().unwrap().0;
    let date1 = kwargs_struct.query_params.clone().unwrap().1;
    let date2 = kwargs_struct.query_params.clone().unwrap().2;

    let data: Vec<polars::frame::row::Row> = crsp::GlobalDailyIndex::read_gdi_batch(
        conn.clone(),
        "tic".to_string(),
        tic_vec,
        (date1, date2),
    )
    .await
    .unwrap();
    let schema = Schema::from_iter([
        Field::new("tic".into(), DataType::String),
        Field::new("datadate".into(), DataType::Date),
        Field::new("gvkeyx".into(), DataType::String),
        Field::new("conm".into(), DataType::String),
        Field::new("indextype".into(), DataType::String),
        Field::new("indexid".into(), DataType::String),
        Field::new("indexcat".into(), DataType::String),
        Field::new("idxiddesc".into(), DataType::String),
        Field::new("dvpsxd".into(), DataType::Float64),
        Field::new("newnum".into(), DataType::Int32),
        Field::new("oldnum".into(), DataType::Int32),
        Field::new("prccd".into(), DataType::Float64),
        Field::new("prccddiv".into(), DataType::Float64),
        Field::new("prccddivn".into(), DataType::Float64),
        Field::new("prchd".into(), DataType::Float64),
        Field::new("prcld".into(), DataType::Float64),
    ]);
    let mut df = DataFrame::from_rows_and_schema(&data, &schema).unwrap();
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
    // Create feature columns by excluding reserved columns  and keepingthe engineered functions
    //from the feature engineering function taht applies polar expressions
    let feature_cols: Vec<String> = df
        .get_column_names_str()
        .iter()
        .cloned()
        .filter(|c| !kwargs_struct.exclude_cols.iter().any(|ex| ex == c))
        .map(|c| c.to_string())
        .collect();
    println!("Dataframe Columns: {:?}", df.get_column_names_str());
    println!("Potential Feature Columns: {:?}", feature_cols);
    /*
    let py_df = PyDataFrame(df);
    let writer = kwargs_struct
        .writer_path
        .unwrap_or_else(|| "../ml-project/py/pl2tfrecord_writer.py".to_string());
    let out_path = kwargs_struct
        .out_path
        .unwrap_or_else(|| "../tmp_data/data.tfrecord".to_string());

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
    */
    Ok(())
}

#[cfg(feature = "server")]
pub async fn training_pipe(
    reader_struct: TseriesTfRecLoad,
    train_struct: MlsLstmTrain,
) -> Result<(serde_json::Value), ServerFnError> {
    let ds = pyexec::load_tfrecord_dataset(reader_struct)
        .map_err(|e| ServerFnError::new(e.to_string()))?;
    println!("The Data was loaded");
    //define the inputs to be passed to the python training function
    let kwargs = Python::with_gil(|py| -> PyResult<Py<PyDict>> {
        let mut kw: Bound<'_, PyDict> = train_struct.to_pydict(py)?; // Bound<PyDict>, not PyAny
        kw.set_item("full_ds", ds)?;
        // set other items...
        Ok(kw.unbind()) // back to Py<PyDict>
    })?;
    let res = pyexec::train::run_mls_lstm_training(
        Path::new(train_struct.trainer_path.as_str()),
        train_struct.class.as_str(),
        train_struct.attr.as_str(),
        kwargs,
    )?;
    println!("Model saved at: {}", res.model_dir);
    println!("Metrics: {}", res.metrics);
    Ok((res.metrics))
}

#[component]
fn TrainingForm() -> Element {
    let mut trainer_py = use_signal(|| "../ml-project/py/mls_lstm_trainer.py".to_string());
    let mut tfrecord_path = use_signal(|| "../tmp_data/data.tfrecord".to_string());
    let mut callable = use_signal(|| "train".to_string());

    let run_training = move |evt: FormEvent| {
        evt.prevent_default();
        // In web builds, you can wire this to server functions via fullstack routing.
        // Left as a stub to keep compilation simple.
        println!(
            "Submit clicked: trainer={}, tfrec={}, call={}",
            trainer_py(),
            tfrecord_path(),
            callable()
        );
    };

    rsx! {
        form { onsubmit: run_training,
            label { "Trainer Python file" }
            input { value: trainer_py(), oninput: move |evt| trainer_py.set(evt.value()) }
            label { "TFRecord file" }
            input { value: tfrecord_path(), oninput: move |evt| tfrecord_path.set(evt.value()) }
            label { "Callable name" }
            input { value: callable(), oninput: move |evt| callable.set(evt.value()) }
            button { r#type: "submit", "Run Training" }
        }
    }
}

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    TrainingForm,
}

#[component]
pub fn app() -> Element {
    rsx! { Router::<Route> {} }
}
