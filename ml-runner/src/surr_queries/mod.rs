#[cfg(feature = "server")]
use ml_backend::{polars_ops, surreal_queries};
//#[cfg(feature = "server")]
use polars::prelude::*;
#[cfg(feature = "server")]
use surrealdb::{prelude::*,engine::any};
use serde::{Serialize,Deserialize};

#[derive(Deserialize, Serialize, Clone, Debug)]
struct Model {
    name: String,
}

#[server]
#[cfg(feature = "server")]
async fn query_feature_bin_demo(db: &any::Any, column_set: vec<&str>, bin_size: &str, where_cond: polars_ops::PartEqSurr) -> surrealdb::Result<(DataFrame)> {
	where_cond.string.unwrap().push("bin_size", bin_size);
	let df = polars_ops::select_table_as_df(db, "equities_returns_temp", column_set, where_cond)?;
	Ok(df)
}


#[server]
#[cfg(feature = "server")]
async fn fetch_models(prop: Model) -> Result<Vec<String>, ServerFnError<Db>> {
    db.use_ns("research").use_db("deeplearning").await?;
    let sql = format!(
        "SELECT name FROM models WHERE name CONTAINS '{}';",
        prop.name
    );
    let mut resp = db.query(sql).await?;
    let rows: Vec<ModelRow> = resp.take(0).map_err(ServerFnError::from)?;
    Ok(rows.into_iter().map(|r| r.name).collect())
}

