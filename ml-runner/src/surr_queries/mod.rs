#[cfg(feature = "server")]
use ml_backend::{polars_ops, surreal_queries};
//#[cfg(feature = "server")]
use polars::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use surrealdb::{engine::any, Surreal};

#[derive(Deserialize, Serialize, Clone, Debug)]
struct Model {
    name: String,
}

#[cfg(feature = "server")]
pub async fn query_feature_bin_demo(
    db: &Surreal<any::Any>,
    column_set: Vec<&str>,
    bin_size: String,
    inst_id: Vec<i64>,
    srt: Option<Vec<String>>,
) -> surrealdb::Result<(DataFrame)> {
    let where_cond = polars_ops::PartEqSurr {
        int: Some(
            inst_id
                .iter()
                .map(|x| ("instrument_id".to_string(), *x, polars_ops::Logic::Or))
                .collect::<Vec<(String, i64,polars_ops::Logic)>>(),
        ),
        string: Some(vec![(
            "bin_size".to_string(),
            bin_size.to_string(),
            polars_ops::Logic::And,
        )]),
        float: None,
        time_range: None,
    };
    let df = polars_ops::select_table_as_df(db, "equities_returns", column_set, where_cond).await?;
    let df = match srt {
        Some(cols) => {
            let exprs: Vec<Expr> = cols.iter().map(|a| col(a)).collect();
            df.lazy()
                .sort_by_exprs(exprs, SortMultipleOptions::default())
                .collect()
                .unwrap()
        }
        None => df,
    };
    Ok(df)
}

#[cfg(feature = "server")]
pub async fn query_feature_bin_time_constrained(
    db: &Surreal<any::Any>,
    column_set: Vec<&str>,
    bin_size: String,
    inst_id: Vec<i64>,
    table: String,
    srt: Option<Vec<String>>,
    time_col: String,    // e.g. Some("bin"), Some("t0"), Some("t1")
    time_start: String,  // e.g. "2025-07-15T10:25:00Z" (RFC3339) or "7/15/2025, 10:25:00 AM"
    time_end: String,    // same format as start
) -> surrealdb::Result<(DataFrame)> {
    let where_cond = polars_ops::PartEqSurr {
        int: Some(
            inst_id
                .iter()
                .map(|x| ("instrument_id".to_string(), *x, polars_ops::Logic::Or))
                .collect::<Vec<(String, i64,polars_ops::Logic)>>(),
        ),
        string: Some(vec![(
            "bin_size".to_string(),
            bin_size.to_string(),
            polars_ops::Logic::And,
        )]),
        float: None,
        // NEW: time range on an arbitrary column
        time_range: match (time_col, time_start, time_end) {
            (col, t0, t1) => Some((col, t0, t1, polars_ops::Logic::And)),
            _ => None,
        },
    };
    let df = polars_ops::select_table_as_df(db, &(table.as_str()), column_set, where_cond).await?;
    let df = match srt {
        Some(cols) => {
            let exprs: Vec<Expr> = cols.iter().map(|a| col(a)).collect();
            df.lazy()
                .sort_by_exprs(exprs, SortMultipleOptions::default())
                .collect()
                .unwrap()
        }
        None => df,
    };
    Ok(df)
}


#[cfg(feature = "server")]
pub async fn query_feature_time_constrained(
    db: &Surreal<any::Any>,
    column_set: Vec<&str>,
    inst_id: Vec<i64>,
    table: String,
    srt: Option<Vec<String>>,
    time_col: String,    // e.g. Some("bin"), Some("t0"), Some("t1")
    time_start: String,  // e.g. "2025-07-15T10:25:00Z" (RFC3339) or "7/15/2025, 10:25:00 AM"
    time_end: String,    // same format as start
) -> surrealdb::Result<(DataFrame)> {
    let where_cond = polars_ops::PartEqSurr {
        int: Some(
            inst_id
                .iter()
                .map(|x| ("instrument_id".to_string(), *x, polars_ops::Logic::Or))
                .collect::<Vec<(String, i64,polars_ops::Logic)>>(),
        ),
        string: None,
        float: None,
        // NEW: time range on an arbitrary column
        time_range: match (time_col, time_start, time_end) {
            (col, t0, t1) => Some((col, t0, t1, polars_ops::Logic::And)),
            _ => None,
        },
    };
    let df = polars_ops::select_table_as_df(db, &(table.as_str()), column_set, where_cond).await?;
    let df = match srt {
        Some(cols) => {
            let exprs: Vec<Expr> = cols.iter().map(|a| col(a)).collect();
            df.lazy()
                .sort_by_exprs(exprs, SortMultipleOptions::default())
                .collect()
                .unwrap()
        }
        None => df,
    };
    Ok(df)
}