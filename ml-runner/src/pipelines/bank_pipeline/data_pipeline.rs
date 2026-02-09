#[cfg(feature = "server")]
use std::path::PathBuf;

#[cfg(feature = "server")]
use dioxus::prelude::ServerFnError;

#[cfg(feature = "server")]
use polars::prelude::*;

#[cfg(feature = "server")]
use crate::streaming::strmpolars::streaming_pipe_merge_wrds_duck_polars;

#[cfg(feature = "server")]
use ml_backend::{
    featscreate::{
        apply_by_names,
        est_returns::Returns,
        mean_by_groups::MeanByGroups,
        prod_by_groups::ProdByGroups,
        sharpes::{RarRatios, RatioTypes},
        DynamicGroupCfg, FeatList, GroupingCfg, PrcDataSource,
    },
    polars_ops,
};

#[cfg(feature = "server")]
use wrds_io::createdatasets::{
    usbanks::{BankCrspDly, BankCrspPaths},
    CreateDuckFls, MergeDuckFls,
};

#[cfg(feature = "server")]
pub async fn create_daily_dataset(
    bcp: Option<BankCrspPaths>,
    bcd: Option<BankCrspDly>,
    dyn_cfg: DynamicGroupCfg,
    duck_size: &str,
    duckdb_source_dir: PathBuf,
    cache_file: PathBuf,
    duckdb_merge_dir: PathBuf,
) -> Result<[DataFrame; 2], ServerFnError> {
    let table = "bank_securities_dly";
    let cdf: Option<(CreateDuckFls, &std::path::Path)> = bcp.map(CreateDuckFls::UsBankCrsp).map(|create| (create, duckdb_source_dir.as_path()));
    let mdf: Option<MergeDuckFls> = bcd.map(MergeDuckFls::UsBankCrspDly);

    let merged_db_path = duckdb_merge_dir.join(format!("{table}.duckdb"));
    if mdf.is_none() && !merged_db_path.exists() && !cache_file.exists() {
        return Err(ServerFnError::new(format!(
            "cache miss and merged duckdb not found at `{}`; pass `bcd` to merge or create the merged DB first",
            merged_db_path.display()
        )));
    }
    let df_path = if mdf.is_some() { duckdb_merge_dir.as_path() } else { merged_db_path.as_path() };

    let df = streaming_pipe_merge_wrds_duck_polars(cdf, mdf, Some(cache_file), None, duck_size, 10, df_path, table, None)
        .await
        .map_err(|e| ServerFnError::new(format!("{e:?}")))?;

    let df_daily = polars_ops::utils::add_year_month_week_cols(
        df.lazy().select([
            col("rssd9001").cast(DataType::Int64),
            col("permno").cast(DataType::Int64),
            col("permco").cast(DataType::Int64),
            col("date"),
            col("ret"),
            col("mktrf"),
            col("hml"),
            col("rmw"),
            col("smb"),
            col("umd"),
            col("rf"),
            col("prc"),
        ]),
        col("date"),
    )
    .with_columns([col("prc").abs()])
    .collect()
    .map_err(|e| ServerFnError::new(format!("{e:?}")))?;

    let ret_spec: Vec<FeatList> = vec![FeatList::Returns(Returns {
        source: PrcDataSource::Crsp,
        grpby: Some(vec!["rssd9001".to_string(), "permco".to_string(), "permno".to_string()]),
        grouping: GroupingCfg::Dynamic(dyn_cfg.clone()),
    })];
    let mbg_spec: Vec<FeatList> = vec![FeatList::MeanByGrp(MeanByGroups {
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        dynamic_group: Some(dyn_cfg.clone()),
        columns: vec!["mktrf".to_string(), "rf".to_string()],
    })];
    let pbg_spec: Vec<FeatList> = vec![FeatList::ProdByGrp(ProdByGroups {
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        dynamic_group: Some(dyn_cfg),
        columns: vec!["mktrf".to_string(), "rf".to_string()],
    })];

    let df_ret = apply_by_names(df_daily.clone(), ret_spec).await.map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    let df_pbg = apply_by_names(df_daily.clone(), pbg_spec).await.map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    let df_mbg = apply_by_names(df_daily.clone(), mbg_spec).await.map_err(|e| ServerFnError::new(format!("{e:?}")))?;

    let join_on = [col("rssd9001"), col("permco"), col("permno"), col("date")];
    let df_ret = df_ret
        .lazy()
        .join(df_pbg.lazy(), join_on.clone(), join_on.clone(), JoinArgs::new(JoinType::Left))
        .with_columns([col("mktrf_cr").alias("mktrf"), col("rf_cr").alias("rf")])
        .drop(by_name(["rssd9001_right", "permco_right", "permno_right", "date_right", "mktrf_cr", "rf_cr"], false))
        .join(df_mbg.lazy(), join_on.clone(), join_on, JoinArgs::new(JoinType::Left))
        .drop(by_name(["rssd9001_right", "permco_right", "permno_right", "date_right"], false))
        .collect()
        .map_err(|e| ServerFnError::new(format!("{e:?}")))?;

    Ok([df_daily, df_ret])
}

#[cfg(feature = "server")]
pub async fn create_rar(mut df_ret: DataFrame, dyn_cfg: DynamicGroupCfg) -> Result<DataFrame, ServerFnError> {
    let rar_spec = vec![FeatList::Sharpe(RarRatios {
        ratios: [Some(RatioTypes::Sharpe), None, None], //Some(RatioTypes::Sortino), None],
        source: PrcDataSource::Crsp,
        grpby: Some(vec![String::from("rssd9001"), String::from("permco"), String::from("permno")]),
        grouping: Some(GroupingCfg::Dynamic(dyn_cfg)),
    })];
    df_ret = apply_by_names(df_ret, rar_spec)
        .await
        .map_err(|e| ServerFnError::new(format!("{e:?}")))?
        .lazy()
        .with_columns([col("date").dt().year().alias("year")])
        .collect()
        .map_err(|e| ServerFnError::new(format!("{e:?}")))?;
    Ok(df_ret)
}

#[cfg(feature = "server")]
pub async fn fama_french_panel(_df: DataFrame) {
    todo!()
}
