#[cfg(feature = "server")]
use ml_backend::surreal_queries::DbParams;

#[cfg(feature = "server")]
use crate::pyexec::pydictstructs::{};
#[cfg(feature = "server")]
use crate::{
    backtest::{run_backtest_series, BacktestKind, BacktestParams, RunBacktestRequest},
    streaming::streaming_pipe,
    training::training_pipe,
};
#[cfg(feature = "server")]
use dioxus::prelude::ServerFnError;

#[cfg(feature = "server")]
pub async fn run_gi_sim_dnn() -> Result<(serde_json::Value), ServerFnError> {
    todo!()
}

pub async fn back_test_gi_sim_dnn() -> Result<(serde_json::Value), ServerFnError> {
    todo!()
}
