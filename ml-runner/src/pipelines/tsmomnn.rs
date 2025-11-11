#[cfg(feature = "server")]
use ml_backend::surreal_queries::DbParams;

#[cfg(feature = "server")]
use crate::{
    backtest::{run_backtest_series, BacktestKind, BacktestParams, RunBacktestRequest},
    streaming::streaming_pipe,
    training::training_pipe,
};

#[cfg(feature = "server")]
use crate::pyexec::pydictstructs::{MlsLstmTrain, TseriesTfRecBento, TseriesTfRecLoad};
#[cfg(feature = "server")]
use dioxus::prelude::ServerFnError;
#[cfg(feature = "server")]
use tokio::{runtime::Handle, task};
//login and query data using streaming pipe
//train using training pipe
#[cfg(feature = "server")]
pub async fn run_time_series_momentum_lstm(
    db_params: DbParams,
    write_params: TseriesTfRecBento,
    load_params: TseriesTfRecLoad,
    train_params: MlsLstmTrain,
) -> Result<(serde_json::Value), ServerFnError> {
    let _handle = task::spawn_blocking(move || {
        // We're on a dedicated blocking thread now.
        let rt = Handle::current();
        rt.block_on(async move { streaming_pipe(db_params, write_params).await })
    });
    let val = training_pipe(load_params, train_params).await?;
    Ok((val))
}

#[cfg(feature = "server")]
pub async fn back_test_time_series_momentum_lstm(
    backtest_params: RunBacktestRequest,
) -> Result<(serde_json::Value), ServerFnError> {
    let output = run_backtest_series(backtest_params).await?;
    Ok((output))
}
//run back test
