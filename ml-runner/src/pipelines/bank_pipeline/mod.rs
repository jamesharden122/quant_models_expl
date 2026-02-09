#[cfg(feature = "server")]
pub mod data_pipeline;

#[cfg(feature = "server")]
pub mod analysis;
pub mod test;
pub mod train;

#[cfg(feature = "server")]
pub use data_pipeline::{create_daily_dataset, create_rar, fama_french_panel};
