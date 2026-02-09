use polars::error::PolarsError;
use std::path::Path;
use wrds_io::instantiatedb::polars_utils::load_cache;

#[derive(Debug)]
pub enum AnlysErr {
    Polars(PolarsError),
}

impl std::fmt::Display for AnlysErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Polars(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for AnlysErr {}

impl From<PolarsError> for AnlysErr {
    fn from(value: PolarsError) -> Self {
        Self::Polars(value)
    }
}

pub fn rar_regressions(path: &Path) -> Result<(), AnlysErr> {
    let _df = load_cache(path)?;
    Ok(())
}
