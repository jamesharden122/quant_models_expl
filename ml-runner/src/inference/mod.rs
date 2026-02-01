#[cfg(feature = "server")]
pub mod onnx;
#[cfg(feature = "server")]
pub mod ort_candle;

#[cfg(feature = "server")]
pub use onnx::onnx_infer;
#[cfg(feature = "server")]
pub use ort_candle::{init_candle_backend, onnx_infer_candle};
