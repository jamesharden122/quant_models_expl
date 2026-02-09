#![allow(dead_code)]

#[cfg(feature = "server")]
use crate::error::{msg, Result};
#[cfg(feature = "server")]
use ndarray::ArrayD;
#[cfg(feature = "server")]
use ort::{session::Session, value::ValueType};

/// Run inference with a standalone ONNX model using standard `ort`.
/// Attempts to use CUDA if the `cuda` feature is available; otherwise falls back to CPU.
#[cfg(feature = "server")]
pub fn onnx_infer(file_path: &str, input: ArrayD<f32>) -> Result<Vec<f32>> {
    let builder = Session::builder()?;

    // Try to register CUDA if compiled with it (ignored at compile-time if not available)
    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        if let Err(e) = CUDAExecutionProvider::default().register(&builder) {
            eprintln!("CUDA provider registration failed, falling back to CPU: {e}");
        }
    }

    let mut session = builder.commit_from_file(file_path)?;
    let dims = match &session.inputs[0].input_type {
        ValueType::Tensor { shape, .. } => shape.iter().map(|d| if *d < 0 { 1usize } else { *d as usize }).collect::<Vec<_>>(),
        _ => vec![1],
    };

    let input = input.into_shape_with_order(dims).map_err(|_| msg("failed to reshape input to model input shape"))?;

    let tensor = ort::value::Tensor::from_array(input).map_err(|e| msg(format!("failed to convert ndarray to ort tensor: {e}")))?;
    let x = ort::inputs![tensor];
    let mut outputs = session.run(x).map_err(|e| msg(format!("onnx runtime failed: {e}")))?;

    if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
        return Ok(data.to_vec());
    }
    if let Ok((_shape, data_i64)) = outputs[0].try_extract_tensor::<i64>() {
        return Ok(data_i64.iter().map(|v| *v as f32).collect());
    }
    Err(msg("unsupported output tensor type"))
}
