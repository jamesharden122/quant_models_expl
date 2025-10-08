#![allow(dead_code)]

#[cfg(feature = "server")]
use anyhow::{anyhow, Result};
#[cfg(feature = "server")]
use ndarray::ArrayD;
#[cfg(feature = "server")]
use ort::{session::Session, value::ValueType};
#[cfg(feature = "server")]
use std::collections::HashMap;
#[cfg(feature = "server")]
use surrealml_core::execution::compute::ModelComputation;
#[cfg(feature = "server")]
use surrealml_core::storage::surml_file::SurMlFile;

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
        ValueType::Tensor { shape, .. } => shape
            .iter()
            .map(|d| if *d < 0 { 1usize } else { *d as usize })
            .collect::<Vec<_>>(),
        _ => vec![1],
    };

    let input = input
        .into_shape_with_order(dims)
        .map_err(|_| anyhow!("failed to reshape input to model input shape"))?;

    let tensor = ort::value::Tensor::from_array(input)
        .map_err(|e| anyhow!("failed to convert ndarray to ort tensor: {e}"))?;
    let x = ort::inputs![tensor];
    let mut outputs = session
        .run(x)
        .map_err(|e| anyhow!("onnx runtime failed: {e}"))?;

    if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
        return Ok(data.to_vec());
    }
    if let Ok((_shape, data_i64)) = outputs[0].try_extract_tensor::<i64>() {
        return Ok(data_i64.iter().map(|v| *v as f32).collect());
    }
    Err(anyhow!("unsupported output tensor type"))
}

/// Run inference with a `.surml` packaged model using `surrealml-core`.
#[cfg(feature = "server")]
pub fn surml_infer(file_path: &str, mut inputs: HashMap<String, f32>) -> Result<Vec<f32>> {
    let mut file =
        SurMlFile::from_file(file_path).map_err(|e| anyhow!("failed to open surml file: {e}"))?;
    let compute = ModelComputation {
        surml_file: &mut file,
    };
    let out = compute
        .buffered_compute(&mut inputs)
        .map_err(|e| anyhow!("surml compute failed: {e}"))?;
    Ok(out)
}

/// Raw `.surml` inference bypassing header mapping/normalisers.
#[cfg(feature = "server")]
pub fn surml_infer_raw(file_path: &str, input: ArrayD<f32>) -> Result<Vec<f32>> {
    let mut file =
        SurMlFile::from_file(file_path).map_err(|e| anyhow!("failed to open surml file: {e}"))?;
    let compute = ModelComputation {
        surml_file: &mut file,
    };
    let out = compute
        .raw_compute(input, None)
        .map_err(|e| anyhow!("surml raw compute failed: {e}"))?;
    Ok(out)
}
