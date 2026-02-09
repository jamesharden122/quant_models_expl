#![allow(dead_code)]

#[cfg(feature = "server")]
use crate::error::{msg, Result};
#[cfg(feature = "server")]
use ndarray::ArrayD;
#[cfg(feature = "server")]
use ort::{session::Session, value::ValueType};

/// Initialize the ort-candle backend. Call once early before any `ort` usage.
#[cfg(feature = "server")]
pub fn init_candle_backend() {
    // Safe to call multiple times; subsequent calls are ignored internally.
    ort::set_api(ort_candle::api());
}

#[cfg(feature = "server")]
fn validate_against_model_shape(model_shape: &[i64], got: &[usize]) -> Result<()> {
    if model_shape.len() != got.len() {
        return Err(msg(format!("rank mismatch: model={model_shape:?}, got={got:?}")));
    }
    for (i, (m, g)) in model_shape.iter().zip(got).enumerate() {
        if *m >= 0 && (*m as usize) != *g {
            return Err(msg(format!("dim {i} mismatch: model expects {m}, got {g}")));
        }
    }
    println!("{:?}", "validated");
    Ok(())
}

/// Run inference using the `ort-candle` alternative backend.
/// Ensure `init_candle_backend()` is called once before using this function.
#[cfg(feature = "server")]
pub fn onnx_infer_candle(file_path: &str, input: ArrayD<f32>) -> Result<Vec<f32>> {
    init_candle_backend();

    let mut session = Session::builder()?.commit_from_file(file_path)?;
    let model_shape = match &session.inputs[0].input_type {
        ValueType::Tensor { shape, .. } => shape.clone(),
        _ => return Err(msg("first input is not a tensor")),
    };

    validate_against_model_shape(&model_shape, input.shape())?;

    let tensor = ort::value::Tensor::from_array(input).map_err(|e| msg(format!("failed to convert ndarray to ort tensor: {e}")))?;
    let x = ort::inputs![tensor];
    let mut outputs = session.run(x).map_err(|e| msg(format!("onnx runtime (candle backend) failed: {e}")))?;

    let val = &outputs[0];
    match val.try_extract_tensor::<f32>() {
        Ok((_shape, data)) => Ok(data.to_vec()),
        Err(_) => match val.try_extract_tensor::<i64>() {
            Ok((_shape, data_i64)) => Ok(data_i64.iter().map(|&v| v as f32).collect()),
            Err(_) => Err(msg("unsupported output tensor type")),
        },
    }
}
