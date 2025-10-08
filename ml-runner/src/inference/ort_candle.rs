#![allow(dead_code)]

#[cfg(feature = "server")]
use anyhow::{anyhow, Result};
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
        return Err(anyhow!(
            "rank mismatch: model={:?}, got={:?}",
            model_shape,
            got
        ));
    }
    for (i, (m, g)) in model_shape.iter().zip(got).enumerate() {
        if *m >= 0 && (*m as usize) != *g {
            return Err(anyhow!(
                "dim {} mismatch: model expects {}, got {}",
                i,
                m,
                g
            ));
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
        _ => return Err(anyhow!("first input is not a tensor")),
    };

    validate_against_model_shape(&model_shape, input.shape())?;

    let tensor = ort::value::Tensor::from_array(input)
        .map_err(|e| anyhow!("failed to convert ndarray to ort tensor: {e}"))?;
    let x = ort::inputs![tensor];
    let mut outputs = session
        .run(x)
        .map_err(|e| anyhow!("onnx runtime (candle backend) failed: {e}"))?;

    let val = &outputs[0];
    match val.try_extract_tensor::<f32>() {
        Ok((_shape, data)) => Ok(data.to_vec()),
        Err(_) => match val.try_extract_tensor::<i64>() {
            Ok((_shape, data_i64)) => Ok(data_i64.iter().map(|&v| v as f32).collect()),
            Err(_) => Err(anyhow!("unsupported output tensor type")),
        },
    }
}

/*#[cfg(feature = "server")]
pub fn onnx_infer_candle(file_path: &str, input: ArrayD<f32>) -> Result<Vec<f32>> {
    // If the caller forgot to init, try to set it here as a convenience.
    init_candle_backend();

    let mut session = Session::builder()?.commit_from_file(file_path)?;

    let dims = match &session.inputs[0].input_type {
        ValueType::Tensor { shape, .. } => shape
            .iter()
            .map(|d| if *d < 0 { 1usize } else { *d as usize })
            .collect::<Vec<_>>(),
        _ => vec![1],
    };
    println!("{:?}", &session.inputs);
    let input = input
        .into_shape_with_order(dims)
        .map_err(|_| anyhow!("failed to reshape input to model input shape"))?;
    let tensor = ort::value::Tensor::from_array(input)
        .map_err(|e| anyhow!("failed to convert ndarray to ort tensor: {e}"))?;
    let x = ort::inputs![tensor];
    let mut outputs = session
        .run(x)
        .map_err(|e| anyhow!("onnx runtime (candle backend) failed: {e}"))?;

    if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
        return Ok(data.to_vec());
    }
    if let Ok((_shape, data_i64)) = outputs[0].try_extract_tensor::<i64>() {
        return Ok(data_i64.iter().map(|v| *v as f32).collect());
    }
    Err(anyhow!("unsupported output tensor type"))
}
*/
