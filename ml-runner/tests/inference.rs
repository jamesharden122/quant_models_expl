//! Inference smoke tests for ONNX and SURML.
//! Uses the repo paths per request:
//! - SURML: "/../tmp_data/final_model.surml" (resolved relative to crate)
//! - ONNX:  "./../ml-project/models/saved/mls_lstm_20250903_211900/final_model.onnx"
//!
//! Run with server feature:
//!   cargo test --manifest-path ml-runner/Cargo.toml --features server --test inference -- --ignored --nocapture

#![allow(unused)]

#[cfg(feature = "server")]
mod tests {
    use ndarray::ArrayD;
    use ort::session::Session;
    use ort::value::ValueType;
    use std::path::{Path, PathBuf};

    const SURML_REQ: &str = "/../tmp_data/final_model.surml";
    const ONNX_REQ: &str = "./../ml-project/models/saved/test/final_model.onnx";
    fn resolve_repo_path(req: &str) -> PathBuf {
        // Interpret leading "/../" as "../" relative to the crate to match user intent.
        let trimmed = if req.starts_with("/../") {
            &req[1..]
        } else {
            req
        };
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(trimmed)
    }

    fn input_shape_from_session(session: &Session) -> Vec<usize> {
        match &session.inputs[0].input_type {
            ValueType::Tensor { shape, .. } => shape
                .iter()
                .map(|d| if *d < 0 { 1usize } else { *d as usize })
                .collect::<Vec<_>>(),
            _ => vec![1],
        }
    }

    #[test]
    fn onnx_infer_standard_ort() {
        let path_buf = resolve_repo_path(ONNX_REQ);
        let path = path_buf.to_string_lossy().to_string();
        assert!(
            Path::new(&path).exists(),
            "onnx path does not exist: {}",
            path
        );

        // Determine input shape from the model
        let builder = Session::builder().expect("session builder");
        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::CUDAExecutionProvider;
            let _ = CUDAExecutionProvider::default().register(&builder);
        }
        let session = builder.commit_from_file(&path).expect("commit session");
        let shape = input_shape_from_session(&session);

        // Build ones input to resemble [1.0, 1.0, 1.0, 1.0, 1.0],
        // expanded to the model's expected total size.
        let total: usize = shape.iter().product::<usize>().max(1);
        let arr = ndarray::Array::from_vec(vec![1.0f32; total])
            .into_shape(shape.clone())
            .unwrap()
            .into_dyn();

        let out = ml_runner::inference::onnx_infer(&path, arr).expect("onnx infer");
        assert!(!out.is_empty(), "inference produced no outputs");
    }

    #[test]
    fn onnx_infer_with_candle_backend() {
        let path_buf = resolve_repo_path(ONNX_REQ);
        let path = path_buf.to_string_lossy().to_string();
        assert!(
            Path::new(&path).exists(),
            "onnx path does not exist: {}",
            path
        );

        // Initialize candle backend and prepare shape
        ml_runner::inference::init_candle_backend();
        let session = Session::builder().unwrap().commit_from_file(&path).unwrap();
        let shape = input_shape_from_session(&session);

        let total: usize = shape.iter().product::<usize>().max(1);
        println!("Total {:?}", total);
        let arr = ndarray::Array::from_vec(vec![1.0f32; total])
            .into_shape(shape.clone())
            .unwrap()
            .into_dyn();

        let out = ml_runner::inference::onnx_infer_candle(&path, arr).expect("onnx infer candle");
        println!("out {:?}", &out);
        assert!(!out.is_empty(), "inference produced no outputs");
    }

    #[test]
    #[ignore]
    fn surml_infer_buffered() {
        let path_buf = resolve_repo_path(SURML_REQ);
        let path = path_buf.to_string_lossy().to_string();
        assert!(
            Path::new(&path).exists(),
            "surml path does not exist: {}",
            path
        );

        // Build input map using header keys
        let mut file = surrealml_core::storage::surml_file::SurMlFile::from_file(&path)
            .expect("open surml file");
        let keys = file.header.keys.store.clone();
        let mut inputs = std::collections::HashMap::<String, f32>::new();
        for k in keys {
            inputs.insert(k, 1.0); // per request, use ones-like inputs
        }

        let out = ml_runner::inference::surml_infer(&path, inputs).expect("surml buffered infer");
        assert!(!out.is_empty(), "inference produced no outputs");
    }

    #[test]
    #[ignore]
    fn surml_infer_raw_ndarray() {
        let path_buf = resolve_repo_path(SURML_REQ);
        let path = path_buf.to_string_lossy().to_string();
        assert!(
            Path::new(&path).exists(),
            "surml path does not exist: {}",
            path
        );

        // Use the embedded ONNX to determine shape
        let mut file = surrealml_core::storage::surml_file::SurMlFile::from_file(&path)
            .expect("open surml file");
        let session = surrealml_core::execution::session::get_session(file.model.clone())
            .expect("session from bytes");
        let shape = match &session.inputs[0].input_type {
            ValueType::Tensor { shape, .. } => shape
                .iter()
                .map(|d| if *d < 0 { 1usize } else { *d as usize })
                .collect::<Vec<_>>(),
            _ => vec![1],
        };

        let total: usize = shape.iter().product::<usize>().max(1);
        let arr = ndarray::Array::from_vec(vec![1.0f32; total])
            .into_shape(shape.clone())
            .unwrap()
            .into_dyn();

        let out = ml_runner::inference::surml_infer_raw(&path, arr).expect("surml raw infer");
        assert!(!out.is_empty(), "inference produced no outputs");
    }
}
