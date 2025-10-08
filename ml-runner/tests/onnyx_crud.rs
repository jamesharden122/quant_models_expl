// Integration test that uploads an existing ONNX file via the
// `pyexec::package_and_upload_surml` flow (build .surml then Python upload).
//
// Env overrides:
// - ONNX_PATH: absolute/relative path to .onnx
// - SURML_OUT: output path for .surml (default: ../tmp_data/final_model.surml)
// - SUR_URL, SUR_NS, SUR_DB, SUR_USER, SUR_PASS, SUR_CHUNK

#[cfg(feature = "server")]
use ml_runner::pyexec;

#[tokio::test]
#[cfg(feature = "server")]
async fn onnyx_package_and_upload_surml() {
    pyo3::prepare_freethreaded_python();
    // Resolve paths
    let onnx_path_default = 
        "/home/yakaman/Dropbox/Desktop/tesero-sol/software_development/trading/quant_models_expl/ml-project/models/saved/mls_lstm_20250911_184101/final_model.onnx";
    let onnx_path = std::env::var("ONNX_PATH").unwrap_or_else(|_| onnx_path_default.to_string());
    let surml_out = std::env::var("SURML_OUT").unwrap_or_else(|_| "../tmp_data/final_model.surml".to_string());

    // Surreal connection (env-overridable)
    let url = "https://quant-platform-06cb0tpcrpsspao10de28go15s.aws-use1.surreal.cloud/ml/import".to_string();
    let namespace = std::env::var("SUR_NS").unwrap_or_else(|_| "equities".to_string());
    let database = std::env::var("SUR_DB").unwrap_or_else(|_| "historical".to_string());
    let username = std::env::var("SUR_USER").unwrap_or_else(|_| "root".to_string());
    let password = std::env::var("SUR_PASS").unwrap_or_else(|_| "root".to_string());
    let chunk_size: usize = std::env::var("SUR_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1024*1024);

    // Execute pack + upload
    pyexec::package_and_upload_surml(
        &onnx_path,
        &surml_out,
        &url,
        chunk_size,
        &namespace,
        &database,
        &username,
        &password,
    )
    .await.unwrap();

    // Optional cleanup
    if std::env::var("CLEAN_SURML").ok().as_deref() == Some("1") {
        let _ = std::fs::remove_file(&surml_out);
    }
}
