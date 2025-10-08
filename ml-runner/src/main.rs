use dioxus::prelude::*;
use dioxus_router::prelude::*;
use dioxus_logger::tracing;
use ml_runner::app;

fn main() {
    #[cfg(feature = "server")]
    {
        // Optionally switch to the ort-candle backend via env.
        if std::env::var("ORT_BACKEND").ok().as_deref() == Some("candle") {
            ml_runner::inference::init_candle_backend();
        }
        tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(launch_server());
    }
    #[cfg(not(feature = "server"))]
    dioxus::launch(app);
}

async fn launch_server() {
    dioxus::logger::initialize_default();

    let socket_addr = dioxus_cli_config::fullstack_address_or_localhost();

    let router = axum::Router::new()
        .serve_dioxus_application(ServeConfigBuilder::new(), app)
        .into_make_service();

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
