use dioxus::prelude::*;
use dioxus_router::prelude::*;
use dioxus_logger::tracing;
use ml_runner::app;

fn main() {
    #[cfg(feature = "server")]
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(launch_server());
    #[cfg(not(feature = "server"))]
    dioxus::launch(app);
}

#[cfg(feature = "server")]
async fn launch_server() {
    dioxus::logger::initialize_default();

    let socket_addr = dioxus_cli_config::fullstack_address_or_localhost();

    let router = axum::Router::new()
        .serve_dioxus_application(ServeConfigBuilder::new(), app)
        .into_make_service();

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
