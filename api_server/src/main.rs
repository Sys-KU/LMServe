use std::sync::Arc;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
use clap::Parser;
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::info;

use api_server::args::APIServerArgs;
use api_server::router::EngineRouter;
use api_server::{APIServer, GenerateParams};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    utils::logging::init_tracing();

    let args = APIServerArgs::parse();

    let socket_addr = args.address;

    let url = format!("http://{}", socket_addr);

    let tokenizer =
        Tokenizer::from_pretrained(&args.model_name, None).expect("Failed to load tokenizer");

    let mut engine_router = EngineRouter::new();
    for addr in args.llm_server_addresses.iter() {
        let llm_server_url = format!("http://{}", addr);
        let _ = engine_router
            .add_node(&llm_server_url)
            .await
            .unwrap_or_else(|e| panic!("Unable to establish connection to engine ({addr}): {e}"));

        info!("Successfully connected to LLM server at {}", addr);
    }

    let api_server = Arc::new(APIServer::new(engine_router, tokenizer));

    // Build API server with a route.
    let app = Router::new().route(
        "/generate",
        post(move |Json(params): Json<GenerateParams>| async move {
            match api_server.generate(params).await {
                Ok(output) => (StatusCode::OK, Json(output)).into_response(),
                Err(err) => {
                    eprintln!("Failed to generate: {:?}", err);
                    (StatusCode::INTERNAL_SERVER_ERROR, "Generation failed").into_response()
                }
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    info!("API Server started to {url}");

    axum::serve(listener, app)
        .with_graceful_shutdown(utils::signal_handler::wait_shutdown_signal())
        .await
        .unwrap();

    info!("API Server terminated.");

    Ok(())
}
