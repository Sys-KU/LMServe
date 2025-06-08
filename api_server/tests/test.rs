use tokio::sync::OnceCell;
use tokio::sync::oneshot::Sender;
use tokio::task::JoinHandle;

use api_server::pb::llm::llm_server::{Llm, LlmServer};
use api_server::pb::llm::{GenerateRequest, GenerateResponse, GetKindResponse};
use api_server::{APIServer, GenerateParams, router::EngineRouter};
use std::time::Duration;
use tokenizers::tokenizer::Tokenizer;
use tonic::transport::Server;
use utils::logging::init_tracing;
use utils::signal_handler::wait_shutdown_signal;

const TEST_MODEL_NAME: &str = "Qwen/Qwen2.5-0.5B";
const DEFAULT_MAX_LEN: u64 = 64;
const DUMMY_TOKEN_ID: u32 = 1;
const DUMMY_PROMPT: &str = "Hello World";
const DUMMY_ENGINE_ADDR: &str = "0.0.0.0:12300";
const TEST_PROMPT_LEN_STRIDE: usize = 100;
const TEST_PROMPT_MAX_LEN: usize = 1000;

static DUMMY_FULL_LLM_SERVER_HANDLE: OnceCell<(Sender<()>, JoinHandle<()>)> = OnceCell::const_new();

struct DummyLLMService {
    kind: String,
}

#[tonic::async_trait]
impl Llm for DummyLLMService {
    #[allow(unused_variables)]
    async fn get_kind(
        &self,
        request: tonic::Request<()>,
    ) -> anyhow::Result<tonic::Response<GetKindResponse>, tonic::Status> {
        Ok(tonic::Response::new(GetKindResponse {
            kind: self.kind.clone(),
        }))
    }

    async fn generate(
        &self,
        request: tonic::Request<GenerateRequest>,
    ) -> anyhow::Result<tonic::Response<GenerateResponse>, tonic::Status> {
        let generate_request = request.into_inner();

        let session_id = generate_request.session_id;
        // let input_ids = generate_request.input_ids;
        let _ = generate_request.num_samples;
        let max_output_len = match generate_request.max_output_len {
            Some(n) => n,
            _ => DEFAULT_MAX_LEN,
        };
        let _ = generate_request.ignore_eos;

        let output_ids: Vec<u32> = vec![DUMMY_TOKEN_ID; max_output_len as usize]; // Dummy output, replace with actual generation

        let output = GenerateResponse {
            session_id,
            output_ids,
        };
        Ok(tonic::Response::new(output))
    }
}

async fn run_dummy_llm_server(
    engine_kind: String,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    let llm_service = DummyLLMService { kind: engine_kind };
    let llm_server = LlmServer::new(llm_service);
    let addr = DUMMY_ENGINE_ADDR
        .parse()
        .expect("Failed to parse dummy llm server address");

    tracing::info!("Initialized dummy llm server");

    Server::builder()
        .add_service(llm_server)
        .serve_with_shutdown(addr, async move {
            tokio::select! {
                _ = shutdown_rx=>{
                    tracing::info!("Received shutdown signal, gracefully exiting dummy llm server.");
                },
                _ = wait_shutdown_signal()=>{
                    tracing::info!("Received terminate signal, exiting dummy llm server.");
                },
            }
        })
        .await
        .expect("Failed to start dummy llm server");
}

async fn try_init_full_dummy_server() -> &'static (Sender<()>, JoinHandle<()>) {
    DUMMY_FULL_LLM_SERVER_HANDLE
        .get_or_init(|| async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let handle = tokio::spawn(async { run_dummy_llm_server("full".to_string(), rx).await });
            (tx, handle)
        })
        .await
}

async fn init_api_server(tokenizer_name: &str, urls: Vec<&str>) -> APIServer {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
        .expect(format!("Failed to load tokenizer {tokenizer_name}").as_str());
    let mut engine_router = EngineRouter::new();
    for url in urls.iter() {
        let llm_server_url = format!("http://{}", url);
        engine_router
            .add_node(&llm_server_url)
            .await
            .unwrap_or_else(|e| panic!("Unable to establish connection to engine ({url}): {e}"));
        tracing::info!("Successfully connected to LLM server at {}", url);
    }
    APIServer::new(engine_router, tokenizer)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_max_output_length() {
    init_tracing();
    try_init_full_dummy_server().await;

    let api_server = init_api_server(TEST_MODEL_NAME, vec![DUMMY_ENGINE_ADDR]).await;
    let input_token_len = api_server
        .tokenizer
        .encode(DUMMY_PROMPT, false)
        .expect("Dummy prompt tokenization failed")
        .get_ids()
        .len();
    for max_len in (0..TEST_PROMPT_MAX_LEN).step_by(TEST_PROMPT_LEN_STRIDE) {
        let output = api_server
            .generate(GenerateParams {
                prompt: DUMMY_PROMPT.to_string(),
                num_samples: 1,
                session_id: None,
                max_output_len: Some(max_len as u64),
                ignore_eos: false,
            })
            .await
            .expect("Failed to generate");
        assert!(
            max_len == 0 || output.token_ids.len() == (max_len + input_token_len),
            "Testing {} output length, Expected output length to be {} but got {}",
            max_len,
            max_len + input_token_len,
            output.token_ids.len()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_threading() {
    init_tracing();
    try_init_full_dummy_server().await;
    let mut cnt = 0;
    loop {
        if cnt > 2 {
            tracing::info!("Main loop reached count {cnt}, sending shutdown signal.");
            break;
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(3)) => {
                cnt += 1;
                tracing::info!("Here from main loop{cnt}");
            }
        }
    }
}
