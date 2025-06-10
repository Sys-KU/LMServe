use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing::debug;

use llmserve::args::LLMEngineArgs;
use llmserve::engine::LLMEngineWrapper;
use llmserve::pb::llm::llm_server::{Llm, LlmServer};
use llmserve::pb::llm::{
    GenerateRequest, GenerateResponse, GetDescriptorsRequest, GetDescriptorsResponse,
    GetKindResponse, GetKvAgentMetadataResponse,
};

pub struct LLMService {
    kind: String,
    engine: Arc<LLMEngineWrapper>,
}

#[tonic::async_trait]
impl Llm for LLMService {
    #[allow(unused_variables)]
    async fn get_kind(&self, request: Request<()>) -> Result<Response<GetKindResponse>, Status> {
        Ok(Response::new(GetKindResponse {
            kind: self.kind.clone(),
        }))
    }

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let generate_request = request.into_inner();

        let session_id = generate_request.session_id;
        let input_ids = generate_request.input_ids;
        let num_samples = generate_request.num_samples;
        let max_output_len = generate_request.max_output_len;
        let ignore_eos = generate_request.ignore_eos;
        let server_url = generate_request.server_url;

        let output = self
            .engine
            .generate(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
                server_url,
            )
            .await
            .expect("Failed to generate");

        Ok(Response::new(GenerateResponse {
            session_id: output.session_id,
            output_ids: output.output_ids,
        }))
    }

    #[allow(unused_variables)]
    async fn get_kv_agent_metadata(
        &self,
        request: Request<()>,
    ) -> Result<Response<GetKvAgentMetadataResponse>, Status> {
        let local_agent_metadata = self
            .engine
            .get_kv_agent_metadata()
            .await
            .expect("Failed to get KV agent metadat");

        Ok(Response::new(GetKvAgentMetadataResponse {
            metadata: local_agent_metadata,
        }))
    }

    async fn get_descriptors(
        &self,
        request: Request<GetDescriptorsRequest>,
    ) -> Result<Response<GetDescriptorsResponse>, Status> {
        let get_desc_request = request.into_inner();

        let session_id = get_desc_request.session_id;

        let (descs, num_blocks) = self
            .engine
            .get_descriptors(session_id)
            .await
            .expect("Failed to get descriptors");

        Ok(Response::new(GetDescriptorsResponse {
            descs,
            num_blocks: num_blocks as u64,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::logging::init_tracing();

    let args = LLMEngineArgs::parse();

    let address = args.address.parse().unwrap_or_else(|_| {
        panic!(
            "Invalid address '{}'. Expected format: <host>:<port>",
            &args.address
        )
    });

    let engine = Arc::new(
        LLMEngineWrapper::new(
            args.model_name,
            args.block_size,
            args.gpu_memory_fraction,
            args.host_kv_cache_size,
            args.max_batch_size,
            args.max_seq_len,
            args.max_num_batched_tokens,
            args.tp_size,
        )
        .await
        .expect("Failed to start API Server"),
    );

    let llm_service = LLMService {
        kind: args.kind,
        engine: engine.clone(),
    };

    // Run the engine asynchronously in the background
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        engine_clone.run_engine().await.unwrap();
    });

    let svc = LlmServer::new(llm_service);

    debug!("LLMServer listening on: {address}");

    Server::builder()
        .add_service(svc)
        .serve_with_shutdown(address, utils::signal_handler::wait_shutdown_signal())
        .await?;

    Ok(())
}
