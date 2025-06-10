use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Uri};
use tracing::debug;

use crate::pb::llm::llm_client::LlmClient;
use crate::pb::llm::{GenerateRequest, GenerateResponse};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EngineKind {
    FULL,
    PREFILL,
    DECODE,
}

impl FromStr for EngineKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" => Ok(EngineKind::FULL),
            "prefill" => Ok(EngineKind::PREFILL),
            "decode" => Ok(EngineKind::DECODE),
            _ => Err(format!("Unknown engine kind: {}", s)),
        }
    }
}

pub struct EngineRouter {
    mapping_table: Arc<Mutex<HashMap<EngineKind, VecDeque<Endpoint>>>>,
}

impl EngineRouter {
    pub fn new() -> Self {
        Self {
            mapping_table: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn add_node(&mut self, url: &str) -> Result<()> {
        let url: String = url.parse().unwrap();
        let endpoint = Channel::from_shared(url.clone().into_bytes())?;

        // TODO(jinu): Add timeout.
        let mut client = loop {
            match LlmClient::connect(endpoint.clone()).await {
                Ok(client) => {
                    break client;
                }
                Err(error) => {
                    debug!("Trying to connect to llm server ({}): {:?}", url, error);
                }
            };

            sleep(Duration::from_millis(500)).await;
        };

        let response = client.get_kind({}).await?.into_inner();
        let kind =
            EngineKind::from_str(&response.kind).unwrap_or_else(|e| panic!("Invalid kind: {e}"));

        self.mapping_table
            .lock()
            .await
            .entry(kind)
            .and_modify(|endpoints| endpoints.push_back(endpoint.clone()))
            .or_insert(vec![endpoint].into());

        Ok(())
    }

    async fn get_client(&self, kind: EngineKind) -> Result<(LlmClient<Channel>, Uri)> {
        let endpoint = {
            let mut mapping_table_guard = self.mapping_table.lock().await;

            let endpoints = mapping_table_guard
                .get_mut(&kind)
                .ok_or_else(|| anyhow::anyhow!("No endpoints for {kind:?}"))?;

            endpoints.rotate_left(1);

            endpoints
                .get(0)
                .ok_or_else(|| anyhow::anyhow!("Endpoint list for {kind:?} is empty"))?
                .clone()
        };

        let uri = endpoint.uri().clone();

        Ok((LlmClient::connect(endpoint).await?, uri))
    }

    // FIXME(jinu)
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let response;
        if let Ok((mut client, _)) = self.get_client(EngineKind::FULL).await {
            response = client.generate(request).await?.into_inner();
        } else {
            response = self.generate_dist(request).await?;
        }

        Ok(response)
    }

    pub async fn generate_dist(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let (mut prefill_client, prefill_uri) = self.get_client(EngineKind::PREFILL).await?;
        let (mut decode_client, _) = self.get_client(EngineKind::DECODE).await?;

        let num_samples = request.num_samples;
        let max_output_len = request.max_output_len;
        let ignore_eos = request.ignore_eos;

        let p_request = GenerateRequest {
            session_id: request.session_id.clone(),
            input_ids: request.input_ids.clone(),
            num_samples: 1,
            max_output_len: Some(1),
            ignore_eos,
            server_url: None,
        };

        let p_response: GenerateResponse = prefill_client.generate(p_request).await?.into_inner();

        let output_ids = p_response.output_ids;
        let max_output_len = match max_output_len {
            Some(max_output_len) => Some(max_output_len - 1),
            None => None,
        };

        let d_request = GenerateRequest {
            session_id: request.session_id.clone(),
            input_ids: [request.input_ids.clone(), output_ids].concat(),
            num_samples,
            max_output_len,
            ignore_eos,
            server_url: Some(prefill_uri.to_string()),
        };

        let d_response: GenerateResponse = decode_client.generate(d_request).await?.into_inner();
        Ok(d_response)
    }
}
