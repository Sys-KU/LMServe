use anyhow::Result;

use crate::pb::llm::GetDescriptorsRequest;
use crate::pb::llm::llm_client::LlmClient;

type Bytes = Vec<u8>;

pub struct LLMEngineStub {}

impl LLMEngineStub {
    pub async fn get_descriptors(
        remote_url: String,
        session_id: String,
    ) -> Result<(Vec<Bytes>, usize)> {
        let request = GetDescriptorsRequest { session_id };

        let mut client = LlmClient::connect(remote_url.clone()).await?;
        let response = client.get_descriptors(request).await?.into_inner();

        let descs = response.descs;
        let num_blocks = response.num_blocks;

        Ok((descs, num_blocks as usize))
    }

    pub async fn get_remote_kv_agent_metadata(remote_url: String) -> Result<Vec<Bytes>> {
        let mut client = LlmClient::connect(remote_url.clone()).await?;
        let response = client.get_kv_agent_metadata(()).await?.into_inner();

        let metadata = response.metadata;

        Ok(metadata)
    }
}
