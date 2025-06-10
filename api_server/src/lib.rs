pub mod args;
pub mod pb;
pub mod router;

use pb::llm::GenerateRequest;
use router::EngineRouter;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};

#[derive(Serialize)]
pub struct GenerateOutput {
    pub session_id: String,
    pub token_ids: Vec<u32>,
    pub output_text: String,
    pub output_len: usize,
}

#[derive(Deserialize)]
pub struct GenerateParams {
    pub prompt: String,
    pub num_samples: u16,
    pub session_id: Option<String>,
    pub max_output_len: Option<u64>,
    pub ignore_eos: bool,
}

pub struct APIServer {
    pub router: EngineRouter,
    pub tokenizer: Tokenizer,
}

impl APIServer {
    pub fn new(router: EngineRouter, tokenizer: Tokenizer) -> Self {
        Self { router, tokenizer }
    }

    pub async fn generate(&self, params: GenerateParams) -> Result<GenerateOutput> {
        // If no session ID is given, generate a new one
        let session_id = params
            .session_id
            .unwrap_or(utils::random::generate_session_id());

        let input_ids = self
            .tokenizer
            .encode(params.prompt, false)?
            .get_ids()
            .to_vec();

        let response = self
            .router
            .generate(GenerateRequest {
                session_id: session_id,
                input_ids: input_ids.clone(),
                num_samples: params.num_samples as u32,
                max_output_len: params.max_output_len,
                ignore_eos: params.ignore_eos,
                server_url: None,
            })
            .await?;

        let session_id = response.session_id;
        let output_ids = response.output_ids;
        let output_text = self.tokenizer.decode(&output_ids, false)?;
        let output_len = output_ids.len();

        Ok(GenerateOutput {
            session_id,
            token_ids: [input_ids, output_ids].concat(),
            output_text,
            output_len,
        })
    }
}
