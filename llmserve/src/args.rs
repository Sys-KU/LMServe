use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct LLMEngineArgs {
    #[arg(long, default_value = "full")]
    pub kind: String,

    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value_t = 8)]
    pub block_size: usize,

    /// Fraction of total GPU memory to use (0.0 ~ 1.0)
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_fraction: f32,

    // Host-side KV cache size in GB
    #[arg(long, default_value_t = 16)]
    pub host_kv_cache_size: usize,

    #[arg(long, default_value_t = 256)]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = 4096)]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = 5120)]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = 1)]
    pub tp_size: u8,

    #[arg(long, default_value = "127.0.0.1:7000")]
    pub address: String,
}
