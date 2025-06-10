use clap::Parser;
use serde::{Deserialize, Serialize};

macro_rules! default_fn {
    ($name:ident, $ty:ty, $val:expr) => {
        fn $name() -> $ty {
            $val
        }
    };
}

default_fn!(default_block_size, usize, 8);
default_fn!(default_gpu_memory_fraction, f32, 0.8);
default_fn!(default_host_kv_cache_size, usize, 16);
default_fn!(default_max_batch_size, usize, 256);
default_fn!(default_max_seq_len, usize, 4096);
default_fn!(default_max_num_batched_tokens, usize, 5120);
default_fn!(default_tp_size, u8, 1);

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct CLIArgs {
    #[arg(long)]
    pub config: Option<String>,

    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value_t = default_block_size())]
    pub block_size: usize,

    /// Fraction of total GPU memory to use (0.0 ~ 1.0)
    #[arg(long, default_value_t = default_gpu_memory_fraction())]
    pub gpu_memory_fraction: f32,

    // Host-side KV cache size in GB
    #[arg(long, default_value_t = default_host_kv_cache_size())]
    pub host_kv_cache_size: usize,

    #[arg(long, default_value_t = default_max_batch_size())]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = default_max_seq_len())]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = default_max_num_batched_tokens())]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = default_tp_size())]
    pub tp_size: u8,

    #[arg(long, default_value_t = 1)]
    pub num_worker_groups: u32,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub address: String,
}

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub model_name: String,

    pub api_server: APIServerConfig,

    pub llm_servers: Vec<LLMServerConfig>,
}

#[derive(Debug, Deserialize)]
pub struct APIServerConfig {
    pub address: String,
}

#[derive(Debug, Deserialize)]
pub struct LLMServerConfig {
    pub kind: String,

    pub address: String,

    #[serde(default = "default_block_size")]
    pub block_size: usize,

    #[serde(default = "default_gpu_memory_fraction")]
    pub gpu_memory_fraction: f32,

    #[serde(default = "default_host_kv_cache_size")]
    pub host_kv_cache_size: usize,

    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,

    #[serde(default = "default_tp_size")]
    pub tp_size: u8,
}
