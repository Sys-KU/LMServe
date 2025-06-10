use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::process::{Child, Command};

use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use tracing::info;

use api_server::args::APIServerArgs;
use launcher::args::{APIServerConfig, AppConfig, CLIArgs, LLMServerConfig};
use llmserve::args::LLMEngineArgs;

fn to_cmd_args<T: Serialize>(args: &T) -> Vec<String> {
    let value = serde_json::to_value(args).expect("serialization failed");
    let obj = value.as_object().expect("expected a struct-like object");

    obj.iter()
        .map(|(k, v)| {
            let key = k.replace("_", "-");
            let val = match v {
                Value::String(s) => s.clone(),
                Value::Array(vec) => {
                    let vec: Vec<String> = vec
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        })
                        .collect();
                    vec.join(" ")
                }
                _ => v.to_string(),
            };
            format!("--{}={}", key, val)
        })
        .collect()
}

const LLMSERVE_BASE_PORT: u32 = 7000;

const WORKER_GROUP_UDS_PATH_PREFIX: &str = "/tmp/llmserve/group";

const MASTER_ADDR: &str = "localhost";
const MASTER_PORT: u32 = 6000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::logging::init_tracing();

    let args = CLIArgs::parse();

    let config: AppConfig = match args.config {
        Some(config) => {
            let file = fs::File::open(config)?;
            let reader = BufReader::new(file);

            serde_yaml::from_reader(reader)?
        }
        None => {
            let api_server_config = APIServerConfig {
                address: args.address.clone(),
            };

            let llm_server_configs = (0..args.num_worker_groups)
                .map(|i| {
                    let port = LLMSERVE_BASE_PORT + i;

                    LLMServerConfig {
                        kind: "full".to_string(),
                        block_size: args.block_size,
                        gpu_memory_fraction: args.gpu_memory_fraction,
                        host_kv_cache_size: args.host_kv_cache_size,
                        max_batch_size: args.max_batch_size,
                        max_seq_len: args.max_seq_len,
                        max_num_batched_tokens: args.max_num_batched_tokens,
                        tp_size: args.tp_size,
                        address: format!("127.0.0.1:{port}"),
                    }
                })
                .collect();

            AppConfig {
                model_name: args.model_name,
                api_server: api_server_config,
                llm_servers: llm_server_configs,
            }
        }
    };

    let model_name = config.model_name;

    let mut llm_servers: Vec<Child> = Vec::with_capacity(config.llm_servers.len() as usize);
    let mut workers: Vec<Child> = Vec::new();
    for (group_id, llm_server) in config.llm_servers.iter().enumerate() {
        for rank in 0..llm_server.tp_size {
            let uds_path = format!(
                "{}-{}/model-{}",
                WORKER_GROUP_UDS_PATH_PREFIX, group_id, rank
            );

            std::fs::create_dir_all(Path::new(&uds_path).parent().unwrap())?;
            if Path::new(&uds_path).exists() {
                fs::remove_file(Path::new(&uds_path))
                    .expect("Failed to remove existing UDS socket file");
            }

            let worker_args = vec![
                model_name.clone(),
                args.block_size.to_string(),
                /*device=*/ workers.len().to_string(),
                uds_path.to_string(),
            ];

            let mut worker_envs = HashMap::new();
            worker_envs.insert("RANK", rank.to_string());
            worker_envs.insert("WORLD_SIZE", llm_server.tp_size.to_string());
            worker_envs.insert("MASTER_ADDR", MASTER_ADDR.to_string());
            worker_envs.insert("MASTER_PORT", (MASTER_PORT + group_id as u32).to_string());

            info!("Launching llmserve-worker (group_id = {group_id}, rank = {rank})...");
            let worker = Command::new("llmserve-worker")
                .args(worker_args)
                .envs(worker_envs)
                .spawn()?;

            workers.push(worker);
        }

        let engine_args = LLMEngineArgs {
            kind: llm_server.kind.clone(),
            model_name: model_name.clone(),
            block_size: llm_server.block_size,
            gpu_memory_fraction: llm_server.gpu_memory_fraction,
            host_kv_cache_size: llm_server.host_kv_cache_size,
            max_batch_size: llm_server.max_batch_size,
            max_seq_len: llm_server.max_seq_len,
            max_num_batched_tokens: llm_server.max_num_batched_tokens,
            tp_size: llm_server.tp_size,
            address: llm_server.address.clone(),
        };

        let mut engine_envs = HashMap::new();
        let worker_group_uds_path = format!("{}-{}", WORKER_GROUP_UDS_PATH_PREFIX, group_id);
        engine_envs.insert("WORKER_GROUP_UDS_PATH", worker_group_uds_path.to_string());

        info!("Launching llmserve (group_id = {group_id})...");
        let engine = Command::new("llmserve")
            .args(&to_cmd_args(&engine_args))
            .envs(engine_envs)
            .spawn()?;

        llm_servers.push(engine);
    }

    let llm_server_addresses = config
        .llm_servers
        .iter()
        .map(|x| x.address.clone())
        .collect();

    let api_server_args = APIServerArgs {
        model_name: model_name.clone(),
        address: config.api_server.address,
        llm_server_addresses,
    };

    info!("Launching api_server...");
    let mut api_server = Command::new("api_server")
        .args(&to_cmd_args(&api_server_args))
        .spawn()?;

    utils::signal_handler::wait_shutdown_signal().await;

    for mut worker in workers {
        worker.wait()?;
    }
    for mut llm_server in llm_servers {
        llm_server.wait()?;
    }
    api_server.wait()?;

    Ok(())
}
