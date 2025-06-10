use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::join;
use tokio::sync::{Mutex, Notify};
use tracing::info;

use crate::infer_task::InferTask;
use crate::scheduler::Scheduler;
use crate::sequence::{SeqStatus, Sequence};
use crate::session_manager::SessionManager;
use crate::stub::LLMEngineStub;
use crate::worker::{KVWorkerGroup, ModelWorkerGroup};

pub fn norm_log_probs(probs: &[f32]) -> f32 {
    let log_sum: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .sum();

    log_sum / probs.len() as f32
}

type Bytes = Vec<u8>;

#[allow(dead_code)]
pub struct LLMEngine {
    model_name: String,
    block_size: usize,
    gpu_memory_fraction: f32,
    max_batch_size: usize,
    max_seq_len: usize,
    max_num_batched_tokens: usize,

    tp_size: u8,

    model_worker_group: ModelWorkerGroup,
    kv_worker_group: KVWorkerGroup,

    kv_local_agent_metadata: Vec<Bytes>,
    kv_remote_agent_table: Arc<Mutex<HashMap<String, Vec<String>>>>,

    scheduler: Arc<Mutex<Scheduler>>,
    session_manager: Arc<Mutex<SessionManager>>,

    last_log_time: u64,
}

impl LLMEngine {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_cache_size: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
    ) -> Result<LLMEngine> {
        let model_worker_group = ModelWorkerGroup::init(tp_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize model worker group: {e}"));

        let (total, peak) = model_worker_group
            .warmup(max_batch_size, max_seq_len, max_num_batched_tokens)
            .await
            .unwrap_or_else(|e| panic!("Failed to warmup worker: {e}"));

        let available = (total as f32 * gpu_memory_fraction) as u64;
        let cache_size = available.saturating_sub(peak) as usize;
        info!(
            "Allocating {:.2} GB of memory per GPU for KV cache.",
            cache_size as f64 / (1 << 30) as f64
        );

        // Convert GBs -> Bytes
        let host_cache_size = host_cache_size * (1 << 30);

        let num_blocks = model_worker_group
            .init_cache(cache_size, host_cache_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to init KV cache: {e}"));
        info!("Created {num_blocks} KV cache blocks.");

        let kv_worker_group = KVWorkerGroup::init(tp_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let kv_local_agent_metadata = kv_worker_group
            .get_local_agent_metadata()
            .await
            .unwrap_or_else(|e| panic!("Failed to get local agent metadata: {e}"));

        let scheduler = Scheduler::new(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_size,
            num_blocks as usize,
        );

        Ok(LLMEngine {
            model_name,
            block_size,
            gpu_memory_fraction,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
            model_worker_group,
            kv_worker_group,
            kv_local_agent_metadata,
            kv_remote_agent_table: Arc::new(Mutex::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(scheduler)),
            session_manager: Arc::new(Mutex::new(SessionManager::new())),
            last_log_time: 0,
        })
    }

    pub async fn add_request(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
        server_url: Option<String>,
    ) -> Result<()> {
        let mut seqs: Vec<Sequence> = Vec::new();
        for _ in 0..num_samples {
            let seq = Sequence::new(
                session_id.clone(),
                input_ids.clone(),
                max_output_len,
                ignore_eos,
            );
            seqs.push(seq);
        }

        let seq_ids: Vec<u64> = seqs.iter().map(|s| s.seq_id).collect();

        {
            self.session_manager
                .lock()
                .await
                .add_session(session_id.clone(), seq_ids.clone());
        }

        let cached = match server_url {
            Some(server_url) => {
                let peer_names = match self.check_remote_agent_metadata(server_url.clone()).await {
                    Some(peer_names) => peer_names,
                    None => {
                        let remote_kv_agent_metadata = self
                            .get_remote_kv_agent_metadata(server_url.clone())
                            .await?;
                        let new_peer_names = self
                            .add_remote_agent_metadata(server_url.clone(), remote_kv_agent_metadata)
                            .await?;
                        new_peer_names
                    }
                };

                let (remote_descs, num_blocks) = self
                    .get_remote_descriptors(server_url.clone(), session_id.clone())
                    .await?;

                self.pull_kv(
                    peer_names,
                    remote_descs,
                    num_blocks,
                    session_id.clone(),
                    seq_ids,
                )
                .await?
            }
            None => false,
        };

        for seq in seqs.iter_mut() {
            seq.cached = cached;
        }

        let infer_task = InferTask::new(session_id.clone(), seqs, utils::time::now());

        {
            self.scheduler.lock().await.add(infer_task);
        }

        Ok(())
    }

    pub async fn iter(&self) -> Option<Vec<InferTask>> {
        let (infer_inputs, fetch_block_mappings, write_back_block_mappings) =
            { self.scheduler.lock().await.schedule() };
        if infer_inputs.len() == 0 {
            return None;
        }

        let write_back_block_mappings = match fetch_block_mappings.len() > 0 {
            true => vec![],
            false => write_back_block_mappings,
        };

        let wait_before_execute = fetch_block_mappings.len() > 0;
        let record_after_execute = write_back_block_mappings.len() > 0;

        let (infer_result, transfer_result) = join!(
            self.model_worker_group
                .infer(infer_inputs, wait_before_execute, record_after_execute),
            self.kv_worker_group
                .transfer_kv(fetch_block_mappings, write_back_block_mappings),
        );

        let infer_outputs =
            infer_result.unwrap_or_else(|e| panic!("Failed to execute worker: {e}"));
        let _ = transfer_result.unwrap_or_else(|e| panic!("Failed to transfer KV: {e}"));

        let finished_tasks = { self.scheduler.lock().await.commit(infer_outputs) };

        if finished_tasks.len() > 0 {
            Some(finished_tasks)
        } else {
            None
        }
    }

    pub fn get_local_agent_metadata(&self) -> Vec<Bytes> {
        self.kv_local_agent_metadata.clone()
    }

    pub async fn check_remote_agent_metadata(&self, server_url: String) -> Option<Vec<String>> {
        let kv_agent_table = self.kv_remote_agent_table.lock().await;

        kv_agent_table.get(&server_url).cloned()
    }

    pub async fn add_remote_agent_metadata(
        &self,
        server_url: String,
        remote_agent_metadata: Vec<Bytes>,
    ) -> Result<Vec<String>> {
        let mut kv_agent_table = self.kv_remote_agent_table.lock().await;

        let peer_names = kv_agent_table
            .entry(server_url.clone())
            .or_insert_with(|| Vec::new());

        if !peer_names.is_empty() {
            return Ok(peer_names.clone());
        }

        let new_peer_names = self
            .kv_worker_group
            .add_remote_agent_metadata(remote_agent_metadata)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to set KV agent metadata: {e}"))?;

        kv_agent_table.insert(server_url, new_peer_names.clone());

        Ok(new_peer_names)
    }

    pub async fn get_descriptors(&self, session_id: String) -> Result<(Vec<Bytes>, usize)> {
        let seq_ids = {
            self.session_manager
                .lock()
                .await
                .get_seq_ids(session_id)
                .expect("Not found session id")
        };

        let (descs, num_blocks) = self
            .kv_worker_group
            .get_descriptors(seq_ids)
            .await
            .unwrap_or_else(|e| panic!("Failed to get descriptors: {e}"));

        Ok((descs, num_blocks))
    }

    pub async fn get_remote_descriptors(
        &self,
        remote_url: String,
        session_id: String,
    ) -> Result<(Vec<Bytes>, usize)> {
        let (descs, num_blocks) = LLMEngineStub::get_descriptors(remote_url, session_id).await?;

        Ok((descs, num_blocks))
    }

    pub async fn get_remote_kv_agent_metadata(&self, remote_url: String) -> Result<Vec<Bytes>> {
        Ok(LLMEngineStub::get_remote_kv_agent_metadata(remote_url).await?)
    }

    pub async fn pull_kv(
        &self,
        peer_names: Vec<String>,
        remote_descs: Vec<Bytes>,
        num_blocks: usize,
        session_id: String,
        seq_ids: Vec<u64>,
    ) -> Result<bool> {
        let ret = self
            .kv_worker_group
            .pull_kv(peer_names, remote_descs, num_blocks, session_id, seq_ids)
            .await
            .unwrap_or_else(|e| panic!("Failed to pull KVs: {e}"));

        Ok(ret)
    }
}

pub struct LLMEngineOutput {
    pub session_id: String,
    pub output_ids: Vec<u32>,
}

pub struct LLMEngineWrapper {
    engine: Arc<LLMEngine>,

    request_events: Mutex<HashMap<String, Arc<Notify>>>,
    request_outputs: Mutex<HashMap<String, InferTask>>,
}

impl LLMEngineWrapper {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_kv_cache_size: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
    ) -> Result<LLMEngineWrapper> {
        let llm_engine = LLMEngine::new(
            model_name,
            block_size,
            gpu_memory_fraction,
            host_kv_cache_size,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
        )
        .await?;

        Ok(LLMEngineWrapper {
            engine: Arc::new(llm_engine),
            request_events: Mutex::new(HashMap::new()),
            request_outputs: Mutex::new(HashMap::new()),
        })
    }

    pub async fn generate(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
        server_url: Option<String>,
    ) -> Result<LLMEngineOutput> {
        let notify = Arc::new(Notify::new());
        {
            self.request_events
                .lock()
                .await
                .insert(session_id.clone(), notify.clone());
        }

        self.engine
            .add_request(
                input_ids,
                num_samples,
                session_id.clone(),
                max_output_len,
                ignore_eos,
                server_url,
            )
            .await?;

        notify.notified().await;

        let request_output = {
            self.request_outputs
                .lock()
                .await
                .remove(&session_id)
                .unwrap()
        };
        let seqs = request_output.get_seqs(SeqStatus::FINISHED);
        let selected_seq = seqs
            .iter()
            .max_by(|a, b| {
                norm_log_probs(a.get_token_probs().as_ref())
                    .partial_cmp(&norm_log_probs(b.get_token_probs().as_ref()))
                    .unwrap()
            })
            .expect("Failed to select sequence: max_by() returned None.");

        let output_ids = selected_seq.get_output_ids();

        Ok(LLMEngineOutput {
            session_id: session_id.clone(),
            output_ids: output_ids.to_vec(),
        })
    }

    pub async fn get_descriptors(&self, session_id: String) -> Result<(Vec<Bytes>, usize)> {
        let (descs, num_blocks) = self.engine.get_descriptors(session_id).await?;

        Ok((descs, num_blocks))
    }

    pub async fn get_kv_agent_metadata(&self) -> Result<Vec<Bytes>> {
        Ok(self.engine.kv_local_agent_metadata.clone())
    }

    pub async fn run_engine(&self) -> Result<()> {
        loop {
            let outputs = self.engine.iter().await;
            let Some(outputs) = outputs else {
                continue;
            };
            for infer_task in outputs.into_iter() {
                let session_id = infer_task.get_session_id();

                self.request_outputs
                    .lock()
                    .await
                    .insert(session_id.clone(), infer_task);

                let request_event = self
                    .request_events
                    .lock()
                    .await
                    .remove(&session_id)
                    .unwrap();
                request_event.notify_waiters()
            }
        }
    }
}
