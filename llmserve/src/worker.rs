use std::{collections::HashMap, env, sync::Arc, time::Duration};

use futures::future::join_all;

use tokio::{sync::Mutex, time::sleep};

use tonic::transport::Channel;

use tracing::{debug, info};

use crate::infer_task::{InferInput, InferOutput};
use crate::pb::worker::kv_worker_client::KvWorkerClient;
use crate::pb::worker::worker_client::WorkerClient;
use crate::pb::worker::{
    AgentMetadata, BlockMapping, GetDescriptorsRequest, InferRequest, InitCacheRequest,
    KvTransferRequest, PullKvRequest, WarmupRequest,
};

type Bytes = Vec<u8>;

type StdError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = StdError> = std::result::Result<T, E>;

#[tonic::async_trait]
pub trait GrpcClient: Send + Sync + Sized + 'static {
    async fn connect(address: String) -> Result<Self, tonic::transport::Error>;
}

#[tonic::async_trait]
impl GrpcClient for WorkerClient<Channel> {
    async fn connect(address: String) -> Result<Self, tonic::transport::Error> {
        WorkerClient::connect(address).await
    }
}

#[tonic::async_trait]
impl GrpcClient for KvWorkerClient<Channel> {
    async fn connect(address: String) -> Result<Self, tonic::transport::Error> {
        KvWorkerClient::connect(address).await
    }
}

#[tonic::async_trait]
trait Worker: Send + Sync + Sized + 'static {
    type Client: GrpcClient;

    async fn connect(address: String) -> Result<Self>;
}

pub struct WorkerImpl<T: GrpcClient> {
    client: Arc<Mutex<T>>,
}

#[tonic::async_trait]
impl<T: GrpcClient> Worker for WorkerImpl<T> {
    type Client = T;
    async fn connect(address: String) -> Result<Self> {
        // TODO(jinu): Add timeout
        let client = loop {
            match T::connect(address.clone()).await {
                Ok(client) => {
                    info!("Successfully connected to {address}");
                    break client;
                }
                Err(error) => {
                    debug!("Trying to connect to worker ({}): {:?}", address, error);
                    sleep(Duration::from_millis(500)).await;
                }
            }
        };

        Ok(Self {
            client: Arc::new(Mutex::new(client)),
        })
    }
}

type ModelWorker = WorkerImpl<WorkerClient<Channel>>;
type KVWorker = WorkerImpl<KvWorkerClient<Channel>>;

impl ModelWorker {
    async fn warmup(&self, request: WarmupRequest) -> Result<(u64, u64)> {
        let response = self.client.lock().await.warmup(request).await?;
        let response = response.into_inner();
        Ok((response.gpu_total_mem_size, response.gpu_peak_mem_size))
    }

    async fn infer(&self, request: InferRequest) -> Result<HashMap<u64, InferOutput>> {
        let response = self.client.lock().await.infer(request).await?;
        let outputs = response.into_inner().outputs;
        Ok(outputs)
    }

    async fn init_cache(&self, request: InitCacheRequest) -> Result<u64> {
        let response = self.client.lock().await.init_cache(request).await?;
        let num_blocks = response.into_inner().num_blocks;
        Ok(num_blocks)
    }
}

impl KVWorker {
    async fn transfer_kv(&self, request: KvTransferRequest) -> Result<()> {
        let _ = self.client.lock().await.transfer_kv(request).await?;
        Ok(())
    }

    async fn get_local_agent_metadata(&self) -> Result<Vec<u8>> {
        let response = self
            .client
            .lock()
            .await
            .get_local_agent_metadata(())
            .await?;
        let metadata = response.into_inner().data;
        Ok(metadata)
    }

    async fn add_remote_agent_metadata(&self, request: AgentMetadata) -> Result<String> {
        let response = self
            .client
            .lock()
            .await
            .add_remote_agent_metadata(request)
            .await?;
        let peer_name = response.into_inner().peer_name;
        Ok(peer_name)
    }

    async fn get_descriptors(&self, request: GetDescriptorsRequest) -> Result<(Bytes, usize)> {
        let response = self
            .client
            .lock()
            .await
            .get_descriptors(request)
            .await?
            .into_inner();

        let descs = response.descs;
        let num_blocks = response.num_blocks;
        Ok((descs, num_blocks as usize))
    }

    async fn pull_kv(&self, request: PullKvRequest) -> Result<bool> {
        let response = self
            .client
            .lock()
            .await
            .pull_kv(request)
            .await?
            .into_inner();

        Ok(response.success)
    }
}

#[tonic::async_trait]
trait WorkerGroup: Send + Sync + Sized + 'static {
    type Worker: Worker;

    async fn run_workers_gather<V, P, E, Fut, F>(&self, params: P, task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<Self::Worker>, P) -> Fut + Copy + Send + Sync + 'static;

    async fn run_workers<V, P, E, Fut, F>(
        &self,
        params_list: &[P],
        task_fn: F,
    ) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<Self::Worker>, P) -> Fut + Copy + Send + Sync + 'static;
}

pub struct WorkerGroupImpl<T> {
    workers: Vec<Arc<T>>,
}

#[tonic::async_trait]
impl<T: Worker> WorkerGroup for WorkerGroupImpl<T> {
    type Worker = T;

    async fn run_workers_gather<V, P, E, Fut, F>(&self, params: P, task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<T>, P) -> Fut + Copy + Send + Sync + 'static,
    {
        let mut handles = Vec::with_capacity(self.workers.len());

        for worker in self.workers.iter() {
            let w = Arc::clone(worker);
            let p = params.clone();

            let handle = tokio::spawn(async move { task_fn(w, p).await });

            handles.push(handle);
        }

        let results = join_all(handles).await;

        let mut outputs = Vec::with_capacity(results.len());
        for res in results {
            match res {
                Ok(Ok(val)) => outputs.push(val),
                Ok(Err(_)) => panic!("Failed to run worker"),
                Err(join_err) => {
                    panic!("tokio::spawn task panicked: {:?}", join_err);
                }
            }
        }

        Ok(outputs)
    }

    async fn run_workers<V, P, E, Fut, F>(&self, params_list: &[P], task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<T>, P) -> Fut + Copy + Send + Sync + 'static,
    {
        assert!(
            self.workers.len() == params_list.len(),
            "Mismatch in number of workers and number of requests: workers = {}, requests = {}",
            self.workers.len(),
            params_list.len(),
        );

        let mut handles = Vec::with_capacity(self.workers.len());

        for (i, worker) in self.workers.iter().enumerate() {
            let w = Arc::clone(worker);
            let p = params_list[i].clone();

            let handle = tokio::spawn(async move { task_fn(w, p).await });

            handles.push(handle);
        }

        let results = join_all(handles).await;

        let mut outputs = Vec::with_capacity(results.len());
        for res in results {
            match res {
                Ok(Ok(val)) => outputs.push(val),
                Ok(Err(_)) => panic!("Failed to run worker"),
                Err(join_err) => {
                    panic!("tokio::spawn task panicked: {:?}", join_err);
                }
            }
        }

        Ok(outputs)
    }
}

fn extract_if_all_equal<T>(items: &[T]) -> Result<T, &'static str>
where
    T: PartialEq + Clone,
{
    if let Some(first) = items.first() {
        if items.iter().all(|x| x == first) {
            Ok(first.clone())
        } else {
            Err("Values are not all equal")
        }
    } else {
        Err("Vector is empty")
    }
}

pub type ModelWorkerGroup = WorkerGroupImpl<ModelWorker>;
pub type KVWorkerGroup = WorkerGroupImpl<KVWorker>;

impl ModelWorkerGroup {
    pub async fn init(num_workers: u8) -> Result<Self> {
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        let worker_group_uds_path =
            env::var("WORKER_GROUP_UDS_PATH").expect("WORKER_GROUP_UDS_PATH env is not set");
        for rank in 0..num_workers {
            let address = format!("unix://{}/model-{}", worker_group_uds_path, rank);
            let worker = ModelWorker::connect(address.parse().unwrap())
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(Self { workers })
    }

    pub async fn warmup(
        &self,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
    ) -> Result<(u64, u64)> {
        let request = WarmupRequest {
            max_batch_size: max_batch_size as u64,
            max_seq_len: max_seq_len as u64,
            max_num_batched_tokens: max_num_batched_tokens as u64,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.warmup(request).await
            })
            .await?;

        let (total, peak) = match outputs.into_iter().min_by_key(|x| x.0) {
            Some(min_values) => min_values,
            None => panic!("Outputs is empty"),
        };

        Ok((total, peak))
    }

    pub async fn init_cache(&self, gpu_cache_size: usize, host_cache_size: usize) -> Result<u64> {
        let request = InitCacheRequest {
            gpu_cache_size: gpu_cache_size as u64,
            host_cache_size: host_cache_size as u64,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.init_cache(request).await
            })
            .await?;

        let num_blocks = extract_if_all_equal(&outputs).expect("Error: not all values are equal");

        Ok(num_blocks)
    }

    pub async fn infer(
        &self,
        inputs: Vec<InferInput>,
        wait_before_execute: bool,
        record_after_execute: bool,
    ) -> Result<HashMap<u64, InferOutput>> {
        let request = InferRequest {
            inputs,
            use_cache: true,
            wait_before_execute,
            record_after_execute,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.infer(request).await
            })
            .await?;

        let infer_outputs =
            extract_if_all_equal(&outputs).expect("Error: not all values are equal");

        Ok(infer_outputs)
    }
}

impl KVWorkerGroup {
    pub async fn init(num_workers: u8) -> Result<Self> {
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        let worker_group_uds_path =
            env::var("WORKER_GROUP_UDS_PATH").expect("WORKER_GROUP_UDS_PATH env is not set");
        for rank in 0..num_workers {
            let address = format!("unix://{}/model-{}-kv", worker_group_uds_path, rank);
            let worker = KVWorker::connect(address.parse().unwrap())
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(Self { workers })
    }

    pub async fn transfer_kv(
        &self,
        fetch_block_mappings: Vec<BlockMapping>,
        write_back_block_mappings: Vec<BlockMapping>,
    ) -> Result<()> {
        let request = KvTransferRequest {
            fetch_block_mappings,
            write_back_block_mappings,
        };

        let _ = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.transfer_kv(request).await
            })
            .await?;

        Ok(())
    }

    pub async fn get_local_agent_metadata(&self) -> Result<Vec<Vec<u8>>> {
        let metadata: Vec<Vec<u8>> = self
            .run_workers_gather((), move |worker, _| async move {
                worker.get_local_agent_metadata().await
            })
            .await?;

        Ok(metadata)
    }

    pub async fn add_remote_agent_metadata(
        &self,
        agent_metadata: Vec<Vec<u8>>,
    ) -> Result<Vec<String>> {
        let mut requests: Vec<_> = Vec::with_capacity(self.workers.len());
        for md in agent_metadata {
            requests.push(AgentMetadata { data: md })
        }

        let peer_names = self
            .run_workers(&requests, move |worker, request| async move {
                worker.add_remote_agent_metadata(request).await
            })
            .await?;

        Ok(peer_names)
    }

    pub async fn get_descriptors(&self, seq_ids: Vec<u64>) -> Result<(Vec<Bytes>, usize)> {
        let num_seqs = seq_ids.len();
        let request = GetDescriptorsRequest { seq_ids };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.get_descriptors(request).await
            })
            .await?;

        let mut descs: Vec<Bytes> = Vec::with_capacity(num_seqs);
        let mut num_blocks_list: Vec<usize> = Vec::with_capacity(num_seqs);

        for (desc, num_blocks) in outputs {
            descs.push(desc);
            num_blocks_list.push(num_blocks);
        }
        let num_blocks =
            extract_if_all_equal(&num_blocks_list).expect("Error: not all values are equal");

        Ok((descs, num_blocks))
    }

    pub async fn pull_kv(
        &self,
        peer_names: Vec<String>,
        descs: Vec<Bytes>,
        num_blocks: usize,
        session_id: String,
        seq_ids: Vec<u64>,
    ) -> Result<bool> {
        assert!(descs.len() == self.workers.len());

        let mut requests: Vec<_> = Vec::with_capacity(self.workers.len());
        for (peer_name, desc) in peer_names.into_iter().zip(descs) {
            requests.push(PullKvRequest {
                peer_name,
                descs: desc,
                num_blocks: num_blocks as u64,
                session_id: session_id.clone(),
                seq_ids: seq_ids.clone(),
            })
        }

        let rets = self
            .run_workers(&requests, move |worker, request| async move {
                worker.pull_kv(request).await
            })
            .await?;

        let all_success = rets.iter().all(|&x| x);
        Ok(all_success)
    }
}
