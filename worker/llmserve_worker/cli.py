import typer
import grpc
import asyncio
import signal
import torch
import torch.multiprocessing as mp
import logging

from loguru import logger
from google.protobuf.empty_pb2 import Empty

from llmserve_worker import ModelWorker, KVWorker
from llmserve_worker.kv_worker import KVWorkerParams
from llmserve_worker.pb import worker_pb2_grpc
from llmserve_worker.pb.worker_pb2 import (
    WarmupRequest,
    WarmupResponse,
    InferRequest,
    InferResponse,
    InitCacheRequest,
    InitCacheResponse,
    KVTransferRequest,
    KVTransferResponse,
    AgentMetadata,
    AddRemoteAgentMetadataResponse,
    GetDescriptorsRequest,
    GetDescriptorsResponse,
    PullKVRequest,
    PullKVResponse,
)


app = typer.Typer()


class WorkerService(worker_pb2_grpc.WorkerServicer):

    def __init__(self, worker: ModelWorker, uds_path: str):
        self.worker = worker
        self.uds_path = uds_path
        self.kv_worker = None

    async def Warmup(
        self,
        request: WarmupRequest,
        context: grpc.aio.ServicerContext,
    ) -> WarmupResponse:
        max_batch_size = request.max_batch_size
        max_seq_len = request.max_seq_len
        max_num_batched_tokens = request.max_num_batched_tokens

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.worker.warmup(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        _, total = torch.cuda.mem_get_info()

        return WarmupResponse(gpu_total_mem_size=total, gpu_peak_mem_size=peak)

    async def Infer(
        self,
        request: InferRequest,
        context: grpc.aio.ServicerContext,
    ) -> InferResponse:
        inputs = request.inputs
        use_cache = request.use_cache
        wait_before_execute = request.wait_before_execute
        record_after_execute = request.record_after_execute

        outputs = self.worker.execute(
            inputs,
            use_cache,
            wait_before_execute,
            record_after_execute,
        )

        return InferResponse(outputs=outputs)

    async def InitCache(
        self,
        request: InitCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> InitCacheResponse:
        gpu_cache_size = request.gpu_cache_size
        host_cache_size = request.host_cache_size

        num_blocks, kv_worker_params = self.worker.init_cache(gpu_cache_size)

        kv_uds_path = "{}-kv".format(self.uds_path)

        kv_worker = mp.Process(target=init_kv_cache,
                               args=(
                                   kv_worker_params,
                                   host_cache_size,
                                   torch.cuda.current_device(),
                                   kv_uds_path,
                               ))
        kv_worker.start()
        self.kv_worker = kv_worker

        return InitCacheResponse(num_blocks=num_blocks)


class KVWorkerService(worker_pb2_grpc.KVWorkerServicer):

    def __init__(self, worker: KVWorker):
        self.worker = worker

    async def TransferKV(
        self,
        request: KVTransferRequest,
        context: grpc.aio.ServicerContext,
    ) -> KVTransferResponse:
        fetch_block_mappings = request.fetch_block_mappings
        write_back_block_mappings = request.write_back_block_mappings

        self.worker.transfer(fetch_block_mappings, write_back_block_mappings)

        return KVTransferResponse(success=True)

    async def GetLocalAgentMetadata(
        self,
        request: Empty,
        context: grpc.aio.ServicerContext,
    ) -> AgentMetadata:
        return AgentMetadata(data=self.worker.kv_agent_metadata)

    async def AddRemoteAgentMetadata(
        self,
        request: AgentMetadata,
        context: grpc.aio.ServicerContext,
    ) -> AddRemoteAgentMetadataResponse:
        remote_agent_metadata = request.data

        # TODO(jinu): Add error handling
        peer_name = self.worker.kv_agent.add_remote_agent(
            remote_agent_metadata)

        return AddRemoteAgentMetadataResponse(peer_name=peer_name)

    async def GetDescriptors(
        self,
        request: GetDescriptorsRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetDescriptorsResponse:
        seq_ids = request.seq_ids

        descs, num_blocks = self.worker.get_descriptors(seq_ids)

        return GetDescriptorsResponse(descs=descs, num_blocks=num_blocks)

    async def PullKV(
        self,
        request: PullKVRequest,
        context: grpc.aio.ServicerContext,
    ) -> PullKVResponse:
        peer_name = request.peer_name
        descs = request.descs
        num_blocks = request.num_blocks
        session_id = request.session_id
        seq_ids = request.seq_ids

        ret = self.worker.pull_kv(
            peer_name,
            descs,
            num_blocks,
            session_id,
            seq_ids,
        )

        return PullKVResponse(success=ret)


def init_kv_cache(
    params: KVWorkerParams,
    host_cache_size: int,
    device: int,
    uds_path: str,
):
    # Enable console output of errors from gRPC methods
    logging.basicConfig(level=logging.INFO)

    torch.cuda.set_device(device)

    # TODO(jinu): Set the name using local IP instead of uds path.
    worker = KVWorker(
        name=uds_path,
        params=params,
        host_cache_size=host_cache_size,
    )
    logger.info(f"Launching KV Cache on GPU {device}...")

    async def kv_cache_inner(worker: KVWorker, uds_path: str):
        server = grpc.aio.server()
        worker_pb2_grpc.add_KVWorkerServicer_to_server(KVWorkerService(worker),
                                                       server)
        address = f"unix://{uds_path}"
        server.add_insecure_port(address)

        logger.info(f"KV worker is ready and listening on: {address}")

        await server.start()

        shutdown_event = asyncio.Event()

        def handle_signal():
            logger.info("Shutdown signal received.")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        await shutdown_event.wait()

        logger.info("Shutting the KV worker.")

    asyncio.run(kv_cache_inner(worker, uds_path))


@app.command()
def serve(
    model_name: str,
    block_size: int,
    device: int,
    uds_path: str,
):
    # Enable console output of errors from gRPC methods
    logging.basicConfig(level=logging.INFO)

    mp.set_start_method("spawn", force=True)

    torch.cuda.set_device(device)

    logger.info(f"Launching {model_name} model on GPU {device}...")
    worker = ModelWorker(model_name, block_size)

    async def serve_inner(worker: ModelWorker, uds_path: str):

        server = grpc.aio.server()
        worker_pb2_grpc.add_WorkerServicer_to_server(
            WorkerService(worker, uds_path), server)
        address = f"unix://{uds_path}"
        server.add_insecure_port(address)

        logger.info(f"Model worker is ready and listening on: {address}")

        await server.start()

        shutdown_event = asyncio.Event()

        def handle_signal():
            logger.info("Shutdown signal received.")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        await shutdown_event.wait()

        logger.info("Shutting the model worker.")

    asyncio.run(serve_inner(worker, uds_path))


if __name__ == "__main__":
    app()
