import torch
import pycuda.driver as cuda
import numpy as np
import torch.multiprocessing as mp
import enum

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple, Any
from nixl._api import nixl_agent

from llmserve_worker.pb.worker_pb2 import BlockMapping

dtype_map = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,
    torch.int32: np.int32,
}


@dataclass
class KVWorkerHandle:
    pre_events: List[cuda.Event]
    post_events: List[cuda.Event]
    model_queue: mp.Queue
    kv_queue: mp.Queue


@dataclass
class KVWorkerParams:
    kv_cache_metadata: Tuple[Any, ...]
    num_layers: int
    num_blocks: int
    dtype: torch.dtype
    pre_event_handles: List[bytes]
    post_event_handles: List[bytes]
    model_queue: mp.Queue
    kv_queue: mp.Queue


class BlockStatus(enum.Enum):
    FREE = enum.auto()
    USED = enum.auto()


@dataclass
class KVBlock:
    block_id: int
    data: np.ndarray
    ref_cnt: int = 0
    status: BlockStatus = BlockStatus.FREE

    def __repr__(self):
        return (f"KVBlock(block_id={self.block_id}, shape={self.data.shape},"
                f"ref_cnt={self.ref_cnt}, status={self.status.name})")


@dataclass
class KVBlockMap:
    blocks: List[KVBlock] = field(default_factory=list)
    block_offset: int = 0

    def __repr__(self):
        return (f"KVBlockMap(block_offset={self.block_offset}, "
                f"num_blocks={len(self.blocks)}, "
                f"blocks=[{', '.join(b.block_id for b in self.blocks)}])")


class KVStore:

    def __init__(self, gpu_kv_caches, host_cache_size: int):
        self.gpu_kv_caches = gpu_kv_caches
        # NOTE(jinu): This block_size does not refer to PagedAttention's block size,
        # but to the total number of elements in a block (block_size × hidden_dim).
        num_layers, num_blocks, block_size = gpu_kv_caches.shape

        # NOTE(jinu): Cache each layer's base pointer to avoid repeated .data_ptr() calls,
        # which can incur non-negligible overhead inside tight loops.
        self.gpu_kv_cache_ptrs = [
            self.gpu_kv_caches[layer_idx].data_ptr()
            for layer_idx in range(num_layers)
        ]

        num_host_blocks = host_cache_size // (num_layers * block_size)

        torch_dtype = gpu_kv_caches.dtype
        np_dtype = dtype_map.get(torch_dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype: {torch_dtype}")

        self.host_kv_caches = cuda.pagelocked_empty(
            (num_host_blocks, num_layers, block_size),
            np_dtype,
            order="C",
        )

        self.free_blocks: List[KVBlock] = [
            KVBlock(i, self.host_kv_caches[i]) for i in range(num_host_blocks)
        ]
        self.block_mapping_table: OrderedDict[str, KVBlockMap] = OrderedDict()

        self.block_stride = block_size * gpu_kv_caches.element_size()

    def free(self, block_map: KVBlockMap) -> None:
        for block in block_map.blocks:
            block.ref_cnt -= 1
            if block.ref_cnt < 0:
                raise RuntimeError("Double free detected for host block")
            elif block.ref_cnt == 0:
                block.status = BlockStatus.FREE
                self.free_blocks.append(block)

    def alloc_blocks(self, num_blocks: int) -> List[KVBlock]:
        # Evict blocks if needed.
        while len(self.free_blocks) < num_blocks:
            if not self.block_mapping_table:
                raise RuntimeError(
                    "Cannot allocate block: all host blocks are already in use"
                )
            victim_seq_id, victim_block_map = self.block_mapping_table.popitem(
                last=False)
            self.free(victim_block_map)

        blocks = self.free_blocks[-num_blocks:]
        del self.free_blocks[-num_blocks:]

        for block in blocks:
            if block.status is not BlockStatus.FREE:
                raise RuntimeError(
                    "Attempted to allocate a block that is already in use")
            block.status = BlockStatus.USED
            block.ref_cnt = 1

        return blocks

    def get_blocks(self, seq_id: str,) -> List[KVBlock]:
        block_map = self.block_mapping_table.get(seq_id, None)
        if block_map is None:
            raise RuntimeError(f"No blocks found for sequence '{seq_id}'")

        return block_map.blocks

    def put_blocks(
        self,
        seq_id: str,
        block_ids: List[int],
    ) -> None:
        block_map = self.block_mapping_table.get(seq_id, None)
        if block_map is None:
            block_map = KVBlockMap()

        num_blocks = len(block_map.blocks)
        required_blocks = len(block_ids)

        if num_blocks < required_blocks:
            for i in range(num_blocks, required_blocks):
                # TODO(jinu): Replace with allocating blocks one by one
                cpu_block = self.alloc_blocks(1)[0]

                block_map.blocks.append(cpu_block)

            if seq_id in self.block_mapping_table:
                self.block_mapping_table.move_to_end(seq_id)
            self.block_mapping_table[seq_id] = block_map

    def write_back(
        self,
        seq_id: int,
        layer_idx: int,
        block_ids: List[int],
        stream: cuda.Stream,
    ) -> None:
        block_map = self.block_mapping_table.get(seq_id, None)
        if block_map is None:
            raise RuntimeError(f"No blocks found for sequence '{seq_id}'")

        blocks = block_map.blocks
        block_offset = block_map.block_offset

        gpu_kv_cache_base_ptr = self.gpu_kv_cache_ptrs[layer_idx]
        block_stride = self.block_stride

        for i in range(block_offset, len(block_ids)):
            block_id = block_ids[i]
            block = blocks[i]

            cuda.memcpy_dtoh_async(
                block.data[layer_idx],
                gpu_kv_cache_base_ptr + block_id * block_stride,
                stream=stream,
            )

    def commit_write_back(self, seq_id) -> None:
        block_map = self.block_mapping_table.get(seq_id, None)
        if block_map is None:
            raise RuntimeError(f"No blocks found for sequence '{seq_id}'")

        block_map.block_offset = len(block_map.blocks) - 1

    def fetch(
        self,
        seq_id: int,
        layer_idx: int,
        block_ids: List[int],
        stream: cuda.Stream,
    ) -> None:
        block_map = self.block_mapping_table.get(seq_id, None)
        if block_map is None:
            raise RuntimeError(f"No blocks found for sequence '{seq_id}'")

        blocks = block_map.blocks
        num_blocks = len(blocks)

        gpu_kv_cache_base_ptr = self.gpu_kv_cache_ptrs[layer_idx]
        block_stride = self.block_stride

        for i in range(min(len(block_ids), num_blocks)):
            block_id = block_ids[i]
            block = blocks[i]

            cuda.memcpy_htod_async(
                gpu_kv_cache_base_ptr + block_id * block_stride,
                block.data[layer_idx],
                stream=stream,
            )


class KVWorker:

    def __init__(
        self,
        name: str,
        params: KVWorkerParams,
        host_cache_size: int,
    ):
        storage = torch.UntypedStorage._new_shared_cuda(
            *params.kv_cache_metadata)
        kv_caches = torch.empty(0, dtype=params.dtype, device='cuda')
        kv_caches.set_(storage)
        kv_caches = kv_caches.view(params.num_layers, params.num_blocks, -1)

        self.gpu_kv_caches = kv_caches

        self.ctx = cuda.Context.attach()

        self.kv_store = KVStore(gpu_kv_caches=kv_caches,
                                host_cache_size=host_cache_size)

        pre_events = [
            cuda.Event.from_ipc_handle(handle)
            for handle in params.pre_event_handles
        ]
        post_events = [
            cuda.Event.from_ipc_handle(handle)
            for handle in params.post_event_handles
        ]

        self.kv_worker_handle = KVWorkerHandle(
            pre_events,
            post_events,
            params.model_queue,
            params.kv_queue,
        )

        self.num_layers = params.num_layers

        self.dtoh_stream = cuda.Stream()
        self.htod_stream = cuda.Stream()

        self.host_kv_caches_tensor = torch.from_numpy(
            self.kv_store.host_kv_caches)

        kv_agent = nixl_agent(name)
        reg_desc = kv_agent.register_memory(
            self.host_kv_caches_tensor,
            is_sorted=True,
        )
        if not reg_desc:
            raise RuntimeError("KV memory registration failed")

        self.reg_desc = reg_desc
        self.kv_agent_metadata = kv_agent.get_agent_metadata()
        self.kv_agent = kv_agent

    def transfer(
        self,
        fetch_block_mappings: List[BlockMapping],
        write_back_block_mappings: List[BlockMapping],
    ) -> None:
        for block_mapping in write_back_block_mappings:
            self.kv_store.put_blocks(
                block_mapping.seq_id,
                block_ids=block_mapping.block_ids,
            )

        for layer_idx in range(self.num_layers):
            if len(fetch_block_mappings) > 0:
                for block_mapping in fetch_block_mappings:
                    self.kv_store.fetch(
                        block_mapping.seq_id,
                        layer_idx,
                        block_mapping.block_ids,
                        self.htod_stream,
                    )
                self.kv_worker_handle.pre_events[layer_idx].record()
                self.kv_worker_handle.model_queue.put(b'')

            if len(write_back_block_mappings) > 0:
                self.kv_worker_handle.kv_queue.get()
                self.dtoh_stream.wait_for_event(
                    self.kv_worker_handle.post_events[layer_idx])
                for block_mapping in write_back_block_mappings:
                    self.kv_store.write_back(
                        block_mapping.seq_id,
                        layer_idx,
                        block_mapping.block_ids,
                        self.dtoh_stream,
                    )

        for block_mapping in write_back_block_mappings:
            self.kv_store.commit_write_back(block_mapping.seq_id)

        self.htod_stream.synchronize()
        self.dtoh_stream.synchronize()

    def get_descriptors(self, seq_ids: List[int]) -> Tuple[bytes, int]:
        """
        Return Tuple[descriptor, num_blocks]
        """
        # FIXME(jinu)
        seq_id = seq_ids[0]

        blocks = self.kv_store.get_blocks(seq_id)
        tensors = [self.host_kv_caches_tensor[b.block_id] for b in blocks]

        xfer_desc = self.kv_agent.get_xfer_descs(tensors, is_sorted=True)
        descs = self.kv_agent.get_serialized_descs(xfer_desc)

        return descs, len(blocks)

    def pull_kv(
        self,
        peer_name: str,
        remote_descs: bytes,
        num_blocks: int,
        session_id: str,
        seq_ids: List[int],
    ) -> None:
        dummy_block_ids = [-1] * num_blocks
        # FIXME(jinu)
        seq_id = seq_ids[0]
        self.kv_store.put_blocks(seq_id, dummy_block_ids)

        blocks = self.kv_store.get_blocks(seq_id)
        tensors = [self.host_kv_caches_tensor[b.block_id] for b in blocks]

        local_descs = self.kv_agent.get_xfer_descs(tensors, is_sorted=True)
        remote_descs = self.kv_agent.deserialize_descs(remote_descs)

        xfer_handle = self.kv_agent.initialize_xfer(
            "READ",
            local_descs,
            remote_descs,
            peer_name,
            session_id,
        )
        if not xfer_handle:
            print("Creating transfer failed.")
            return False

        state = self.kv_agent.transfer(xfer_handle)
        if state == "ERR":
            print("Posting transfer failed.")
            return False

        while True:
            state = self.kv_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer Error.")
                return False
            elif state == "DONE":
                break

        return True

    def __del__(self):
        self.ctx.pop()
        self.kv_agent.deregister_memory(self.reg_desc)
