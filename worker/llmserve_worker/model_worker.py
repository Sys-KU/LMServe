import torch
import flashinfer
import numpy as np
import pycuda.driver as cuda
import torch.multiprocessing as mp

from typing import List, Tuple, Dict, Optional, Any
from transformers import AutoTokenizer

from llmserve_worker.models import get_model, ModelConfig
from llmserve_worker.models.utils import travel_layers
from llmserve_worker.models.input_params import InputParams
from llmserve_worker.models.sampler import sample
from llmserve_worker.ops.cache_ops import copy_blocks
from llmserve_worker.pb.worker_pb2 import InferInput, InferOutput
from llmserve_worker.dist_utils import init_distributed
from llmserve_worker.kv_worker import KVWorkerHandle, KVWorkerParams


def build_prefill_wrapper(
    inputs: List[InferInput],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_data_type: torch.dtype,
) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:

    seqlens_q: List[int] = []
    seqlens_k: List[int] = []
    num_blocks_per_seqs: List[int] = []
    kv_block_indices: List[int] = []
    kv_last_block_lens: List[int] = []
    for data in inputs:
        seqlen_q = data.input_len
        seqlen_k = data.context_len

        seqlens_q.append(seqlen_q)
        seqlens_k.append(seqlen_k)
        kv_block_indices += data.block_ids
        num_blocks_per_seqs.append(len(data.block_ids))
        kv_last_block_lens.append(((seqlen_k - 1) % page_size) + 1)

    cu_seqlens_q_tensor = torch.tensor(
        np.cumsum([0] + seqlens_q),
        dtype=torch.int,
        device='cuda',
    )
    kv_block_indptrs_tensor = torch.tensor(
        np.cumsum([0] + num_blocks_per_seqs),
        dtype=torch.int,
        device='cuda',
    )
    kv_block_indices_tensor = torch.tensor(
        kv_block_indices,
        dtype=torch.int,
        device='cuda',
    )
    kv_last_block_lens_tensor = torch.tensor(
        kv_last_block_lens,
        dtype=torch.int,
        device='cuda',
    )

    workspace_buffer = torch.empty(
        128 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD")

    prefill_wrapper.plan(
        qo_indptr=cu_seqlens_q_tensor,
        paged_kv_indptr=kv_block_indptrs_tensor,
        paged_kv_indices=kv_block_indices_tensor,
        paged_kv_last_page_len=kv_last_block_lens_tensor,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=q_data_type,
    )

    return prefill_wrapper


def build_decode_wrapper(
    inputs: List[InferInput],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_data_type: torch.dtype,
) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:

    num_blocks_per_seqs: List[int] = []
    kv_block_indices: List[int] = []
    kv_last_block_lens: List[int] = []
    for data in inputs:
        seqlen_q = data.input_len

        assert seqlen_q == 1, "seqlen_q of decode inputs should be 1"

        seqlen_k = data.context_len
        kv_block_indices += data.block_ids
        num_blocks_per_seqs.append(len(data.block_ids))
        kv_last_block_lens.append(((seqlen_k - 1) % page_size) + 1)

    kv_block_indptrs_tensor = torch.tensor(
        np.cumsum([0] + num_blocks_per_seqs),
        dtype=torch.int,
        device='cuda',
    )
    kv_block_indices_tensor = torch.tensor(
        kv_block_indices,
        dtype=torch.int,
        device='cuda',
    )
    kv_last_block_lens_tensor = torch.tensor(
        kv_last_block_lens,
        dtype=torch.int,
        device='cuda',
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024,
                                   dtype=torch.uint8,
                                   device="cuda")

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True)

    decode_wrapper.plan(
        indptr=kv_block_indptrs_tensor,
        indices=kv_block_indices_tensor,
        last_page_len=kv_last_block_lens_tensor,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=q_data_type,
    )

    return decode_wrapper


class ModelWorker:

    def __init__(
        self,
        model_name: str,
        block_size: int,
        tokenizer=None,
    ):
        init_distributed()

        self.model_config = ModelConfig(model_name)
        self.model = get_model(self.model_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_cache: Dict[int, str] = {}

        eos_token_ids = self.tokenizer.eos_token_ids
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids_tensor = torch.tensor(eos_token_ids, device='cuda')

        self.block_size = block_size
        # The kv_caches & events are initialized at init_cache()
        self.kv_caches: torch.Tensor = None
        self.kv_worker_handle: KVWorkerHandle = None

        self.enable_wait_before_execute: bool = False
        self.enable_record_after_execute: bool = False

        # Sampling parameters
        self.top_k: int = 1
        self.top_p: float = 0.0
        self.temperature: float = 0.0

        self.ctx = cuda.Context.attach()

    def get_config(self) -> ModelConfig:
        self.model_config.clone()

    def update_sampling_params(
        self,
        top_k: int,
        top_p: float,
        temperature: float,
    ):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def init_cache(self,
                   cache_size: int = None,
                   num_blocks: int = None) -> Tuple[int, KVWorkerParams]:
        if cache_size is None and num_blocks is None:
            raise ValueError(
                "At least one of 'cache_size' or 'num_blocks' must be given.")

        model_config = self.model_config

        block_size = self.block_size
        num_layers = model_config.num_layers
        head_dim = model_config.head_dim
        num_kv_heads = model_config.num_kv_heads
        dtype = model_config.dtype

        dtype_byte = torch.finfo(dtype).bits // 8
        kv_block_size = (block_size * num_kv_heads * head_dim * dtype_byte)

        if cache_size is not None:
            num_available_blocks = (int(cache_size) //
                                    (num_layers * 2 * kv_block_size))
            if num_blocks is not None:
                num_blocks = min(num_blocks, num_available_blocks)
            else:
                num_blocks = num_available_blocks

        self.kv_caches = torch.empty(num_layers,
                                     num_blocks,
                                     2,
                                     block_size,
                                     num_kv_heads,
                                     head_dim,
                                     dtype=dtype,
                                     device='cuda')

        # TODO(jinu): Make KVCacheConfig
        # Init KVWorkerHandle
        pre_events = [
            cuda.Event(flags=(cuda.event_flags.DISABLE_TIMING
                              | cuda.event_flags.INTERPROCESS))
            for _ in range(num_layers)
        ]
        post_events = [
            cuda.Event(flags=(cuda.event_flags.DISABLE_TIMING
                              | cuda.event_flags.INTERPROCESS))
            for _ in range(num_layers)
        ]

        model_queue = mp.Queue()
        kv_queue = mp.Queue()

        self.kv_worker_handle = KVWorkerHandle(
            pre_events,
            post_events,
            model_queue,
            kv_queue,
        )

        # Register hook to synchronize with KVWorker
        attn_layers = travel_layers(self.model, includes=["Attention"])
        for i, layer in enumerate(attn_layers):
            def make_hook(i):
                def pre_hook_fn(mod, input):
                    if self.enable_wait_before_execute:
                        self.kv_worker_handle.model_queue.get()
                        self.kv_worker_handle.pre_events[i].synchronize()

                def post_hook_fn(mod, input, output):
                    if self.enable_record_after_execute:
                        self.kv_worker_handle.post_events[i].record()
                        # Put a dummy value to trigger KV transfer
                        self.kv_worker_handle.kv_queue.put(b'')
                return pre_hook_fn, post_hook_fn

            pre_hook, post_hook = make_hook(i)
            layer.register_forward_pre_hook(pre_hook)
            layer.register_forward_hook(post_hook)

        # Generate KVWorkerParams
        kv_cache_metadata = self.kv_caches.untyped_storage()._share_cuda_()
        pre_event_handles = [
            e.ipc_handle() for e in self.kv_worker_handle.pre_events
        ]
        post_event_handles = [
            e.ipc_handle() for e in self.kv_worker_handle.post_events
        ]

        kv_worker_params = KVWorkerParams(
            kv_cache_metadata,
            num_layers,
            num_blocks,
            dtype,
            pre_event_handles,
            post_event_handles,
            model_queue,
            kv_queue,
        )

        return num_blocks, kv_worker_params

    def _make_inputs(
            self, inputs: List[InferInput], use_cache=True,
    ) -> Tuple[torch.Tensor, InputParams]:

        flatten_input_ids = []
        position_ids = []
        prefill_input_len = 0

        seqlens_q = []
        seqlens_k = []

        prefill_data_list: List[InferInput] = []
        decode_data_list: List[InferInput] = []

        # NOTE(jinu):
        # Sort the list according to phase type to compact the inputs.
        inputs.sort(
            key=lambda x: -x.input_len)

        for data in inputs:
            input_ids = data.input_ids
            cache_seqlen = data.filled_token_len

            seqlen_q = data.input_len

            if seqlen_q > 1:
                prefill_input_len += seqlen_q
                prefill_data_list.append(data)

            elif seqlen_q == 1:
                decode_data_list.append(data)

            seqlen_k = data.context_len

            flatten_input_ids += input_ids
            position_ids += [i for i in range(cache_seqlen, seqlen_k)]

            seqlens_q.append(seqlen_q)
            seqlens_k.append(seqlen_k)

        flatten_input_ids_tensor = torch.tensor(flatten_input_ids,
                                                dtype=torch.long,
                                                device='cuda')
        position_ids_tensor = torch.tensor(position_ids,
                                           dtype=torch.int,
                                           device='cuda')
        cu_seqlens_q_tensor = torch.tensor(
            np.cumsum([0] + seqlens_q),
            dtype=torch.int,
            device='cuda',
        )

        if self.kv_caches is not None and use_cache is True:
            num_qo_heads = self.model_config.num_heads
            num_kv_heads = self.model_config.num_kv_heads
            head_dim = self.model_config.head_dim
            page_size = self.block_size
            q_data_type = self.model_config.dtype

            if len(prefill_data_list) > 0:
                prefill_wrapper = build_prefill_wrapper(
                    prefill_data_list,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    q_data_type=q_data_type,
                )
            else:
                prefill_wrapper = None

            if len(decode_data_list) > 0:
                decode_wrapper = build_decode_wrapper(
                    decode_data_list,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    q_data_type=q_data_type,
                )
            else:
                decode_wrapper = None

            kv_block_indices = []
            num_blocks_per_seqs: List[int] = []
            kv_last_block_lens: List[int] = []
            for data in inputs:
                seqlen_k = data.context_len

                kv_block_indices += data.block_ids
                num_blocks_per_seqs.append(len(data.block_ids))
                last_block_len = (seqlen_k - 1) % page_size + 1
                kv_last_block_lens.append(last_block_len)

            kv_block_indptrs_tensor = torch.tensor(
                np.cumsum([0] + num_blocks_per_seqs),
                dtype=torch.int,
                device='cuda',
            )
            kv_block_indices_tensor = torch.tensor(
                kv_block_indices,
                dtype=torch.int,
                device='cuda',)
            kv_last_block_lens_tensor = torch.tensor(
                kv_last_block_lens,
                dtype=torch.int,
                device='cuda',
            )

            batch_indices, positions = flashinfer.get_batch_indices_positions(
                cu_seqlens_q_tensor,
                flashinfer.get_seq_lens(
                    kv_block_indptrs_tensor,
                    kv_last_block_lens_tensor,
                    page_size,
                ),
                sum(seqlens_q),
            )

            input_params = InputParams(
                position_ids=position_ids_tensor,
                cu_seqlens_q=cu_seqlens_q_tensor,
                prefill_input_len=prefill_input_len,
                prefill_wrapper=prefill_wrapper,
                decode_wrapper=decode_wrapper,
                batch_indices=batch_indices,
                positions=positions,
                kv_block_indices=kv_block_indices_tensor,
                kv_block_indptrs=kv_block_indptrs_tensor,
                kv_last_block_lens=kv_last_block_lens_tensor,
            )

        else:
            input_params = InputParams(
                position_ids=position_ids_tensor,
                cu_seqlens_q=cu_seqlens_q_tensor,
                prefill_input_len=prefill_input_len,
            )

        return flatten_input_ids_tensor, input_params

    @torch.inference_mode()
    def warmup(
        self,
        max_batch_size: int,
        max_seq_len: int,
        max_num_batched_tokens: int,
    ) -> None:
        inputs: List[InferInput] = []

        if max_num_batched_tokens <= max_batch_size * max_seq_len:
            num_total_tokens = max_num_batched_tokens
        else:
            num_total_tokens = max_batch_size * max_seq_len

        total_token_ids = [0] * num_total_tokens
        token_ids_list = [[] for _ in range(max_batch_size)]
        for i, token_id in enumerate(total_token_ids):
            token_ids_list[i % max_batch_size].append(token_id)

        for i, token_ids in enumerate(token_ids_list):
            inputs.append(
                InferInput(
                    seq_id=i,
                    input_ids=token_ids,
                    input_len=len(token_ids),
                    filled_token_len=0,
                    context_len=len(token_ids),
                    block_ids=[],
                ))

        input_ids, input_params = self._make_inputs(inputs)
        logits = self.model(
            input_ids=input_ids,
            input_params=input_params,
            kv_caches=None,
        )

        # The code below does sampling.
        cu_output_lens = input_params.cu_seqlens_q[1:]
        indices = cu_output_lens - 1
        last_hidden_states = logits[indices].unsqueeze(1)

        next_token_ids, probs, all_probs = sample(
            last_hidden_states,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )

    @torch.inference_mode()
    def execute(
        self,
        inputs: List[InferInput],
        use_cache: bool,
        wait_before_execute: bool,
        record_after_execute: bool,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        input_ids, input_params = self._make_inputs(inputs, use_cache)

        if blocks_to_copy is not None and len(blocks_to_copy) > 0:
            copy_blocks(self.kv_caches, blocks_to_copy)

        self.enable_wait_before_execute = wait_before_execute
        self.enable_record_after_execute = record_after_execute

        logits = self.model(input_ids=input_ids,
                            kv_caches=self.kv_caches if use_cache else None,
                            input_params=input_params)

        last_token_indptrs = input_params.cu_seqlens_q[1:] - 1
        last_hidden_states = logits[last_token_indptrs]

        next_token_ids_tensor, probs_tensor, all_probs_tensor = sample(
            last_hidden_states,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )

        next_token_ids = next_token_ids_tensor.cpu().tolist()
        probs = probs_tensor.cpu().tolist()

        eos_mask_tensor = torch.isin(next_token_ids_tensor,
                                     self.eos_token_ids_tensor)
        eos_mask = eos_mask_tensor.cpu().tolist()

        next_words = []
        for next_token_id in next_token_ids:
            if next_token_id in self.vocab_cache:
                next_word = self.vocab_cache[next_token_id]
            else:
                next_word = self.tokenizer.decode(next_token_id)
                self.vocab_cache[next_token_id] = next_word
            next_words.append(next_word)

        outputs: Dict[InferOutput] = {}
        for i, data in enumerate(inputs):
            seq_id = data.seq_id
            output_id = next_token_ids[i]
            outputs[seq_id] = InferOutput(
                output_id=output_id,
                prob=probs[i],
                output_word=next_words[i],
                is_eos=eos_mask[i],
            )
        return outputs

    def get_model_size(self) -> int:
        def get_total_size(mod):
            mod_size = 0
            if len(list(mod.children())) == 0:
                param_size = 0
                for param in mod.parameters():
                    param_size += param.numel() * param.element_size()
                buffer_size = 0
                for buffer in mod.buffers():
                    buffer_size += buffer.numel() * buffer.element_size()
                mod_size = param_size + buffer_size
                return mod_size
            else:
                for child in mod.children():
                    mod_size += get_total_size(child)

                return mod_size

        return get_total_size(self.model)

    def __del__(self):
        self.ctx.pop()
