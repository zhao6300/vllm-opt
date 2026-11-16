# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def stage_sync_hooks(model: nn.Module, name: str):
    """Hooks that synchronize after every submodule of `model` runs eagerly.

    Used with `VLLM_DRAFT_CAPTURE_STAGE_SYNC`: a CUDA fault is reported by the
    first synchronization after the kernel that caused it, so syncing after each
    module names that module. Captured passes are skipped, as a synchronize
    is illegal during stream capture. Returns the hook handles to remove.
    """
    handles = []

    def make_hook(module_name: str):
        def hook(module, args, output):
            if torch.cuda.is_current_stream_capturing():
                return
            try:
                torch.accelerator.synchronize()
            except Exception:
                logger.error("CUDA fault after %s.%s", name, module_name)
                raise
            logger.info("%s.%s ok", name, module_name)

        return hook

    for module_name, module in model.named_modules():
        if module_name:
            handles.append(module.register_forward_hook(make_hook(module_name)))
    return handles


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        if block_tables.cp_size > 1:
            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                input_batch.num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
                num_reqs_padded=input_batch.num_reqs_after_padding,
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        prepare_forward: Callable[[int, dict[str, Any] | None], None],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """`prepare_forward(num_reqs, attn_metadata)` is captured ahead of the
        query forward."""

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            def forward(cg_mode: CUDAGraphMode) -> None:
                prepare_forward(num_reqs, attn_metadata)
                if (
                    envs.VLLM_DRAFT_CAPTURE_STAGE_SYNC
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    torch.accelerator.synchronize()
                    logger.info(
                        "prepare_forward ok: mode=%s num_reqs=%d num_tokens=%d",
                        desc.cg_mode.name,
                        num_reqs,
                        num_tokens,
                    )
                forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )

            return forward

        super().capture(create_forward_fn, progress_bar_desc)
