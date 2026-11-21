# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DSA indexer's paged MQA-logits kernel on both K-cache dtypes.

The kernel reads one 68-byte row per compressed state with the MXFP4 cache and
132 bytes with the FP8 one, so the speedup tracks how bandwidth-bound the
step is: it approaches 2x at long contexts and large batches.

Run on an sm_10x or sm_12x GPU:

    python benchmarks/kernels/benchmark_indexer_mqa_logits.py
"""

import argparse

import torch

from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    mqa_logits_use_mx_scales,
)

HEADS = 32  # DeepSeek-V4.1 index_n_heads
HEAD_DIM = 128  # DeepSeek-V4.1 index_head_dim
BLOCK_SIZE = 64  # tokens per indexer page
NEXT_N = 1

# (batch, context length per request)
SCENARIOS = [(1, 8192), (8, 8192), (32, 8192), (8, 32768), (8, 131072)]


def make_inputs(batch: int, context_len: int, use_fp4: bool):
    blocks_per_request = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_request
    row_bytes = HEAD_DIM // 2 + 4 if use_fp4 else HEAD_DIM + 4
    cache = torch.randint(
        0,
        256,
        (num_blocks, BLOCK_SIZE, row_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        batch, blocks_per_request
    )
    context_lens = torch.full(
        (batch, NEXT_N), context_len, dtype=torch.int32, device="cuda"
    )
    weights_dtype = torch.bfloat16 if mqa_logits_use_mx_scales() else torch.float32
    weights = torch.ones(batch * NEXT_N, HEADS, dtype=weights_dtype, device="cuda")
    if use_fp4:
        q = torch.randint(
            0,
            256,
            (batch, NEXT_N, HEADS, HEAD_DIM // 2),
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.int8)
        q_scale = torch.full(
            (batch, NEXT_N, HEADS), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
        )
    else:
        q = torch.randn(
            batch, NEXT_N, HEADS, HEAD_DIM, dtype=torch.float32, device="cuda"
        ).to(torch.float8_e4m3fn)
        q_scale = None
    return cache, block_table, context_lens, q, q_scale, weights


def bench(batch: int, context_len: int, use_fp4: bool) -> float:
    cache, block_table, context_lens, q, q_scale, weights = make_inputs(
        batch, context_len, use_fp4
    )
    cache_view = kv_cache_as_quant_view(cache, HEAD_DIM, use_fp4)
    schedule = get_paged_mqa_logits_metadata(context_lens, BLOCK_SIZE, get_num_sms())
    ms, _, _ = triton.testing.do_bench(
        lambda: fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            cache_view,
            weights,
            context_lens,
            block_table,
            schedule,
            context_len,
        ),
        quantiles=[0.5, 0.2, 0.8],
        rep=100,
    )
    return ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Unused; accepted for parity with the other kernel benchmarks.",
    )
    parser.parse_args()

    print(
        f"{'batch':>5} {'ctx':>7} | {'fp8 ms':>8} {'fp4 ms':>8} {'speedup':>8}"
        f" | {'fp8 GB/s':>9} {'fp4 GB/s':>9}"
    )
    for batch, context_len in SCENARIOS:
        fp8_ms = bench(batch, context_len, use_fp4=False)
        fp4_ms = bench(batch, context_len, use_fp4=True)
        torch.cuda.empty_cache()
        fp8_bytes = batch * context_len * (HEAD_DIM + 4)
        fp4_bytes = batch * context_len * (HEAD_DIM // 2 + 4)
        print(
            f"{batch:>5} {context_len:>7} | {fp8_ms:8.3f} {fp4_ms:8.3f}"
            f" {fp8_ms / fp4_ms:8.2f} | {fp8_bytes / fp8_ms / 1e6:9.1f}"
            f" {fp4_bytes / fp4_ms / 1e6:9.1f}"
        )


if __name__ == "__main__":
    main()
