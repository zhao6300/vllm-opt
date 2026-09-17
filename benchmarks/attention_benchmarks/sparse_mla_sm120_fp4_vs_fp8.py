# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark FlashInfer SM120 DSv4.1 sparse MLA extra-cache modes."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from flashinfer.mla._sparse_mla_sm120 import SparseMLASm120Wrapper


def _build_inputs(num_tokens, num_heads, topk, kv_blocks, device):
    kv_tokens = 64
    fp8_extra = dict(num_blocks=kv_blocks, block_size=kv_tokens, width=528)
    fp4_extra = dict(num_blocks=kv_blocks, block_size=kv_tokens, width=288)
    main_cache = torch.randint(
        0, 32, (kv_blocks, kv_tokens, 528), dtype=torch.uint8, device=device
    )
    extra_caches = {
        "swa_fp8_extra_fp8": torch.randint(
            0, 32, (kv_blocks, kv_tokens, 1, 528), dtype=torch.uint8, device=device
        ),
        "swa_fp8_extra_fp4": torch.randint(
            0, 32, (kv_blocks, kv_tokens, 1, 288), dtype=torch.uint8, device=device
        ),
    }
    indices = torch.randint(
        0, kv_tokens, (num_tokens, topk), dtype=torch.int32, device=device
    )
    extra_indices = torch.randint(
        0, kv_tokens, (num_tokens, 1, topk), dtype=torch.int32, device=device
    )
    inputs = dict(
        q=torch.randn(
            (num_tokens, num_heads, 512), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty(
            (num_tokens, num_heads, 512), dtype=torch.bfloat16, device=device
        ),
        topk_length=torch.full((num_tokens,), topk, dtype=torch.int32, device=device),
        extra_topk_length=torch.full(
            (num_tokens,), topk, dtype=torch.int32, device=device
        ),
    )
    return main_cache, extra_caches, indices, extra_indices, inputs


def _benchmark_mode(
    extra_fp4, num_tokens, num_heads, topk, kv_blocks, warmup, iterations, device
):
    wrapper = SparseMLASm120Wrapper(
        max_num_tokens=num_tokens,
        max_num_heads=num_heads,
        d_v=512,
        kv_scale_format="ue8m0_g32",
        kv_cache_format="fp8",
        extra_kv_fp4=extra_fp4,
        compute_precision="fp8",
        device=device,
    )
    main_cache, extra_caches, indices, extra_indices, qout = _build_inputs(
        num_tokens, num_heads, topk, kv_blocks, device
    )
    mode = "swa_fp8_extra_fp4" if extra_fp4 else "swa_fp8_extra_fp8"

    def run_once():
        wrapper.run(
            q=qout["q"],
            kv_cache=main_cache,
            indices=indices,
            output=qout["output"],
            sm_scale=512**-0.5,
            topk_length=qout["topk_length"],
            attn_sink=None,
            extra_kv_cache=extra_caches[mode],
            extra_indices=extra_indices,
            extra_topk_length=qout["extra_topk_length"],
            prefill_impl=None,
        )

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms), times_ms, extra_caches[mode].shape[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[8, 32, 48])
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=1152)
    parser.add_argument("--kv-blocks", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/sparse_mla_sm120_fp4_vs_fp8.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    results = []
    for num_tokens in args.num_tokens:
        for mode, extra_fp4 in (
            ("swa_fp8_extra_fp8", False),
            ("swa_fp8_extra_fp4", True),
        ):
            median_ms, samples_ms, extra_width = _benchmark_mode(
                extra_fp4,
                num_tokens,
                args.num_heads,
                args.topk,
                args.kv_blocks,
                args.warmup,
                args.iterations,
                device,
            )
            results.append(
                dict(
                    mode=mode,
                    extra_kv_fp4=extra_fp4,
                    num_tokens=num_tokens,
                    num_heads=args.num_heads,
                    topk=args.topk,
                    kv_blocks=args.kv_blocks,
                    median_ms=median_ms,
                    p50_ms=statistics.quantiles(samples_ms, n=100)[49],
                    min_ms=min(samples_ms),
                    max_ms=max(samples_ms),
                    iterations=args.iterations,
                    extra_cache_width=extra_width,
                )
            )
            current = results[-1]
            print(
                f"{mode} tokens={num_tokens}: {current['median_ms'] * 1000:.2f} us median"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
