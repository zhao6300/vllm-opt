# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 with PP=2 + attention DP=4 + EP=4 + DSpark on 8 SM120 GPUs.

Stage 0 holds layers 0-19 and stage 1 holds layers 20-39 plus the DSpark
drafter, whose target layers 37/38/39 are captured locally on stage 1. Each
stage's four ranks form one EP group. No DCP, DBO or microbatching.

Needs 8 SM120 GPUs and a local checkpoint:

    DSV41_MODEL=/path/to/DeepSeek-V4.1-Flash \
        pytest tests/v1/distributed/test_dsv41_pp2_dp4_ep4_dspark.py
"""

import asyncio
import os
from contextlib import ExitStack
from typing import Any

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = os.environ.get("DSV41_MODEL")
PP, DP = 2, 4
DSPARK_CONFIG = {"method": "dspark", "num_speculative_tokens": 5}
PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain pipeline parallelism in one paragraph.",
    "1, 1, 2, 3, 5, 8,",
]
# Different PP/EP layouts reduce in a different order, so greedy outputs may
# drift after a few tokens; require most tokens to match, not all.
MIN_TOKEN_MATCH_RATIO = float(os.environ.get("DSV41_MIN_TOKEN_MATCH_RATIO", "0.9"))
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 8192


def _skip_reason() -> str | None:
    if MODEL is None:
        return "set DSV41_MODEL to a local DeepSeek V4.1 checkpoint"
    if not current_platform.is_cuda():
        return "requires CUDA"
    if current_platform.device_count() < PP * DP:
        return f"requires {PP * DP} GPUs, got {current_platform.device_count()}"
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return f"requires SM120, got {capability}"
    return None


_SKIP = _skip_reason()

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") == "0",
        reason="DSpark needs the V2 model runner",
    ),
    pytest.mark.skipif(_SKIP is not None, reason=_SKIP or ""),
]


def _engine_args(
    speculative: bool = True, async_scheduling: bool = True
) -> AsyncEngineArgs:
    assert MODEL is not None
    return AsyncEngineArgs(
        model=MODEL,
        pipeline_parallel_size=PP,
        tensor_parallel_size=1,
        data_parallel_size=DP,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        enable_dbo=False,
        # The SM120 settings serve.sh uses.
        attention_backend="FLASHINFER_MLA_SPARSE_DSV41",
        kv_cache_dtype="nvfp4_ds_mla",
        block_size=64,
        all2all_backend="allgather_reducescatter",
        language_model_only=True,
        speculative_config=dict(DSPARK_CONFIG) if speculative else None,
        async_scheduling=async_scheduling,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=32,
        gpu_memory_utilization=0.92,
    )


async def _generate(
    engine: AsyncLLM, request_id: str, prompt: str, max_tokens: int
) -> list[int]:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )
    token_ids: list[int] = []
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        token_ids.extend(out.outputs[0].token_ids)
    return token_ids


async def _generate_all(engine: AsyncLLM, tag: str, max_tokens: int):
    return await asyncio.gather(
        *[
            _generate(engine, f"{tag}-{i}", prompt, max_tokens)
            for i, prompt in enumerate(PROMPTS)
        ]
    )


async def _worker_topologies(engine: AsyncLLM) -> list[dict[str, Any]]:
    per_engine = await engine.engine_core.call_utility_all_async(
        "collective_rpc", "parallel_topology", None, (), None
    )
    return [worker for workers in per_engine for worker in workers]


def check_pp2_dp4_ep4_topology(workers: list[dict[str, Any]]) -> None:
    """Every DP rank owns one PP chain; each PP stage is one EP group."""
    assert len(workers) == PP * DP, workers
    by_rank = {w["global_rank"]: w for w in workers}
    assert sorted(by_rank) == list(range(PP * DP)), sorted(by_rank)
    physical = [w["physical_gpu"] for w in workers]
    assert len(set(physical)) == PP * DP, f"GPUs shared between ranks: {physical}"
    for w in workers:
        assert w["tp_rank"] == 0
        chain = [by_rank[r] for r in w["pp_ranks"]]
        assert len(chain) == PP
        assert {m["dp_rank"] for m in chain} == {w["dp_rank"]}, (
            f"PP chain of rank {w['global_rank']} mixes DP replicas: {chain}"
        )
        assert [m["pp_rank"] for m in chain] == list(range(PP))
        ep_group = [by_rank[r] for r in w["ep_ranks"]]
        assert len(ep_group) == DP
        assert {m["pp_rank"] for m in ep_group} == {w["pp_rank"]}, (
            f"EP group of rank {w['global_rank']} spans PP stages: {ep_group}"
        )
        assert sorted(m["dp_rank"] for m in ep_group) == list(range(DP))


def _token_match_ratio(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_scheduling", [True, False])
async def test_topology_and_dspark_generation(async_scheduling: bool):
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(
            _engine_args(async_scheduling=async_scheduling)
        )
        after.callback(engine.shutdown)

        check_pp2_dp4_ep4_topology(await _worker_topologies(engine))

        # Single request: the low-concurrency latency path.
        single = await _generate(engine, "single", PROMPTS[0], max_tokens=32)
        assert len(single) == 32

        # Concurrent requests spread over the 4 DP replicas; a long decode
        # runs many DSpark rounds with mixed accepted/rejected drafts.
        results = await _generate_all(engine, "multi", max_tokens=128)
        assert all(len(r) == 128 for r in results), [len(r) for r in results]

        # Greedy decoding is deterministic per request on a fixed layout.
        again = await _generate(engine, "single-again", PROMPTS[0], max_tokens=32)
        assert _token_match_ratio(single, again) >= MIN_TOKEN_MATCH_RATIO


async def _generate_text(
    engine: AsyncLLM, request_id: str, prompt: str, max_tokens: int
) -> str:
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, output_kind=RequestOutputKind.FINAL_ONLY
    )
    text = ""
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        text = out.outputs[0].text
    return text


def _needle_prompt(num_filler_lines: int, code: str) -> str:
    filler = [
        f"Line {i}: the warehouse inventory lists {i * 7 % 97} crates of item {i}."
        for i in range(num_filler_lines)
    ]
    filler.insert(num_filler_lines // 3, f"The secret access code is {code}.")
    return "\n".join(filler) + "\nQuestion: What is the secret access code?\nAnswer:"


@pytest.mark.asyncio
async def test_sm120_kv_cache_answers():
    """Answer checks for SM120 KV addressing, which token counts cannot catch.

    A wrong SM120 sparse-KV row mapping still yields full-length outputs, but
    the model misses arithmetic, loses recalled facts and drifts into repeated
    text. The long needle prompt is several prefill chunks
    (max_num_batched_tokens) over many 64-token pages: it exercises chunked
    prefill, the compressed caches and indexer top-k, and the 19 -> 20 PP
    transfer at large T.
    """
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args())
        after.callback(engine.shutdown)

        arithmetic = await _generate_text(engine, "arith", "17 * 23 =", 8)
        assert "391" in arithmetic, arithmetic

        for i, lines in enumerate((40, 900)):  # short, then ~20k tokens
            code = f"{48213 + i * 1111}"
            answer = await _generate_text(
                engine, f"needle-{i}", _needle_prompt(lines, code), 16
            )
            assert code in answer, (lines, answer)


@pytest.mark.asyncio
async def test_pp_send_buffers_and_microbatch_balance_keep_outputs(monkeypatch):
    """Phase 3/4 options change when sends wait and which step a request
    decodes in, never what it computes: greedy outputs must match."""
    max_tokens = 64
    outputs = {}
    for options in (False, True):
        monkeypatch.setenv("VLLM_PP_SEND_BUFFERS", "2" if options else "0")
        monkeypatch.setenv("VLLM_PP_MICROBATCH_BALANCE", "1" if options else "0")
        monkeypatch.setenv("VLLM_PP_STEP_PROFILE_INTERVAL", "50" if options else "0")
        with ExitStack() as after:
            engine = AsyncLLM.from_engine_args(_engine_args())
            after.callback(engine.shutdown)
            # A burst, so balancing has a lopsided microbatch to split.
            outputs[options] = await _generate_all(
                engine, f"opts{int(options)}", max_tokens
            )
    for base, opt in zip(outputs[False], outputs[True]):
        ratio = _token_match_ratio(base, opt)
        assert ratio >= MIN_TOKEN_MATCH_RATIO, (ratio, base, opt)


@pytest.mark.asyncio
async def test_dspark_matches_target_only_greedy():
    max_tokens = 64
    outputs = {}
    for speculative in (False, True):
        with ExitStack() as after:
            engine = AsyncLLM.from_engine_args(_engine_args(speculative=speculative))
            after.callback(engine.shutdown)
            outputs[speculative] = await _generate_all(
                engine, f"spec{int(speculative)}", max_tokens
            )
    for target_only, with_dspark in zip(outputs[False], outputs[True]):
        ratio = _token_match_ratio(target_only, with_dspark)
        assert ratio >= MIN_TOKEN_MATCH_RATIO, (ratio, target_only, with_dspark)
