# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which token counts keep the shared experts on the aux stream.

A FULL decode graph bakes this decision in at capture time (replay runs no
Python), so the threshold that applies at capture decides the overlap for
every request count the graph serves.
"""

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def make_shared_experts(monkeypatch):
    # Keep the test off the GPU: the layer builds CUDA events only when the
    # aux stream is enabled.
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")

    def make() -> SharedExperts:
        parallel_config = FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        moe_config = FusedMoEConfig(
            num_experts=256,
            experts_per_token=8,
            hidden_dim=64,
            intermediate_size=64,
            num_local_experts=256,
            num_logical_experts=256,
            activation=MoEActivation.SILU,
            device="cpu",
            routing_method=RoutingMethodType.TopK,
            moe_parallel_config=parallel_config,
            in_dtype=torch.bfloat16,
            moe_backend="deep_gemm",
        )
        return SharedExperts(
            layer=torch.nn.Identity(),
            moe_config=moe_config,
            enable_dbo=False,
            mk_can_overlap_shared_experts=lambda: False,
        )

    return make


def _forward_context(mode: CUDAGraphMode) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
    )


def test_decode_threshold_applies_to_full_graphs_only(make_shared_experts, monkeypatch):
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    # Outside a forward pass, and in non-graph steps, prefill keeps the cutoff.
    assert shared_experts._token_threshold() == 256
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 256
    with override_forward_context(_forward_context(CUDAGraphMode.PIECEWISE)):
        assert shared_experts._token_threshold() == 256

    # Capture and replay of a FULL decode graph use the decode cutoff.
    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        assert shared_experts._token_threshold() == 384


def test_decode_threshold_never_lowers_the_prefill_cutoff(
    make_shared_experts, monkeypatch
):
    """A smaller decode cutoff must not shrink the prefill threshold."""
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "512")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        assert shared_experts._token_threshold() == 512
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 512


def test_unset_decode_threshold_follows_the_prefill_one(
    make_shared_experts, monkeypatch
):
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "128")
    monkeypatch.delenv(
        "VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", raising=False
    )
    shared_experts = make_shared_experts()

    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        assert shared_experts._token_threshold() == 128


def test_disabled_stream_never_overlaps(make_shared_experts, monkeypatch):
    """VLLM_DISABLE_SHARED_EXPERTS_STREAM must win over any token count."""
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        order = shared_experts._determine_shared_experts_order(torch.empty(8, 64))

    assert order == SharedExpertsOrder.NO_OVERLAP
