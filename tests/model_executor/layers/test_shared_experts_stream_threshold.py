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
from vllm.model_executor.layers.fused_moe.runner import (
    shared_experts as shared_experts_module,
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


@pytest.fixture
def recording_graph(monkeypatch):
    """Put the process in the state the runner records a FULL graph in.

    The capture path passes CUDAGraphMode.NONE to the model forward so compiled
    wrappers keep running eagerly inside the capture; only the active stream
    capture reveals that a FULL graph is being recorded.
    """

    def start() -> None:
        monkeypatch.setattr(
            shared_experts_module.current_platform, "is_cuda_alike", lambda: True
        )
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    return start


def _forward_context(mode: CUDAGraphMode) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
    )


def test_decode_threshold_applies_to_recorded_full_graphs(
    make_shared_experts, monkeypatch, recording_graph
):
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    # Outside a forward pass, and in eager and PIECEWISE steps, prefill keeps
    # the cutoff.
    assert shared_experts._token_threshold() == 256
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 256
    with override_forward_context(_forward_context(CUDAGraphMode.PIECEWISE)):
        assert shared_experts._token_threshold() == 256

    # Recording a FULL decode graph uses the decode cutoff. The runner keeps
    # cudagraph_runtime_mode at NONE throughout capture.
    recording_graph()
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 384
    # A replay under a runtime that does report FULL keeps working too.
    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        assert shared_experts._token_threshold() == 384


def test_piecewise_capture_keeps_the_prefill_cutoff(make_shared_experts, monkeypatch):
    """PIECEWISE graphs serve prefill batches even while they are recorded."""
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    with override_forward_context(_forward_context(CUDAGraphMode.PIECEWISE)):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        assert shared_experts._token_threshold() == 256


def test_decode_threshold_replaces_the_prefill_cutoff(
    make_shared_experts, monkeypatch, recording_graph
):
    """The decode knob is independent: it may lower the cutoff as well."""
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "512")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    recording_graph()
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 384
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        assert shared_experts._token_threshold() == 512


def test_unset_decode_threshold_follows_the_prefill_one(
    make_shared_experts, monkeypatch, recording_graph
):
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "128")
    monkeypatch.delenv(
        "VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", raising=False
    )
    shared_experts = make_shared_experts()

    recording_graph()
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        assert shared_experts._token_threshold() == 128


def test_disabled_stream_never_overlaps(
    make_shared_experts, monkeypatch, recording_graph
):
    """VLLM_DISABLE_SHARED_EXPERTS_STREAM must win over any token count."""
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD", "384")
    shared_experts = make_shared_experts()

    recording_graph()
    with override_forward_context(_forward_context(CUDAGraphMode.NONE)):
        order = shared_experts._determine_shared_experts_order(torch.empty(8, 64))

    assert order == SharedExpertsOrder.NO_OVERLAP
