# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification gating of the DeepSeek-V4 indexer on SM12x."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
)


def _make_indexer_config(*, architecture: str, adaptive: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        num_speculative_tokens=1,
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive),
    )


def _mock_cuda_platform(monkeypatch: pytest.MonkeyPatch, family: int) -> None:
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: True)


@pytest.mark.cpu_test
@pytest.mark.parametrize("family", [120, 121])
@pytest.mark.parametrize(
    ("architecture", "backend_cls"),
    [
        ("DeepseekV4ForCausalLM", DeepseekV4IndexerBackend),
        ("DeepseekV41ForCausalLM", DeepseekV41IndexerBackend),
    ],
)
def test_deepseek_sm12x_adaptive_indexer_flattens(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    architecture: str,
    backend_cls: type[DeepseekV4IndexerBackend],
):
    # SM12x has no varlen paged MQA logits kernel; adaptive verification takes
    # the flattened path, even at next_n=2 where native decode would run.
    _mock_cuda_platform(monkeypatch, family)
    adaptive_config = _make_indexer_config(architecture=architecture, adaptive=True)
    fixed_config = _make_indexer_config(architecture=architecture, adaptive=False)

    assert backend_cls.supports_device_cpu_query_lens_mismatch()
    assert indexer._use_flattening(adaptive_config)
    assert (
        DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
            adaptive_config, SimpleNamespace()
        )
        == AttentionCGSupport.ALWAYS
    )
    # next_n=2 runs natively on SM12x without adaptive verification.
    assert not indexer._use_flattening(fixed_config)


@pytest.mark.cpu_test
def test_sm12x_flattening_is_limited_to_deepseek_v4(monkeypatch: pytest.MonkeyPatch):
    _mock_cuda_platform(monkeypatch, 120)
    adaptive_config = _make_indexer_config(
        architecture="DeepseekV32ForCausalLM", adaptive=True
    )
    assert not indexer._sm12x_supports_flattened_device_query_lens(adaptive_config)
    assert not indexer._use_flattening(adaptive_config)


@pytest.mark.cpu_test
def test_sm12x_without_deep_gemm_keeps_rejecting(monkeypatch: pytest.MonkeyPatch):
    _mock_cuda_platform(monkeypatch, 120)
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: False)
    assert not DeepseekV41IndexerBackend.supports_device_cpu_query_lens_mismatch()
