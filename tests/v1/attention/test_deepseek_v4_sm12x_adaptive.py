# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification gating of the DeepSeek-V4 indexer on SM12x."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from tests.v1.core.utils import new_kv_cache_group
from vllm.config import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    get_block_table_width,
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


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform.is_cuda()
        and has_deep_gemm()
        and not indexer._supports_varlen_paged_mqa_logits()
    ),
    reason="needs the flattened (non-varlen) CUDA indexer path with DeepGEMM",
)
@pytest.mark.parametrize(
    ("num_speculative_tokens", "device_lens", "cpu_lens"),
    [
        # next_n=2 is the case the SM12x opt-in newly flattens; the CPU even
        # split and the device admission disagree on every request boundary.
        (1, [1, 2, 2, 1], [2, 2, 1, 1]),
        # Adaptive replay can concentrate all drafts on one request.
        (5, [4, 2, 1, 1], [2, 2, 2, 2]),
    ],
)
def test_adaptive_flattened_decode_follows_device_boundaries(
    num_speculative_tokens: int,
    device_lens: list[int],
    cpu_lens: list[int],
):
    """Adaptive verification trims drafts on device, so the CPU per-request
    lengths are only upper bounds. The flattened decode expansion must place
    every row by the device boundaries and stay inside the builder buffers."""
    device = torch.device("cuda")
    num_reqs = len(device_lens)
    num_decode_tokens = sum(cpu_lens)
    assert sum(device_lens) == num_decode_tokens

    kv_cache_spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    vllm_config = create_vllm_config(
        model_name="vllm-project/tiny-random-LlamaForCausalLM", max_model_len=1024
    )
    # The SM12x opt-in is architecture-gated; the builder only reads the name.
    vllm_config.model_config.model_arch_config.architectures = [
        "DeepseekV41ForCausalLM"
    ]
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_speculative_tokens,
        enable_adaptive_verification=True,
    )
    assert indexer._use_flattening(vllm_config)

    group = new_kv_cache_group(["layer"], kv_cache_spec)
    block_table_width = get_block_table_width(
        group.max_num_blocks_per_req(vllm_config, 1024), kv_cache_spec.block_size
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    device_lens_t = torch.tensor(device_lens, dtype=torch.int32, device=device)
    cpu_lens_t = torch.tensor(cpu_lens, dtype=torch.int32)
    qsl_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), cpu_lens_t.cumsum(0)))
    qsl_device = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), device_lens_t.cumsum(0))
    )
    # Keep the context start positive for every request.
    seq_lens = device_lens_t + 100
    # Distinct block ids per request so a wrong row is visible.
    block_table = (
        torch.arange(
            num_reqs * block_table_width, dtype=torch.int32, device=device
        ).view(num_reqs, block_table_width)
        + 1
    )
    metadata = builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=qsl_device,
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens.cpu(),
            num_reqs=num_reqs,
            num_actual_tokens=num_decode_tokens,
            max_query_len=max(cpu_lens),
            max_seq_len=int(seq_lens.max()),
            block_table_tensor=block_table,
            slot_mapping=torch.arange(
                num_decode_tokens, dtype=torch.int64, device=device
            ),
            causal=True,
        ),
    )

    decode = metadata.decode
    assert decode is not None
    assert not decode.requires_padding
    assert decode.seq_lens.shape == (num_decode_tokens, 1)

    expected_rows: list[int] = []
    expected_block_rows: list[int] = []
    for req, (decode_len, seq_len) in enumerate(zip(device_lens, seq_lens.tolist())):
        expected_rows.extend(
            seq_len - decode_len + offset + 1 for offset in range(decode_len)
        )
        expected_block_rows.extend([req] * decode_len)
    torch.testing.assert_close(
        decode.seq_lens.view(-1),
        torch.tensor(expected_rows, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        decode.block_table[:num_decode_tokens], block_table[expected_block_rows]
    )
    assert torch.all(decode.decode_lens == 1)
    torch.testing.assert_close(decode.per_req_decode_lens[:num_reqs], device_lens_t)
    # Adaptive replay can put up to next_n rows in one request.
    assert decode.write_max_decode_len == num_speculative_tokens + 1
    assert not decode.decode_is_uniform
