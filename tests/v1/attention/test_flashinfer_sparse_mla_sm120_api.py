# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.models.deepseek_v41 import nvidia
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class _FakeSm120Wrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, **kwargs):
        self.run_kwargs = kwargs


def _install_sm120_wrapper(monkeypatch) -> None:
    import flashinfer.mla._sparse_mla_sm120 as flashinfer_sm120

    monkeypatch.setattr(
        flashinfer_sm120,
        "SparseMLASm120Wrapper",
        _FakeSm120Wrapper,
    )
    monkeypatch.setattr(
        nvidia.flashinfer_sparse,
        "_sparse_mla_sm120_wrappers",
        {},
    )


def _sm120_attention_tensors():
    query = torch.zeros((48, 64, 512), dtype=torch.bfloat16)
    output = torch.zeros_like(query)
    swa_cache = torch.empty((8, 64, 528), dtype=torch.uint8)
    extra_cache = torch.empty((8, 64, 1, 288), dtype=torch.uint8)
    return query, swa_cache, extra_cache, output


def _sm120_indices() -> torch.Tensor:
    return torch.empty((48, 1152), dtype=torch.int32)


def _sm120_lengths(num_tokens: int = 48) -> torch.Tensor:
    return torch.empty((num_tokens,), dtype=torch.int32)


def _sm120_extra_indices() -> torch.Tensor:
    return torch.empty((48, 1, 1152), dtype=torch.int32)


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_sm120_backend_exposes_masked_mha_available_false() -> None:
    # The prefill dispatcher reads ``impl.masked_mha_available`` for any sparse
    # impl; SM120 has no masked-MHA prefill kernel, so the attribute must exist
    # and be False rather than AttributeError at startup.
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.masked_mha_available is False


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


def test_sm120_dsv4_1_uses_explicit_fp8_precision_wrapper(monkeypatch) -> None:
    class FakeWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            self.run_kwargs = kwargs

    import flashinfer.mla._sparse_mla_sm120 as flashinfer_sm120

    monkeypatch.setattr(flashinfer_sm120, "SparseMLASm120Wrapper", FakeWrapper)
    monkeypatch.setattr(
        nvidia.flashinfer_sparse,
        "_sparse_mla_sm120_wrappers",
        {},
    )

    query = torch.zeros((48, 64, 512), dtype=torch.bfloat16)
    output = query.clone()
    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(
        query=query,
        swa_kv_cache=torch.empty((8, 64, 528), dtype=torch.uint8),
        sparse_indices=torch.empty((48, 1152), dtype=torch.int32),
        output=output,
        sm_scale=1.0,
        topk_length=torch.empty((48,), dtype=torch.int32),
        attn_sink=None,
        extra_kv_cache=torch.empty((8, 64, 1, 288), dtype=torch.uint8),
        extra_sparse_indices=torch.empty((48, 1, 1152), dtype=torch.int32),
        extra_sparse_topk_lens=torch.empty((48,), dtype=torch.int32),
        extra_kv_fp4=True,
        prefill_impl="auto",
    )
    cached = nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers[query.device, True]

    assert isinstance(cached, FakeWrapper)
    assert cached.kwargs["kv_scale_format"] == "ue8m0_g32"
    assert cached.kwargs["extra_kv_fp4"] is True
    assert cached.kwargs["compute_precision"] == "fp8"


def test_sm120_dsv4_1_isolate_flag_when_extra_cache_is_absent(
    monkeypatch,
) -> None:
    import flashinfer.mla._sparse_mla_sm120 as flashinfer_sm120

    monkeypatch.setattr(
        flashinfer_sm120,
        "SparseMLASm120Wrapper",
        _FakeSm120Wrapper,
    )
    monkeypatch.setattr(
        nvidia.flashinfer_sparse,
        "_sparse_mla_sm120_wrappers",
        {},
    )

    query = torch.zeros((48, 64, 512), dtype=torch.bfloat16)
    output = query.clone()
    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(
        query=query,
        swa_kv_cache=torch.empty((8, 64, 528), dtype=torch.uint8),
        sparse_indices=torch.empty((48, 1152), dtype=torch.int32),
        output=output,
        sm_scale=1.0,
        topk_length=torch.empty((48,), dtype=torch.int32),
        attn_sink=None,
        extra_kv_cache=None,
        extra_sparse_indices=None,
        extra_sparse_topk_lens=None,
        extra_kv_fp4=True,
        prefill_impl="auto",
    )
    cached = nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers[query.device, False]

    assert isinstance(cached, _FakeSm120Wrapper)
    assert cached.kwargs["extra_kv_fp4"] is False
    assert cached.run_kwargs["extra_kv_cache"] is None
    assert cached.run_kwargs["extra_indices"] is None


def test_sm120_dsv4_1_requires_extra_cache_for_fp4_extra_wrapper(
    monkeypatch,
) -> None:
    _install_sm120_wrapper(monkeypatch)

    query, swa_cache, extra_cache, output = _sm120_attention_tensors()
    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(
        query=query,
        swa_kv_cache=swa_cache,
        sparse_indices=_sm120_indices(),
        output=output,
        sm_scale=1.0,
        topk_length=_sm120_lengths(),
        attn_sink=None,
        extra_kv_cache=None,
        extra_sparse_indices=None,
        extra_sparse_topk_lens=None,
        extra_kv_fp4=False,
        prefill_impl="auto",
    )
    assert set(nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers) == {
        (query.device, False)
    }

    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(
        query=query,
        swa_kv_cache=swa_cache,
        sparse_indices=_sm120_indices(),
        output=output,
        sm_scale=1.0,
        topk_length=_sm120_lengths(),
        attn_sink=None,
        extra_kv_cache=extra_cache,
        extra_sparse_indices=_sm120_extra_indices(),
        extra_sparse_topk_lens=_sm120_lengths(num_tokens=48),
        extra_kv_fp4=True,
        prefill_impl="auto",
    )
    assert set(nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers) == {
        (query.device, False),
        (query.device, True),
    }
    assert (
        nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers[query.device, True].kwargs[
            "extra_kv_fp4"
        ]
        is True
    )


def test_sm120_dsv4_1_reuses_wrapper_for_same_extra_cache_mode(
    monkeypatch,
) -> None:
    _install_sm120_wrapper(monkeypatch)

    query, swa_cache, extra_cache, output = _sm120_attention_tensors()
    args = dict(
        query=query,
        swa_kv_cache=swa_cache,
        sparse_indices=_sm120_indices(),
        output=output,
        sm_scale=1.0,
        topk_length=_sm120_lengths(),
        attn_sink=None,
        extra_kv_cache=extra_cache,
        extra_sparse_indices=_sm120_extra_indices(),
        extra_sparse_topk_lens=_sm120_lengths(),
        extra_kv_fp4=True,
        prefill_impl="auto",
    )
    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(**args)
    cached = nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers[query.device, True]
    first = len(cached.run_kwargs)

    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(**args)
    assert cached.run_kwargs["q"] is args["query"]
    assert cached.run_kwargs["extra_kv_cache"] is args["extra_kv_cache"]
    assert len(cached.run_kwargs) == first


def test_sm120_dsv4_1_forwards_prefill_impl_and_sink(monkeypatch) -> None:
    _install_sm120_wrapper(monkeypatch)
    query, swa_cache, extra_cache, output = _sm120_attention_tensors()
    sink = torch.full((64,), 0.25, dtype=torch.float32)

    nvidia.flashinfer_sparse._sparse_mla_sm120_paged_attention(
        query=query,
        swa_kv_cache=swa_cache,
        sparse_indices=_sm120_indices(),
        output=output,
        sm_scale=0.25,
        topk_length=_sm120_lengths(),
        attn_sink=sink,
        extra_kv_cache=extra_cache,
        extra_sparse_indices=_sm120_extra_indices(),
        extra_sparse_topk_lens=_sm120_lengths(),
        extra_kv_fp4=True,
        prefill_impl="swapab",
    )
    cached = nvidia.flashinfer_sparse._sparse_mla_sm120_wrappers[query.device, True]

    assert cached.run_kwargs["sm_scale"] == 0.25
    assert cached.run_kwargs["attn_sink"] is sink
    assert cached.run_kwargs["prefill_impl"] == "swapab"
