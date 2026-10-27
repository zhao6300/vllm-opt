# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.models.deepseek_v41 import nvidia
from vllm.models.deepseek_v41.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
    _flashinfer_dsv4_workspace_bytes,
    _get_flashinfer_dsv4_workspace,
)
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


def _sm12x_platform(mock_platform):
    mock_platform.is_device_capability_family.side_effect = lambda family: (
        family
        in (
            120,
            121,
        )
    )


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
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)}),
        _DECODE_DSV4_1_DISPATCH=frozenset({(64, 256), (64, 384)}),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    # Legacy DSV4 checks the legacy table only.
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(64, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)

    # DSv4.1 checks its independent table only.
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(64, 256, dsv41=True)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(64, 384, dsv41=True)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128, dsv41=True)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(64, 512, dsv41=True)

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


def test_sm120_dsv4_auto_kv_cache_resolves_to_nvfp4() -> None:
    assert DeepseekV4FlashInferSM120Attention.packed_kv_cache_dtype() == "nvfp4_ds_mla"

    cache_config = SimpleNamespace(cache_dtype="auto")

    resolved, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        True,
        cache_config.cache_dtype,
        cache_config,
        DeepseekV4FlashInferSM120Attention.packed_kv_cache_dtype(),
    )

    assert resolved == "nvfp4_ds_mla"
    assert cache_config.cache_dtype == "nvfp4_ds_mla"
    assert torch_dtype == torch.uint8


@pytest.mark.parametrize(
    ("kv_cache_dtype", "use_fp4_extra_kv", "expected_reasons"),
    [
        ("nvfp4_ds_mla", True, []),
        (
            "fp8_ds_mla",
            False,
            ["FLASHINFER_SM120 requires nvfp4_ds_mla for the FP4 compressed cache"],
        ),
        (
            "fp8_ds_mla",
            True,
            ["FLASHINFER_SM120 requires nvfp4_ds_mla for the FP4 compressed cache"],
        ),
    ],
)
def test_sm120_dsv4_rejects_mismatched_fp4_extra_cache(
    kv_cache_dtype,
    use_fp4_extra_kv,
    expected_reasons,
) -> None:
    with mock.patch(
        "vllm.models.deepseek_v41.sparse_mla.current_platform"
    ) as mock_platform:
        _sm12x_platform(mock_platform)
        invalid_reasons = DeepseekV4FlashInferMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype=kv_cache_dtype,
            block_size=128,
            use_mla=True,
            has_sink=True,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == expected_reasons


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
    assert cached.kwargs["kv_scale_format"] == "ue8m0_g32_inline"
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


def _dsv4_workspace_config(
    *,
    max_model_len: int,
    index_topk: int = 512,
    num_speculative_tokens: int | None = 7,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(index_topk=index_topk),
        ),
        attention_config=SimpleNamespace(
            use_non_causal=num_speculative_tokens is not None
        ),
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        ),
    )


def test_sm120_c128a_workspace_covers_reporter_config() -> None:
    # DeepSeek-V4-Flash-0731 at its native 1M context with DSpark k=7: the
    # SM120 C128A decode plan spans the whole 8192-entry compressed buffer, so
    # it needs far more than the old 128 MiB floor even for a few tokens.
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        _flashinfer_dsv4_workspace_bytes,
    )

    computed = _flashinfer_dsv4_workspace_bytes(
        _dsv4_workspace_config(max_model_len=1_048_576),
        padded_heads=64,
        window_size=128,
        compress_ratio=128,
    )
    assert computed == 551_616_512
    assert computed > 128 * 1024 * 1024


def test_sm120_dsv4_workspace_scales_with_compressed_width() -> None:
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        _flashinfer_dsv4_workspace_bytes,
    )

    config = _dsv4_workspace_config(max_model_len=1_048_576)
    c128a = _flashinfer_dsv4_workspace_bytes(
        config, padded_heads=64, window_size=128, compress_ratio=128
    )
    c4a = _flashinfer_dsv4_workspace_bytes(
        config, padded_heads=64, window_size=128, compress_ratio=4
    )
    swa_only = _flashinfer_dsv4_workspace_bytes(
        config, padded_heads=64, window_size=128, compress_ratio=1
    )
    assert c128a > c4a > swa_only > 0

    shorter = _flashinfer_dsv4_workspace_bytes(
        _dsv4_workspace_config(max_model_len=131_072),
        padded_heads=64,
        window_size=128,
        compress_ratio=128,
    )
    assert shorter < c128a


def test_sm120_dsv4_workspace_tracks_dspark_and_vision_widths() -> None:
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        _flashinfer_dsv4_workspace_bytes,
    )

    causal_config = _dsv4_workspace_config(
        max_model_len=1_048_576, num_speculative_tokens=None
    )
    causal = _flashinfer_dsv4_workspace_bytes(
        causal_config,
        padded_heads=64,
        window_size=128,
        compress_ratio=4,
    )
    dspark = _flashinfer_dsv4_workspace_bytes(
        _dsv4_workspace_config(max_model_len=1_048_576),
        padded_heads=64,
        window_size=128,
        compress_ratio=4,
    )
    vision = _flashinfer_dsv4_workspace_bytes(
        causal_config,
        padded_heads=64,
        window_size=128,
        compress_ratio=4,
        max_image_tokens=1024,
    )
    assert dspark > causal
    assert vision > causal


def test_dsv4_workspace_grows_but_never_shrinks() -> None:
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
        _get_flashinfer_dsv4_workspace,
    )

    device = torch.device("cpu")
    floor = _get_flashinfer_dsv4_workspace(device)
    assert floor.numel() >= _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE

    grown = _get_flashinfer_dsv4_workspace(device, floor.numel() * 2)
    assert grown.numel() == floor.numel() * 2

    assert _get_flashinfer_dsv4_workspace(device) is grown


def _dsv41_workspace_config(
    *,
    index_topk: int = 1024,
    num_speculative_tokens: int | None = 7,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=index_topk),
        ),
        attention_config=SimpleNamespace(
            use_non_causal=num_speculative_tokens is not None
        ),
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        ),
    )


def test_sm120_dsv41_workspace_covers_shared_index_topk() -> None:
    # Every compressed V4.1 layer gathers top-k over the shared index_topk-wide
    # buffer, so the SM120 decode plan needs more than the 128 MiB floor.
    computed = _flashinfer_dsv4_workspace_bytes(
        _dsv41_workspace_config(index_topk=2048),
        padded_heads=64,
        window_size=128,
        compress_ratio=1,
    )
    assert computed == 147_390_464
    assert computed > _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE


def test_sm120_workspace_scales_with_dspark_and_index_topk() -> None:
    causal = _dsv41_workspace_config(num_speculative_tokens=None)
    swa_causal = _flashinfer_dsv4_workspace_bytes(
        causal, padded_heads=64, window_size=128, compress_ratio=0
    )
    swa_dspark = _flashinfer_dsv4_workspace_bytes(
        _dsv41_workspace_config(), padded_heads=64, window_size=128, compress_ratio=0
    )
    ratio1 = _flashinfer_dsv4_workspace_bytes(
        _dsv41_workspace_config(), padded_heads=64, window_size=128, compress_ratio=1
    )
    ratio2 = _flashinfer_dsv4_workspace_bytes(
        _dsv41_workspace_config(), padded_heads=64, window_size=128, compress_ratio=2
    )
    wide = _flashinfer_dsv4_workspace_bytes(
        _dsv41_workspace_config(index_topk=2048),
        padded_heads=64,
        window_size=128,
        compress_ratio=1,
    )
    assert swa_causal < swa_dspark < ratio1 < wide
    assert ratio1 == ratio2


def test_dsv41_workspace_grows_but_never_shrinks() -> None:
    device = torch.device("cpu")
    floor = _get_flashinfer_dsv4_workspace(device)
    assert floor.numel() >= _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE

    grown = _get_flashinfer_dsv4_workspace(device, floor.numel() * 2)
    assert grown.numel() == floor.numel() * 2

    assert _get_flashinfer_dsv4_workspace(device) is grown


def _padded_swa_cache(
    pages: int = 4,
    page_rows: int = 32,
    page_span: int = 36,
    row_bytes: int = 528,
) -> torch.Tensor:
    """SWA cache with the interleaved V4.1 layout's per-page row padding."""
    backing = torch.zeros(pages * page_span * row_bytes, dtype=torch.uint8)
    return backing.as_strided(
        (pages, page_rows, row_bytes), (page_span * row_bytes, row_bytes, 1)
    )


def _remap_reference(
    indices: torch.Tensor, block_size: int, page_rows: int, page_span: int
) -> torch.Tensor:
    safe = indices.clamp(min=0).long()
    pages_per_block = block_size // page_rows
    page = (safe // block_size) * pages_per_block + (safe % block_size) // page_rows
    rows = page * page_span + (safe % block_size) % page_rows
    return torch.where(indices < 0, -1, rows).to(torch.int32)


def test_storage_offset_indices_remap_padded_pages() -> None:
    from vllm.models.deepseek_v41.common.ops import (
        remap_flashinfer_sparse_indices,
    )

    compact = torch.tensor([[0, 31, 32, 63, -1]], dtype=torch.int32)
    assert remap_flashinfer_sparse_indices(compact, 32, 32, 32) is compact

    if not torch.cuda.is_available():
        pytest.skip("the remap kernel is a CUDA kernel")
    indices = torch.tensor(
        [[0, 1, 31, 32, 33, 63, 64, 127, -1]], dtype=torch.int32, device="cuda"
    )
    remapped = remap_flashinfer_sparse_indices(indices, 32, 32, 36)
    expected = _remap_reference(indices, 32, 32, 36)
    assert torch.equal(remapped, expected)
    assert remapped.tolist() == [[0, 1, 31, 36, 37, 67, 72, 139, -1]]

    # [num_tokens, 1, width] metadata rows and blocks spanning several pages.
    rows = indices[:, None, :]
    assert torch.equal(
        remap_flashinfer_sparse_indices(rows, 32, 32, 36), expected[:, None, :]
    )
    assert torch.equal(
        remap_flashinfer_sparse_indices(indices, 128, 64, 72),
        _remap_reference(indices, 128, 64, 72),
    )


def test_sm120_dsv41_remaps_sparse_indices_of_padded_swa_cache(monkeypatch) -> None:
    """A padded page stores slot b*32+o at row b*36+o; FlashInfer's inline-scale
    kernels read records as flat rows, so the SM120 path must hand them storage
    offsets or attention reads the wrong KV (garbled, repeating output)."""
    calls = []

    def fake_remap(indices, block_size, page_rows, page_span, out=None):
        calls.append((block_size, page_rows, page_span))
        out.copy_(
            _remap_reference(indices, block_size, page_rows, page_span).reshape(
                out.shape
            )
        )
        return out if out.shape == indices.shape else out.view(indices.shape)

    monkeypatch.setattr(
        nvidia.flashinfer_sparse, "remap_flashinfer_sparse_indices", fake_remap
    )
    captured = {}
    monkeypatch.setattr(
        nvidia.flashinfer_sparse,
        "_sparse_mla_sm120_paged_attention",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(nvidia.flashinfer_sparse, "_storage_offset_index_scratch", {})

    attn = DeepseekV4FlashInferSM120Attention.__new__(
        DeepseekV4FlashInferSM120Attention
    )
    attn.kv_cache_torch_dtype = torch.bfloat16
    attn.scale = 1.0
    attn.attn_sink = None
    attn.use_fp4_extra_kv = False
    attn.max_num_batched_tokens = 8

    slots = torch.tensor([[0, 31, 32, 33, -1], [64, 95, -1, -1, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=slots[:, None, :],
        decode_swa_lens=torch.tensor([4, 2], dtype=torch.int32),
        block_size=32,
    )
    attn.swa_cache_layer = SimpleNamespace(kv_cache=_padded_swa_cache())
    query = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    output = torch.zeros_like(query)

    attn._forward_decode(
        q=query,
        kv_cache=None,
        swa_metadata=metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert calls == [(32, 32, 36)]
    # Same values as the reference, in the metadata's [tokens, 1, width] shape.
    assert torch.equal(
        captured["sparse_indices"], _remap_reference(slots, 32, 32, 36)[:, None, :]
    )
    # The remapped rows live in the CUDA-graph-safe scratch, not a fresh tensor.
    assert captured["sparse_indices"].shape == (2, 1, 5)

    # A compact page needs no remap and keeps the metadata's own buffer.
    calls.clear()
    attn.swa_cache_layer = SimpleNamespace(kv_cache=_padded_swa_cache(page_span=32))
    attn._forward_decode(
        q=query,
        kv_cache=None,
        swa_metadata=metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )
    assert calls == []
    assert captured["sparse_indices"].data_ptr() == slots.data_ptr()
