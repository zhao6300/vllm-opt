# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

from vllm.v1.attention.backend import MultipleOf


def _sm12x_platform(mock_platform):
    mock_platform.is_device_capability_family.side_effect = lambda family: family in (
        120,
        121,
    )


def _sm90_platform(mock_platform):
    mock_platform.is_device_capability_family.side_effect = lambda family: family == 90


def _ds41_sparse_mock_platform(mock_platform):
    mock_platform.is_device_capability_family.side_effect = lambda family: family in (
        90,
        120,
        121,
    )


def _other_platform(mock_platform):
    mock_platform.is_device_capability_family.side_effect = lambda family: False


def test_v41_sparse_mla_accepts_ratio_token_block_sizes():
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend

    with mock.patch(
        "vllm.models.deepseek_v41.sparse_mla.current_platform"
    ) as mock_platform:
        _sm12x_platform(mock_platform)
        sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()

    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 64


def test_v41_indexer_accepts_ratio_token_block_sizes():
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

    with mock.patch(
        "vllm.v1.attention.backends.mla.indexer.current_platform"
    ) as mock_platform:
        _sm12x_platform(mock_platform)
        sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()

    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 64


def test_v41_select_common_block_size_uses_ratio_page_width():
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
    from vllm.v1.worker.utils import select_common_block_size

    backends = [
        DeepseekV4FlashInferMLASparseBackend,
        DeepseekV41IndexerBackend,
        DeepseekSparseSWABackend,
    ]

    with mock.patch(
        "vllm.models.deepseek_v41.nvidia.flashinfer_sparse.current_platform"
    ) as mock_platform:
        _sm12x_platform(mock_platform)
        assert select_common_block_size(64, backends) == 64
        assert select_common_block_size(128, backends) == 128


def test_v41_flashinfer_shared_backends_use_common_minimum():
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.worker.utils import select_common_block_size

    backends = [DeepseekV4SparseMLABackend, DeepseekV41IndexerBackend]

    with (
        mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as sparse_platform,
        mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as indexer_platform,
    ):
        _sm90_platform(sparse_platform)
        _sm90_platform(indexer_platform)
        assert select_common_block_size(64, backends) == 64
        assert select_common_block_size(128, backends) == 128


def test_v41_other_architectures_keep_128_token_pages():
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.worker.utils import select_common_block_size

    backends = [DeepseekV4SparseMLABackend, DeepseekV41IndexerBackend]

    with (
        mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as sparse_platform,
        mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as indexer_platform,
    ):
        _other_platform(sparse_platform)
        _other_platform(indexer_platform)
        assert select_common_block_size(128, backends) == 128


@pytest.mark.parametrize(
    ("cache_dtype", "state_content_bytes", "alignment"),
    [
        ("fp8_ds_mla", 528, 512),
        ("nvfp4_ds_mla", 288, 512),
    ],
)
def test_v41_compressed_cache_spec_uses_layout_bytes_per_token(
    cache_dtype,
    state_content_bytes,
    alignment,
):
    from vllm.models.deepseek_v41.attention import _compressed_cache_spec

    vllm_config = mock.Mock()
    vllm_config.cache_config.block_size = 64
    with mock.patch(
        "vllm.models.deepseek_v41.attention.current_platform"
    ) as mock_platform:
        _sm12x_platform(mock_platform)
        spec = _compressed_cache_spec(
            vllm_config,
            512,
            1,
            cache_dtype,
            torch.uint8,
        )

    assert spec.dtype == torch.uint8
    assert spec.state_content_bytes == state_content_bytes
    assert spec.alignment == alignment


@pytest.mark.parametrize(
    ("cache_dtype", "kv_mxfp8", "expected"),
    [
        ("fp8_ds_mla", True, 528),
        ("fp8_ds_mla", False, 584),
        ("nvfp4_ds_mla", True, 528),
        ("nvfp4_ds_mla", False, 584),
    ],
)
def test_v41_swa_cache_bytes_per_token_follows_cache_dtype(
    cache_dtype,
    kv_mxfp8,
    expected,
):
    from vllm.models.deepseek_v41.attention import (
        _swa_bytes_per_token_for_cache_dtype,
    )

    assert _swa_bytes_per_token_for_cache_dtype(cache_dtype, kv_mxfp8) == expected


@pytest.mark.parametrize(
    ("config_block_size", "compress_ratio", "expected"),
    [
        (32, 1, 64),
        (64, 1, 64),
        (128, 1, 128),
        (64, 2, 128),
        (128, 2, 128),
    ],
)
def test_v41_compressed_cache_spec_sizes_state_page(
    config_block_size,
    compress_ratio,
    expected,
):
    from vllm.models.deepseek_v41.attention import _compressed_cache_spec

    vllm_config = mock.Mock()
    vllm_config.cache_config.block_size = config_block_size
    spec = _compressed_cache_spec(
        vllm_config,
        512,
        compress_ratio,
        "fp8_ds_mla",
        torch.uint8,
    )

    assert spec.block_size == expected
    assert spec.tokens_per_state == compress_ratio
    with mock.patch(
        "vllm.v1.attention.backends.mla.indexer.current_platform"
    ) as mock_platform:
        _ds41_sparse_mock_platform(mock_platform)
        selected_block_size = min(spec.block_size, 64 * compress_ratio)
    assert spec.get_num_kernel_states(selected_block_size) == 64


@pytest.mark.parametrize(
    ("config_block_size", "compress_ratio", "expected"),
    [
        (32, 1, 64),
        (64, 1, 64),
        (128, 1, 128),
        (64, 2, 128),
        (128, 2, 128),
    ],
)
def test_v41_attention_cache_spec_sizes_state_page(
    config_block_size,
    compress_ratio,
    expected,
):
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferSM120Attention,
    )

    vllm_config = mock.Mock()
    vllm_config.cache_config = SimpleNamespace(block_size=config_block_size)
    vllm_config.model_config = mock.Mock()

    attention = DeepseekV4FlashInferSM120Attention.__new__(
        DeepseekV4FlashInferSM120Attention
    )
    attention.is_kv_source = True
    attention.kv_cache_dtype = "fp8_ds_mla"
    attention.kv_cache_torch_dtype = torch.uint8
    attention.head_dim = 512
    attention.compress_ratio = compress_ratio
    attention.kv_mxfp8 = False
    attention.compressed_bytes_per_token = 584
    attention.kv_page_alignment = 576

    spec = attention.get_kv_cache_spec(vllm_config)
    selected_block_size = min(spec.block_size, 64 * compress_ratio)
    assert spec.block_size == expected
    assert spec.get_num_kernel_states(selected_block_size) == 64


@pytest.mark.parametrize(
    ("config_block_size", "compress_ratio", "expected"),
    [
        (32, 1, 64),
        (64, 1, 64),
        (64, 2, 128),
        (128, 2, 128),
    ],
)
def test_v41_indexer_cache_matches_compressed_state_page(
    config_block_size,
    compress_ratio,
    expected,
):
    from vllm.models.deepseek_v41.attention import (
        DeepseekV4IndexerCache,
        _indexer_k_cache_head_dim,
    )

    vllm_config = mock.Mock()
    vllm_config.cache_config.block_size = config_block_size
    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    cache = DeepseekV4IndexerCache.__new__(DeepseekV4IndexerCache)
    cache.head_dim = _indexer_k_cache_head_dim(128, False)
    cache.compress_ratio = compress_ratio
    cache.dtype = torch.uint8
    cache.sparse_logits = False

    spec = cache.get_kv_cache_spec(vllm_config)

    assert spec.block_size == expected
    assert spec.get_num_kernel_states(expected) == 64


@pytest.mark.parametrize(
    ("block_size", "tokens_per_state", "expected"),
    [
        (64, 1, 64),
        (128, 1, 128),
        (64, 2, 32),
        (128, 2, 64),
    ],
)
def test_mla_spec_num_states_matches_compressed_pages(
    block_size, tokens_per_state, expected
):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=tokens_per_state,
    )
    assert spec.num_states == expected
