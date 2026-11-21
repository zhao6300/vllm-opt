# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware and model gates for the sparse indexer's K-cache dtype."""

import types

import pytest

import vllm.v1.attention.backends.mla.indexer as mla_indexer
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4


def _config(kv_dtype: str, model_type: str = "deepseek_v41"):
    return types.SimpleNamespace(
        attention_config=types.SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: kv_dtype
        ),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=model_type)
        ),
    )


def _platform(monkeypatch, family: int):
    fake = types.SimpleNamespace(
        is_rocm=lambda: False,
        is_device_capability_family=lambda value: value == family,
    )
    monkeypatch.setattr(mla_indexer, "current_platform", fake)


@pytest.mark.parametrize("family", [100, 120, 121])
@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
def test_mxfp4_indexer_cache_enabled_on_supported_sms(monkeypatch, family, model_type):
    """sm_10x and sm_12x ship the MXFP4 MQA-logits kernels V4/V4.1 read."""
    _platform(monkeypatch, family)
    assert dsa_indexer_uses_fp4(_config("mxfp4", model_type))


def test_mxfp4_indexer_cache_rejected_before_blackwell(monkeypatch):
    _platform(monkeypatch, 90)
    with pytest.raises(ValueError, match="sm_10x datacenter"):
        dsa_indexer_uses_fp4(_config("mxfp4"))


def test_mxfp4_indexer_cache_rejected_without_fused_store(monkeypatch):
    """Only V4/V4.1 write FP4 pages; other DSA models assert at the first step."""
    _platform(monkeypatch, 120)
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        dsa_indexer_uses_fp4(_config("mxfp4", "deepseek_v32"))


def test_fp8_indexer_cache_needs_no_fp4_support(monkeypatch):
    _platform(monkeypatch, 90)
    assert dsa_indexer_uses_fp4(_config("fp8")) is False


def test_unknown_indexer_cache_dtype_rejected(monkeypatch):
    _platform(monkeypatch, 120)
    with pytest.raises(ValueError, match="is not supported by the DeepSeek"):
        dsa_indexer_uses_fp4(_config("nvfp4"))
