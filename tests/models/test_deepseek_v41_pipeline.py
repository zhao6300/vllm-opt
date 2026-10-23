# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.pipeline import (
    get_sharing_dependencies,
    validate_local_sharing,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        mhc_stream=None,
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix


def _config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
    )


@pytest.mark.parametrize(
    "ranges", [[(0, 40)], [(0, 20), (20, 40)], [(0, 8), (8, 14), (14, 20), (20, 40)]]
)
def test_group_aligned_pipeline_cuts_keep_all_sources_local(ranges):
    validate_local_sharing(get_sharing_dependencies(_config(), ranges))


def test_equal_pp4_reports_cross_stage_kv_index_and_candidates():
    dependencies = get_sharing_dependencies(
        _config(), [(0, 10), (10, 20), (20, 30), (30, 40)]
    )
    cross = {
        (d.kind, d.source_layer, d.consumer_layer)
        for d in dependencies
        if d.source_stage != d.consumer_stage
    }
    assert {("kv", 8, 10), ("index", 28, 30), ("candidate", 20, 32)} <= cross
    with pytest.raises(NotImplementedError, match="source layer 8.*stage 0.*stage 1"):
        validate_local_sharing(dependencies)


@pytest.mark.parametrize("values", [[8, 2], [2, 2], [0, 2], [-1, 2], [2, 40]])
def test_invalid_source_lists_fail_before_layer_construction(values):
    config = _config()
    config.kv_source_layer_ids = values
    with pytest.raises(ValueError, match="kv_source_layer_ids"):
        get_sharing_dependencies(config, [(0, 40)])


def test_compressed_consumer_requires_an_earlier_source():
    config = _config()
    config.kv_source_layer_ids = [8, 14, 20]
    with pytest.raises(ValueError, match="layer 2 has no preceding kv source"):
        get_sharing_dependencies(config, [(0, 40)])


def test_candidate_source_must_publish_indices():
    config = _config()
    config.candidate_source_layer_id = 21
    with pytest.raises(ValueError, match="candidate source must be an index source"):
        get_sharing_dependencies(config, [(0, 40)])
