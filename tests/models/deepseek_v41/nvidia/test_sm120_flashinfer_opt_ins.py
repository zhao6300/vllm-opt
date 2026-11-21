# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in SM120 FlashInfer kernels against the paths they replace: the
fused MoE gate against `fused_topk_bias`, the paged indexer logits against
DeepGEMM's, and the FP8 record writer against the Triton one."""

import pytest
import torch
import torch.nn.functional as F

import vllm.utils.flashinfer as fi
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120),
    reason="the opt-in FlashInfer kernels target SM120",
)

HIDDEN, EXPERTS, TOPK = 5120, 384, 6


class _Gate(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = torch.randn(EXPERTS, HIDDEN, generator=generator, device="cuda")
        self.weight = torch.nn.Parameter((weight * 0.01).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x.float(), self.weight.float()), None


def _routers(monkeypatch):
    import vllm.model_executor.layers.fused_moe.router.sm12x_mega_gate_router as mg
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.sm12x_mega_gate_router import (
        Sm12xMegaGateRouter,
    )

    monkeypatch.setenv("VLLM_FLASHINFER_DSV41_SM120_MEGA_GATE", "1")
    if not mg.sm12x_mega_gate_available():
        pytest.skip("FlashInfer has no usable SM12x fused gate here")
    generator = torch.Generator(device="cuda").manual_seed(1)
    bias = torch.randn(EXPERTS, generator=generator, device="cuda") * 0.02
    bias = bias.contiguous()
    kwargs = dict(
        top_k=TOPK,
        global_num_experts=EXPERTS,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
    )
    fused = Sm12xMegaGateRouter(**kwargs)
    stock = FusedTopKBiasRouter(**kwargs)
    fused.attach_gate(_Gate(0).cuda(), skipped=True)
    return mg, fused, stock


def _stock_routing(stock, gate, x):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )

    logits, _ = gate(x)
    return FusedTopKBiasRouter._compute_routing(stock, x, logits, None)


@pytest.mark.parametrize("rows", [1, 8, 64, 512])
def test_mega_gate_matches_fused_topk_bias(monkeypatch, rows):
    _, fused, stock = _routers(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router."
        "sm12x_mega_gate_router.get_moe_padding_mask",
        lambda _: None,
    )
    gate = fused._gate()
    x = (torch.randn(rows, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    ref_weights, ref_ids = _stock_routing(stock, gate, x)
    weights, ids = fused._compute_routing(x, None, None)
    assert fused.uses_fused_gate
    assert torch.equal(ids.long(), ref_ids.long())
    torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


def test_mega_gate_masks_padded_rows(monkeypatch):
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as bm

    mg, fused, stock = _routers(monkeypatch)
    gate = fused._gate()
    rows = 16
    padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
    padding[::3] = True
    for module in (mg, bm):
        monkeypatch.setattr(module, "get_moe_padding_mask", lambda _: padding)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    ref_weights, ref_ids = _stock_routing(stock, gate, x)
    weights, ids = fused._compute_routing(x, None, None)
    assert torch.equal(ids.long(), ref_ids.long())
    assert not ids[padding].ge(0).any(), "padding rows must route nowhere"
    assert not weights[padding].any()
    torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


def test_mega_gate_falls_back_when_the_kernel_declines(monkeypatch):
    _, fused, stock = _routers(monkeypatch)
    gate = fused._gate()
    x = (torch.randn(32, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    # A stride-2 view is not contiguous, so the fused kernel declines and the
    # router must run the gate GEMM the runner skipped.
    x = x.repeat_interleave(2, dim=0)[::2]
    ref_weights, ref_ids = _stock_routing(stock, gate, x)
    weights, ids = fused._compute_routing(x, None, None)
    assert torch.equal(ids.long(), ref_ids.long())
    torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("rows", [16, 64, 1025])
def test_mega_gate_routes_fresh_hidden_states_each_call(monkeypatch, rows):
    """An eager step hands in a new hidden-states tensor every call.

    The gate reuses a plan for small batches, so each call must rank its own
    rows; `1025` covers the shapes that still rebuild a plan per tensor.
    """
    _, fused, stock = _routers(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router."
        "sm12x_mega_gate_router.get_moe_padding_mask",
        lambda _: None,
    )
    gate = fused._gate()
    for _ in range(3):
        x = (torch.randn(rows, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
        ref_weights, ref_ids = _stock_routing(stock, gate, x)
        weights, ids = fused._compute_routing(x, None, None)
        assert torch.equal(ids.long(), ref_ids.long())
        torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


def test_indexer_logits_match_deepgemm(monkeypatch):
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
    )

    monkeypatch.setenv("VLLM_FLASHINFER_DSV41_SM120_INDEXER_LOGITS", "1")
    fi.sm120_indexer_logits_enabled.cache_clear()
    if not fi.sm120_indexer_logits_enabled():
        pytest.skip("FlashInfer has no SM120 paged MQA logits here")
    try:
        torch.manual_seed(0)
        heads, dim, row_bytes, page_kv, pages = 32, 128, 132, 64, 12
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        for batch, next_n in ((1, 1), (8, 2), (32, 4)):
            kv_cache = torch.zeros(
                batch * pages,
                page_kv,
                1,
                row_bytes,
                dtype=torch.uint8,
                device="cuda",
            )
            values = torch.randn(batch * pages, page_kv, 1, dim, device="cuda")
            kv_cache[..., :dim] = values.to(torch.float8_e4m3fn).view(torch.uint8)
            scale = torch.full((1,), 0.125, device="cuda")
            kv_cache[..., dim:] = scale.view(torch.uint8).expand(
                kv_cache[..., dim:].shape
            )
            block_table = torch.stack(
                [
                    torch.arange(pages, dtype=torch.int32, device="cuda") + i * pages
                    for i in range(batch)
                ]
            )
            context = torch.full(
                (batch, next_n),
                (pages - 2) * page_kv + 1,
                dtype=torch.int32,
                device="cuda",
            )
            q = torch.randn(batch, next_n, heads, dim, device="cuda").to(
                torch.float8_e4m3fn
            )
            weights = torch.randn(batch * next_n, heads, device="cuda")
            ref = fp8_fp4_paged_mqa_logits(
                (q, None),
                kv_cache,
                weights,
                context,
                block_table,
                get_paged_mqa_logits_metadata(context, page_kv, num_sms),
                max_model_len=pages * page_kv,
            )
            width = min(pages * page_kv, block_table.shape[1] * page_kv)
            meta = torch.empty((num_sms + 1, 2), dtype=torch.int32, device="cuda")
            logits = fi.flashinfer_sm120_fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                context,
                block_table,
                meta,
                width,
                page_kv=page_kv,
            )
            # The kernels accumulate in different orders, so logits that
            # cancel to ~0 differ by a few 1e-38; everything else is
            # bit-identical (the host routes also flush denormals).
            for r in range(batch * next_n):
                columns = int(context.reshape(-1)[r])
                torch.testing.assert_close(
                    logits[r, :columns], ref[r, :columns], atol=1e-33, rtol=0
                )
    finally:
        monkeypatch.delenv("VLLM_FLASHINFER_DSV41_SM120_INDEXER_LOGITS")
        fi.sm120_indexer_logits_enabled.cache_clear()


def test_fp8_cache_writer_matches_triton_writer(monkeypatch):
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    if not fi.has_flashinfer_dsv41_fp8_cache_writer():
        pytest.skip("FlashInfer has no DSv4.1 FP8 record writer here")
    torch.manual_seed(0)
    tokens, ratio, pages, page_tokens = 24, 2, 8, 32
    latent = (torch.randn(tokens, 512, device="cuda") * 3).to(torch.bfloat16)
    positions = torch.randint(0, 4096, (tokens,), dtype=torch.int64, device="cuda")
    positions[:8] = (
        torch.arange(8, dtype=torch.int64, device="cuda") * ratio + ratio - 1
    )
    cos_sin = torch.randn(4096, 64, device="cuda")
    slots = torch.randperm(pages * page_tokens, device="cuda")[:tokens]
    slots = slots.to(torch.int32).contiguous()
    slots[::7] = -1

    def write(enabled: bool) -> torch.Tensor:
        monkeypatch.setenv(
            "VLLM_FLASHINFER_DSV41_SM120_FP8_CACHE_WRITER",
            "1" if enabled else "0",
        )
        fi.dsv41_fp8_cache_writer_enabled.cache_clear()
        cache = torch.zeros(pages, page_tokens, 528, dtype=torch.uint8, device="cuda")
        rope_quant_insert(
            latent.clone(),
            positions.clone(),
            cos_sin,
            cache,
            slots.clone(),
            ratio,
            footer_scale=False,
        )
        torch.cuda.synchronize()
        return cache

    try:
        reference = write(False)
        written = write(True)
    finally:
        monkeypatch.delenv("VLLM_FLASHINFER_DSV41_SM120_FP8_CACHE_WRITER")
        fi.dsv41_fp8_cache_writer_enabled.cache_clear()

    # The shared rotation kernel rounds to BF16 before FlashInfer quantizes,
    # so the RoPE dims can be one e4m3 ULP off; NoPE values and every group
    # scale must match exactly.
    assert torch.equal(reference[..., :448], written[..., :448])
    assert torch.equal(reference[..., 512:], written[..., 512:])
    rope = reference[..., 448:512].to(torch.int16) - written[..., 448:512]
    assert int(rope.abs().max()) <= 1
