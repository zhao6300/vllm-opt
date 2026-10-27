# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    as_fp32_logits,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
    native_next_n_supported,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv


def encode_indexer_scales(sf: torch.Tensor) -> torch.Tensor:
    """Per-token power-of-two fp32 scales as the indexer K cache stores them:
    four equal packed UE8M0 bytes under MX scales, else the fp32 itself."""
    if not mqa_logits_use_mx_scales():
        return sf
    ue8m0 = (sf.view(torch.int32) >> 23).to(torch.uint8)
    return ue8m0[..., None].repeat_interleave(4, dim=-1).view(torch.int32)[..., 0]


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = encode_indexer_scales(
        sf.view(num_blocks, block_size)
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def _packed_requests_test_data(seq_len: int, seq_len_kv: int):
    # Two causal prefill requests packed into one KV buffer, so the second
    # request's rows start at a nonzero K position.
    half, half_kv = seq_len // 2, seq_len_kv // 2
    rows = torch.arange(seq_len, dtype=torch.int, device="cuda")
    ks = torch.where(rows < half, 0, half_kv).to(torch.int)
    ke = ks + half_kv - half + 1 + rows % half
    return ks, ke


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("layout", ["cp", "no_cp", "packed"])
def test_deepgemm_fp8_mqa_logits(layout: str):
    """FP8 indexer prefill logits, compressed to each row's valid K span."""
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128
    seq_len, seq_len_kv = 512, 1024
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)

    if layout == "cp":
        ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)
    elif layout == "no_cp":
        ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
        ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
            seq_len_kv - seq_len
        )
    else:
        ks, ke = _packed_requests_test_data(seq_len, seq_len_kv)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), True)
    q_scale, kernel_weights = mqa_logits_mx_inputs(q_fp8, None, weights)
    max_seqlen_k = int((ke - ks).max())
    logits = fp8_fp4_mqa_logits(
        (q_fp8, q_scale),
        (kv_fp8, encode_indexer_scales(kv_sf)),
        kernel_weights,
        ks,
        ke,
        max_seqlen_k,
    )
    assert logits.shape == (seq_len, max_seqlen_k)
    logits = as_fp32_logits(logits, ke - ks)

    ref_logits = _ref_fp8_mqa_logits(
        q=q,
        kv=kv,
        weights=kernel_weights.float(),
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    # Compressed row i holds K positions [ks[i], ke[i]) from column zero.
    cols = ks[:, None] + torch.arange(max_seqlen_k, device="cuda")[None, :]
    valid = cols < ke[:, None]
    ref_logits = ref_logits.gather(1, cols.clamp(max=seq_len_kv - 1).long())
    ref_logits = ref_logits.masked_fill(~valid, 0)
    logits = logits.masked_fill(~valid, 0)
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"


def _ref_fp8_fp4_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
# next_n = 1 + num_speculative_tokens, so next_n=4 is MTP=3 (issue #35878).
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (2, 4)])
def test_deepgemm_fp8_fp4_paged_mqa_logits(batch_size: int, next_n: int):
    if not native_next_n_supported(next_n):
        pytest.skip(f"next_n={next_n} has no native kernel on this architecture")

    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for heads, index_dim in [(32, 128)]:
        for avg_kv in (2048,):
            num_blocks, blocksize = max_model_len * 2, 64

            q = torch.randn(
                (batch_size, next_n, heads, index_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )
            kv_cache = torch.randn(
                (num_blocks, blocksize, 1, index_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )
            weights = torch.randn(
                (batch_size * next_n, heads),
                device="cuda",
                dtype=torch.float32,
            )

            context_lens = (
                torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                .cuda()
                .to(torch.int32)
            )
            max_block_len = (
                (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
            )
            block_tables = torch.zeros(
                (batch_size, max_block_len),
                device="cuda",
                dtype=torch.int32,
            )

            counter = 0
            block_idx_pool = list(range(num_blocks))
            random.shuffle(block_idx_pool)
            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                for j in range((ctx_len + blocksize - 1) // blocksize):
                    block_tables[i][j] = block_idx_pool[counter]
                    counter += 1

            q_fp8 = q.to(torch.float8_e4m3fn)
            kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

            # deep_gemm paged MQA logits requires 2D context_lens of
            # shape (B, next_n) (csrc/apis/attention.hpp:332-335);
            # see indexer.py:607-608. For each batch/next_n token, the
            # effective context length is context_lens[b] - next_n + j + 1.
            next_n_arange = torch.arange(next_n, device="cuda", dtype=torch.int32)
            context_lens_2d = (
                context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange
            ).contiguous()
            q_scale, kernel_weights = mqa_logits_mx_inputs(q_fp8, None, weights)
            kernel_q = (q_fp8, q_scale)
            kernel_context_lens, kernel_block_tables = context_lens_2d, block_tables
            indices = None
            if mqa_logits_use_mx_scales():
                # SM100 only takes varlen rows: one Q token each, tied to its
                # request by `indices`.
                assert q_scale is not None
                kernel_q = (
                    q_fp8.flatten(0, 1)[:, None],
                    q_scale.flatten(0, 1)[:, None],
                )
                kernel_context_lens = context_lens_2d.view(-1, 1)
                kernel_block_tables = block_tables.repeat_interleave(next_n, dim=0)
                indices = torch.arange(batch_size, device="cuda", dtype=torch.int32)
                indices = indices.repeat_interleave(next_n)
            schedule_metadata = get_paged_mqa_logits_metadata(
                kernel_context_lens, blocksize, get_num_sms(), indices=indices
            )
            logits = fp8_fp4_paged_mqa_logits(
                kernel_q,
                kv_cache_fp8,
                kernel_weights,
                kernel_context_lens,
                kernel_block_tables,
                schedule_metadata,
                max_model_len,
                indices=indices,
            )
            if indices is not None:
                logits = as_fp32_logits(logits, kernel_context_lens.view(-1))

            ref_logits = _ref_fp8_fp4_paged_mqa_logits(
                q,
                kv_cache,
                kernel_weights.float(),
                context_lens,
                block_tables,
                max_model_len,
            )

            positions = (
                torch.arange(max_model_len, device="cuda")
                .unsqueeze(0)
                .expand(batch_size * next_n, -1)
            )
            row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
            next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
            mask = positions <= (
                context_lens[row_indices] - next_n + next_n_offset
            ).unsqueeze(1)

            logits = logits.float().masked_fill(~mask, 0)
            ref_logits = ref_logits.masked_fill(~mask, 0)
            diff = calc_diff(logits, ref_logits)
            assert diff < 1e-3, f"{diff=}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
def test_deepgemm_paged_mqa_packed_manager_block_stride():
    torch.manual_seed(0)
    num_blocks, block_size, packed_rows = 4, 256, 320
    num_heads, head_dim, row_bytes = 64, 128, 132
    page_size, pages_per_block = 64, block_size // 64
    dense_stride, packed_stride = pages_per_block, packed_rows // page_size
    compact_pages = kv_cache_cast_to_fp8(
        torch.randn(
            num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
        ).view(num_blocks * pages_per_block, page_size, 1, head_dim)
    )
    packed_backing = torch.zeros(
        num_blocks,
        packed_rows,
        row_bytes,
        device="cuda",
        dtype=torch.uint8,
    )
    packed_manager = packed_backing[:, :block_size]
    packed_manager.copy_(compact_pages.squeeze(2).view(num_blocks, block_size, -1))
    packed_pages = packed_backing.view(-1, page_size, row_bytes).unsqueeze(2)
    offsets = torch.arange(pages_per_block, device="cuda", dtype=torch.int32)
    tables = [
        (stride + offsets).unsqueeze(0) for stride in (dense_stride, packed_stride)
    ]

    q = torch.randn(1, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    q_scale, weights = mqa_logits_mx_inputs(
        q, None, torch.randn(1, num_heads, device="cuda", dtype=torch.float32)
    )
    context_lens = torch.full((1, 1), block_size, device="cuda", dtype=torch.int32)
    indices = (
        torch.zeros(1, device="cuda", dtype=torch.int32)
        if mqa_logits_use_mx_scales()
        else None
    )
    schedule = get_paged_mqa_logits_metadata(
        context_lens, page_size, get_num_sms(), indices=indices
    )
    logits = [
        fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            cache,
            weights,
            context_lens,
            table,
            schedule,
            block_size,
            indices=indices,
        )
        for cache, table in zip((compact_pages, packed_pages), tables)
    ]
    torch.testing.assert_close(*logits, rtol=0, atol=0)


# Nearest-value E2M1 codes, as `cvt.rn.satfinite.e2m1x2.f32` emits them.
_MXFP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between consecutive codes: a magnitude falls in the bucket whose
# representative is nearest, ties rounding up.
_MXFP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the last dim of `x` to MXFP4, as the indexer kernels store it.

    Two E2M1 values share a byte (low nibble first) and every 32 values share
    one UE8M0 scale of ``2**(e - 127)`` where ``e`` is the ceil-log2 block
    scale of the reference kernels.

    Returns:
        Packed values ``[..., dim // 2]`` uint8, scale bytes ``[..., dim // 32]``
        uint8, and the dequantized values as float32 ``[..., dim]``.
    """
    assert x.shape[-1] % 32 == 0
    blocks = x.float().view(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(exponent)
    scaled = (blocks / scale).clamp(-6.0, 6.0)
    code = torch.bucketize(
        scaled.abs().contiguous(),
        torch.tensor(_MXFP4_MIDPOINTS, device=x.device, dtype=torch.float32),
    ).to(torch.uint8)
    code |= (scaled < 0).to(torch.uint8) << 3
    values = torch.tensor(_MXFP4_E2M1, device=x.device, dtype=torch.float32)
    magnitude = values[(code & 7).long()]
    dequant = magnitude * torch.where(code < 8, 1.0, -1.0) * scale
    packed = (code[..., 1::2] << 4) | code[..., 0::2]
    scales = (exponent + 127).to(torch.uint8).squeeze(-1)
    return packed.flatten(-2), scales, dequant.flatten(-2)


def mxfp4_indexer_kv_cache(
    kv: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an MXFP4 indexer K cache holding `kv` (`[tokens, head_dim]` bf16).

    Each page stores its ``block_size`` packed value rows first and its
    ``block_size`` four-byte UE8M0 scale rows after them, the layout both the
    indexer's K-store kernel and DeepGEMM use.

    Returns:
        The uint8 cache ``[num_blocks, block_size, head_dim // 2 + 4]`` and the
        dequantized keys ``[tokens, head_dim]`` float32.
    """
    num_tokens, head_dim = kv.shape
    num_blocks = num_tokens // block_size
    row_bytes = head_dim // 2
    packed, scales, dequant = quantize_mxfp4(kv)
    cache = torch.zeros(
        num_blocks, block_size, row_bytes + 4, device=kv.device, dtype=torch.uint8
    )
    flat = cache.view(-1)
    page_bytes = block_size * (row_bytes + 4)
    for block in range(num_blocks):
        base = block * page_bytes
        rows = slice(block * block_size, (block + 1) * block_size)
        flat[base : base + block_size * row_bytes] = packed[rows].reshape(-1)
        scale_region = slice(base + block_size * row_bytes, base + page_bytes)
        flat[scale_region] = scales[rows].reshape(-1)
    return cache, dequant


def _mxfp4_logits_supported() -> bool:
    """Whether DeepGEMM ships the MXFP4 MQA-logits kernels for this GPU."""
    return current_platform.is_cuda() and (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
        or current_platform.is_device_capability_family(121)
    )


@pytest.mark.skipif(not _mxfp4_logits_supported(), reason="No MXFP4 logits kernel")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_deepgemm_fp4_paged_mqa_logits(batch_size: int):
    """MXFP4 indexer K cache through the paged decode logits kernel.

    The cache is the 68 B/token page the indexer allocates, so a layout
    mismatch between the K-store kernel and the logits kernel shows up here as
    wrong scores rather than as a crash.
    """
    torch.manual_seed(0)
    num_heads, head_dim, block_size = 32, 128, 64
    context_len = 128
    blocks_per_request = cdiv(context_len, block_size)
    num_blocks = batch_size * blocks_per_request

    q = torch.randn(
        batch_size, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(
        num_blocks * block_size, head_dim, device="cuda", dtype=torch.bfloat16
    )
    cache, kv_dequant = mxfp4_indexer_kv_cache(kv, block_size)
    q_packed, q_scales, q_dequant = quantize_mxfp4(q)

    weights = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.float32)
    if mqa_logits_use_mx_scales():
        # SM100's kernel takes BF16 weights; sm_12x asserts FP32.
        weights = weights.to(torch.bfloat16)
    context_lens = torch.full(
        (batch_size, 1), context_len, device="cuda", dtype=torch.int32
    )
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, blocks_per_request
    )
    indices = (
        torch.arange(batch_size, device="cuda", dtype=torch.int32)
        if mqa_logits_use_mx_scales()
        else None
    )
    schedule = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms(), indices=indices
    )
    logits = fp8_fp4_paged_mqa_logits(
        (q_packed.view(torch.int8), q_scales.view(torch.int32).squeeze(-1)),
        kv_cache_as_quant_view(cache, head_dim, True),
        weights,
        context_lens,
        block_table,
        schedule,
        context_len,
        indices=indices,
    )
    assert logits.shape == (batch_size, context_len)

    keys = kv_dequant.view(batch_size, context_len, head_dim)
    q_heads = q_dequant.view(batch_size, num_heads, head_dim)
    scores = torch.einsum("bhd,bkd->bhk", q_heads, keys)
    ref_logits = (scores.relu() * weights.float().unsqueeze(-1)).sum(dim=1)
    assert calc_diff(logits.float(), ref_logits) < 1e-6


@pytest.mark.skipif(not _mxfp4_logits_supported(), reason="No MXFP4 logits kernel")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
def test_deepgemm_fp4_mqa_logits():
    """MXFP4 prefill logits over each row's valid K span (non-paged kernel)."""
    torch.manual_seed(0)
    num_heads, head_dim = 32, 128
    seq_len, seq_len_kv = 32, 128
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    if mqa_logits_use_mx_scales():
        weights = weights.to(torch.bfloat16)

    q_packed, q_scales, q_dequant = quantize_mxfp4(q)
    kv_packed, kv_scales, kv_dequant = quantize_mxfp4(kv)
    ks = torch.arange(seq_len, device="cuda", dtype=torch.int32)
    ke = torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32)
    logits = fp8_fp4_mqa_logits(
        (q_packed.view(torch.int8), q_scales.view(torch.int32).squeeze(-1)),
        (
            kv_packed.view(torch.int8),
            kv_scales.view(torch.int32).squeeze(-1),
        ),
        weights,
        ks,
        ke,
        seq_len_kv - seq_len + 1,
    )
    logits = as_fp32_logits(logits, ke - ks)

    scores = torch.einsum("thd,kd->thk", q_dequant, kv_dequant)
    ref = (scores.relu() * weights.float()[:, :, None]).sum(dim=1)
    cols = ks[:, None] + torch.arange(logits.shape[1], device="cuda")[None, :]
    valid = cols < ke[:, None]
    ref = ref.gather(1, cols.clamp(max=seq_len_kv - 1).long())
    logits = logits.masked_fill(~valid, 0)
    ref = ref.masked_fill(~valid, 0)
    assert calc_diff(logits, ref) < 1e-6
