# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend."""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    get_flashinfer_dsv4_workspace,
    packed_block_span,
    pad_to_supported_q_heads,
    required_sm120_sparse_topk,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    build_flashinfer_mixed_sparse_indices,
    compute_global_topk_indices_and_lens,
    remap_flashinfer_sparse_indices,
)
from vllm.models.deepseek_v41.common.ops import (
    paged_cache_block_span as _packed_block_span,
)
from vllm.models.deepseek_v41.common.ops import (
    paged_cache_page_rows as _paged_cache_page_rows,
)
from vllm.models.deepseek_v41.nvidia.ops.o_proj import (
    dsv41_o_proj,
    register_dsv41_o_proj_warmup,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
    max_decode_query_len,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


def _debug_phase_dir(root: str) -> str:
    try:
        with open(os.path.join(root, "phase")) as f:
            phase = f.read().strip() or "default"
    except OSError:
        phase = "default"
    path = os.path.join(root, phase)
    os.makedirs(path, exist_ok=True)
    return path


# FlashInfer resolves its decode plan -- the only plan that carves scratch out
# of the caller-owned workspace -- for at most this many rows; wider batches
# resolve to a prefill plan whose scratch is allocated by the caller.
_FLASHINFER_DSV4_DECODE_MAX_TOKENS = 64
# Decode-plan scratch is ``tokens * heads * chunk_capacity`` slots of bf16
# partials (``value_dim`` wide) plus one fp32 LSE entry per slot. Both are
# fixed for the DeepSeek V4 family.
_FLASHINFER_DSV4_VALUE_DIM = 512
_FLASHINFER_DSV4_DECODE_CHUNK_WIDTH = 64
# The compiled split-K resolver tiles at 32 candidates; sizing the caller
# scratch with this width bounds every plan the wrapper can resolve.
_FLASHINFER_DSV4_SPLITK_TILE = 32
_sparse_mla_sm120_wrappers: dict[tuple[torch.device, bool, str], object] = {}


@dataclass
class _Sm120CallerBuffers:
    """Caller-owned scratch the wrapper must not free.

    `prepare` drops the previous plan, and any tensors it owns, whenever a
    calibration update changes the resolved plan. A captured graph keeps the
    addresses it baked, so the caller has to hold the scratch for the life of
    the process. All layers share one bundle and use it sequentially on one
    stream: the plan runs on this stream, so its kernels consume the scratch
    before the next step's launch. That holds under pipeline parallelism too --
    PP microbatches are staggered across steps (a request decodes every M
    steps), so a worker never runs two forwards at once, and the PP handler's
    side stream only carries sampled tokens. Two forwards on different streams
    (dual-batch overlap) would need one bundle per stream; `in_use` fails
    closed on the Python-level re-entry but cannot see graph replays.
    The plan object itself is a host-side descriptor (the C++
    `ExecutionPlan` is plain data), so replacing it frees no device memory.
    """

    lse: torch.Tensor
    mid: torch.Tensor | None
    mlse: torch.Tensor | None
    stream: torch.cuda.Stream | None = None
    in_use: bool = False
    # Set by the first attention call; the buffers are frozen from then on.
    used: bool = False
    # (topk, extra_topk, kv_scale_format, extra_fp4) of every layer sharing
    # the bundle: the target and the DSpark drafter resolve different widths.
    bounds: set[tuple[int, int, str, bool]] = field(default_factory=set)


_sparse_mla_sm120_caller_buffers: dict[
    tuple[torch.device, int, bool], _Sm120CallerBuffers
] = {}


def _sm120_scratch_slots(
    *,
    padded_heads: int,
    max_tokens: int,
    topk: int,
    extra_topk: int,
    sm_count: int,
) -> tuple[int, int]:
    """Upper bound on decode-plan scratch slots and its rounded head count.

    Decode plans are the only ones that carve scratch. The FP8 split-K plan
    sizes `tokens * decode_scratch_heads(heads) * chunk_capacity` with a
    64-candidate chunk; the compiled resolver's split-K variant tiles at 32 and
    is selected only while `2 * tokens * ceil(heads / 32) < SMs`. Its 8-head
    CTA is selected at every token count, with the split count capped by the
    wave budget. Taking the maximum covers both.
    """
    # Match ``decode_scratch_heads``: 8 heads stay at 8, everything else
    # rounds up to a multiple of 16 (attention_plan.h:99).
    scratch_heads = 8 if padded_heads == 8 else cdiv(padded_heads, 16) * 16
    profile_slots = (
        _FLASHINFER_DSV4_DECODE_MAX_TOKENS
        * scratch_heads
        * (
            cdiv(topk, _FLASHINFER_DSV4_DECODE_CHUNK_WIDTH)
            + cdiv(extra_topk, _FLASHINFER_DSV4_DECODE_CHUNK_WIDTH)
        )
    )
    if padded_heads <= 8:
        compiled_slots = padded_heads * max(max_tokens, 2 * sm_count)
    else:
        head_blocks = cdiv(padded_heads, _FLASHINFER_DSV4_SPLITK_TILE)
        splitk_rows = max(1, (sm_count - 1) // (2 * head_blocks))
        compiled_slots = splitk_rows * padded_heads * 8
    return max(profile_slots, compiled_slots), scratch_heads


def _validate_sm120_caller_buffers(
    buffers: _Sm120CallerBuffers,
    *,
    device: torch.device,
    padded_heads: int,
    topk: int,
    extra_topk: int,
    kv_scale_format: str,
    extra_fp4: bool,
) -> None:
    """Resolve every decode plan at startup and check the scratch fits.

    Scratch is not largest at the widest batch: fewer tokens can buy more
    split-K partitions, and a calibrated profile may select a different
    variant per token bucket. Resolve every decode token count under the
    stored profiles, the no-profile fallback, and a forced split-K route,
    then compare the worst workspace with the caller buffers. This turns a
    sizing mistake into a startup failure instead of a mid-serving
    `prefix_scratch` error.
    """
    from flashinfer.mla._sparse_mla_sm120 import _execution as fi_exec
    from flashinfer.mla._sparse_mla_sm120 import _policy as fi_policy
    from flashinfer.mla._sparse_mla_sm120._prepared import prepare as prepare_plan

    model = fi_exec.resolve_model_type(_FLASHINFER_DSV4_VALUE_DIM, kv_scale_format)
    facts = fi_exec.query("format_info", model)
    row_bytes = int(facts["bytes_per_token"])
    fp4_row_bytes = int(facts["fp4_bytes_per_token"])
    page_size = 64

    def page_stride(rows: int, row: int) -> int:
        return (rows * row + 15) // 16 * 16

    def resolve(tokens: int, pass_extra_topk: int, pass_extra_fp4: bool):
        pass_row = fp4_row_bytes if pass_extra_fp4 else row_bytes
        metadata = fi_exec.AttentionMetadata(
            model=model,
            tokens=tokens,
            heads=padded_heads,
            topk=topk,
            extra_topk=pass_extra_topk,
            page_size=page_size,
            extra_page_size=page_size if pass_extra_topk else 0,
            page_stride_bytes=page_stride(page_size, row_bytes),
            extra_page_stride_bytes=(
                page_stride(page_size, pass_row) if pass_extra_topk else 0
            ),
            row_stride_bytes=row_bytes,
            indices_stride=topk,
            extra_indices_stride=pass_extra_topk,
            lse_stride=padded_heads,
            has_lengths=True,
            has_extra_lengths=pass_extra_topk > 0,
            has_sink=False,
            extra_fp4=pass_extra_fp4,
            variant=0,
            extra_row_stride_bytes=pass_row if pass_extra_topk else 0,
        )
        return prepare_plan(
            metadata,
            device,
            "fp8",
            False,
            0,
            owned=False,
            caller_mid=True,
            caller_mlse=True,
            caller_lse=True,
        )

    forced_splitk = fi_policy.PlannedCall(fi_policy.KernelVariant.DECODE_SPLITK, -1)

    def no_profile(*_args, **_kwargs):
        return None

    def splitk_route(*_args, **_kwargs):
        return forced_splitk

    # The buffers are shared by compressed and SWA-only layers; check both the
    # FP4-extra layout and the no-extra one (which can take the compiled split-K
    # resolver).
    passes = [(extra_topk, extra_fp4)]
    if extra_topk:
        passes.append((0, False))

    stored_selection = fi_policy.profile_selection
    routes = (stored_selection, no_profile, splitk_route)
    worst = (0, 0, (0, 0), 0, 0)
    try:
        for pass_extra_topk, pass_extra_fp4 in passes:
            for tokens in range(1, _FLASHINFER_DSV4_DECODE_MAX_TOKENS + 1):
                for route in routes:
                    fi_policy.profile_selection = route
                    try:
                        plan = resolve(tokens, pass_extra_topk, pass_extra_fp4)
                    except ValueError:
                        # This route cannot serve this token count.
                        continue
                    (
                        (_, _, mid_bytes, _),
                        (_, _, mlse_bytes, _),
                        (
                            lse_shape,
                            _,
                            _,
                            _,
                        ),
                    ) = plan.workspace
                    worst = (
                        max(worst[0], mid_bytes),
                        max(worst[1], mlse_bytes),
                        (
                            max(worst[2][0], lse_shape[0]),
                            max(worst[2][1], lse_shape[1]),
                        ),
                        tokens,
                        pass_extra_topk,
                    )
    finally:
        fi_policy.profile_selection = stored_selection

    mid_bytes, mlse_bytes, lse_shape, tokens, worst_extra_topk = worst
    mid_fits = buffers.mid is not None and (
        buffers.mid.numel() * buffers.mid.element_size() >= mid_bytes
    )
    mlse_fits = buffers.mlse is not None and (
        buffers.mlse.numel() * buffers.mlse.element_size() >= mlse_bytes
    )
    if not (
        buffers.lse.shape[0] >= lse_shape[0]
        and buffers.lse.shape[1] >= lse_shape[1]
        and mid_fits
        and mlse_fits
    ):
        raise RuntimeError(
            "SM120 sparse-MLA caller scratch is too small for the widest "
            f"decode plan (tokens={tokens}, extra_topk={worst_extra_topk}, "
            f"mid={mid_bytes} B, mlse={mlse_bytes} B, lse={lse_shape}); "
            "allocated lse="
            f"{tuple(buffers.lse.shape)}, mid="
            f"{buffers.mid.numel() if buffers.mid is not None else 0}, mlse="
            f"{buffers.mlse.numel() if buffers.mlse is not None else 0}"
        )


def _sm120_caller_buffers_for_layer(
    *,
    device: torch.device,
    padded_heads: int,
    max_tokens: int,
    topk: int,
    extra_topk: int,
    kv_scale_format: str,
    extra_fp4: bool,
    scratch: bool,
) -> _Sm120CallerBuffers:
    """The process-wide caller buffers of every layer with `padded_heads` heads.

    Layers built with different bounds share one bundle: the DSpark drafter's
    non-causal SWA index row is wider than the target's. The bundle grows to
    the largest bound while the model is being built, which is before any
    attention call or graph capture, and is frozen once a call has used it.
    """
    key = (device, padded_heads, scratch)
    buffers = _sparse_mla_sm120_caller_buffers.get(key)
    bound = (topk, extra_topk, kv_scale_format, extra_fp4)
    if buffers is None:
        buffers = _Sm120CallerBuffers(
            torch.empty((0, padded_heads), dtype=torch.float32, device=device),
            None,
            None,
        )
        _sparse_mla_sm120_caller_buffers[key] = buffers
    elif bound in buffers.bounds and buffers.lse.shape[0] >= max_tokens:
        return buffers

    lse_rows = max(buffers.lse.shape[0], max_tokens)
    slots = 0
    if scratch:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        slots = max(
            _sm120_scratch_slots(
                padded_heads=padded_heads,
                max_tokens=lse_rows,
                topk=b_topk,
                extra_topk=b_extra_topk,
                sm_count=sm_count,
            )[0]
            for b_topk, b_extra_topk, _, _ in buffers.bounds | {bound}
        )
    grow = buffers.lse.shape[0] < lse_rows or (
        scratch and (buffers.mlse is None or buffers.mlse.numel() < slots)
    )
    if grow:
        if buffers.used or (
            device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "the SM120 caller scratch must grow after it was used; build "
                "every layer that shares it before the first attention call"
            )
        # Layers hold the bundle, not its tensors, so replacing them here
        # reaches every layer built earlier.
        buffers.lse = torch.empty(
            (lse_rows, padded_heads), dtype=torch.float32, device=device
        )
        if scratch:
            buffers.mid = torch.empty(
                slots * _FLASHINFER_DSV4_VALUE_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            buffers.mlse = torch.empty(slots, dtype=torch.float32, device=device)
    buffers.bounds.add(bound)
    if scratch:
        for b_topk, b_extra_topk, b_format, b_extra_fp4 in sorted(buffers.bounds):
            _validate_sm120_caller_buffers(
                buffers,
                device=device,
                padded_heads=padded_heads,
                topk=b_topk,
                extra_topk=b_extra_topk,
                kv_scale_format=b_format,
                extra_fp4=b_extra_fp4,
            )
    return buffers


# Layer-0 attention I/O for the precise CPU reference
# (`VLLM_DEBUG_ATTN_DUMP=1`). Records are normalized to 512 value bytes plus
# the 16 UE8M0 scale bytes, whatever the cache layout.
_debug_attn_calls: dict[tuple[str, int], int] = {}


def _debug_attn_dump(
    tag: str,
    layer_prefix: str,
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor | None,
    scale: float,
    output: torch.Tensor,
    *,
    footer: bool,
    block: int,
) -> None:
    root = os.environ.get("VLLM_DEBUG_TF_DIR")
    if not root or not os.environ.get("VLLM_DEBUG_ATTN_DUMP"):
        return
    if ".layers.0." not in layer_prefix:
        return
    # Device-to-host copies and file writes are invalid while capturing.
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        with open(os.path.join(root, "phase")) as f:
            phase = f.read().strip() or "off"
    except OSError:
        phase = "off"
    if phase == "off":
        return
    try:
        torch.cuda.current_stream().synchronize()
    except RuntimeError as error:
        raise RuntimeError(
            f"{layer_prefix} {tag}: the sparse MLA kernel faulted before the "
            "attention dump read its inputs"
        ) from error
    out_dir = os.path.join(root, phase)
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.cuda.current_device()
    call = _debug_attn_calls.get((tag, dev), 0)
    _debug_attn_calls[(tag, dev)] = call + 1

    # The cache is a strided view into the shared block allocation (padding
    # rows, other layers' pages), so a record is addressed in its storage:
    # flattening the view would compact it and shift every later page.
    storage = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
        cache.untyped_storage()
    )
    base = cache.storage_offset() * cache.element_size()
    page_bytes = cache.stride(0) * cache.element_size()
    page_rows = _paged_cache_page_rows(cache)
    idx = indices.detach().reshape(-1).cpu()
    uniq = torch.unique(idx[idx >= 0]).long()
    if footer:
        # Footer pages: `page_rows` 512-byte value rows, then 16-byte scales;
        # the kernel's indices are paged slot ids.
        pages, entries = uniq // page_rows, uniq % page_rows
        starts = base + pages * page_bytes
        value_off = starts + entries * 512
        scale_off = starts + page_rows * 512 + entries * 16
    else:
        # Inline records: the kernel's indices are flat cache rows.
        row_bytes = page_bytes // _packed_block_span(cache)
        value_off = base + uniq * row_bytes
        scale_off = value_off + 512
    in_range = (value_off >= 0) & (scale_off + 16 <= storage.numel())
    recs = torch.zeros((uniq.numel(), 528), dtype=torch.uint8)
    if in_range.any():
        d512, d16 = torch.arange(512), torch.arange(16)
        v_idx = (value_off[in_range][:, None] + d512).to(storage.device)
        s_idx = (scale_off[in_range][:, None] + d16).to(storage.device)
        recs[in_range] = torch.cat([storage[v_idx], storage[s_idx]], 1).cpu()
    torch.save(
        {
            "tag": tag,
            "layer": layer_prefix,
            "phase": phase,
            "q": q.detach().cpu(),
            "indices": indices.detach().cpu(),
            "lens": lens.detach().cpu(),
            "sink": None if sink is None else sink.detach().cpu(),
            "scale": float(scale),
            "output": output.detach().cpu(),
            "recs": recs.cpu(),
            "uniq": uniq,
            "in_range": in_range,
            "footer": footer,
            "block": int(page_rows),
            "cache_shape": tuple(cache.shape),
            "cache_stride": tuple(cache.stride()),
        },
        os.path.join(
            out_dir, f"attn0__{tag}__dev{dev}__rows{q.shape[0]}__call{call}.pt"
        ),
    )


# Row-padded copies of a short wrapper call's operands
# (`VLLM_FLASHINFER_DSV41_SM120_MIN_TOKENS`), one per device, role, row tail
# and dtype. Layers share them: the copy-in, the kernel and the copy-out run in
# order on one stream. Kept for the process so captured graphs keep addresses.
_sm120_min_tokens_scratch: dict[tuple, torch.Tensor] = {}


def _sm120_min_tokens_buffer(
    role: str, like: torch.Tensor, rows: int, fill: int | float | None
) -> torch.Tensor:
    """A persistent contiguous `[rows, *like.shape[1:]]` buffer for `role`."""
    key = (like.device, role, tuple(like.shape[1:]), like.dtype, rows)
    buf = _sm120_min_tokens_scratch.get(key)
    if buf is None:
        if like.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"the SM120 row-padding buffer for {role} {tuple(like.shape)} "
                "must be created by the eager warmup before graph capture"
            )
        buf = torch.empty((rows, *like.shape[1:]), dtype=like.dtype, device=like.device)
        if fill is not None:
            buf.fill_(fill)
        _sm120_min_tokens_scratch[key] = buf
    return buf


def _pad_sm120_rows(
    role: str,
    tensor: torch.Tensor,
    rows: int,
    pad: int | float | None,
) -> torch.Tensor:
    """`tensor`'s rows first, then `rows - len(tensor)` rows set to `pad`
    (left as they are when `pad` is None)."""
    buf = _sm120_min_tokens_buffer(role, tensor, rows, 0 if pad is None else pad)
    n = tensor.shape[0]
    buf[:n].copy_(tensor)
    if pad is not None:
        buf[n:].fill_(pad)
    return buf


def _sparse_mla_sm120_paged_attention(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor | None,
    extra_kv_cache: torch.Tensor | None,
    extra_sparse_indices: torch.Tensor | None,
    extra_sparse_topk_lens: torch.Tensor | None,
    extra_kv_fp4: bool,
    prefill_impl: str | None,
    kv_scale_format: str,
    caller: _Sm120CallerBuffers | None = None,
    layer_prefix: str = "",
    phase_tag: str = "",
) -> None:
    """Run FlashInfer's SM120 sparse MLA wrapper in-place."""
    from flashinfer.mla._sparse_mla_sm120 import SparseMLASm120Wrapper

    has_extra_kv_cache = extra_kv_cache is not None
    extra_kv_flag = extra_kv_fp4 and has_extra_kv_cache
    # `d_v` is baked into the wrapper at construction, so layers with different
    # value widths need their own; the per-call plans carry heads/topk.
    device_key = (query.device, query.shape[-1], extra_kv_flag, kv_scale_format)
    wrapper = _sparse_mla_sm120_wrappers.get(device_key)
    if wrapper is None:
        wrapper = SparseMLASm120Wrapper(
            d_v=query.shape[-1],
            kv_scale_format=kv_scale_format,
            kv_cache_format="fp8",
            extra_kv_fp4=extra_kv_flag,
            compute_precision="fp8",
            device=query.device,
        )
        _sparse_mla_sm120_wrappers[device_key] = wrapper
    num_tokens = query.shape[0]
    min_tokens = envs.VLLM_FLASHINFER_DSV41_SM120_MIN_TOKENS
    padded_output = None
    if 0 < num_tokens < min_tokens:
        # Padded rows have zero-length windows, so the kernel reads no KV for
        # them; only the first `num_tokens` output rows are copied back.
        query = _pad_sm120_rows("q", query, min_tokens, None)
        sparse_indices = _pad_sm120_rows("indices", sparse_indices, min_tokens, -1)
        topk_length = _pad_sm120_rows("lens", topk_length, min_tokens, 0)
        if extra_sparse_indices is not None:
            extra_sparse_indices = _pad_sm120_rows(
                "extra_indices", extra_sparse_indices, min_tokens, -1
            )
        if extra_sparse_topk_lens is not None:
            extra_sparse_topk_lens = _pad_sm120_rows(
                "extra_lens", extra_sparse_topk_lens, min_tokens, 0
            )
        padded_output = output
        output = _sm120_min_tokens_buffer("output", output, min_tokens, 0)
        if caller is not None and caller.lse.shape[0] < min_tokens:
            raise RuntimeError(
                f"VLLM_FLASHINFER_DSV41_SM120_MIN_TOKENS={min_tokens} exceeds the "
                f"{caller.lse.shape[0]} LSE rows of the SM120 caller scratch"
            )
    if caller is not None:
        if caller.in_use:
            raise RuntimeError(
                "SM120 sparse-MLA caller scratch is re-entered; every layer must "
                "call it sequentially."
            )
        caller.in_use = True
        caller.used = True
    try:
        if (
            caller is not None
            and caller.mid is not None
            and query.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        ):
            stream = torch.cuda.current_stream(query.device)
            if caller.stream is None:
                caller.stream = stream
            elif caller.stream.cuda_stream != stream.cuda_stream:
                # A strict single-stream assertion is wrong: vLLM's graph
                # capture runs the eager warmup pass on the capture stream
                # (cudagraph_utils.py:477-505). Drain the old stream before the
                # shared scratch is touched from the new one.
                caller.stream.synchronize()
                caller.stream = stream
        wrapper.run(
            q=query,
            kv_cache=swa_kv_cache,
            indices=sparse_indices,
            output=output,
            sm_scale=sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_sparse_indices,
            extra_topk_length=extra_sparse_topk_lens,
            prefill_impl=prefill_impl,
            out_lse=None if caller is None else caller.lse,
            mid_out=None if caller is None else caller.mid,
            mid_lse=None if caller is None else caller.mlse,
        )
    finally:
        if caller is not None:
            caller.in_use = False
    if padded_output is not None:
        padded_output.copy_(output[:num_tokens])
        output = padded_output
    if phase_tag:
        _debug_attn_dump(
            phase_tag,
            layer_prefix,
            query[:num_tokens],
            swa_kv_cache,
            sparse_indices[:num_tokens],
            topk_length[:num_tokens],
            attn_sink,
            sm_scale,
            output,
            footer=kv_scale_format == "ue8m0_g32",
            block=swa_kv_cache.shape[1] if swa_kv_cache.dim() > 1 else 1,
        )


def _cake_dsv41_sm120_paged_attention(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor | None,
    extra_kv_cache: torch.Tensor | None,
    extra_sparse_indices: torch.Tensor | None,
    extra_sparse_topk_lens: torch.Tensor | None,
    workspace: torch.Tensor,
    layer_prefix: str = "",
    phase_tag: str = "",
) -> None:
    """Run FlashInfer's SM120 DSv4.1 `cake` mixed-cache decode in-place.

    The split-K scratch is carved from the caller-owned workspace, so a captured
    graph never re-allocates it.
    """
    from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import functional_run

    functional_run(
        query,
        swa_kv_cache,
        sparse_indices,
        output,
        workspace,
        sm_scale,
        lengths=topk_length,
        sink=attn_sink,
        extra=extra_kv_cache,
        extra_indices=extra_sparse_indices,
        extra_lengths=extra_sparse_topk_lens,
    )
    if phase_tag:
        _debug_attn_dump(
            phase_tag,
            layer_prefix,
            query,
            swa_kv_cache,
            sparse_indices,
            topk_length,
            attn_sink,
            sm_scale,
            output,
            footer=True,
            block=swa_kv_cache.shape[1] if swa_kv_cache.dim() > 1 else 1,
        )


# Flat-row (storage-offset) sparse indices of a padded inline-scale cache. One
# buffer per device, role and width — a dense row is required for prefill — and
# resized only by rows, which keeps every captured graph's buffer in place.
# The remap kernel and the attention kernel reading its output run back to back
# on the step's stream, so layers can share a buffer.
_storage_offset_index_scratch: dict[tuple[torch.device, str, int], torch.Tensor] = {}


def _get_storage_offset_index_scratch(
    device: torch.device, role: str, capacity_rows: int, width: int
) -> torch.Tensor:
    """Reusable index scratch, sized so captured graphs never see it move."""
    key = (device, role, width)
    scratch = _storage_offset_index_scratch.get(key)
    if scratch is None:
        scratch = torch.empty((capacity_rows, width), dtype=torch.int32, device=device)
        _storage_offset_index_scratch[key] = scratch
    elif scratch.shape[0] < capacity_rows:
        raise RuntimeError(
            "the storage-offset index scratch was allocated with a smaller "
            "capacity; every caller must size it with the same row bound"
        )
    return scratch


def _flashinfer_dsv4_workspace_bytes(
    vllm_config: VllmConfig,
    *,
    padded_heads: int,
    window_size: int,
    compress_ratio: int,
) -> int:
    """Workspace FlashInfer's SM120 DSV4 decode plan needs for this layer.

    The plan sizes its partial and LSE scratch from the widest dense index row it
    resolves -- the SWA window plus the compressed top-k width -- times the
    number of rows, heads and candidate chunks. Every V4.1 layer that owns a
    compressed cache gathers its top-k over the shared `index_topk`-wide
    buffer, so a long-context model needs more than the 128 MiB floor even for a
    small decode batch.
    """
    topk = required_sm120_sparse_topk(vllm_config, window_size)
    extra_topk = (
        int(getattr(vllm_config.model_config.hf_config, "index_topk", 0))
        if compress_ratio > 0
        else 0
    )
    chunk_capacity = cdiv(topk, _FLASHINFER_DSV4_DECODE_CHUNK_WIDTH) + cdiv(
        extra_topk, _FLASHINFER_DSV4_DECODE_CHUNK_WIDTH
    )
    slots = _FLASHINFER_DSV4_DECODE_MAX_TOKENS * padded_heads * chunk_capacity
    scratch = slots * (_FLASHINFER_DSV4_VALUE_DIM * 2 + 4)
    metadata = _FLASHINFER_DSV4_DECODE_MAX_TOKENS * padded_heads * 4
    return scratch + metadata


def _cake_dsv41_workspace_bytes(
    vllm_config: VllmConfig,
    *,
    padded_heads: int,
    window_size: int,
    compress_ratio: int,
    max_tokens: int,
) -> int:
    """Split-K scratch FlashInfer's `cake` DSv4.1 decode needs for this layer.

    Same shape of requirement as the sparse decode plan -- rows x candidate
    chunks -- but with its own chunk width and merge buffers. `max_tokens` is
    the rows one launch takes, so the buffer covers the widest step.
    """
    from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
        cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes,
    )

    extra_topk = (
        int(getattr(vllm_config.model_config.hf_config, "index_topk", 0))
        if compress_ratio > 0
        else 0
    )
    return cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(
        max_tokens,
        padded_heads,
        required_sm120_sparse_topk(vllm_config, window_size),
        extra_topk,
    )


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4SparseMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout.

    Inherits the base and backend reuses its`DeepseekV4SparseMLAMetadataBuilder`
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # SM12x kernels only instantiate 64-token sparse-MLA blocks (128
        # trips the block_kv assert, #59203); SM90 pages must contain 64
        # compressed states. Wider manager blocks may hold several.
        if (
            current_platform.is_device_capability_family(90)
            or current_platform.is_device_capability_family(120)
            or current_platform.is_device_capability_family(121)
        ):
            return [MultipleOf(64)]
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [10, 12]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major == 10:
            if kv_cache_dtype == "fp8_ds_mla":
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM10x uses the plain "
                    "per-tensor FP8 KV layout, not fp8_ds_mla"
                )
            if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
                return "kv_cache_dtype not supported"
            return None
        if device_capability.major == 12:
            if kv_cache_dtype not in (
                "fp8",
                "fp8_e4m3",
                "fp8_ds_mla",
                "nvfp4_ds_mla",
            ):
                return "kv_cache_dtype not supported"
            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

            if not has_flashinfer_sparse_mla_sm120():
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                    "sparse MLA decode API"
                )
            return None
        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashInferSparseMLAMetadataBuilder"]:
        return DeepseekV4FlashInferSparseMLAMetadataBuilder


class DeepseekV4FlashInferSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """Varlen-decode-capable metadata builder for the FlashInfer sparse MLA backend."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Decode replays use device request boundaries; prefill metadata is
        # not graph-safe.
        return max_decode_query_len(vllm_config)


class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """SWA metadata for the FlashInfer sparse decode path (varlen decode)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Decode replays use device request boundaries; prefill metadata is
        # not graph-safe.
        return max_decode_query_len(vllm_config)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Graphs retain these addresses while each build refreshes their contents.
        self._decode_topk_lens = torch.empty(
            self._max_tokens, dtype=torch.int32, device=self.device
        )
        self._decode_seq_lens = torch.empty_like(self._decode_topk_lens)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> "DeepseekSparseSWAMetadata":
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build, replay_start
        )
        if not common_attn_metadata.causal and metadata.num_decode_tokens > 0:
            self._fill_noncausal_decode_lens(metadata)
        return metadata

    def update_draft_decode_metadata(
        self, metadata: "DeepseekSparseSWAMetadata"
    ) -> None:
        super().update_draft_decode_metadata(metadata)
        if metadata.flashinfer_decode_topk_lens is not None:
            self._fill_noncausal_decode_lens(metadata)

    def _fill_noncausal_decode_lens(
        self, metadata: "DeepseekSparseSWAMetadata"
    ) -> None:
        num_tokens = metadata.num_decode_tokens
        assert metadata.decode_swa_lens is not None
        assert metadata.seq_lens is not None
        assert metadata.token_to_req_indices is not None
        topk_lens = self._decode_topk_lens[:num_tokens]
        seq_lens = self._decode_seq_lens[:num_tokens]
        torch.clamp(metadata.decode_swa_lens, min=self.window_size, out=topk_lens)
        torch.index_select(
            metadata.seq_lens,
            0,
            metadata.token_to_req_indices[:num_tokens],
            out=seq_lens,
        )
        metadata.flashinfer_decode_topk_lens = topk_lens
        metadata.flashinfer_decode_seq_lens = seq_lens


class DeepseekSparseSWAFlashInferBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAFlashInferMetadataBuilder]:
        return DeepseekSparseSWAFlashInferMetadataBuilder


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):
    """FlashInfer TRTLLM-gen sparse MLA attention layer for SM100 DeepSeek V4."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return pad_to_supported_q_heads(num_heads)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        register_dsv41_o_proj_warmup(self)
        # Per-tensor FP8 scale buffers + precomputed scalar BMM scales. Only the
        # per-tensor FP8 cache path consumes these; bf16 reads ``self.scale``.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # TRTLLM-gen takes scalar scale args on a distinct C++ path vs
        # one-element tensors, so these are Python floats.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The TRTLLM-gen kernel requires h_q in {64, 128}, so the output buffer
        # is allocated at the padded head count while q arrives at the local
        # head count; _forward pads q to match before the launcher.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: FlashInfer reads the cache directly and lazily
            # allocates its workspace, so nothing to reserve here.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward(
            q=q,
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
            output=output,
        )

    def _build_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the combined sparse-index tensors for the mixed batch.

        Returns `(compressed_kv_cache, seq_lens, sparse_indices,
        sparse_topk_lens)`.
        """
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None
        assert swa_metadata.replay_start is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, swa_metadata.decode_swa_width
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            assert self.topk_indices_buffer is not None
            assert swa_metadata.is_valid_token is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            # Local indices filled by the index-source layer's indexer.
            if num_prefill_tokens > 0:
                prefill_topk_indices = self.topk_indices_buffer[
                    num_decode_tokens:num_tokens
                ]
                top_k = prefill_topk_indices.shape[-1]
            else:
                prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                top_k = 0

            decode_compressed_indices_are_local = True
            decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
            if num_decode_tokens > 0:
                decode_compressed_indices = self.topk_indices_buffer[:num_decode_tokens]
            else:
                # Keep the logical width aligned with the mixed-batch case so
                # pure-prefill steps reuse the same Triton specialization.
                decode_compressed_indices = prefill_topk_indices[:0]

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        # SWA-only layers all build the same mixed sparse indices, so the first
        # one caches them for the step; indexer layers depend on their own topk
        # indices and stay uncached.
        cached_sparse = (
            swa_metadata.flashinfer_sparse_index_cache.get("swa_only")
            if swa_only
            else None
        )
        if cached_sparse is None:
            swa_block_span = packed_block_span(swa_k_cache)
            compressed_block_span = packed_block_span(compressed_kv_cache)
            sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
                decode_swa_indices,
                decode_compressed_indices,
                decode_compressed_topk_lens,
                prefill_topk_indices[:num_prefill_tokens],
                query_start_loc,
                seq_lens,
                swa_metadata.token_to_req_indices[:num_tokens],
                swa_metadata.block_table[:num_reqs],
                swa_metadata.block_size,
                compressed_block_table,
                compressed_block_size,
                self.window_size,
                self.compress_ratio,
                top_k,
                decode_compressed_indices_are_local=decode_compressed_indices_are_local,
                decode_is_valid_token=decode_is_valid_token,
                swa_block_span=swa_block_span,
                compressed_block_span=compressed_block_span,
                replay_start=swa_metadata.replay_start[:num_reqs],
            )
            if swa_only:
                swa_metadata.flashinfer_sparse_index_cache["swa_only"] = (
                    sparse_indices,
                    sparse_topk_lens,
                )
        else:
            sparse_indices, sparse_topk_lens = cached_sparse
        return compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens

    def _forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        (
            compressed_kv_cache,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )

        # CUDA graph execution can pad q/output past the scheduled token count;
        # restrict to the real tokens (the launcher validates sparse indices).
        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        # The TRTLLM-gen sparse-MLA kernel requires h_q in {64, 128}; zero-pad
        # the query heads to the allocated output head count. Padded heads attend
        # to the shared KV and are sliced off downstream (output is padded too).
        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        workspace = get_flashinfer_dsv4_workspace(q.device)
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc is not None and query_start_loc_cpu is not None

        # Keep the TRTLLM-gen decode/prefill split: the launcher is tuned for
        # uniform-q batches, and this avoids flattening mixed batches into one call.
        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            decode_output = output[:num_decode_tokens]
            decode_cu = query_start_loc[: num_decodes + 1]
            decode_seq_lens = seq_lens[:num_decodes]
            decode_topk_lens = sparse_topk_lens[:num_decode_tokens]
            max_decode_query_len = swa_metadata.max_decode_query_len
            if swa_metadata.decode_swa_width > self.window_size:
                # DSpark's non-causal window extends past the fixed 128 SWA
                # columns into the aliased compressed pool. Exclude padding,
                # and expose the full block to each query instead of letting
                # TRTLLM derive a causal SWA length from its query position.
                assert swa_only
                assert swa_metadata.flashinfer_decode_topk_lens is not None
                assert swa_metadata.flashinfer_decode_seq_lens is not None
                decode_topk_lens = swa_metadata.flashinfer_decode_topk_lens
                decode_seq_lens = swa_metadata.flashinfer_decode_seq_lens
                decode_query = decode_query.unsqueeze(1)
                decode_output = decode_output.unsqueeze(1)
                decode_cu = None
                max_decode_query_len = 1
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=decode_query,
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                sparse_indices_are_storage_offsets=True,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=decode_topk_lens,
                seq_lens=decode_seq_lens,
                out=decode_output,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=max_decode_query_len,
            )

        if num_prefill_tokens > 0:
            # The prefill query view re-anchors at offset 0, so rebase the
            # cumulative query offsets to start at 0.
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[num_decode_tokens:num_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                sparse_indices_are_storage_offsets=True,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=output[num_decode_tokens:num_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_cu,
                max_q_len=int(prefill_lens_cpu.max().item()),
            )


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    use_fp4_extra_kv: bool = current_platform.is_device_capability_family(
        120
    ) or current_platform.is_device_capability_family(121)
    # Built in ``__init__``; None when a test constructs a bare instance.
    _sm120_caller_buffers: _Sm120CallerBuffers | None = None

    @classmethod
    def packed_kv_cache_dtype(cls) -> CacheDType:
        return "nvfp4_ds_mla"

    @classmethod
    def uses_footer_scale_layout(cls) -> bool:
        """`cake` reads page-footer scale records; the sparse kernels do not."""
        return envs.VLLM_FLASHINFER_DSV41_SM120_CAKE

    @staticmethod
    def _get_workspace(device: torch.device) -> torch.Tensor:
        return get_flashinfer_dsv4_workspace(device)

    @staticmethod
    def _as_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if kv_cache.dim() == 4:
            return kv_cache
        return kv_cache.unsqueeze(-2)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return pad_to_supported_q_heads(num_heads)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config

        # Upper bound on a step's tokens: the flat-row index scratch is sized
        # once from it so a captured graph never outgrows its buffer.
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        if (
            current_platform.is_device_capability_family(120)
            or current_platform.is_device_capability_family(121)
        ) and self.kv_cache_dtype != "nvfp4_ds_mla":
            raise ValueError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires nvfp4_ds_mla "
                "for the FP4 compressed cache"
            )
        required_topk = required_sm120_sparse_topk(vllm_config, self.window_size)
        if not has_flashinfer_sparse_mla_sm120_config(
            self.padded_heads,
            required_topk,
            dsv41=True,
        ):
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV41 on SM120 requires a FlashInfer "
                "DSv4.1 sparse MLA decode specialization for "
                f"(num_q_heads={self.padded_heads}, top_k={required_topk}). "
                "Install a FlashInfer build containing the DSV4.1 SM120 "
                "dispatch specialization."
            )
        self.kv_scale_format = (
            "ue8m0_g32" if self.kv_footer_scale else "ue8m0_g32_inline"
        )
        if self.kv_footer_scale:
            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_cake

            if not has_flashinfer_sparse_mla_sm120_cake(self.padded_heads):
                raise RuntimeError(
                    "VLLM_FLASHINFER_DSV41_SM120_CAKE requires a FlashInfer "
                    "build with the SM120 DeepSeek-V4.1 `cake` mixed-cache "
                    f"decode kernel for {self.padded_heads} query heads, and "
                    "it rewrites the KV cache into the layout that kernel "
                    "reads. Unset it to keep the default inline-scale records."
                )
            # Load the kernel family now: it must not land inside a captured
            # step.
            from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
                get_cake_sparse_mla_sm120_dsv41_mixed_module,
            )

            get_cake_sparse_mla_sm120_dsv41_mixed_module()
            self.cake_max_tokens = envs.VLLM_FLASHINFER_DSV41_SM120_CAKE_MAX_TOKENS
            if self.cake_max_tokens < 1:
                raise ValueError(
                    "VLLM_FLASHINFER_DSV41_SM120_CAKE_MAX_TOKENS must be a "
                    f"positive decode row count, got {self.cake_max_tokens}"
                )
            logger.info_once(
                "Sparse MLA decode runs on FlashInfer's cake kernels "
                "(VLLM_FLASHINFER_DSV41_SM120_CAKE), %d decode rows per "
                "launch; wider steps take several launches of the same cache, "
                "and prefill stays on the sparse kernels. The V4.1 KV caches "
                "hold page-footer scale records, so caches written by the "
                "default inline records cannot be reused.",
                self.cake_max_tokens,
            )
        # The workspace is per device and shared by every layer, so size it here
        # (outside CUDA graph capture) for the widest plan this layer can ask for.
        # Cake carves its split-K scratch from the same buffer.
        workspace_bytes = _flashinfer_dsv4_workspace_bytes(
            vllm_config,
            padded_heads=self.padded_heads,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
        )
        if self.kv_footer_scale:
            workspace_bytes = max(
                workspace_bytes,
                _cake_dsv41_workspace_bytes(
                    vllm_config,
                    padded_heads=self.padded_heads,
                    window_size=self.window_size,
                    compress_ratio=self.compress_ratio,
                    max_tokens=self.cake_max_tokens,
                ),
            )
        workspace = get_flashinfer_dsv4_workspace(
            torch.device("cuda", torch.accelerator.current_device_index()),
            workspace_bytes,
        )
        if self.kv_footer_scale:
            self._cake_workspace_tensor = workspace
        self.use_fp4_extra_kv = self.kv_cache_dtype == "nvfp4_ds_mla" and (
            current_platform.is_device_capability_family(120)
            or current_platform.is_device_capability_family(121)
        )
        # The wrapper drops a replaced plan's owned tensors. Anything a captured
        # graph baked must outlive the plan, so the caller owns the scratch.
        self._sm120_caller_buffers = _sm120_caller_buffers_for_layer(
            device=torch.device("cuda", torch.accelerator.current_device_index()),
            padded_heads=self.padded_heads,
            max_tokens=self.max_num_batched_tokens,
            topk=required_sm120_sparse_topk(vllm_config, self.window_size),
            extra_topk=int(
                getattr(vllm_config.model_config.hf_config, "index_topk", 0)
            ),
            kv_scale_format=self.kv_scale_format,
            extra_fp4=self.use_fp4_extra_kv,
            scratch=True,
        )

        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        # Per-tensor FP8 cache path scales.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # FlashInfer expects scalar scale arguments for this path.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _cake_workspace(self) -> torch.Tensor:
        # Resolved at init: growing the shared workspace inside a captured step
        # would move a buffer an earlier graph already points at.
        return self._cake_workspace_tensor

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        debug_mixed = (
            os.environ.get("VLLM_DEBUG_MIXED_ATTN") == "1"
            and num_decode_tokens > 0
            and num_prefill_tokens > 0
        )
        if debug_mixed and (layers := os.environ.get("VLLM_DEBUG_MIXED_ATTN_LAYERS")):
            wanted = [x.strip() for x in layers.split(",") if x.strip()]
            if wanted and not any(x in self.prefix for x in wanted):
                debug_mixed = False
        if debug_mixed:
            fc = get_forward_context()
            bd = getattr(fc, "batch_descriptor", None)
            print(
                f"[mixed] {self.prefix} dec={num_decode_tokens} "
                f"pref={num_prefill_tokens} "
                f"batch={getattr(bd, 'num_tokens', None)} "
                f"mode={getattr(fc, 'cudagraph_runtime_mode', None)}",
                flush=True,
            )
        prefill_before = None
        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
            if debug_mixed:
                prefill_before = output[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ].clone()
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )
            if debug_mixed:
                prefill_after = output[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
                changed = (prefill_before - prefill_after).abs().max().item()
                print(
                    f"[mixed] {self.prefix} decode changed prefill output: {changed}",
                    flush=True,
                )
                root = os.environ.get("VLLM_DEBUG_TF_DIR")
                if root:
                    phase_dir = _debug_phase_dir(root)
                    stem = (
                        f"{self.prefix}__dev{torch.cuda.current_device()}"
                        f"__dec{num_decode_tokens}_pref{num_prefill_tokens}"
                    )
                    torch.save(
                        prefill_before.detach().cpu(),
                        os.path.join(phase_dir, f"attnprefbefore__{stem}.pt"),
                    )
                    torch.save(
                        prefill_after.detach().cpu(),
                        os.path.join(phase_dir, f"attnprefafter__{stem}.pt"),
                    )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def _storage_offset_indices(
        self,
        indices: torch.Tensor | None,
        cache: torch.Tensor | None,
        block_size: int,
        role: str,
        row_offset: int = 0,
    ) -> torch.Tensor | None:
        """Rewrite paged slot ids as flat cache rows for an inline-scale cache.

        FlashInfer's DSv4.1 kernels address a record as `row * row_pitch`, so
        a page padded to whole rows (the interleaved V4.1 layout) or sharing its
        block with another layer needs storage-offset indices. Compact pages map
        ids to themselves and keep the metadata's buffer. Footer-scale pages are
        addressed by paged slot id, so they keep the metadata's buffer too.
        """
        if indices is None or cache is None or self.kv_footer_scale:
            return indices
        page_rows = _paged_cache_page_rows(cache)
        page_span = _packed_block_span(cache)
        if page_rows == block_size and page_span == block_size:
            return indices
        width = indices.shape[-1]
        rows = indices.numel() // width
        scratch = _get_storage_offset_index_scratch(
            indices.device, role, self.max_num_batched_tokens, width
        )
        return remap_flashinfer_sparse_indices(
            indices,
            block_size,
            page_rows,
            page_span,
            out=scratch[row_offset : row_offset + rows],
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        compressed_block_size = 0
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer decode requires top-k indices from the indexer."
                )
            # Local indices filled by the index-source layer's indexer.
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio
            global_indices, extra_sparse_lengths = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                compressed_block_size,
                is_valid,
            )
            extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        swa_indices = self._storage_offset_indices(
            swa_indices, swa_cache, swa_metadata.block_size, "swa"
        )
        extra_sparse_indices = self._storage_offset_indices(
            extra_sparse_indices, extra_cache, compressed_block_size, "extra"
        )
        if self.kv_footer_scale:
            # One cake launch takes as many rows as the shared split-K scratch
            # covers; a wider step splits into launches of that size, which
            # still beats the sparse kernels (~2x at 512 rows). Each launch
            # resolves its own split plan, so a chunked step can differ from a
            # single wide launch by one rounding step of the split merge.
            cap = self.cake_max_tokens
            logger.info_once(
                "A decode step of at most %d tokens runs on the cake kernels; "
                "wider steps take one launch per %d tokens.",
                cap,
                cap,
            )
            for lo in range(0, num_decode_tokens, cap):
                hi = min(lo + cap, num_decode_tokens)
                _cake_dsv41_sm120_paged_attention(
                    query=q[lo:hi],
                    swa_kv_cache=swa_cache,
                    sparse_indices=cast(torch.Tensor, swa_indices)[lo:hi],
                    output=output[lo:hi],
                    sm_scale=self.scale,
                    topk_length=swa_lens[lo:hi],
                    attn_sink=self.attn_sink,
                    extra_kv_cache=extra_cache,
                    extra_sparse_indices=(
                        extra_sparse_indices[lo:hi]
                        if extra_sparse_indices is not None
                        else None
                    ),
                    extra_sparse_topk_lens=(
                        extra_sparse_lengths[lo:hi]
                        if extra_sparse_lengths is not None
                        else None
                    ),
                    workspace=self._cake_workspace(),
                    layer_prefix=self.swa_cache_layer.prefix,
                    phase_tag="dec",
                )
            return
        _sparse_mla_sm120_paged_attention(
            query=q,
            swa_kv_cache=swa_cache,
            sparse_indices=swa_indices,
            output=output,
            sm_scale=self.scale,
            topk_length=swa_lens,
            attn_sink=self.attn_sink,
            extra_kv_cache=extra_cache,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
            extra_kv_fp4=self.use_fp4_extra_kv,
            # Same preference as prefill: a padded call takes a prefill kernel.
            prefill_impl=envs.VLLM_FLASHINFER_SM120_PREFILL_IMPL,
            kv_scale_format=self.kv_scale_format,
            caller=self._sm120_caller_buffers,
            layer_prefix=self.swa_cache_layer.prefix,
            phase_tag="dec",
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        compressed_block_size = 0
        if not swa_only:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer prefill requires top-k indices from the indexer."
                )
            if attn_metadata is None:
                raise RuntimeError("Compressed-layer prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError(
                    "Compressed-layer prefill request mapping is missing."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "Compressed-layer prefill validity metadata is missing."
                )
            # Local indices filled by the index-source layer's indexer.
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            compressed_block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    compressed_block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_query(q, output)
        swa_kv_paged = self._as_sparse_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = self._as_sparse_cache(compressed_k_cache)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            # Flat cache rows keep the step's global row index: decode rows own
            # [0, num_decode_tokens), so a mixed step's prefills must not
            # overwrite them in the shared remap scratch.
            row_offset = num_decode_tokens + query_start
            swa_indices_chunk = self._storage_offset_indices(
                swa_metadata.prefill_swa_indices[query_start:query_end],
                swa_kv_paged,
                swa_metadata.block_size,
                "swa",
                row_offset=row_offset,
            )
            extra_sparse_indices_chunk = self._storage_offset_indices(
                extra_sparse_indices_chunk,
                extra_kv_paged,
                compressed_block_size,
                "extra",
                row_offset=row_offset,
            )
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            _sparse_mla_sm120_paged_attention(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                sparse_indices=swa_indices_chunk,
                output=output[query_start:query_end],
                sm_scale=self.scale,
                topk_length=swa_lens_chunk,
                attn_sink=self.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
                extra_kv_fp4=self.use_fp4_extra_kv,
                prefill_impl=envs.VLLM_FLASHINFER_SM120_PREFILL_IMPL,
                kv_scale_format=self.kv_scale_format,
                caller=self._sm120_caller_buffers,
                layer_prefix=self.swa_cache_layer.prefix,
                phase_tag="pref",
            )
