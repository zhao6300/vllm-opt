"""CPU differential simulation of the DSv4.1 mixed-batch inputs.

A = [TF prefill 37]                 (alone)
B = [bg decode 6 rows + TF prefill 37]   (mixed, decode rows first)

Runs the real metadata builders and Triton kernels with TRITON_INTERPRET=1 and
compares every TF-row input.  A builder's returned tensors are often views into
its persistent buffers, so A's outputs are cloned before B is built.
"""

import os
import sys

import torch

# Run against the editable/compiled tree (a source-only worktree has no
# `vllm/_C` extension). Override with VLLM_SIM_ROOT if needed.
_ROOT = os.environ.get("VLLM_SIM_ROOT", "/workspace/vllm-opt")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from tests.v1.attention.utils import create_vllm_config  # noqa: E402
from vllm.config import SpeculativeConfig  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec  # noqa: E402

DEV = torch.device("cpu")
W = 8


def make_cfg():
    c = create_vllm_config(
        model_name="/models/DeepSeek-V4.1-Flash-1017",
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        hf_config_override={
            "sliding_window": 128,
            "compress_ratios": [0, 1, 2] * 16,
            "index_topk": 512,
            "index_head_dim": 128,
            "index_n_heads": 32,
        },
    )
    c.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=5
    )
    return c


CFG = make_cfg()


def common(seq_lens, query_lens, slot_lists, block_tags):
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    positions = torch.cat(
        [torch.arange(s - q, s, dtype=torch.int64)
         for s, q in zip(seq_lens, query_lens)]
    )
    slots = torch.cat(slot_lists)
    block_table = torch.stack(
        [torch.arange(t * 100, t * 100 + W, dtype=torch.int32) for t in block_tags]
    )
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(seq_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slots,
        causal=True,
        positions=positions,
    )


TF_SLOTS = [torch.arange(0, 37, dtype=torch.int64)]
BG_SLOTS = [torch.arange(1000, 1006, dtype=torch.int64)]


def batches():
    """Fresh metadata per builder: builders may mutate the common object."""
    a = common([37], [37], TF_SLOTS, [1])
    b = common([300, 37], [6, 37], [BG_SLOTS[0], TF_SLOTS[0]], [2, 1])
    return a, b


def snap(obj, names):
    out = {}
    for n in names:
        v = getattr(obj, n)
        out[n] = v.clone() if isinstance(v, torch.Tensor) else v
    return out


def report(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'DIFF'}] {name} {detail}", flush=True)
    return ok


def swa():
    from vllm.models.deepseek_v41.sparse_mla import (
        DeepseekV41SparseSWAMetadataBuilder,
    )

    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.bfloat16,
        cache_dtype_str="auto", model_version="deepseek_v4", tokens_per_state=1,
    )
    b = DeepseekV41SparseSWAMetadataBuilder(
        kv_cache_spec=spec, layer_names=["layer0"], vllm_config=CFG, device=DEV
    )
    A, B = batches()
    ma = b.build(0, A)
    a = snap(ma, ["prefill_swa_indices", "prefill_swa_lens",
                  "is_valid_token", "token_to_req_indices"])
    mb = b.build(0, B)
    ok = True
    ok &= report("SWA prefill_swa_indices", torch.equal(
        a["prefill_swa_indices"], mb.prefill_swa_indices))
    ok &= report("SWA prefill_swa_lens", torch.equal(
        a["prefill_swa_lens"], mb.prefill_swa_lens))
    ok &= report("SWA is_valid_token TF rows", torch.equal(
        a["is_valid_token"][:37], mb.is_valid_token[6:43]))
    ok &= report("SWA token_to_req TF rows (0 vs 1)", bool(
        torch.all(a["token_to_req_indices"][:37] == 0)
        and torch.all(mb.token_to_req_indices[6:43] == 1)))
    print(f"       A dec/pref={ma.num_decode_tokens}/{ma.num_prefill_tokens} "
          f"B dec/pref={mb.num_decode_tokens}/{mb.num_prefill_tokens}", flush=True)
    return ok


def indexer(compress_ratio):
    import vllm.v1.attention.backends.mla.indexer as idx

    idx.get_paged_mqa_logits_metadata = (
        lambda *a, **k: torch.zeros((1, 2), dtype=torch.int32)
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.uint8,
        cache_dtype_str="auto", model_version="deepseek_v4",
        tokens_per_state=compress_ratio,
    )
    b = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=spec, layer_names=["layer0"], vllm_config=CFG,
        device=DEV, block_table_width=W,
    )
    names = ["block_table", "cu_seqlen_ks", "cu_seqlen_ke", "cu_seq_lens",
             "token_to_seq", "local_cu_seq_lens"]
    A, B = batches()
    ma = b.build(0, A)
    ca = ma.prefill.chunks[-1]
    a = snap(ca, names)
    a_host = {n: getattr(ca, n) for n in
              ("token_start", "token_end", "max_seqlen_k", "max_local_total_seq_lens",
               "local_total_seq_lens", "num_reqs", "skip_kv_gather")}
    mb = b.build(0, B)
    cb = mb.prefill.chunks[-1]
    ok = True
    for n in names:
        va, vb = a[n], getattr(cb, n)
        if va is None and vb is None:
            continue
        ok &= report(f"indexer ratio{compress_ratio} {n}", torch.equal(va, vb))
    host_names = [n for n in a_host if n not in ("token_start", "token_end")]
    ok &= report(f"indexer ratio{compress_ratio} host fields", all(
        a_host[n] == getattr(cb, n) for n in host_names),
        f"A={a_host} B=" + str({n: getattr(cb, n) for n in a_host}))
    ok &= report(f"indexer ratio{compress_ratio} token_start offset",
                 int(cb.token_start) == int(ca.token_start) + 6)
    return ok


def compressor():
    from vllm.models.deepseek_v41.compressor import CompressorMetadataBuilder
    from vllm.v1.kv_cache_interface import AttentionSpec

    spec = AttentionSpec(
        block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0,
        dtype=torch.float32,
    )
    b = CompressorMetadataBuilder(
        kv_cache_spec=spec, layer_names=["state"], vllm_config=CFG, device=DEV
    )
    A, B = batches()
    ma = b.build(0, A)
    a_slots = ma.slot_mapping.clone()
    mb = b.build(0, B)
    ok = report("compressor TF ring slots", torch.equal(
        mb.slot_mapping[6:43], a_slots), f"A[:4]={a_slots[:4].tolist()} "
        f"B[6:10]={mb.slot_mapping[6:10].tolist()} B[:6]={mb.slot_mapping[:6].tolist()}")
    return ok


def remap():
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        remap_flashinfer_sparse_indices,
    )

    idx = torch.arange(37 * 4, dtype=torch.int32).view(37, 4) * 1  # paged slots
    scratch = torch.full((64, 4), -99, dtype=torch.int32)
    remap_flashinfer_sparse_indices(
        idx, 64, 64, 80, out=scratch[0:37])
    prefill_alone = scratch[0:37].clone()
    scratch.fill_(-99)
    # mixed: decode remap owns rows 0..5, prefill owns 6..42
    dec = torch.arange(6 * 4, dtype=torch.int32).view(6, 4)
    remap_flashinfer_sparse_indices(dec, 64, 64, 80, out=scratch[0:6])
    remap_flashinfer_sparse_indices(idx, 64, 64, 80, out=scratch[6:43])
    ok = report("remap prefill rows equal alone", torch.equal(
        scratch[6:43], prefill_alone))
    ok &= report("remap scratch disjoint rows", not torch.equal(
        scratch[0:6], torch.full((6, 4), -99, dtype=torch.int32)))
    return ok


def engram_hash():
    from vllm.models.deepseek_v41.common.engram import DEAD_ID, _hash_ids_kernel

    LAYERS, MAX_NGRAM, HEADS, DEPTH = 2, 4, 2, 8
    torch.manual_seed(0)
    tok_map = torch.arange(1000, dtype=torch.int32)
    mult = torch.randint(1, 7, (LAYERS, MAX_NGRAM), dtype=torch.int64)
    param_n = LAYERS * (MAX_NGRAM - 1) * HEADS
    primes = torch.randint(11, 97, (param_n,), dtype=torch.int32)
    offsets = torch.zeros(param_n, dtype=torch.int32)
    look_a = torch.randint(2, 900, (1, DEPTH), dtype=torch.int32)
    look_b = torch.cat([torch.randint(2, 900, (1, DEPTH), dtype=torch.int32),
                        look_a])

    tf_ids = (torch.arange(37, dtype=torch.int32) * 7 + 11) % 900 + 2
    bg_ids = (torch.arange(6, dtype=torch.int32) * 13 + 50) % 900 + 2

    def run(ids, positions, qsl, lookback, dead):
        n = ids.numel()
        out = torch.full((n, LAYERS, (MAX_NGRAM - 1) * HEADS), -99,
                         dtype=torch.int32)
        _hash_ids_kernel[(triton_cdiv(n, 32), LAYERS)](
            ids, tok_map, dead, positions, qsl, mult, primes, offsets,
            lookback, torch.zeros_like(lookback, dtype=torch.bool), out,
            n, 1, 1, 1, 1, 1, qsl.numel() - 1,
            MAX_NGRAM=MAX_NGRAM, num_heads=HEADS, BLOCK_T=32, BLOCK_H=2,
            dead_id=DEAD_ID, lookback_depth=DEPTH,
            lookback_row_stride=lookback.stride(0),
            lookback_col_stride=lookback.stride(1),
            lookback_mask_row_stride=lookback.stride(0),
            lookback_mask_col_stride=lookback.stride(1),
            num_warps=4,
        )
        return out

    def triton_cdiv(a, b):
        return (a + b - 1) // b

    a = run(tf_ids, torch.arange(37, dtype=torch.int64),
            torch.tensor([0, 37], dtype=torch.int32), look_a,
            torch.zeros(37, dtype=torch.bool))
    b = run(torch.cat([bg_ids, tf_ids]),
            torch.cat([torch.arange(294, 300, dtype=torch.int64),
                       torch.arange(37, dtype=torch.int64)]),
            torch.tensor([0, 6, 43], dtype=torch.int32), look_b,
            torch.zeros(43, dtype=torch.bool))
    return report("engram hash TF rows", torch.equal(a, b[6:43]))


if __name__ == "__main__":
    all_ok = True
    all_ok &= swa()
    all_ok &= indexer(1)
    all_ok &= indexer(2)
    all_ok &= compressor()
    all_ok &= remap()
    all_ok &= engram_hash()
    print("ALL OK" if all_ok else "DIFFERENCES FOUND", flush=True)
