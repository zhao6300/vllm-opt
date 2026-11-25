"""CPU differential simulation of the V2 runner's input assembly.

Runs the real input-preparation kernels for
  A = [TF prefill 37]
  B = [bg decode 6 + TF prefill 37]
and compares the TF rows' input_ids, positions, slot mappings and Engram
lookback.  Run with TRITON_INTERPRET=1 on CPU.
"""

import sys

import torch

sys.path.insert(0, "/workspace/vllm-opt")
sys.path.insert(0, "/workspace/vllm-opt/tests")

from vllm.models.deepseek_v41.nvidia.model_state import (  # noqa: E402
    _gather_lookback_kernel,
)
from vllm.v1.worker.gpu.block_table import (  # noqa: E402
    _compute_slot_mappings_kernel,
)
from vllm.v1.worker.gpu.input_batch import (  # noqa: E402
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)


def real_slots(bt, idx, qsl, pos, pad):
    """Run the real `_compute_slot_mappings_kernel` (one KV cache group)."""
    ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64)
    strides = torch.tensor([bt.stride(0)], dtype=torch.int32)
    block_sizes = torch.tensor([BLOCK], dtype=torch.int32)
    enabled = torch.tensor([1], dtype=torch.int32)
    sharded = torch.tensor([0], dtype=torch.int32)
    slots = torch.full((1, pad), -2, dtype=torch.int64)
    _compute_slot_mappings_kernel[(1, idx.numel() + 1)](
        pad,
        idx,
        qsl,
        pos,
        ptrs,
        strides,
        block_sizes,
        enabled,
        sharded,
        slots,
        pad,
        0,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=32,
    )
    return slots[0]

W = 8
BLOCK = 64
DEPTH = 8
MAXLEN = 512


def make_batch(cold=True):
    prefix = 0 if cold else 8
    tf_len = prefix + 37
    tf_tokens = (torch.arange(tf_len, dtype=torch.int64) + 100).to(torch.int32)
    if cold:
        sizes = ([37], [37], [0], [37], 40)  # query_lens, qsl, computed, prefill
    else:
        sizes = ([37], [37], [prefix], [tf_len], 48)
    return prefix, tf_tokens, sizes


def run_case(cold: bool):
    prefix, tf_tokens, sizes = make_batch(cold)
    tf_query, _, tf_computed, tf_prefill = sizes[0][0], sizes[1], sizes[2], sizes[3]
    bg_len = 300
    bg_tokens = (torch.arange(bg_len, dtype=torch.int64) + 5000).to(torch.int32)
    bg_query = 6

    # A: one prefill request (state 0 = TF)
    a_states = 1
    a_all = torch.zeros((a_states, MAXLEN), dtype=torch.int32)
    a_all[0, : tf_prefill[0]] = tf_tokens
    a_idx = torch.tensor([0], dtype=torch.int32)
    a_qsl = torch.tensor([0, tf_query], dtype=torch.int32)
    a_computed = torch.tensor([tf_computed[0]], dtype=torch.int32)
    a_prefill_len = torch.tensor([tf_prefill[0]], dtype=torch.int32)
    a_pad = 40
    a_ids = torch.zeros(a_pad, dtype=torch.int32)
    a_next = torch.zeros((8, a_states), dtype=torch.int64)
    prepare_prefill_inputs(
        a_ids, a_next, a_idx, a_qsl, a_all, a_prefill_len, a_computed
    )
    a_pos = torch.zeros(a_pad, dtype=torch.int64)
    a_seq = torch.zeros(a_pad, dtype=torch.int32)
    prepare_pos_seq_lens(a_idx, a_qsl, a_computed, a_pos, a_seq)
    a_look = torch.full((1, DEPTH), -99, dtype=torch.int32)
    _gather_lookback_kernel[(1,)](
        a_look,
        a_idx,
        a_qsl,
        a_pos,
        a_all,
        a_all.stride(0),
        1,
        DEPTH=DEPTH,
        BLOCK_DEPTH=DEPTH,
    )
    a_bt = torch.stack([torch.arange(100, 100 + W, dtype=torch.int32)])
    a_slots = real_slots(a_bt, a_idx, a_qsl, a_pos, a_pad)

    # B: bg decode (state 0) + TF prefill (state 1)
    b_states = 2
    b_all = torch.zeros((b_states, MAXLEN), dtype=torch.int32)
    b_all[0, :bg_len] = bg_tokens
    b_all[1, : tf_prefill[0]] = tf_tokens
    b_idx = torch.tensor([0, 1], dtype=torch.int32)
    b_qsl = torch.tensor([0, bg_query, bg_query + tf_query], dtype=torch.int32)
    b_computed = torch.tensor([bg_len, tf_computed[0]], dtype=torch.int32)
    b_prefill_len = torch.tensor([bg_len, tf_prefill[0]], dtype=torch.int32)
    b_pad = 48
    b_ids = torch.zeros(b_pad, dtype=torch.int32)
    b_next = torch.zeros((8, b_states), dtype=torch.int64)
    prepare_prefill_inputs(
        b_ids, b_next, b_idx, b_qsl, b_all, b_prefill_len, b_computed
    )
    b_pos = torch.zeros(b_pad, dtype=torch.int64)
    b_seq = torch.zeros(b_pad, dtype=torch.int32)
    prepare_pos_seq_lens(b_idx, b_qsl, b_computed, b_pos, b_seq)
    b_tf_before = b_ids[bg_query : bg_query + tf_query].clone()
    last_sampled = torch.tensor([7777, 0], dtype=torch.int64)
    drafts = torch.zeros((b_states, 5), dtype=torch.int64)
    drafts[0] = torch.tensor([8100, 8101, 8102, 8103, 8104])
    cu_logits = torch.tensor([0, 6, 6], dtype=torch.int32)
    combine_sampled_and_draft_tokens(
        b_ids,
        b_idx,
        last_sampled,
        b_qsl,
        b_seq,
        b_prefill_len,
        drafts,
        cu_logits,
        num_logits=6,
        num_new_sampled_tokens=1,
    )
    b_look = torch.full((2, DEPTH), -99, dtype=torch.int32)
    _gather_lookback_kernel[(2,)](
        b_look,
        b_idx,
        b_qsl,
        b_pos,
        b_all,
        b_all.stride(0),
        2,
        DEPTH=DEPTH,
        BLOCK_DEPTH=DEPTH,
    )
    b_bt = torch.stack(
        [
            torch.arange(200, 200 + W, dtype=torch.int32),
            torch.arange(100, 100 + W, dtype=torch.int32),
        ]
    )
    b_slots = real_slots(b_bt, b_idx, b_qsl, b_pos, b_pad)

    tf_slice = slice(bg_query, bg_query + tf_query)
    print(f"case {'cold' if cold else 'prefixed'}")
    for name, va, vb in (
        ("input_ids", a_ids[:tf_query], b_ids[tf_slice]),
        ("positions", a_pos[:tf_query], b_pos[tf_slice]),
        ("slot_mapping", a_slots[:tf_query], b_slots[tf_slice]),
        ("lookback", a_look[0], b_look[1]),
    ):
        ok = torch.equal(va, vb)
        print(f"  [{'OK ' if ok else 'DIFF'}] {name} A={va[:4].tolist()} "
              f"B={vb[:4].tolist()}")
        if not ok:
            print("     A full", va.tolist())
            print("     B full", vb.tolist())
    # the decode write must stay in rows [0, bg_query)
    tf_ids_before = b_tf_before
    print("  [OK ] decode rows", b_ids[:bg_query].tolist())
    print("  [OK ] TF rows untouched", torch.equal(
        tf_ids_before, b_ids[tf_slice]))


if __name__ == "__main__":
    run_case(cold=True)
    run_case(cold=False)
