# Mixed-batch TF-row bisect

Debug-only branch (`debug/dsv41-tf-dump`, not for push). It dumps the
teacher-forced (TF) request's rows of every decoder layer and of the SM120
attention output, so the first layer where a co-batched decode row changes the
TF stream can be found.

## What the hook does

- `vllm/models/deepseek_v41/nvidia/model.py`
  - after attention (after the SP reduce-scatter) dumps `x` as `attn__<layer>`;
  - before the FFN/MoE (after the mHC post/pre merge) dumps `x` as
    `ffnin__<layer>`;
  - after the FFN/MoE dumps `x` (tensor only) as `ffn__<layer>`.
  - Rows dumped: `[num_decode_tokens, num_decode_tokens + num_prefill_tokens)`
    (the TF request's rows). The file name carries the TP rank, the CUDA
    device (DP/PP rank), the global start row and the row count:
    `attn__<layer>__r<rank>__dev<N>__start<S>__rows<R>.pt`; the compare script
    concatenates the TP shards per device.
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` dumps the TF
  rows of `router_logits`, the routed output before the EP combine
  (`moeprecombine`), and the combined output (`moepostcombine`).
- `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`
  - logs `[mixed] <layer> dec=... pref=... batch=... mode=...` per mixed
    attention call;
  - clones `output[num_decode_tokens:num_decode_tokens+num_prefill_tokens]`
    after `_forward_prefill` and after `_forward_decode`, prints the max change,
    and (with `VLLM_DEBUG_TF_DIR`) saves
    `attnprefbefore/after__<layer>__dev<N>__dec<D>_pref<P>.pt`.
- `attention.py` (v4 + v41): when `VLLM_DEBUG_TF_DIR` is set,
  `_short_context_topk_is_uncaptured` returns False, so the indexer runs the
  full scoring path in both the alone and mixed runs.
- `vllm/v1/worker/gpu/debug_step_dump.py` (also on this branch): with
  `VLLM_DEBUG_STEP_DUMP=1` it logs one `DSV41DEBUG {json}` line per step
  (composition, scheduled tokens, input ids/positions, PP payload checksums);
  with `VLLM_DEBUG_TF_DIR` it additionally saves the TF rows of the runner
  inputs (`runnerinputs_ids/positions`), the PP intermediate payload
  (`runnerinputs_intermediate_<key>` / `runneroutputs_<key>`) and the model
  output, per stage/device.
- Layer filters: `VLLM_DEBUG_TF_LAYERS=2,20,23,36` limits the per-decoder-layer
  dumps; `VLLM_DEBUG_MIXED_ATTN_LAYERS=2,20,23,36` limits the
  `_forward_sparse_impl` prefill-output dumps and `[mixed]` log lines.

All dumps read `<VLLM_DEBUG_TF_DIR>/phase` on every call and write to
`<root>/<phase>/`, so one server run can dump both phases.

## Server config for the debug run

Eager mixed steps are required for the Python hook to run at every step:

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_DEBUG_STEP_DUMP=1 \
VLLM_DEBUG_TF_DIR=/dev/shm/tf_dump \
VLLM_DEBUG_MIXED_ATTN=1 \
# optional: VLLM_DEBUG_TF_LAYERS=2,20,23,36
# optional: VLLM_DEBUG_MIXED_ATTN_LAYERS=2,20,23,36
# same serve command, but compilation config:
#   {"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}
```

The absolute logits differ from the PIECEWISE/production run; the target is the
**first layer at which alone and mixed diverge**, not the absolute values.

## Run

```bash
mkdir -p /dev/shm/tf_dump
pid=$(pgrep -f 'vllm serve' | head -1)

# 1) alone: no background decode
echo alone > /dev/shm/tf_dump/phase
/opt/venv/bin/python /workspace/debug-vllm/teacher_force.py "$pid" \
  --codes cobalt-raven-0225,ginger-raven-0024,hollow-raven-6153 \
  --phase alone

# 2) mixed: one co-batched decode row on the same DP rank
echo mixed > /dev/shm/tf_dump/phase
/opt/venv/bin/python /workspace/debug-vllm/teacher_force.py "$pid" \
  --codes cobalt-raven-0225,ginger-raven-0024,hollow-raven-6153 \
  --phase both --bg-rank 0 --bg-len 64 --bg-max-tokens 1000 --trials 1

# 3) compare
/opt/venv/bin/python debug/compare_tf_dumps.py /dev/shm/tf_dump
```

`teacher_force.py --phase both` runs its own alone phase first and then the bg
phase. With the phase file reading `mixed`, the hook skips the alone steps
(`num_decode_tokens == 0`) and dumps only the co-batched steps, so the two
directories stay clean. The `attnprefbefore/after` files are written only for
mixed steps as well.

## Reading the result

- Dump table: layers sorted by index, `attn` then `ffn`. The first `max_abs`
  above the bf16-noise threshold is the layer where the streams first diverge.
  `attnprefbefore/after` tells whether the SM120 `_forward_decode` call changed
  the already-computed prefill output (`max_abs > 0`) or not.
- `attnprefbefore != after` → the shared SM120 wrapper / caller scratch is
  implicated; `== 0` → look outside attention (compressor/indexer/Engram/PP
  relay/MoE·EP).
- The `[mixed] ...` lines identify the step: `dec=6 pref=37 batch=48` for the
  production mixed step, `dec=0 pref=37` for the alone prefill.
- `runnerinputs_*` vs `runneroutputs_*` answer whether the input assembly
  changed: `runnerinputs_ids/positions` identical and both stages' PP payload
  rows identical → the runner is clean. Stage 0's `runneroutputs_*` vs stage
  1's `runnerinputs_intermediate_*` (same phase, same device pair) is the PP
  boundary comparison.

## Caveats

- Based on `33452c74d5` (includes the committed short-context gate and the
  step dump). The debug hook forces the full indexer path, which is the control
  we want.
- `FULL_DECODE_ONLY` still captures the pure-decode FULL graphs; their Python
  only runs at capture. The TF request's prefill/mixed steps are eager and are
  the ones dumped.
- Dumps go to `/dev/shm`; remove the directory between runs.
