#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Phase 1: serve DeepSeek V4.1 on 8 SM120 GPUs (e.g. RTX PRO 6000 96GB) as
#   PP=2 x TP=1 x attention-DP=4, EP=4 per PP stage, DSpark (5 drafts).
# Stage 0 = layers 0-19 on 4 GPUs, stage 1 = layers 20-39 + DSpark on 4 GPUs.
# No DCP, DBO/microbatching off.
#
#   MODEL=/models/DeepSeek-V4.1-Flash bash serve.sh
# Toggles (for the phase 2 matrix): DSPARK=0, ASYNC=0. EP=0 is only for
# smaller test checkpoints: the full model's experts do not fit without EP.
set -euo pipefail

MODEL=${MODEL:?set MODEL to the DeepSeek V4.1 checkpoint path or HF id}
PORT=${PORT:-8000}
EP=${EP:-1}
DSPARK=${DSPARK:-1}
ASYNC=${ASYNC:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}

# The 40 layers split 20/20 by default; pin it so the cut stays on the
# layer 20 KV/index/candidate source boundary.
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-20,20}
export VLLM_USE_V2_MODEL_RUNNER=1

args=(
    --pipeline-parallel-size 2
    --tensor-parallel-size 1
    --data-parallel-size 4
    --data-parallel-backend mp
    # SM120 FlashInfer sparse MLA requires an FP8/FP4 KV cache.
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41"}'
    --kv-cache-dtype fp8
    # Engram tables stay in pinned host memory, one copy shared by the
    # four DP replicas of stage 0 (layers 1 and 14).
    --engram-config '{"cpu_offload":true}'
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --language-model-only
    --port "$PORT"
)
if [ "$EP" = 1 ]; then
    args+=(--enable-expert-parallel)
fi
if [ "$DSPARK" = 1 ]; then
    # dspark_block_size=5, target layers 37/38/39, 128 draft experts (top 3)
    # come from the checkpoint config.
    args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5}')
fi
if [ "$ASYNC" = 1 ]; then
    args+=(--async-scheduling)
else
    args+=(--no-async-scheduling)
fi
if [ -n "${PROFILE_DIR:-}" ]; then
    # Per-rank torch traces for the phase 2 breakdown (stage forward, PP
    # send/recv, EP all-to-all, DSpark draft); drive with
    # `vllm bench serve --profile`.
    args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\"}")
fi

# Each worker logs "Parallel topology: global_rank=... dp_rank=... pp_rank=...
# ep_ranks=... physical_gpu=..." at startup; check that both EP groups stay
# inside one PP stage ({0,2,4,6} and {1,3,5,7}).
exec vllm serve "$MODEL" "${args[@]}" "$@"
