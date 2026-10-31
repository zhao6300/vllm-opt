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
# Per DP rank: the server admits up to 4 x MAX_NUM_SEQS concurrent requests.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
# Largest decode step (tokens) replayed as a CUDA graph; larger steps run
# eager. vLLM's default on SM120 is min(MAX_NUM_SEQS x 6 x 2, 512): with five
# DSpark drafts and two PP microbatches that covers ~85 requests per
# microbatch (~170 per DP rank). Raise it for higher concurrency, at the cost
# of graph memory and capture time. Empty keeps vLLM's default.
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
# MXFP4 routed experts: "auto" resolves to DEEPGEMM_MXFP4 on SM120 when
# DeepGEMM is installed. Set MOE_BACKEND=marlin if that kernel fails its
# SM120 test (tests/kernels/moe/test_deepgemm.py::test_deepgemm_fp4_vs_triton).
MOE_BACKEND=${MOE_BACKEND:-auto}
# Indexer K cache: "auto" is FP8 for the DeepSeek indexer. mxfp4 is opt-in on
# sm_12x (DeepGEMM >= 2.8.1) and still lacks long-context accuracy evals.
INDEXER_KV_DTYPE=${INDEXER_KV_DTYPE:-auto}
# Phase 3: PP send buffers on stage 0. 0 keeps the per-step send wait; 2
# double-buffers the 19 -> 20 activations (2 x max_num_batched_tokens rows of
# hidden_states + pre_mix on each stage-0 GPU).
export VLLM_PP_SEND_BUFFERS=${PP_SEND_BUFFERS:-0}
# Phase 4: decode microbatches M (default = PP size, 2). Each request decodes
# every M steps and M + 1 batches are in flight. 3 fills a pipeline whose
# round trip through the stages exceeds one microbatch step (see
# token_wait / recv_wait in the step profile); TPOT grows with M. Set PP_SEND_BUFFERS
# to M with it.
export VLLM_PP_NUM_MICROBATCHES=${PP_NUM_MICROBATCHES:-0}
# Phase 4: keep the decode microbatches equal in size.
export VLLM_PP_MICROBATCH_BALANCE=${PP_MICROBATCH_BALANCE:-0}
# Phase 2: per-rank step timings every N steps (0 = off).
export VLLM_PP_STEP_PROFILE_INTERVAL=${VLLM_PP_STEP_PROFILE_INTERVAL:-0}

# GPU placement. DP rank d gets visible GPUs [2d, 2d+1] (stage 0, stage 1),
# so with the default order 0..7 the EP groups are {0,2,4,6} and {1,3,5,7}:
# on a two-socket host each spans both NUMA nodes, and the EP all-to-all of
# every MoE layer crosses the socket link. GPU_ORDER=auto (default) puts each
# stage's EP group on one NUMA node / the closest PCIe paths (gpu_order.py);
# only the per-step PP hand-off crosses. An explicit CUDA_VISIBLE_DEVICES or
# GPU_ORDER=default keeps the order as is.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU_ORDER=${GPU_ORDER:-auto}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$GPU_ORDER" = auto ]; then
    CUDA_VISIBLE_DEVICES=$(python "$HERE/gpu_order.py" --pp 2 --dp 4)
    export CUDA_VISIBLE_DEVICES
    echo "GPU order (stage 0 = 1st of each pair): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
# Pin each worker's CPU threads and host memory to its GPU's NUMA node
# (numactl). Stage 0's workers then also first-touch the shared Engram host
# tables on their own node. With --membind those tables (~200 GiB) must fit
# in that node's free memory; set NUMA_BIND=0 if they do not.
NUMA_BIND=${NUMA_BIND:-1}

# The 40 layers split 20/20 by default; pin it so the cut stays on the
# layer 20 KV/index/candidate source boundary.
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-20,20}
export VLLM_USE_V2_MODEL_RUNNER=1

args=(
    --pipeline-parallel-size 2
    --tensor-parallel-size 1
    --data-parallel-size 4
    --data-parallel-backend mp
    # SM120 picks DeepseekV4FlashInferSM120Attention, which accepts only
    # nvfp4_ds_mla: FP8 (MXFP8 record) sliding-window cache + NVFP4 compressed
    # cache. "fp8" aliases to it; "fp8_ds_mla" and "auto"/bf16 are rejected.
    --attention-config "{\"backend\":\"FLASHINFER_MLA_SPARSE_DSV41\",\"indexer_kv_dtype\":\"$INDEXER_KV_DTYPE\"}"
    --kv-cache-dtype nvfp4_ds_mla
    # SM120 kernel pages hold 64 compressed states: 64-token blocks give
    # ratio-1 layers 64-token and ratio-2 layers 128-token pages. This is the
    # block size the SM120 FP4 KV path was validated with.
    --block-size 64
    # NCCL all-gather/reduce-scatter dispatch: works over PCIe. DeepEP, MoRI,
    # NIXL-EP and the FlashInfer NVLink backends need NVLink/NVSHMEM/RDMA.
    --all2all-backend allgather_reducescatter
    --kernel-config "{\"moe_backend\":\"$MOE_BACKEND\"}"
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
if [ "$NUMA_BIND" = 1 ]; then
    args+=(--numa-bind)
fi
if [ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]; then
    args+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
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
