#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Phase 2: baseline measurements for PP=2 + DP=4 on 8 SM120 GPUs.
# For each configuration this starts serve.sh, waits for /health, runs
# `vllm bench serve` (TTFT, TPOT, throughput; DSpark acceptance when on) at
# each concurrency, and stops the server.
#
#   MODEL=/models/DeepSeek-V4.1-Flash bash bench_matrix.sh
# Add PROFILE_DIR=/tmp/dsv41-traces for one profiled run per configuration.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:?set MODEL}
PORT=${PORT:-8000}
RESULT_DIR=${RESULT_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
# Total concurrency across the 4 DP ranks. The sweep crosses the points where
# behaviour is expected to change: ~80 (about 20 requests per rank, where the
# SM120 FlashInfer decode plan's 64-row limit is passed with 6 tokens per
# request) and ~680 (about 170 per rank, the default CUDA graph ceiling).
CONCURRENCIES=${CONCURRENCIES:-"1 8 16 32 64 128 256 512 1024"}
# Server limits for the sweep, so admission does not cap it first:
# MAX_NUM_SEQS is per DP rank (x4 total). The capture size covers 1024 tokens
# per microbatch (~170 requests with 6 tokens each).
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-1024}
# Requests per run: enough to reach steady state without hours at c=1024.
PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY:-4}
MIN_PROMPTS=${MIN_PROMPTS:-32}
# Passed to summarize_results.py: highest concurrency whose mean TPOT fits.
TPOT_SLO_MS=${TPOT_SLO_MS:-50}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-512}
LONG_INPUT_LEN=${LONG_INPUT_LEN:-32768}
mkdir -p "$RESULT_DIR"

# Per-rank step timings in every server log (read by summarize_results.py).
export VLLM_PP_STEP_PROFILE_INTERVAL=${VLLM_PP_STEP_PROFILE_INTERVAL:-200}

# name EP DSPARK PP_SEND_BUFFERS PP_MICROBATCH_BALANCE PP_NUM_MICROBATCHES
# C/D are the plan's baselines; E1-E3 add the phase 3/4 PP options to D one at
# a time and together; E4 adds three decode microbatches (M=3, four batches in
# flight). EP=0 is left out: without EP every GPU holds all of its stage's
# routed experts (about half of 259.5 GiB MXFP4), which does not fit in 96 GB.
CONFIGS=(
    "C_ep1_dspark0 1 0 0 0 0"
    "D_ep1_dspark1 1 1 0 0 0"
    "E1_dspark1_sendbuf2 1 1 2 0 0"
    "E2_dspark1_balance 1 1 0 1 0"
    "E3_dspark1_sendbuf2_balance 1 1 2 1 0"
    "E4_dspark1_m3_sendbuf3_balance 1 1 3 1 3"
)
if [ -n "${ONLY:-}" ]; then
    # e.g. ONLY="D_ep1_dspark1 E3_dspark1_sendbuf2_balance"
    filtered=()
    for config in "${CONFIGS[@]}"; do
        for want in $ONLY; do
            [ "${config%% *}" = "$want" ] && filtered+=("$config")
        done
    done
    CONFIGS=("${filtered[@]}")
fi

wait_ready() {
    local pid=$1
    for _ in $(seq 1 360); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server exited during startup" >&2
            return 1
        fi
        sleep 10
    done
    echo "server not ready after 1h" >&2
    return 1
}

num_prompts() {
    local n=$(($1 * PROMPTS_PER_CONCURRENCY))
    echo $((n > MIN_PROMPTS ? n : MIN_PROMPTS))
}

bench() {
    local name=$1 concurrency=$2 input_len=$3 extra=("${@:4}")
    vllm bench serve \
        --model "$MODEL" --port "$PORT" \
        --dataset-name random \
        --random-input-len "$input_len" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$(num_prompts "$concurrency")" --max-concurrency "$concurrency" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${name}_c${concurrency}_in${input_len}.json" \
        "${extra[@]}"
}

for config in "${CONFIGS[@]}"; do
    read -r name ep dspark send_buffers balance microbatches <<<"$config"
    echo "=== $name ==="
    EP=$ep DSPARK=$dspark PORT=$PORT MODEL=$MODEL \
        PP_SEND_BUFFERS=$send_buffers PP_MICROBATCH_BALANCE=$balance \
        PP_NUM_MICROBATCHES=$microbatches \
        PROFILE_DIR=${PROFILE_DIR:+$PROFILE_DIR/$name} \
        bash "$HERE/serve.sh" >"$RESULT_DIR/$name.server.log" 2>&1 &
    pid=$!
    if wait_ready "$pid"; then
        grep "Parallel topology" "$RESULT_DIR/$name.server.log" \
            >"$RESULT_DIR/$name.topology.txt" || true
        for c in $CONCURRENCIES; do
            bench "$name" "$c" "$INPUT_LEN"
        done
        # Long prompts: prefill throughput and the 19->20 activation transfer.
        bench "$name" 4 "$LONG_INPUT_LEN"
        if [ -n "${PROFILE_DIR:-}" ]; then
            bench "${name}_profiled" 8 "$INPUT_LEN" --profile
        fi
    fi
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
done

echo "Results in $RESULT_DIR"
python "$HERE/summarize_results.py" "$RESULT_DIR" --tpot-slo-ms "$TPOT_SLO_MS" || true
if [ -n "${PROFILE_DIR:-}" ]; then
    python "$HERE/analyze_traces.py" "$PROFILE_DIR" || true
fi
