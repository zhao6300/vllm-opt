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
CONCURRENCIES=${CONCURRENCIES:-"1 8 32 64"}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-512}
LONG_INPUT_LEN=${LONG_INPUT_LEN:-32768}
mkdir -p "$RESULT_DIR"

# name EP DSPARK
# EP=0 is left out: without EP every GPU holds all of its stage's routed
# experts (about half of 259.5 GiB MXFP4), which does not fit in 96 GB.
CONFIGS=(
    "pp2_dp4_ep1_dspark0 1 0"
    "pp2_dp4_ep1_dspark1 1 1"
)

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

bench() {
    local name=$1 concurrency=$2 input_len=$3 extra=("${@:4}")
    vllm bench serve \
        --model "$MODEL" --port "$PORT" \
        --dataset-name random \
        --random-input-len "$input_len" --random-output-len "$OUTPUT_LEN" \
        --num-prompts $((concurrency * 8)) --max-concurrency "$concurrency" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${name}_c${concurrency}_in${input_len}.json" \
        "${extra[@]}"
}

for config in "${CONFIGS[@]}"; do
    read -r name ep dspark <<<"$config"
    echo "=== $name ==="
    EP=$ep DSPARK=$dspark PORT=$PORT MODEL=$MODEL \
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
