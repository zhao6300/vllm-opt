#!/usr/bin/env bash
# Sweep the NCCL knobs that decide coll P2P vs SHM for the 4-rank EP comm.
#
#   GPUS=0,1,2,3 bash sweep.sh
#
# Each run writes a full NCCL_DEBUG=INFO log under $OUT and prints the
# transport lines plus the benchmark table. The live server uses all 8 GPUs;
# run this only when they are free.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPUS=${GPUS:-0,1,2,3}
ROWS=${ROWS:-"4 28 1024 4096"}
PY=${PY:-/opt/venv/bin/python}
OUT=${OUT:-/tmp/ep_p2p_sim}
mkdir -p "$OUT"

run() {
    local name=$1
    shift
    local log="$OUT/$name.log"
    echo "=== $name ==="
    if ! env CUDA_VISIBLE_DEVICES="$GPUS" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
        "$@" "$PY" -I "$HERE/p2p_ep_bench.py" --rows $ROWS --check \
        >"$log" 2>&1; then
        echo "FAILED (see $log)"
        tail -5 "$log"
        return
    fi
    grep -E "^rows|^ *[0-9]+ " "$log" || true
    echo "-- transport --"
    grep -E "Check P2P Type|Channel .* via" "$log" \
        | sed -E 's/.*(Check P2P Type.*|Channel .* via .*)/\1/' \
        | sort -u | head -8
    echo
}

run baseline
run level_phb NCCL_P2P_LEVEL=PHB
run level_sys NCCL_P2P_LEVEL=SYS
run direct_disable NCCL_P2P_DIRECT_DISABLE=1
run cumem_off NCCL_CUMEM_ENABLE=0
run sys_direct NCCL_P2P_LEVEL=SYS NCCL_P2P_DIRECT_DISABLE=1
