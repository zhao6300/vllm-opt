#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Phase 0: check the 8-GPU SM120 host before serving DeepSeek V4.1 with
# PP=2 + DP=4 + EP=4 + DSpark. Read-only; changes nothing.
#
#   PYTHON=.venv/bin/python bash check_env.sh
set -uo pipefail

PYTHON=${PYTHON:-python}
NUM_GPUS=${NUM_GPUS:-8}
# Engram tables (~183 GiB FP8 + scales) live in shared pinned host memory.
MIN_SHM_GIB=${MIN_SHM_GIB:-200}
status=0

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "== GPU topology (PCIe/NVLink paths between the 8 GPUs) =="
nvidia-smi topo -m || status=1

echo
echo "== PP stage / EP group placement (serve.sh GPU_ORDER=auto) =="
# Lower intra-group cost is better; an EP group spanning NUMA nodes pays the
# socket link on every MoE layer's all-to-all.
"$PYTHON" "$HERE/gpu_order.py" --pp 2 --dp 4 --explain || status=1
command -v numactl >/dev/null || {
    echo "  numactl not found: serve.sh's --numa-bind (NUMA_BIND=1) needs it"
    status=1
}

echo
echo "== CUDA devices, compute capability, memory =="
"$PYTHON" - "$NUM_GPUS" <<'EOF' || status=1
import sys

import torch

want = int(sys.argv[1])
count = torch.cuda.device_count()
print(f"visible GPUs: {count}")
ok = count >= want
for i in range(count):
    major, minor = torch.cuda.get_device_capability(i)
    total = torch.cuda.get_device_properties(i).total_memory / 2**30
    sm120 = major == 12
    ok &= sm120
    print(
        f"  cuda:{i} {torch.cuda.get_device_name(i)} sm_{major}{minor} "
        f"{total:.1f} GiB {'OK' if sm120 else 'NOT SM120'}"
    )
if count < want:
    print(f"need {want} GPUs")
sys.exit(0 if ok else 1)
EOF

echo
echo "== Peer-to-peer access within EP groups and PP pairs (chosen order) =="
ORDER=${CUDA_VISIBLE_DEVICES:-$("$PYTHON" "$HERE/gpu_order.py" --pp 2 --dp 4)}
# Probe with every GPU visible in PCI order, so CUDA index == physical id.
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$PYTHON" - "$ORDER" <<'EOF' || status=1
import sys

import torch

# Physical GPU ids in visible order; DP rank d's PP chain is entries
# [2d, 2d+1], and each stage's EP group is every other entry.
order = [int(x) for x in sys.argv[1].split(",")]


def p2p(a: int, b: int) -> bool:
    return torch.cuda.can_device_access_peer(a, b)


for d in range(len(order) // 2):
    a, b = order[2 * d], order[2 * d + 1]
    print(f"  DP {d} PP pair GPU{a}->GPU{b}: p2p={p2p(a, b)}")
for stage in (0, 1):
    group = order[stage::2]
    ok = all(p2p(a, b) for a in group for b in group if a != b)
    print(f"  stage {stage} EP group GPUs {group}: all-pairs p2p={ok}")
EOF

echo
echo "== FlashInfer (FLASHINFER_MLA_SPARSE_DSV41 needs an SM120 build) =="
"$PYTHON" -c 'import flashinfer; print("flashinfer", flashinfer.__version__)' \
    || status=1

echo
echo "== Host memory for shared Engram tables =="
shm_gib=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
mem_gib=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
echo "  /dev/shm available: ${shm_gib} GiB, MemAvailable: ${mem_gib} GiB"
if [ "${shm_gib:-0}" -lt "$MIN_SHM_GIB" ]; then
    echo "  /dev/shm is below ${MIN_SHM_GIB} GiB: the DP replicas cannot share" \
        "one host copy of the Engram tables"
    status=1
fi

echo
if [ "$status" -eq 0 ]; then
    echo "All checks passed."
else
    echo "Some checks failed; see above."
fi
exit "$status"
