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

echo "== GPU topology (PCIe/NVLink paths between the 8 GPUs) =="
nvidia-smi topo -m || status=1

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
echo "== Peer-to-peer access between stage-0 and stage-1 GPUs =="
"$PYTHON" - <<'EOF' || status=1
import torch

n = torch.cuda.device_count()
# Global rank r = dp * 2 + pp sits on GPU r: PP pairs are (0,1), (2,3), ...
# and each EP group is {0,2,4,6} or {1,3,5,7}.
pairs = [(2 * d, 2 * d + 1) for d in range(n // 2)]
for a, b in pairs:
    print(f"  PP pair GPU{a}<->GPU{b}: p2p={torch.cuda.can_device_access_peer(a, b)}")
for stage in (0, 1):
    group = list(range(stage, n, 2))
    p2p = all(
        torch.cuda.can_device_access_peer(a, b) for a in group for b in group if a != b
    )
    print(f"  EP group stage {stage} GPUs {group}: all-pairs p2p={p2p}")
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
