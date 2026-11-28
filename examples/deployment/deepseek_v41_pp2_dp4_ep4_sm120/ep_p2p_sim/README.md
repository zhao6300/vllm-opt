# EP 4-rank collective: coll (SHM) vs P2P

Supporting scripts for the `isAllDirectP2p 0` finding in the live
PP2 x TP2 x DP2 + EP4 deployment. The EP all-gather / reduce-scatter run on a
4-rank NCCL **coll** comm, which NCCL puts on `SHM/direct`; the 2-rank
TP/PP/DP comms report `isAllDirectP2p 1` and use `P2P/IPC`. This directory
measures whether a P2P implementation is faster, and which NCCL knob (if any)
flips the coll comm to P2P.

## What the script measures

`p2p_ep_bench.py` creates a 4-rank NCCL comm on 4 GPUs and benchmarks, for
rows = 4 / 28 / 1024 / 4096 (decode and prefill shapes from the live log):

| column | implementation | transport |
| --- | --- | --- |
| `AG coll` | `dist.all_gather_into_tensor` | coll -> `SHM/direct` |
| `AG p2p` | `batch_isend_irecv`: send local rows to every peer | p2p -> `P2P/IPC` |
| `AG a2a` | replicated `all_to_all_single` (input copied world times) | p2p -> `P2P/IPC` |
| `AG ring` | ring all-gather via `batch_isend_irecv` | p2p -> `P2P/IPC` |
| `RS coll` | `dist.reduce_scatter_tensor` | coll -> `SHM/direct` |
| `RS a2a_sum` | `all_to_all_single` + fixed-order fp32 sum | p2p -> `P2P/IPC` |

`AG p2p` is the cheapest replacement for `AgRsAll2AllManager._gather_rows`:
no input copy, 3 sends + 3 receives. `RS a2a_sum` is what
`VLLM_MOE_AG_RS_FIXED_ORDER_COMBINE=1` already does for the combine; the
script uses torch's fp32 sum, while vLLM uses a Triton kernel, so treat it as
an upper bound.

The script uses `batch_isend_irecv` for readability. The production path must
use `pynccl_comm.all_to_all_single` (`pynccl.py:507`), which wraps the same
`ncclSend`/`ncclRecv` in `ncclGroupStart/End` and is CUDA-graph capturable.

`--check` verifies the P2P all-gathers are bit-identical to the coll one and
the P2P combine matches the coll reduce-scatter within bf16 tolerance.

## Run

The live server uses all 8 GPUs. Run this on 4 free GPUs (or stop the server):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
    /opt/venv/bin/python -I p2p_ep_bench.py --rows 4 28 1024 4096 --check
```

Sweep the NCCL knobs:

```bash
GPUS=0,1,2,3 bash sweep.sh
```

The sweep runs `baseline`, `NCCL_P2P_LEVEL=PHB`, `NCCL_P2P_LEVEL=SYS`,
`NCCL_P2P_DIRECT_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, and
`NCCL_P2P_LEVEL=SYS NCCL_P2P_DIRECT_DISABLE=1`, and prints the
`Check P2P Type` / `Channel ... via` lines for each.

## Findings (2026-11-28, GPUs 4-7 while the live server was running)

`NCCL_P2P_LEVEL=PHB` or `SYS` flips the 4-rank comm from `isAllDirectP2p 0`
(`SHM/direct`) to `isAllDirectP2p 1` (`P2P/CUMEM`). `NCCL_P2P_DIRECT_DISABLE=1`
and `NCCL_CUMEM_ENABLE=0` do not. With `SYS`, no `SHM/direct` line remains.

P2P/CUMEM is faster at every size (us, min of 50; contended by the live
server, so read the ratios, not the absolute values):

| rows | AG coll SHM | AG coll P2P | RS coll SHM | RS coll P2P |
| --- | --- | --- | --- | --- |
| 4 | 25.7 | 23.4 | 25.6 | 21.9 |
| 28 | 53.2 | 42.9 | 54.9 | 43.9 |
| 1024 | 933.7 | 729.4 | 938.8 | 740.2 |
| 4096 | 3645.1 | 2880.2 | 3646.8 | 2934.7 |

That is -9%/-14% at 4 rows, -19%/-20% at 28, and about -21% at 1024/4096.
The 1024-row SHM all-gather (933.7 us) matches the earlier idle
`dp_padding_sim/nccl_bench.py` number, so the min-of-50 estimator is sound.

Recommendation: restart the live server with `NCCL_P2P_LEVEL=SYS` (not `PHB`:
the PP hand-off is cross-socket, and `SYS` keeps it on P2P). Then check the
log for `isAllDirectP2p 1` and no `via SHM`, and re-run the determinism and
bad-answer checks before trusting the step-time win.

## How to read it

- `isAllDirectP2p 0` on the 4-rank comm -> coll is on SHM. If a knob makes it
  `1` and the `Channel` lines say `via P2P/IPC`, the knob is a candidate.
- Compare `AG coll` vs `AG p2p` at 4 rows (decode latency) and 4096 rows
  (prefill bandwidth). The P2P path can win on small messages and lose on
  large ones; NCCL's SHM heuristic exists for a reason.
- The host has no PCIe switch (`lspci -tv`: every GPU on its own root port),
  so all pairs are PHB/SYS. NCCL's coll "direct" check may not be satisfiable
  without `NCCL_TOPO_FILE`.

## If no knob works: the code path

`AgRsAll2AllManager` already has a P2P building block:
`pynccl.all_to_all_single` (`pynccl.py:507`) uses `ncclSend`/`ncclRecv`,
which run on p2p channels. `_reduce_scatterv_fixed_order` (`all2all.py:156`)
uses it for the combine behind `VLLM_MOE_AG_RS_FIXED_ORDER_COMBINE=1`.

For the dispatch, replace `_gather_rows`'s `all_gatherv` with either:

```python
# 3 sends + 3 receives, no input copy
ops = []
for peer in range(world):
    if peer == rank:
        continue
    ops.append(P2POp(isend, packed, peer))
    ops.append(P2POp(irecv, out[peer * rows : (peer + 1) * rows], peer))
batch_isend_irecv(ops)
```

or the replicated `all_to_all_single` (needs `packed` copied `world` times).
The dispatch is a pure copy, so it stays bit-identical; the combine's
all-to-all + sum changes the reduction order and is the determinism path
(`VLLM_MOE_AG_RS_FIXED_ORDER_COMBINE`), not a drop-in for NCCL RS.

Measure both before changing the default: for 288 KB messages the 6 p2p ops
may cost more than one coll kernel, and for 4096-row prefill SHM may already
be at the PCIe Gen5 x16 limit.
