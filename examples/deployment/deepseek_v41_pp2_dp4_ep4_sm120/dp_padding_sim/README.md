# DP padding eager-fallback simulation

Supporting scripts for [TIMING.md §5.17](../TIMING.md#517-dp-padding-的-eager-fallback模拟与盈亏平衡2026-11-27).
They reproduce the numbers behind the proposed `VLLM_DP_EAGER_PAD_TOKENS` rule:
a PIECEWISE step should run eager when `dp*G - sum(real)` wasted rows exceed
`P / c_row`.

Run from the repo root with the repo venv (`.venv/bin/python` or `/opt/venv/bin/python`):

| script | what it does |
| --- | --- |
| `sim_logic.py` | Runs the real `sync_cudagraph_and_dp_padding` with a fake manager and a mocked DP group, one call per rank. Checks that graph steps pad every rank to one G, an eager vote makes the whole group eager at real sizes, and every rank agrees. |
| `scenarios.py` | Concrete break-even table for per-rank token vectors (8k/4k/2k/1k prefill on one rank, decode skew, balanced). |
| `sim_workload.py` | Discrete-event mixed workload (Poisson arrivals, least-loaded admission, prefill chunks) + the graph-vs-eager cost model; sweeps the threshold and compares absolute / ratio / min-token rules. `--trigger-table` reports the trigger rate for `dp × chunk cap` (a one-sided step wastes `(dp-1) * chunk` rows), and `--prompt-len` models a burst of long prompts. |
| `nccl_bench.py` | 4-rank NCCL all-gather / reduce-scatter / ragged-broadcast at 28..16384 rows, to measure `c_row`. Run with `CUDA_VISIBLE_DEVICES=0,1,2,3`. Large sizes are only meaningful when the GPUs are otherwise idle. |

The cost constants used in the doc come from `nccl_bench.py` (1024-row point:
933.7 µs all-gather + 933.6 µs reduce-scatter per layer, 44.9 GB/s) and the
eager-vs-graph measurements in TIMING.md §5.7 (P ≈ 40–45 ms, roughly constant).
