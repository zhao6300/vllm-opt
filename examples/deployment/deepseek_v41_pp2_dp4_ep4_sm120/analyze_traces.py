# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split per-rank GPU time from torch profiler traces into PP/EP categories.

The step profiler (VLLM_PP_STEP_PROFILE_INTERVAL) times whole steps. With
FULL CUDA graphs, only kernel-level traces show how a step divides between EP
all-to-all, PP P2P, attention and MoE. Capture traces with
``PROFILE_DIR=... bash bench_matrix.sh`` (``--profiler-config`` +
``vllm bench serve --profile``), then:

    python analyze_traces.py /tmp/dsv41-traces/pp2_dp4_ep1_dspark1

For each trace this prints the GPU time and share of each category, plus
``busy``: the fraction of the traced window in which any kernel runs.
Categories match kernel names. NCCL SendRecv is the PP activation transfer.
NCCL AllGather/ReduceScatter is the EP dispatch/combine of the
``allgather_reducescatter`` backend; with TP=1 there are no TP collectives.
NCCL Broadcast is the sampled-token and draft relay back to stage 0.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import regex as re

CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("pp_p2p (nccl send/recv)", re.compile(r"nccl.*SendRecv|nccl.*Send|nccl.*Recv")),
    (
        "ep_all2all (nccl allgather/reducescatter)",
        re.compile(r"nccl.*(AllGather|ReduceScatter)"),
    ),
    ("pp_token_relay (nccl broadcast)", re.compile(r"nccl.*Broadcast")),
    ("other_nccl", re.compile(r"nccl")),
    ("indexer", re.compile(r"mqa_logits|indexer|topk", re.I)),
    ("attention", re.compile(r"sparse_mla|SparseMLA|mla|attention|flash", re.I)),
    ("moe", re.compile(r"moe|marlin|m_grouped|fp8_fp4|expert", re.I)),
    ("mhc", re.compile(r"mhc|hc_|sinkhorn", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|nvjet|xmma|matmul", re.I)),
]


def categorize(name: str) -> str:
    for category, pattern in CATEGORIES:
        if pattern.search(name):
            return category
    return "other"


def load_kernels(path: Path) -> list[tuple[float, float, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    kernels = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy"):
            continue
        start = float(event["ts"])
        kernels.append((start, start + float(event.get("dur", 0.0)), event["name"]))
    return kernels


def busy_fraction(kernels: list[tuple[float, float, str]]) -> float:
    if not kernels:
        return 0.0
    intervals = sorted((s, e) for s, e, _ in kernels)
    covered, cur_start, cur_end = 0.0, *intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    covered += cur_end - cur_start
    window = max(e for _, e in intervals) - intervals[0][0]
    return covered / window if window > 0 else 0.0


def summarize(path: Path) -> str:
    kernels = load_kernels(path)
    totals: dict[str, float] = defaultdict(float)
    for start, end, name in kernels:
        totals[categorize(name)] += end - start
    total = sum(totals.values())
    header = (
        f"{path.name}: {len(kernels)} kernels, {total / 1e3:.1f} ms GPU time, "
        f"busy {100 * busy_fraction(kernels):.0f}%"
    )
    lines = [header]
    for category, us in sorted(totals.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"  {category:<44} {us / 1e3:10.2f} ms  {100 * us / max(total, 1e-9):5.1f}%"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="trace files or directories")
    args = parser.parse_args()
    files: list[Path] = []
    for p in map(Path, args.paths):
        if p.is_dir():
            files += sorted(p.rglob("*.pt.trace.json*"))
        else:
            files.append(p)
    if not files:
        print("no *.pt.trace.json[.gz] files found", file=sys.stderr)
        return 1
    for path in files:
        print(summarize(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
