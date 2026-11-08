# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick CUDA_VISIBLE_DEVICES so each PP stage's EP group is topologically close.

vLLM orders ranks DP x PP x TP and gives DP rank d the visible GPUs
[d * pp, (d + 1) * pp). With PP=2, DP=4 and the default order 0..7, stage 0
is GPUs {0, 2, 4, 6} and stage 1 is {1, 3, 5, 7}. On a two-socket PCIe host
(GPUs 0-3 on NUMA 0, 4-7 on NUMA 1) every EP group then spans both sockets,
and the per-layer EP all-to-all crosses the inter-socket link.

This script splits the GPUs into one group per PP stage. Each group stays in
one NUMA node where possible, with the closest PCIe paths within it, because
EP all-to-all runs every MoE layer. The PP hand-off (one hidden_states +
pre_mix transfer per step) is the traffic left to cross groups. It then
prints the interleaved order that maps group s to stage s:

    CUDA_VISIBLE_DEVICES=$(python gpu_order.py)       # e.g. 0,4,1,5,2,6,3,7
    python gpu_order.py --explain                      # layout and costs

Set CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA indices match nvidia-smi's.
Topology comes from ``nvidia-smi topo -m`` (PCIe link types) and sysfs
(``/sys/bus/pci/devices/<bus id>/numa_node``). ``--topo-file`` and ``--numa``
replace them for offline checks.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import regex as re

# Relative cost of a GPU pair's path (nvidia-smi topo -m legend): NVLink, a
# single PCIe switch, multiple switches, the host bridge, across PCIe host
# bridges within a NUMA node, across NUMA nodes (SMP interconnect).
LINK_COST = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 8}
UNKNOWN_COST = 4


def link_cost(token: str) -> int:
    if token.startswith("NV"):
        return LINK_COST["NV"]
    return LINK_COST.get(token, UNKNOWN_COST)


def parse_topo(text: str) -> dict[tuple[int, int], int]:
    """Pairwise link costs between GPUs from ``nvidia-smi topo -m``."""
    rows: dict[int, list[str]] = {}
    for line in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)  # Strip colour codes.
        m = re.match(r"^GPU(\d+)\s+(.*)$", line.strip())
        if m:
            rows[int(m.group(1))] = m.group(2).split()
    gpus = sorted(rows)
    costs = {}
    for i in gpus:
        for col, j in enumerate(gpus):
            if i != j and col < len(rows[i]):
                costs[(i, j)] = link_cost(rows[i][col])
    return costs


def numa_nodes_from_sysfs() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    nodes = {}
    for line in out.strip().splitlines():
        index, bus_id = (s.strip() for s in line.split(","))
        # nvidia-smi: 00000000:17:00.0; sysfs: 0000:17:00.0
        sysfs_id = bus_id.lower()[-12:]
        path = Path(f"/sys/bus/pci/devices/{sysfs_id}/numa_node")
        node = int(path.read_text()) if path.exists() else -1
        nodes[int(index)] = max(node, 0)
    return nodes


def pair_cost(a: int, b: int, costs, numa: dict[int, int]) -> int:
    if (a, b) in costs:
        return costs[(a, b)]
    return LINK_COST["SYS"] if numa.get(a) != numa.get(b) else UNKNOWN_COST


def group_cost(group, costs, numa) -> int:
    return sum(
        pair_cost(a, b, costs, numa) for a, b in itertools.combinations(group, 2)
    )


def partitions(items: list[int], size: int):
    """All splits of ``items`` into groups of ``size``, without reorderings."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for others in itertools.combinations(rest, size - 1):
        group = (first, *others)
        remaining = [x for x in rest if x not in others]
        for tail in partitions(remaining, size):
            yield [group, *tail]


def best_layout(gpus: list[int], pp: int, dp: int, costs, numa):
    """Stage groups minimising intra-group (EP) cost, then DP pairing along
    the cheapest PP links. Returns (stage_groups, order)."""
    assert len(gpus) == pp * dp, f"need {pp * dp} GPUs, got {len(gpus)}"
    best = min(
        partitions(sorted(gpus), dp),
        key=lambda groups: (
            sum(group_cost(g, costs, numa) for g in groups),
            [sorted(g) for g in groups],
        ),
    )
    # Stage 0 takes the group with the lowest GPU (and NUMA node) id.
    stages = sorted((sorted(g) for g in best), key=lambda g: (numa.get(g[0], 0), g[0]))
    # DP rank d's PP chain: stage 0 GPU d, then the closest unused GPU of
    # each next stage.
    chains = [[g] for g in stages[0]]
    for stage in stages[1:]:
        free = list(stage)
        for chain in chains:
            nxt = min(free, key=lambda b: (pair_cost(chain[-1], b, costs, numa), b))
            free.remove(nxt)
            chain.append(nxt)
    order = [gpu for chain in chains for gpu in chain]
    return [[c[s] for c in chains] for s in range(pp)], order


def describe(name, stage_groups, costs, numa) -> str:
    lines = [name]
    for s, group in enumerate(stage_groups):
        nodes = sorted({numa.get(g, 0) for g in group})
        lines.append(
            f"  stage {s} (EP group) GPUs {group}: NUMA {nodes}, "
            f"intra-group cost {group_cost(group, costs, numa)}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--topo-file", help="saved `nvidia-smi topo -m` output")
    parser.add_argument("--numa", help="comma-separated NUMA node per GPU index")
    args = parser.parse_args()

    if args.topo_file:
        topo = Path(args.topo_file).read_text()
    else:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
        ).stdout
    costs = parse_topo(topo)
    if args.numa:
        numa = {i: int(n) for i, n in enumerate(args.numa.split(","))}
    else:
        numa = numa_nodes_from_sysfs()

    n = args.pp * args.dp
    gpus = sorted(numa)[:n]
    if len(gpus) < n:
        print(f"need {n} GPUs, found {len(gpus)}", file=sys.stderr)
        return 1
    stage_groups, order = best_layout(gpus, args.pp, args.dp, costs, numa)
    if args.explain:
        default = [
            [gpus[d * args.pp + s] for d in range(args.dp)] for s in range(args.pp)
        ]
        print(
            describe("default order " + ",".join(map(str, gpus)), default, costs, numa)
        )
        print(
            describe(
                "chosen order " + ",".join(map(str, order)), stage_groups, costs, numa
            )
        )
        for d in range(args.dp):
            chain = order[d * args.pp : (d + 1) * args.pp]
            cost = sum(pair_cost(a, b, costs, numa) for a, b in zip(chain, chain[1:]))
            print(f"  DP rank {d} PP chain {chain}: link cost {cost}")
    print(",".join(map(str, order)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
