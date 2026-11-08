# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tabulate a bench_matrix.sh result directory.

    python summarize_results.py results/20261030-120000 --tpot-slo-ms 50

Prints one row per `vllm bench serve` run (concurrency, TTFT, TPOT,
throughput, DSpark acceptance), grouped by configuration and sorted by
concurrency. With ``--tpot-slo-ms`` it also reports, per configuration and
input length, the highest concurrency whose mean TPOT fits the SLO and the
peak output throughput. Last, it prints the final ``PP step profile`` line
each rank logged (from VLLM_PP_STEP_PROFILE_INTERVAL).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import regex as re

COLUMNS = [
    ("mean_ttft_ms", "TTFT ms", "{:.1f}"),
    ("p99_ttft_ms", "p99 TTFT", "{:.1f}"),
    ("mean_tpot_ms", "TPOT ms", "{:.2f}"),
    ("p99_tpot_ms", "p99 TPOT", "{:.2f}"),
    ("output_throughput", "out tok/s", "{:.0f}"),
    ("total_token_throughput", "total tok/s", "{:.0f}"),
    ("spec_decode_acceptance_rate", "accept %", "{:.1f}"),
    ("spec_decode_acceptance_length", "accept len", "{:.2f}"),
]
# bench_matrix.sh names results <config>_c<concurrency>_in<input_len>.json.
RUN_NAME = re.compile(r"^(?P<config>.+)_c(?P<conc>\d+)_in(?P<isl>\d+)$")
PROFILE = re.compile(r"PP step profile (pp_rank=\d+ dp_rank=\d+) over (.*)")


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows
    )


def load_runs(result_dir: Path) -> list[tuple[str, int, int, dict]]:
    runs = []
    for path in result_dir.glob("*.json"):
        m = RUN_NAME.match(path.stem)
        if m is None:
            continue
        result = json.loads(path.read_text())
        conc = int(result.get("max_concurrency") or m["conc"])
        runs.append((m["config"], int(m["isl"]), conc, result))
    runs.sort(key=lambda r: (r[0], r[1], r[2]))
    return runs


def slo_summary(runs, tpot_slo_ms: float) -> list[list[str]]:
    """Per (config, input length): highest concurrency within the TPOT SLO
    and the peak output throughput over the sweep."""
    groups: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for config, isl, conc, result in runs:
        groups[(config, isl)].append((conc, result))
    rows = [
        [
            "config",
            "input",
            f"max conc @ TPOT<={tpot_slo_ms:g}ms",
            "out tok/s there",
            "peak out tok/s",
            "at conc",
        ]
    ]
    for (config, isl), points in sorted(groups.items()):
        if len(points) < 2:
            continue  # A single run (e.g. the long-prompt one) is no sweep.
        fitting = [
            (c, r)
            for c, r in points
            if r.get("mean_tpot_ms") is not None and r["mean_tpot_ms"] <= tpot_slo_ms
        ]
        peak_conc, peak = max(points, key=lambda p: p[1].get("output_throughput", 0))
        if fitting:
            best_conc, best = max(fitting, key=lambda p: p[0])
            within = [str(best_conc), f"{best.get('output_throughput', 0):.0f}"]
        else:
            within = ["none", "-"]
        rows.append(
            [
                config,
                str(isl),
                *within,
                f"{peak.get('output_throughput', 0):.0f}",
                str(peak_conc),
            ]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=None,
        help="report the highest concurrency whose mean TPOT fits this",
    )
    args = parser.parse_args()

    runs = load_runs(args.result_dir)
    if not runs:
        print(f"no results in {args.result_dir}", file=sys.stderr)
        return 1
    rows = [["config", "input", "conc"] + [title for _, title, _ in COLUMNS]]
    for config, isl, conc, result in runs:
        row = [config, str(isl), str(conc)]
        for key, _, fmt in COLUMNS:
            value = result.get(key)
            row.append("-" if value is None else fmt.format(value))
        rows.append(row)
    print(table(rows))

    if args.tpot_slo_ms is not None:
        summary = slo_summary(runs, args.tpot_slo_ms)
        if len(summary) > 1:
            print()
            print(table(summary))

    for log in sorted(args.result_dir.glob("*.server.log")):
        last: dict[str, str] = {}
        for line in log.read_text(errors="replace").splitlines():
            m = PROFILE.search(line)
            if m:
                last[m.group(1)] = m.group(2)
        if last:
            print(f"\n{log.name.removesuffix('.server.log')} (last PP step profile)")
            for rank, text in sorted(last.items()):
                print(f"  {rank}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
