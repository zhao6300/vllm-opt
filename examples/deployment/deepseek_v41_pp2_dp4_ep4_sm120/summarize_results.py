# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tabulate a bench_matrix.sh result directory.

    python summarize_results.py results/20261030-120000

Prints one row per `vllm bench serve` run (TTFT, TPOT, throughput, DSpark
acceptance), then the last ``PP step profile`` line each rank logged per
configuration (from VLLM_PP_STEP_PROFILE_INTERVAL).
"""

import argparse
import json
import re
import sys
from pathlib import Path

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
PROFILE = re.compile(r"PP step profile (pp_rank=\d+ dp_rank=\d+) over (.*)")


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    rows = [["run"] + [title for _, title, _ in COLUMNS]]
    for path in sorted(args.result_dir.glob("*.json")):
        result = json.loads(path.read_text())
        row = [path.stem]
        for key, _, fmt in COLUMNS:
            value = result.get(key)
            row.append("-" if value is None else fmt.format(value))
        rows.append(row)
    if len(rows) == 1:
        print(f"no results in {args.result_dir}", file=sys.stderr)
        return 1
    print(table(rows))

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
