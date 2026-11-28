"""Concrete break-even table for the DP eager fallback.

For each per-rank token vector: graph size G, wasted rows, and the step-time
delta of running eager instead of replaying the padded graph.
"""

from vllm.config.compilation import prefill_cudagraph_capture_grid  # noqa: E402

GRID = prefill_cudagraph_capture_grid(8192)
DP = 4
C_ROW_US = 9.0  # measured: 933.7us AG + 933.6us RS per layer x 20 / 4096 rows
P_MS = 40.0  # measured eager penalty, roughly constant per step


def bucket(n):
    for s in GRID:
        if s >= n:
            return s
    return n


CASES = [
    ("8k prefill on one rank", [8192, 30, 30, 30]),
    ("8k + 3x4k", [8192, 4096, 4096, 4096]),
    ("4k prefill on one rank", [4096, 30, 30, 30]),
    ("2k prefill on one rank", [2048, 30, 30, 30]),
    ("1k prefill on one rank", [1024, 30, 30, 30]),
    ("512 prefill on one rank", [512, 30, 30, 30]),
    ("decode skew 64 vs 1 req", [384, 6, 6, 6]),
    ("balanced 8k", [8192, 8192, 8192, 8192]),
]

print(
    f"c_row={C_ROW_US} us/row, P={P_MS} ms, "
    f"break-even={P_MS * 1000 / C_ROW_US:.0f} wasted rows\n"
)
print(f"{'case':28s} {'G':>5s} {'waste':>7s} {'save_ms':>8s} {'net_ms':>8s}  eager?")
for name, tok in CASES:
    g = bucket(max(tok))
    waste = DP * g - sum(tok)
    save = C_ROW_US * 1e-3 * waste
    net = save - P_MS
    print(
        f"{name:28s} {g:5d} {waste:7d} {save:8.1f} {net:8.1f}  "
        f"{'YES' if net > 0 else 'no'}"
    )
