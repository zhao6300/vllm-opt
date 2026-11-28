"""Focused comparison of the DP LB settings with more seeds, splitting the
short-request tail from the long-request tail."""

import importlib.util
import statistics
from pathlib import Path

spec = importlib.util.spec_from_file_location("sim", Path(__file__).with_name("sim.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CONFIGS = [
    ("off", m.Cfg(token_aware=False)),
    ("default T2048 cap4",
     m.Cfg(token_aware=True, tokens_per_request=2048, max_pending=4)),
    ("T2048 cap16", m.Cfg(token_aware=True, tokens_per_request=2048, max_pending=16)),
    ("T2048 capinf", m.Cfg(token_aware=True, tokens_per_request=2048, max_pending=1e9)),
    ("T8192 cap8", m.Cfg(token_aware=True, tokens_per_request=8192, max_pending=8)),
    ("T16384 cap4", m.Cfg(token_aware=True, tokens_per_request=16384, max_pending=4)),
    ("T16384 capinf",
     m.Cfg(token_aware=True, tokens_per_request=16384, max_pending=1e9)),
    ("budget600 cap4", m.Cfg(token_aware=True, budget_ms=600, max_pending=4)),
    ("budget600 cap16", m.Cfg(token_aware=True, budget_ms=600, max_pending=16)),
]

WLS = ["staggered", "burst", "single_long", "online", "online_hot"]
SEEDS = {"staggered": 4, "burst": 6, "single_long": 4, "online": 8, "online_hot": 8}


def run(cfg, wl, seeds):
    out = [m.Sim(cfg, s, m.WORKLOADS[wl]).run() for s in range(seeds)]
    def mean(k):
        return statistics.mean(r[k] for r in out)
    def rng(k):
        xs = [r[k] for r in out]
        return max(xs) - min(xs)
    keys = [k for k, v in out[0].items() if isinstance(v, (int, float))]
    return {k: mean(k) for k in keys} | {k + "_rng": rng(k) for k in keys}


for wl in WLS:
    print(f"\n===== {wl} (seeds={SEEDS[wl]}) =====")
    print(f"{'config':20s} {'makespan':>9s} {'short_p50':>10s} {'short_p99':>10s} "
          f"{'long_p99':>9s} {'imbal':>6s}")
    for name, cfg in CONFIGS:
        r = run(cfg, wl, SEEDS[wl])
        print(f"{name:20s} {r['makespan']:9.2f} {r['short_p50']:10.1f} "
              f"{r['short_p99']:10.1f} {r['long_p99']:9.1f} {r['imbalance']:6.2f}")
