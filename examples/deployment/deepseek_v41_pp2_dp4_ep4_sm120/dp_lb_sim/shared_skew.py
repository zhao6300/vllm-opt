"""Shared-load benefit under API-server arrival skew."""

import importlib.util
import statistics
from pathlib import Path

spec = importlib.util.spec_from_file_location("sim", Path(__file__).with_name("sim.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CASES = [
    ("skew_burst", dict(kind="burst", count=48, rate=0.0, gap=0.0,
                        sizes=[1000, 2000, 4000, 8000, 16000, 32000] * 8,
                        output=300, hit_frac=0.0, long_frac=0.0, median=2000)),
    ("skew_online", dict(kind="online", count=1200, rate=3.0, gap=0.0, output=300,
                         hit_frac=0.85, long_frac=0.04, median=2000)),
]

for name, wl in CASES:
    m.WORKLOADS[name] = wl
    print(f"\n===== {name} =====")
    print(f"{'config':28s} {'makespan':>9s} {'short_p99':>10s} {'imbal':>6s}")
    for servers, shared, skew in [
        (1, False, 0.0),
        (4, False, 0.0),
        (4, True, 0.0),
        (4, False, 0.75),
        (4, True, 0.75),
        (8, False, 0.75),
        (8, True, 0.75),
    ]:
        cfg = m.Cfg(token_aware=True, tokens_per_request=2048, max_pending=16,
                    servers=servers, shared=shared, server_skew=skew)
        out = [m.Sim(cfg, s, wl).run() for s in range(6)]
        mk = statistics.mean(r["makespan"] for r in out)
        sp = statistics.mean(r["short_p99"] for r in out)
        im = statistics.mean(r["imbalance"] for r in out)
        label = f"srv={servers} shared={int(shared)} skew={skew:g}"
        print(f"{label:28s} {mk:9.2f} {sp:10.1f} {im:6.2f}")
