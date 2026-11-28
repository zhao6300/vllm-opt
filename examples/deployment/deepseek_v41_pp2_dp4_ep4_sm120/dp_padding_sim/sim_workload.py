"""Discrete-event DP=4 workload sim + graph-vs-eager padding cost model.

Purpose: verify the claim that the DP padding waste is the big lever, and test
whether a single absolute waste threshold is enough to decide when to run a
PIECEWISE step eager instead of replaying the padded graph.

Workload model (calibrated to the measured DP2/DP4 behaviour in TIMING.md):
  * 4 DP engines, synchronous steps (EP group is lockstep).
  * Poisson arrivals, 1k-4k-token prompts, ~300-token outputs.
  * least-loaded admission (the core_client rule), no migration.
  * each step: decode first (6 rows/request, DSpark 5 drafts), then prefill
    from the waiting queue up to max_num_batched_tokens=8192.
  * step time = max_i(compute_i) + c_row * (rows the collective moves) + fixed.

Cost model (per step, 20 MoE layers per PP stage):
  graph: collective rows = dp * G,  G = smallest captured prefill size >= max r_i
  eager: collective rows = sum(r_i), plus a constant eager penalty P
  delta = c_row * (dp*G - sum(r_i)) - P        (>0 means eager wins)

Measured anchors (TIMING.md):
  * equal-length 4-rank dispatch+combine: 54.8 us/layer at 28 rows/rank
    -> 1.1 ms/step at 20 layers (latency-dominated at that size).
  * EP collectives are 17% of a ~20 ms decode step.
  * eager prefill is ~+5% at 8k (965 vs 920 ms) and +42 ms/stage at 1 token,
    i.e. P is roughly constant per step (~40 ms), launch-overhead dominated.
  * the marginal per-row collective cost is the uncertain quantity; we sweep
    c_row over 2-11 us/row (the range implied by the measurements).
"""

import argparse
import math
import random
from dataclasses import dataclass, field

from vllm.config.compilation import prefill_cudagraph_capture_grid  # noqa: E402

GRID = prefill_cudagraph_capture_grid(8192)
MAX_BATCHED = 8192
MAX_SEQS = 64
DRAFT_ROWS = 6  # DSpark: 1 + 5 drafts
ACCEPT_PER_STEP = 2.0  # measured ~2.1 tokens/iteration
PREFILL_MS_PER_TOKEN = 0.085
DECODE_MS_PER_REQ = 0.5
FIXED_MS = 10.0
DP = 4


def bucket(n: int) -> int:
    for size in GRID:
        if size >= n:
            return size
    return n


@dataclass
class Req:
    prompt_left: int
    output_left: float


@dataclass
class Engine:
    running: list = field(default_factory=list)
    waiting: list = field(default_factory=list)

    def load(self) -> int:
        return len(self.running) + len(self.waiting)


def simulate(
    seed,
    duration_s,
    rate,
    c_row_us,
    eager_penalty_ms,
    ragged_factor=1.0,
    prefill_share=1.0,
    dp=DP,
    chunk_cap=None,
    prompt_len=None,
):
    """Return per-step records: (tokens_per_rank, G, waste, delta_ms).

    `dp` is the DP group size. `chunk_cap` bounds the prefill tokens one rank
    takes per step, as the scheduler's cost model and
    `VLLM_PREFILL_FILL_MIN_TOKENS` do; None fills to the batch budget.
    `prompt_len` fixes every prompt, to model a burst of long prompts; None
    draws 1k-4k.
    """
    rng = random.Random(seed)
    engines = [Engine() for _ in range(dp)]
    arrivals = []
    t = 0.0
    while t < duration_s:
        t += rng.expovariate(rate)
        if t >= duration_s:
            break
        if prompt_len is not None:
            prompt = prompt_len
        else:
            prompt = int(min(4096, max(1024, rng.lognormvariate(math.log(2000), 0.35))))
        output = max(1.0, rng.expovariate(1 / 300.0))
        arrivals.append((t, prompt, output))
    arrivals.sort()
    ai = 0
    now = 0.0
    records = []
    while now < duration_s:
        while ai < len(arrivals) and arrivals[ai][0] <= now:
            _, prompt, output = arrivals[ai]
            ai += 1
            e = min(range(dp), key=lambda i: engines[i].load())
            engines[e].waiting.append(Req(prompt, output))
        tokens, prefills, decodes, finishing = [], [], [], []
        for eng in engines:
            budget = MAX_BATCHED
            decode_reqs = min(len(eng.running), MAX_SEQS)
            decode_rows = decode_reqs * DRAFT_ROWS
            budget -= decode_rows
            prefill = 0
            done = []
            if rng.random() < prefill_share:
                while eng.waiting and budget > 0:
                    req = eng.waiting[0]
                    take = min(req.prompt_left, budget)
                    if chunk_cap is not None:
                        take = min(take, chunk_cap)
                    req.prompt_left -= take
                    prefill += take
                    budget -= take
                    if req.prompt_left == 0:
                        done.append(eng.waiting.pop(0))
                    else:
                        break
            tokens.append(decode_rows + prefill)
            prefills.append(prefill)
            decodes.append(decode_reqs)
            finishing.append(done)
        compute = [
            PREFILL_MS_PER_TOKEN * p + DECODE_MS_PER_REQ * d + 3.0
            for p, d in zip(prefills, decodes)
        ]
        g = bucket(max(tokens))
        graph_rows = dp * g
        real_rows = sum(tokens)
        step_ms = max(compute) + c_row_us * 1e-3 * graph_rows + FIXED_MS
        delta = (
            c_row_us * 1e-3 * graph_rows
            - c_row_us * 1e-3 * ragged_factor * real_rows
            - eager_penalty_ms
        )
        records.append((tuple(tokens), g, graph_rows - real_rows, delta))
        for eng, done in zip(engines, finishing):
            eng.running.extend(done)
            still = []
            for req in eng.running:
                req.output_left -= ACCEPT_PER_STEP
                if req.output_left > 0:
                    still.append(req)
            eng.running = still
        now += step_ms / 1000.0
    return records


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[k]


def rule_net(records, pred):
    sel = [d for (tok, g, w, d) in records if pred(tok, g, w)]
    net = sum(sel)
    regret = -sum(d for d in sel if d < 0)
    trig = len(sel) / len(records)
    return net, regret, trig


def trigger_table(args):
    """Trigger rate of an absolute threshold as a function of the prefill chunk.

    The live server's chunk is set by the cost model (7104 tokens on the env
    prior, the 2048 floor once fitted), not by the batch budget, and a burst of
    long prompts keeps one rank prefilling while the other has nothing. Both
    change the waste distribution far more than the arrival process does.
    """
    print(
        f"c_row={args.c_row} us/row, P={args.penalty} ms, T={args.threshold} rows, "
        f"prompt_len={args.prompt_len or '1k-4k'}, rate={args.rate}/s"
    )
    print(
        f"{'dp':>2} {'chunk':>6} {'steps':>6} {'trigger%':>8} {'net_ms':>8} "
        f"{'waste_p50':>9} {'waste_p90':>9}"
    )
    for dp in args.dps:
        for cap in args.caps:
            recs = []
            for s in range(args.seeds):
                recs += simulate(
                    s,
                    args.duration,
                    args.rate,
                    args.c_row,
                    args.penalty,
                    dp=dp,
                    chunk_cap=cap,
                    prompt_len=args.prompt_len,
                )
            net, _, trig = rule_net(recs, lambda tok, g, w, T=args.threshold: w >= T)
            wastes = [r[2] for r in recs]
            print(
                f"{dp:2d} {cap:6d} {len(recs):6d} {trig * 100:7.1f}% {net:8.0f} "
                f"{pct(wastes, 50):9.0f} {pct(wastes, 90):9.0f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--trigger-table", action="store_true")
    ap.add_argument("--dps", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--caps", type=int, nargs="+", default=[2048, 4608, 7104, 8192])
    ap.add_argument("--threshold", type=int, default=5500)
    ap.add_argument("--c-row", type=float, default=9.0)
    ap.add_argument("--penalty", type=float, default=40.0)
    ap.add_argument("--prompt-len", type=int, default=None)
    args = ap.parse_args()

    if args.trigger_table:
        trigger_table(args)
        return

    print(
        f"grid: {len(GRID)} prefill sizes, max gap "
        f"{max(b - a for a, b in zip(GRID, GRID[1:]))}"
    )
    # c_row measured from the 1024-row 4-rank all-gather + reduce-scatter:
    # 933.7us + 933.6us per layer x 20 layers / (4*1024 rows) = 9.1 us/row.
    for c_row in (4.0, 6.0, 9.0, 11.0):
        for P in (20.0, 40.0, 60.0):
            recs = []
            for s in range(args.seeds):
                recs += simulate(s, args.duration, args.rate, c_row, P)
            wastes = [r[2] for r in recs]
            deltas = [r[3] for r in recs]
            oracle = sum(max(0.0, d) for d in deltas)
            print(
                f"\n=== c_row={c_row} us/row  P={P} ms  "
                f"break-even={P * 1000 / c_row:.0f} rows  steps={len(recs)} ==="
            )
            print(
                f"  waste p50/p90/p99/max = "
                f"{pct(wastes, 50):.0f}/{pct(wastes, 90):.0f}/"
                f"{pct(wastes, 99):.0f}/{max(wastes)}"
            )
            print(
                f"  oracle: fallback on {sum(d > 0 for d in deltas)} steps, "
                f"net {oracle:.0f} ms"
            )
            print("    T      trigger%   net_ms   regret_ms   vs_oracle%")
            print(
                f"  {'off':>6}   {0.0:6.1f}%   {0:8.0f}   {0:9.0f}   "
                f"{0.0:8.1f}%   (VLLM_DP_EAGER_PAD_TOKENS=0)"
            )
            best = None
            for T in (1024, 2048, 3072, 4096, 5500, 6667, 8192, 10240, 12288):
                net, regret, trig = rule_net(recs, lambda tok, g, w, T=T: w >= T)
                print(
                    f"  {T:6d}   {trig * 100:6.1f}%   {net:8.0f}   "
                    f"{regret:9.0f}   "
                    f"{100 * net / oracle if oracle else 0:8.1f}%"
                )
                if best is None or net > best[1]:
                    best = (T, net)
            for name, pred in (
                ("ratio>50%", lambda tok, g, w: w >= 0.5 * DP * g),
                ("ratio>70%", lambda tok, g, w: w >= 0.7 * DP * g),
                (
                    "min<=64 & max>=2048",
                    lambda tok, g, w: min(tok) <= 64 and max(tok) >= 2048,
                ),
            ):
                net, regret, trig = rule_net(recs, pred)
                print(
                    f"  {name:20s} {trig * 100:5.1f}%   {net:8.0f}   "
                    f"{regret:9.0f}   "
                    f"{100 * net / oracle if oracle else 0:8.1f}%"
                )
            print(f"  best absolute T={best[0]} net={best[1]:.0f} ms")


if __name__ == "__main__":
    main()
