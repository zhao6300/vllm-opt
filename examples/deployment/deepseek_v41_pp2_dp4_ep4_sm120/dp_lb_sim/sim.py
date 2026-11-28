"""Discrete-event simulation of DP prefill routing.

Mirrors vllm/v1/engine/core_client.py DPLBAsyncMPClient:
  score = max(scale*(own_inflight+other_inflight), snap_waiting+snap_running)
          + min(scale*pending_load, MAX_PENDING)
          + snap_waiting*6*max(0, snap_kv-0.5)
  pending_load = pending_ms/budget (budget>0 and ms>0) else pending_tokens/T
  * waiting/running/kv come from the 100 ms snapshot, not live engine state.
  * engine reports pending tokens (remainder after cache hits) and, with a
    budget, pending ms = sum(remaining/chunk * step_ms(chunk, ctx)).
  * dispatch floor: a request this server sent floors its engine's reported
    backlog until 2 snapshots after the dispatch, or its first output.
  * shared load: each server publishes per-engine (inflight, pending tokens,
    pending ms); a router sums the other live servers' slots unscaled.

Step cost calibrated to TIMING.md (PP2, DP4, SM120, fast prefill):
  prefill-form step: 130 ms fixed + 0.087 ms/token (16K cold = 1.55 s)
  decode-only step: 20 ms + 0.15 ms per extra request
  DP is lockstep: a step lasts max over ranks.
"""

import argparse
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field

MAX_PENDING_DEFAULT = 4.0
FLOOR_SNAPSHOTS = 2
SNAPSHOT_S = 0.1
MAX_BATCHED = 8192
DRAFT_ROWS = 6
ACCEPT_PER_STEP = 2.0

PREFILL_FIXED_MS = 130.0
PREFILL_MS_PER_TOKEN = 0.087
DECODE_FIXED_MS = 20.0
DECODE_MS_PER_REQ = 0.15
IDLE_MS = 5.0


@dataclass
class Req:
    rid: int
    prompt: int
    work: int
    remaining: int
    output_left: float
    server: int
    engine: int
    arrival: float
    dispatched: float
    first_token: float | None = None
    finished: float | None = None
    started: bool = False
    tok_times: list = field(default_factory=list)


@dataclass
class Engine:
    reqs: list = field(default_factory=list)

    def counts(self):
        waiting = sum(1 for r in self.reqs if r.remaining > 0 and not r.started)
        return waiting, len(self.reqs) - waiting

    def pending_tokens(self):
        return sum(r.remaining for r in self.reqs)

    def decode_reqs(self):
        return [r for r in self.reqs if r.remaining == 0]


@dataclass
class Server:
    index: int
    inflight: Counter = field(default_factory=Counter)
    pending_tokens: Counter = field(default_factory=Counter)
    pending_ms: Counter = field(default_factory=Counter)
    floor: dict = field(default_factory=dict)
    seq: int = 0
    start_index: int = 0


class Sim:
    def __init__(self, cfg, seed, workload):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.workload = workload
        self.engines = [Engine() for _ in range(cfg.engines)]
        self.servers = [Server(i) for i in range(cfg.servers)]
        self.shared = (
            [[[0, 0, 0] for _ in range(cfg.engines)] for _ in range(cfg.servers)]
            if cfg.shared and cfg.servers > 1
            else None
        )
        self.heartbeat = [0.0] * cfg.servers
        self.now = 0.0
        # per engine: pending tokens, pending ms, ms/token, waiting, running, kv
        self.reported = [[0, 0.0, 0.0, 0, 0, 0.0] for _ in range(cfg.engines)]
        self.next_snapshot = 0.0
        self.reqs = []
        self._next_rid = 0
        self._arrivals = self._make_arrivals()
        self._ai = 0
        self._rr_server = 0
        self.step_ms = []

    # ---------- workload ----------
    def _make_arrivals(self):
        w = self.workload
        out = []
        t = 0.0
        for i in range(w["count"]):
            t += self.rng.expovariate(w["rate"]) if w["rate"] > 0 else w["gap"]
            kind = w["kind"]
            if kind == "uniform_long":
                prompt = w["prompt"]
                remainder = prompt
            elif kind == "burst":
                prompt = self.rng.choice(w["sizes"])
                remainder = prompt
            elif kind == "single_long":
                prompt = w["long"] if i == 0 else int(
                    min(16_000, max(256, self.rng.lognormvariate(math.log(1000), 0.4))))
                remainder = prompt
            else:  # online
                if self.rng.random() < w["long_frac"]:
                    prompt = int(self.rng.uniform(32_000, 128_000))
                else:
                    prompt = int(min(16_000, max(256, self.rng.lognormvariate(
                        math.log(w["median"]), 0.6))))
                if self.rng.random() < w["hit_frac"]:
                    remainder = 0 if self.rng.random() < 0.7 else min(
                        prompt, self.rng.choice([127, 300, 1024]))
                else:
                    remainder = prompt
            out.append((t, prompt, remainder))
        return out

    # ---------- routing ----------
    def _route(self, prompt, remainder):
        cfg = self.cfg
        if cfg.server_skew and self.rng.random() < cfg.server_skew:
            server = self.servers[0]
        else:
            server = self.servers[self._rr_server % len(self.servers)]
            self._rr_server += 1
        n = cfg.engines
        others = None
        if self.shared is not None:
            others = []
            for e in range(n):
                tot = [0, 0, 0]
                for s in range(len(self.servers)):
                    if s == server.index or self.now - self.heartbeat[s] > 2.0:
                        continue
                    slot = self.shared[s][e]
                    tot[0] += slot[0]
                    tot[1] += slot[1]
                    tot[2] += slot[2]
                others.append((tot[0], tot[1], tot[2] / 1000.0))
        scale = 1 if others is not None else len(self.servers)
        start = server.start_index
        best, best_idx = math.inf, 0
        for i in range(n):
            idx = (start + i) % n
            tok, ms, _per, waiting, running, kv = self.reported[idx]
            inflight = server.inflight[idx]
            oi, ot, om = others[idx] if others is not None else (0, 0, 0.0)
            score = max(scale * (inflight + oi), waiting + running)
            if cfg.token_aware:
                tok = max(tok, server.pending_tokens[idx] + ot)
                ms = max(ms, server.pending_ms[idx] + om)
                if cfg.budget_ms > 0 and ms > 0:
                    load = ms / cfg.budget_ms
                else:
                    load = tok / cfg.tokens_per_request
                score += min(scale * load, cfg.max_pending)
            if waiting:
                score += waiting * 6.0 * max(0.0, kv - 0.5)
            if score < best:
                best, best_idx = score, idx
        engine = best_idx
        # The real client rotates the scan start and bumps the snapshot's
        # waiting count so a burst spreads before the next snapshot.
        server.start_index = (server.start_index + 1) % n
        self.reported[engine][3] += scale
        req = Req(self._next_rid, prompt, remainder, remainder, self._output_tokens(),
                  server.index, engine, self.now, self.now)
        self._next_rid += 1
        self.reqs.append(req)
        self.engines[engine].reqs.append(req)
        server.inflight[engine] += 1
        if cfg.token_aware:
            ms_per_tok = self.reported[engine][2] or cfg.cost_ms_per_token
            ms = prompt * ms_per_tok
            server.floor[req.rid] = (engine, prompt, ms, server.seq)
            server.pending_tokens[engine] += prompt
            server.pending_ms[engine] += ms
        self._publish(server, engine)
        return req

    def _output_tokens(self):
        return max(1.0, self.rng.expovariate(1 / self.workload["output"]))

    def _publish(self, server, engine):
        if self.shared is None:
            return
        slot = self.shared[server.index][engine]
        slot[0] = server.inflight[engine]
        slot[1] = server.pending_tokens[engine]
        slot[2] = round(server.pending_ms[engine] * 1000)
        self.heartbeat[server.index] = self.now

    # ---------- engine report ----------
    def _chunk(self, decode_tokens, context):
        if self.cfg.budget_ms <= 0:
            return max(1, MAX_BATCHED - decode_tokens * DRAFT_ROWS)
        room = (
            self.cfg.budget_ms - PREFILL_FIXED_MS
            - PREFILL_MS_PER_TOKEN * decode_tokens
        )
        if room <= 0:
            return 256
        return max(256, min(int(room / PREFILL_MS_PER_TOKEN),
                            MAX_BATCHED - decode_tokens * DRAFT_ROWS))

    def _step_ms(self, chunk, decode_tokens):
        return PREFILL_FIXED_MS + PREFILL_MS_PER_TOKEN * chunk

    def _report(self):
        for e, eng in enumerate(self.engines):
            waiting, running = eng.counts()
            tok = eng.pending_tokens()
            ms = 0.0
            if self.cfg.budget_ms > 0:
                dec = len(eng.decode_reqs())
                for r in eng.reqs:
                    if r.remaining <= 0:
                        continue
                    ctx = r.prompt - r.remaining
                    mid = ctx + r.remaining // 2
                    chunk = self._chunk(dec, mid)
                    ms += r.remaining / chunk * self._step_ms(chunk, dec)
            dec = len(eng.decode_reqs())
            chunk0 = self._chunk(dec, 0)
            per_tok = (
                self._step_ms(chunk0, dec) / chunk0 if self.cfg.budget_ms > 0 else 0.0
            )
            kv = min(1.0, sum(r.prompt for r in eng.reqs) / self.cfg.kv_capacity)
            self.reported[e] = [tok, ms, per_tok, waiting, running, kv]

    def _snapshot(self):
        self._report()
        for s in self.servers:
            s.seq += 1
            seen = s.seq - FLOOR_SNAPSHOTS
            for rid, (eng, tok, ms, seq) in list(s.floor.items()):
                if seq <= seen:
                    del s.floor[rid]
                    s.pending_tokens[eng] -= tok
                    s.pending_ms[eng] -= ms
                    self._publish(s, eng)
            self.heartbeat[s.index] = self.now

    # ---------- step ----------
    def _step(self):
        durations, plans = [], []
        for eng in self.engines:
            dec = eng.decode_reqs()
            cap = MAX_BATCHED - len(dec) * DRAFT_ROWS
            prefill = 0
            for r in eng.reqs:
                if cap <= 0:
                    break
                if r.remaining <= 0:
                    continue
                chunk = self._chunk(len(dec), r.prompt - r.remaining)
                take = min(r.remaining, chunk, cap)
                prefill += take
                cap -= take
            if prefill > 0:
                dur = PREFILL_FIXED_MS + PREFILL_MS_PER_TOKEN * prefill + \
                    DECODE_MS_PER_REQ * len(dec)
            elif dec:
                dur = DECODE_FIXED_MS + DECODE_MS_PER_REQ * max(0, len(dec) - 1)
            else:
                dur = IDLE_MS
            durations.append(dur)
            plans.append(dec)
        dur = max(durations)
        self.step_ms.append(dur)
        for ei, (eng, dec) in enumerate(zip(self.engines, plans)):
            cap = MAX_BATCHED - len(dec) * DRAFT_ROWS
            for r in list(eng.reqs):
                if cap <= 0:
                    break
                if r.remaining <= 0:
                    continue
                chunk = self._chunk(len(dec), r.prompt - r.remaining)
                take = min(r.remaining, chunk, cap)
                r.remaining -= take
                cap -= take
                r.started = True
                if r.remaining == 0 and r.first_token is None:
                    r.first_token = self.now + dur / 1000.0
                    r.tok_times.append(r.first_token)
                    s = self.servers[r.server]
                    if r.rid in s.floor:
                        e2, tok, ms, _ = s.floor.pop(r.rid)
                        s.pending_tokens[e2] -= tok
                        s.pending_ms[e2] -= ms
                        self._publish(s, e2)
            for r in dec:
                r.output_left -= ACCEPT_PER_STEP
                r.tok_times.append(self.now + dur / 1000.0)
                if r.output_left <= 0:
                    r.finished = self.now + dur / 1000.0
                    eng.reqs.remove(r)
                    s = self.servers[r.server]
                    s.inflight[ei] -= 1
                    self._publish(s, ei)
        return dur

    # ---------- loop ----------
    def run(self):
        while True:
            while (
                self._ai < len(self._arrivals)
                and self._arrivals[self._ai][0] <= self.now
            ):
                _, prompt, remainder = self._arrivals[self._ai]
                self._ai += 1
                self._route(prompt, remainder)
            if self.now >= self.next_snapshot:
                self._snapshot()
                self.next_snapshot = self.now + SNAPSHOT_S
            if self._ai >= len(self._arrivals) and not any(
                e.reqs for e in self.engines
            ):
                break
            if not any(e.reqs for e in self.engines):
                self.now = max(self.now, self._arrivals[self._ai][0])
                continue
            self.now += self._step() / 1000.0
        return self.metrics()

    def metrics(self):
        reqs = [r for r in self.reqs if r.finished is not None]
        ttfts = [(r.first_token - r.arrival) * 1000 for r in reqs if r.first_token]
        lat = [(r.finished - r.arrival) * 1000 for r in reqs]
        makespan = max((r.finished for r in reqs), default=0.0)
        busy = [0] * len(self.engines)
        for r in self.reqs:
            busy[r.engine] += r.work
        short_ttft = [(r.first_token - r.arrival) * 1000 for r in reqs
                      if r.first_token and r.prompt <= 16000]
        long_ttft = [(r.first_token - r.arrival) * 1000 for r in reqs
                     if r.first_token and r.prompt > 16000]
        itl = []
        for r in reqs:
            ts = r.tok_times
            for a, b in zip(ts, ts[1:]):
                itl.append((b - a) * 1000)
        return {
            "makespan": makespan,
            "ttft_mean": statistics.mean(ttfts) if ttfts else 0,
            "ttft_p50": pct(ttfts, 50),
            "ttft_p99": pct(ttfts, 99),
            "short_p50": pct(short_ttft, 50),
            "short_p99": pct(short_ttft, 99),
            "long_p50": pct(long_ttft, 50),
            "long_p99": pct(long_ttft, 99),
            "lat_p50": pct(lat, 50),
            "lat_p99": pct(lat, 99),
            "itl_p50": pct(itl, 50),
            "itl_p99": pct(itl, 99),
            "imbalance": (max(busy) / (sum(busy) / len(busy))) if sum(busy) else 1,
            "busy": busy,
        }


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[k]


@dataclass
class Cfg:
    engines: int = 4
    servers: int = 1
    token_aware: bool = False
    tokens_per_request: float = 2048.0
    budget_ms: float = 0.0
    cost_ms_per_token: float = 0.087
    shared: bool = False
    max_pending: float = MAX_PENDING_DEFAULT
    kv_capacity: float = 4 * 262144.0
    server_skew: float = 0.0  # probability an arrival lands on server 0


WORKLOADS = {
    "uniform_long": dict(kind="uniform_long", count=32, prompt=16000, rate=0.0,
                         gap=0.02, output=300, hit_frac=0.0,
                         long_frac=0.0, median=2000),
    "single_long": dict(kind="single_long", count=48, long=128000, rate=0.0,
                        gap=0.15, output=300, hit_frac=0.0, long_frac=0.0, median=1000),
    "burst": dict(kind="burst", count=16, rate=0.0, gap=0.0,
                  sizes=[1000, 1000, 1000, 8000, 8000, 8000, 32000, 32000,
                         1000, 1000, 1000, 8000, 8000, 8000, 32000, 32000],
                  output=300, hit_frac=0.0, long_frac=0.0, median=2000),
    "online": dict(kind="online", count=1200, rate=3.0, gap=0.0, output=300,
                   hit_frac=0.85, long_frac=0.02, median=2000),
    "online_hot": dict(kind="online", count=1600, rate=5.0, gap=0.0, output=300,
                       hit_frac=0.85, long_frac=0.08, median=2000),
    # spread-out equal long prompts: does uncapped routing skew?
    "staggered": dict(kind="uniform_long", count=32, prompt=16000, rate=0.0,
                      gap=0.3, output=300, hit_frac=0.0, long_frac=0.0, median=2000),
    # a burst that lands mostly on one API server
    "skew_burst": dict(kind="burst", count=48, rate=0.0, gap=0.0,
                       sizes=[1000, 2000, 4000, 8000, 16000, 32000] * 8,
                       output=300, hit_frac=0.0, long_frac=0.0, median=2000),
}


def run(cfg, wl_name, seeds=3):
    out = [Sim(cfg, s, WORKLOADS[wl_name]).run() for s in range(seeds)]
    keys = ["makespan", "ttft_mean", "ttft_p50", "ttft_p99", "short_p50", "short_p99",
            "long_p50", "long_p99", "lat_p50", "lat_p99", "itl_p50", "itl_p99",
            "imbalance"]
    return {k: statistics.mean(r[k] for r in out) for k in keys}


def sweep(wl, seeds):
    print(f"\n===== workload: {wl} =====")
    print(f"{'config':26s} {'makespan':>8s} {'ttft_mean':>9s} {'ttft_p99':>8s} "
          f"{'itl_p50':>7s} {'itl_p99':>7s} {'imbal':>6s}")

    def add(name, **kw):
        m = run(Cfg(**kw), wl, seeds)
        print(f"{name:26s} {m['makespan']:8.2f} {m['ttft_mean']:9.1f} "
              f"{m['ttft_p99']:8.1f} {m['itl_p50']:7.1f} {m['itl_p99']:7.1f} "
              f"{m['imbalance']:6.2f}")

    add("off (count only)", token_aware=False)
    for t in (2048, 8192, 16384, 32768):
        add(f"tok T={t} cap=4", token_aware=True, tokens_per_request=t)
    for cap in (1.0, 2.0, 8.0, 1e9):
        add(f"tok T=2048 cap={cap:g}", token_aware=True, max_pending=cap)
    add("tok T=16384 cap=inf", token_aware=True,
        tokens_per_request=16384, max_pending=1e9)
    for b in (300, 600, 1200):
        add(f"budget {b}ms", token_aware=True, budget_ms=b)
    add("shared off 4srv", token_aware=True, servers=4)
    add("shared on  4srv", token_aware=True, servers=4, shared=True)
    add("shared on  8srv", token_aware=True, servers=8, shared=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    wls = list(WORKLOADS) if args.wl == "all" else [args.wl]
    for wl in wls:
        sweep(wl, args.seeds)


if __name__ == "__main__":
    main()
