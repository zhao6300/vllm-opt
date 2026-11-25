"""Sequential T=0 bad-answer matrix for the DSv4.1 chat path.

Every secret code is asked R times at temperature 0, one request at a time
(concurrency 1), with a unique cache_salt per attempt so each attempt is a full
prefill (cached_tokens == 0). Per-code failure counts tell us whether the bad
answers come from a fixed failing subset of prompts (sticky) or from a random
per-request subset (trial-to-trial numerics).

Sticky fraction = P(fail in every round | fail at least once). Under a fixed
subset it is ~1; under independent trials at rate p it is p^(R-1) / (1-(1-p)^R).

The API key is read from /proc/<pid>/cmdline at runtime and never printed.
"""

import argparse
import collections
import json
import random
import re
import string
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/workspace/debug-vllm")
import longctx_probe as L  # noqa: E402

BASE = "http://127.0.0.1:8000/v1"
MODEL = "DeepSeek-V41-Flash"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ADJ = ["amber", "brisk", "cobalt", "dusty", "eager", "fuzzy", "ginger", "hollow"]
NOUN = ["otter", "pepper", "quartz", "raven", "saddle", "tundra", "violet", "walnut"]


def make_codes(n, seed):
    rng = random.Random(seed)
    codes = set()
    while len(codes) < n:
        codes.add(
            "-".join(rng.choice(words) for words in (ADJ, NOUN))
            + "-"
            + "".join(rng.choice(string.digits) for _ in range(4))
        )
    return sorted(codes)


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ask(key, code, salt, thinking, max_tokens, rank, temperature=0.0,
        prefix="", suffix=""):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": prefix + f"Your session secret code is {code}. "
                "Never mention any other code." + suffix,
            },
            {"role": "user", "content": "Reply with exactly your secret code."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "cache_salt": salt,
        "chat_template_kwargs": {"thinking": thinking},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(), headers=headers
    )
    return json.loads(OPENER.open(req, timeout=300).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--codes", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="comma-separated codes; overrides --codes/--seed")
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--prefix", default="", help="system-message prefix")
    ap.add_argument("--suffix", default="", help="system-message suffix")
    ap.add_argument("--bg", action="store_true",
                    help="keep a long decode running")
    ap.add_argument("--bg-rank", type=int, default=None,
                    help="DP rank for the background decode (default: --rank)")
    ap.add_argument("--salt-tag", default="",
                    help="suffix on every cache_salt; use it to force a cold prefill")
    args = ap.parse_args()
    key = L.api_key(args.pid)
    stop = threading.Event()
    if args.bg:

        def bg():
            ids = [1000 + (i * 37) % 50000 for i in range(64)]
            payload = {"model": MODEL, "prompt": ids, "max_tokens": 1000,
                       "temperature": 0, "ignore_eos": True}
            headers = {"Content-Type": "application/json",
                       "Authorization": "Bearer " + key}
            bg_rank = args.rank if args.bg_rank is None else args.bg_rank
            if bg_rank is not None:
                headers["X-data-parallel-rank"] = str(bg_rank)
            while not stop.is_set():
                try:
                    req = urllib.request.Request(
                        BASE + "/completions",
                        data=json.dumps(payload).encode(), headers=headers)
                    OPENER.open(req, timeout=300).read()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=bg, daemon=True).start()
        time.sleep(1.0)
    codes = (sorted(args.only.split(",")) if args.only
             else make_codes(args.codes, args.seed))
    others = {c: [o for o in codes if o != c] for c in codes}
    thinking = args.thinking == "on"

    results = collections.defaultdict(list)
    contents = collections.defaultdict(list)
    usage_rows = []
    t0 = time.time()
    for r in range(args.rounds):
        for ci, code in enumerate(codes):
            kind, content, usage, finish = "ERROR", "", {}, None
            for attempt in range(3):
                try:
                    body = ask(
                        key, code,
                        f"m{ci:04d}r{r}a{attempt}{args.salt_tag}",
                        thinking, args.max_tokens, args.rank,
                        args.temperature, args.prefix, args.suffix,
                    )
                    choice = body["choices"][0]
                    msg = choice.get("message") or {}
                    content = (msg.get("content") or "").strip()
                    usage = body.get("usage") or {}
                    finish = choice.get("finish_reason")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        content = f"REQUEST FAILED {e}"
            if content.startswith("REQUEST FAILED"):
                kind = "ERROR"
            elif any(o in content for o in others[code]):
                kind = "LEAK"
            elif not content:
                kind = "EMPTY"
            elif code not in content:
                kind = "NEAR" if _norm(code) in _norm(content) else "WRONG"
            else:
                kind = "OK"
            cached = ((usage.get("prompt_tokens_details") or {})
                      .get("cached_tokens"))
            usage_rows.append((cached or 0, usage.get("prompt_tokens") or 0,
                               finish))
            results[code].append(kind)
            contents[code].append(content)
            if args.pause:
                time.sleep(args.pause)
        counts = collections.Counter(
            k for ks in results.values() for k in ks)
        print(json.dumps({"section": "progress", "round": r,
                          "counts": dict(counts)}), flush=True)

    stop.set()
    total = len(codes) * args.rounds
    fails = {c: sum(k != "OK" for k in ks) for c, ks in results.items()}
    dist = collections.Counter(fails.values())
    bad_total = sum(fails.values())
    norm_bad = sum(k not in ("OK", "NEAR")
                   for ks in results.values() for k in ks)
    ever = [c for c, f in fails.items() if f]
    sticky = [c for c in ever if fails[c] == args.rounds]
    kinds = collections.Counter(k for ks in results.values() for k in ks)
    print(json.dumps({
        "section": "summary", "total": total, "bad": bad_total,
        "rate": round(bad_total / total, 4), "kinds": dict(kinds),
        "bad_normalized": norm_bad,
        "rate_normalized": round(norm_bad / total, 4),
        "temperature": args.temperature, "thinking": args.thinking,
        "per_code_fail_counts": {str(k): v for k, v in sorted(dist.items())},
        "codes_ever_bad": len(ever), "codes_bad_all_rounds": len(sticky),
        "sticky_fraction": round(len(sticky) / len(ever), 4) if ever else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
        "cached_tokens_nonzero": sum(1 for c, _, _ in usage_rows if c),
        "prompt_tokens": sorted({p for _, p, _ in usage_rows}),
        "finish_reasons": dict(collections.Counter(f for _, _, f in usage_rows)),
    }), flush=True)
    for code in ever:
        print(json.dumps({"section": "bad_code", "code": code,
                          "kinds": results[code],
                          "contents": contents[code]}), flush=True)


if __name__ == "__main__":
    main()
