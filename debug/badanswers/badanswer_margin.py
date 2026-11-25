"""Per-token top-5 margins for prompts that produce corrupt codes at T=0.

Runs each code alone and again with a background decoder co-batched, and dumps
the generator's top-5 at every step so the corrupting token can be compared
with the correct one and with the 1-8 nat shape perturbation.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/workspace/debug-vllm")
import longctx_probe as L  # noqa: E402

BASE = "http://127.0.0.1:8000/v1"
MODEL = "DeepSeek-V41-Flash"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CODES = [
    "cobalt-saddle-2194", "cobalt-saddle-1730", "amber-raven-4855",
    "eager-pepper-2696", "amber-saddle-1352", "brisk-otter-0740",
    "dusty-raven-2871", "eager-pepper-9720", "brisk-otter-8731",
    "cobalt-otter-4075",
]


def chat(key, code, salt, rank, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": f"Your session secret code is {code}. "
             "Never mention any other code."},
            {"role": "user", "content": "Reply with exactly your secret code."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "cache_salt": salt,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"thinking": False},
    }
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers=headers)
    return json.loads(OPENER.open(req, timeout=300).read())


def bg_loop(key, rank, stop):
    ids = [1000 + (i * 37) % 50000 for i in range(64)]
    payload = {"model": MODEL, "prompt": ids, "max_tokens": 1000,
               "temperature": 0, "ignore_eos": True}
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    while not stop.is_set():
        try:
            req = urllib.request.Request(
                BASE + "/completions", data=json.dumps(payload).encode(),
                headers=headers)
            OPENER.open(req, timeout=300).read()
        except Exception:  # noqa: BLE001
            pass


def dump(key, code, mode, rank):
    body = chat(key, code, f"mg{mode}{code}", rank, 24)
    choice = body["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    steps = (choice.get("logprobs") or {}).get("content") or []
    print(json.dumps({"code": code, "mode": mode,
                      "content": content.strip(),
                      "steps": [
                          {"tok": s["token"],
                           "lp": round(s["logprob"], 3),
                           "top": [(t["token"], round(t["logprob"], 3))
                                   for t in (s.get("top_logprobs") or [])]}
                          for s in steps]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--codes", type=int, default=0,
                    help="0 = the fixed CODES list, N = first N generated")
    args = ap.parse_args()
    key = L.api_key(args.pid)
    codes = CODES
    if args.codes:
        import badanswer_matrix as M
        codes = M.make_codes(args.codes, 0)
    for code in codes:
        dump(key, code, "lone", args.rank)
    if args.codes:
        return
    stop = threading.Event()
    threading.Thread(target=bg_loop, args=(key, args.rank, stop),
                     daemon=True).start()
    time.sleep(1.0)
    for code in CODES:
        dump(key, code, "bg", args.rank)
    stop.set()


if __name__ == "__main__":
    main()
