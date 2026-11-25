"""Compare router logits and top-k of a controlled alone/mixed pair.

Both runs must use the same total B/N; the expected TP shard patterns are
selected explicitly so ambient files with other shapes are ignored.

Example (the B=N=48 pair):
  alone: P=48, shards start0/rows24 + start24/rows24
  mixed: P=42 + 6 decode, shards start6/rows18 + start24/rows24
  compare_router.py /dev/shm/tf_cmp \
    --alone-shards 0:24,24:24 --mixed-shards 6:18,24:24
"""

import argparse
import glob
import os
import re

import torch

PAT = re.compile(
    r"router__(.+)__dev(\d+)__start(\d+)__rows(\d+)\.pt$"
)


def parse_shards(text):
    return [tuple(int(x) for x in item.split(":")) for item in text.split(",")]


def collect(root, phase, pairs):
    raw = {}
    for path in glob.glob(os.path.join(root, phase, "router__*.pt")):
        m = PAT.match(os.path.basename(path))
        if not m:
            continue
        layer, dev, start, rows = (
            m.group(1),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        )
        if dev >= 4 or (start, rows) not in pairs:
            continue
        raw.setdefault(layer, {})[start] = torch.load(path)
    out = {}
    for layer, parts in raw.items():
        if set(parts) == {start for start, _ in pairs}:
            out[layer] = torch.cat([parts[s] for s in sorted(parts)], 0)
    return out


def layer_index(layer):
    m = re.search(r"layers\.(\d+)", layer)
    return int(m.group(1)) if m else 1 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--alone-shards", default="0:24,24:24")
    ap.add_argument("--mixed-shards", default="6:18,24:24")
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    alone = collect(args.root, "alone", parse_shards(args.alone_shards))
    mixed = collect(args.root, "mixed", parse_shards(args.mixed_shards))
    k = args.topk
    print(
        f"{'layer':6s} {'rows':>4s} {'logit_max':>10s} {'top6_diff':>9s} "
        f"{'top1_diff':>9s} {'margin_min':>11s} {'diff_margin_min':>15s}"
    )
    first = None
    for layer in sorted(set(alone) & set(mixed), key=layer_index):
        a = alone[layer].float()
        b = mixed[layer].float()
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        logit_max = (a - b).abs().max().item()
        ta = torch.topk(a, k + 1, dim=1)
        tb = torch.topk(b, k + 1, dim=1)
        ids_a = torch.sort(ta.indices[:, :k], dim=1).values
        ids_b = torch.sort(tb.indices[:, :k], dim=1).values
        set_diff = (ids_a != ids_b).any(dim=1)
        top1_diff = ta.indices[:, 0] != tb.indices[:, 0]
        margin = ta.values[:, k - 1] - ta.values[:, k]
        dm = margin[set_diff] if set_diff.any() else None
        print(
            f"{layer_index(layer):6d} {n:4d} {logit_max:10.4f} "
            f"{int(set_diff.sum()):9d} {int(top1_diff.sum()):9d} "
            f"{margin.min().item():11.4f} "
            f"{(dm.min().item() if dm is not None and dm.numel() else float('nan')):15.4f}"
        )
        if set_diff.any() and first is None:
            first = (layer, int(set_diff.sum()), logit_max)
    print(f"\nFIRST top-{k} difference: {first}")


if __name__ == "__main__":
    main()
