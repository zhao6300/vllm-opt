"""Compare the TF-row dumps of the alone and mixed runs.

The server writes `<root>/<phase>/<tag>__<layer>__r<rank>__start<S>__rows<N>.pt`
and `<root>/<phase>/attnpref{before,after}__<layer>.pt`.  Rank shards are
concatenated by their global start row, so the alone and mixed layouts need not
split at the same row.

Usage: compare_tf_dumps.py /dev/shm/tf_dump [--threshold 1e-2]
"""

import argparse
import glob
import os
import re

import torch

DUMP = re.compile(
    r"^(?P<tag>attn|ffn|ffnin|moeprecombine|moepostcombine)__"
    r"(?P<layer>.+?)__r(?P<rank>\d+)"
    r"__dev(?P<dev>\d+)__start(?P<start>\d+)__rows(?P<rows>\d+)\.pt$"
)


def collect(root, phase):
    groups = {}
    for path in glob.glob(os.path.join(root, phase, "*.pt")):
        m = DUMP.match(os.path.basename(path))
        if not m:
            continue
        key = (m["tag"], m["layer"], int(m["dev"]))
        groups.setdefault(key, []).append((int(m["start"]), path))
    out = {}
    for key, items in groups.items():
        items.sort()
        parts = [torch.load(path) for _, path in items]
        starts = [start for start, _ in items]
        rows = [p.shape[0] for p in parts]
        expect = starts[0]
        for start, n in zip(starts, rows):
            if start != expect:
                print(f"[warn] {key}: gap/overlap at row {start} (want {expect})")
            expect = start + n
        out[key] = torch.cat(parts, dim=0)
    return out


def layer_index(layer):
    m = re.search(r"layers\.(\d+)", layer)
    return int(m.group(1)) if m else 1 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--threshold", type=float, default=1e-2)
    args = ap.parse_args()
    alone = collect(args.root, "alone")
    mixed = collect(args.root, "mixed")
    keys = sorted(
        set(alone) & set(mixed),
        key=lambda k: (layer_index(k[1]), k[2], k[0]),
    )
    print(
        f"{'layer':44s} {'dev':>3s} {'phase':4s} {'rows':>4s} {'max_abs':>10s} "
        f"{'mean_abs':>10s} {'rel_max':>8s}"
    )
    first = None
    for key in keys:
        a, b = alone[key].float(), mixed[key].float()
        if a.shape != b.shape:
            print(f"{key} shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        diff = (a - b).abs()
        mx = diff.max().item()
        mean = diff.mean().item()
        rel = mx / max(b.abs().mean().item(), 1e-6)
        over = mx > args.threshold
        if over and first is None:
            first = (key, mx)
        print(
            f"{key[1]:44s} {key[2]:3d} {key[0]:4s} {b.shape[0]:4d} {mx:10.4f} "
            f"{mean:10.5f} {rel:8.3f}{'  <-- first' if over and first and first[0] == key else ''}"
        )

    print("\nprefill output changed by the decode call (should be 0):")
    for path in sorted(
        glob.glob(os.path.join(args.root, "mixed", "attnprefbefore__*.pt"))
    ):
        layer = os.path.basename(path)[len("attnprefbefore__") : -3]
        after = os.path.join(os.path.dirname(path), f"attnprefafter__{layer}.pt")
        if not os.path.exists(after):
            continue
        a = torch.load(path).float()
        b = torch.load(after).float()
        print(f"{layer:56s} max_abs={(a - b).abs().max().item():.6f}")

    print("\nrunner / PP payload rows:")
    for path in sorted(
        glob.glob(os.path.join(args.root, "mixed", "runner*.pt"))
    ):
        base = os.path.basename(path)
        alone_path = os.path.join(args.root, "alone", base)
        if not os.path.exists(alone_path):
            continue
        a = torch.load(alone_path).float()
        b = torch.load(path).float()
        if a.shape != b.shape:
            print(f"{base:64s} shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        print(f"{base:64s} max_abs={(a - b).abs().max().item():.6f}")

    print(f"\nFIRST OVER {args.threshold}: {first}")


if __name__ == "__main__":
    main()
