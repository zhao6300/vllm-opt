"""Measure the 4-rank NCCL cost the DP padding analysis depends on.

Graph path: one fixed-size all-gather of dp*G rows (equal sizes).
Eager path: a ragged all-gatherv, which vLLM implements as one broadcast per
rank, so it is modelled here as world_size broadcasts of the per-rank sizes.

Run with: CUDA_VISIBLE_DEVICES=0,1,2,3 python -I nccl_bench.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 5120
DTYPE = torch.bfloat16
ROW_BYTES = HIDDEN * 2 + 6 * 4 + 6 * 4  # hidden + topk_weights + topk_ids, packed
SIZES = [28, 1024, 4096, 8192, 16384]
WARMUP = 5
ITERS = 40


def bench(fn, iters=ITERS):
    """Min over iterations: the live server shares these GPUs, so the mean is
    contention; the fastest iteration is the closest to the true cost."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[0], times[len(times) // 10]  # ms (min, p10)


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{rank}")
    lines = []
    for n in SIZES:
        x = torch.randn(n, HIDDEN, dtype=DTYPE, device=dev)
        out = torch.empty(world_size * n, HIDDEN, dtype=DTYPE, device=dev)
        ag_ms, ag_p10 = bench(lambda out=out, x=x: dist.all_gather_into_tensor(out, x))
        # reduce-scatter: input [world*n, H], output [n, H]
        big = torch.randn(world_size * n, HIDDEN, dtype=DTYPE, device=dev)
        rs_out = torch.empty(n, HIDDEN, dtype=DTYPE, device=dev)
        rs_ms, rs_p10 = bench(
            lambda rs_out=rs_out, big=big: dist.reduce_scatter_tensor(rs_out, big)
        )
        # ragged: one broadcast per rank, each of its own size. Rank 0 is the
        # small one; every rank uses the same sizes list.
        buf = torch.empty(world_size * n, HIDDEN, dtype=DTYPE, device=dev)
        sizes = [n] * world_size
        sizes[0] = max(1, n // 8)
        ragged_ms, _ = bench(
            lambda buf=buf, sizes=sizes: [
                dist.broadcast(buf[: sizes[r]], src=r) for r in range(world_size)
            ]
        )
        total_bytes = world_size * n * HIDDEN * 2
        lines.append(
            f"{n:6d}  {ag_ms * 1000:8.1f}us (p10 {ag_p10 * 1000:8.1f})"
            f"  {rs_ms * 1000:8.1f}us (p10 {rs_p10 * 1000:8.1f})"
            f"  {ragged_ms * 1000:8.1f}us"
            f"  {total_bytes / ag_ms / 1e6:6.1f}GB/s"
        )
    if rank == 0:
        print("   rows   all_gather(min,p10)      reduce_scatter(min,p10)     ragged")
        for line in lines:
            print(line)
    dist.destroy_process_group()


if __name__ == "__main__":
    port = 29517
    mp.spawn(worker, args=(4, port), nprocs=4, join=True)
