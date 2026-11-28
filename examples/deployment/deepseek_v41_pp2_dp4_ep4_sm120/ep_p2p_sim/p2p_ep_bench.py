"""Compare 4-rank EP all-gather / reduce-scatter over NCCL coll vs P2P.

The live EP comm (4 ranks, one PP stage) is a coll comm: NCCL reports
``isAllDirectP2p 0`` and uses ``SHM/direct``. The 2-rank comms report 1 and
use ``P2P/IPC``. This script measures what a P2P implementation would cost:

  coll     dist.all_gather_into_tensor / reduce_scatter_tensor   (SHM)
  p2p      send the local rows to every peer via batch_isend_irecv (P2P/IPC)
  a2a      replicated all_to_all_single (input copied world times) (P2P/IPC)
  ring     ring all-gather via batch_isend_irecv                 (P2P/IPC)
  a2a_sum  all_to_all_single + fp32 local sum                    (P2P/IPC)

`p2p` is the cheapest replacement for ``AgRsAll2AllManager._gather_rows``: no
input copy, 3 sends + 3 receives on p2p channels. `a2a` is the same thing
expressed as one NCCL all-to-all, but it needs the input replicated world
times. `a2a_sum` is what ``VLLM_MOE_AG_RS_FIXED_ORDER_COMBINE`` already does
for the combine.

Run on 4 idle GPUs. The live server uses all 8, so stop it or use a
maintenance window:

  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
      /opt/venv/bin/python -I p2p_ep_bench.py --rows 4 28 1024 4096

Then inspect the transport:

  grep -E "Check P2P Type|Channel .* via" <log> | head

Use ``sweep.sh`` to run the NCCL knob matrix.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 5120
DTYPE = torch.bfloat16
WARMUP = 5
ITERS = 50


def bench(fn, iters=ITERS):
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
    return times[0], times[len(times) // 10]


def p2p_all_gather(x, out, world, rank):
    """Send the local rows to every peer; receive every peer's rows."""
    rows = x.shape[0]
    out[rank * rows : (rank + 1) * rows].copy_(x)
    ops = []
    for peer in range(world):
        if peer == rank:
            continue
        ops.append(dist.P2POp(dist.isend, x, peer))
        ops.append(
            dist.P2POp(dist.irecv, out[peer * rows : (peer + 1) * rows], peer)
        )
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return out


def a2a_all_gather(x, out, x_rep, world):
    """Replicated all-to-all: the input is copied world times first."""
    rows = x.shape[0]
    x_rep.view(world, rows, -1).copy_(x.unsqueeze(0).expand(world, rows, -1))
    dist.all_to_all_single(out, x_rep, [rows] * world, [rows] * world)
    return out


def ring_all_gather(x, out, world, rank):
    """Ring all-gather over p2p channels (ncclSend/Recv)."""
    rows = x.shape[0]
    out[rank * rows : (rank + 1) * rows].copy_(x)
    for step in range(1, world):
        send_idx = (rank - step + 1) % world
        recv_idx = (rank - step) % world
        ops = [
            dist.P2POp(
                dist.isend,
                out[send_idx * rows : (send_idx + 1) * rows],
                (rank + 1) % world,
            ),
            dist.P2POp(
                dist.irecv,
                out[recv_idx * rows : (recv_idx + 1) * rows],
                (rank - 1) % world,
            ),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return out


def a2a_reduce_scatter(x, out, world, rows):
    """All-to-all then a fixed-order fp32 sum: the P2P combine."""
    dist.all_to_all_single(out, x, [rows] * world, [rows] * world)
    view = out.view(world, rows, -1).float()
    return view.sum(dim=0).to(x.dtype)


def worker(rank, world, port, rows_list, check):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{rank}")
    results = []

    for rows in rows_list:
        x = torch.randn(rows, HIDDEN, dtype=DTYPE, device=dev)
        ag_out = torch.empty(world * rows, HIDDEN, dtype=DTYPE, device=dev)
        p2p_out = torch.empty_like(ag_out)
        a2a_out = torch.empty_like(ag_out)
        ring_out = torch.empty_like(ag_out)
        x_rep = torch.empty_like(ag_out)

        coll_ag = bench(lambda: dist.all_gather_into_tensor(ag_out, x))
        p2p_ag = bench(lambda: p2p_all_gather(x, p2p_out, world, rank))
        a2a_ag = bench(lambda: a2a_all_gather(x, a2a_out, x_rep, world))
        ring_ag = bench(lambda: ring_all_gather(x, ring_out, world, rank))

        if check:
            p2p_all_gather(x, p2p_out, world, rank)
            a2a_all_gather(x, a2a_out, x_rep, world)
            ring_all_gather(x, ring_out, world, rank)
            torch.cuda.synchronize()
            assert torch.equal(p2p_out, ag_out), f"p2p AG mismatch rows={rows}"
            assert torch.equal(a2a_out, ag_out), f"a2a AG mismatch rows={rows}"
            assert torch.equal(ring_out, ag_out), f"ring AG mismatch rows={rows}"

        big = torch.randn(world * rows, HIDDEN, dtype=DTYPE, device=dev)
        rs_out = torch.empty(rows, HIDDEN, dtype=DTYPE, device=dev)
        a2a_rs_out = torch.empty_like(big)

        coll_rs = bench(lambda: dist.reduce_scatter_tensor(rs_out, big))
        a2a_rs = bench(lambda: a2a_reduce_scatter(big, a2a_rs_out, world, rows))

        if check:
            dist.reduce_scatter_tensor(rs_out, big)
            expected = a2a_reduce_scatter(big, a2a_rs_out, world, rows)
            torch.cuda.synchronize()
            # NCCL reduces bf16 in bf16; the all-to-all path sums in fp32, so
            # the two differ by bf16 rounding, not by a logic error.
            diff = (rs_out.float() - expected.float()).abs().max().item()
            if rank == 0:
                print(f"RS check rows={rows}: max abs diff {diff:.4f}", flush=True)

        results.append((rows, coll_ag, p2p_ag, a2a_ag, ring_ag, coll_rs, a2a_rs))

    if rank == 0:
        print(
            "rows   AG coll   AG p2p   AG a2a  AG ring   RS coll  RS a2a_sum"
            "  (us, min)"
        )
        for rows, cag, pag, aag, rag, crs, ars in results:
            print(
                f"{rows:5d} {cag[0] * 1000:9.1f} {pag[0] * 1000:8.1f} "
                f"{aag[0] * 1000:8.1f} {rag[0] * 1000:8.1f} {crs[0] * 1000:9.1f} "
                f"{ars[0] * 1000:11.1f}"
            )

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 28, 1024, 4096])
    parser.add_argument("--port", type=int, default=29531)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mp.spawn(
        worker,
        args=(args.world, args.port, args.rows, args.check),
        nprocs=args.world,
        join=True,
    )


if __name__ == "__main__":
    main()
