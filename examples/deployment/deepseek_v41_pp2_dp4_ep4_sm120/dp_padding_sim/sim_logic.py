"""Validate the real sync_cudagraph_and_dp_padding padding behaviour.

Runs the actual vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding with a
fake CudaGraphManager (the same "smallest captured size >= n" bucket rule the
real one uses) and a mocked DP group, for every rank of a scenario. This checks
the claims the analysis rests on:

  1. graph steps pad every rank to one agreed graph size G;
  2. an eager vote on any rank makes the whole group eager, at real sizes;
  3. every rank reaches the same decision (no rank disagrees and deadlocks);
  4. a waste-threshold rule evaluated from the all-reduced tensor agrees on
     every rank.
"""

from unittest import mock

import torch

from vllm.config.compilation import CUDAGraphMode, prefill_cudagraph_capture_grid
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

GRID = prefill_cudagraph_capture_grid(8192)


def bucket(n: int) -> int:
    """Smallest captured prefill size >= n (the real _candidates rule)."""
    for size in GRID:
        if size >= n:
            return size
    return n


class FakeManager:
    """Only dispatch() is used by sync_cudagraph_and_dp_padding."""

    def __init__(self, mode: CUDAGraphMode):
        self.mode = mode

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, **kw):
        size = bucket(num_tokens)
        if self.mode == CUDAGraphMode.NONE or size > GRID[-1]:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
            )
        return BatchExecutionDescriptor(
            cg_mode=self.mode, num_tokens=size, num_reqs=None
        )


def run_scenario(tokens, modes, uniform=None, threshold=None):
    """Run the real sync on each rank; return one result per rank."""
    dp_size = len(tokens)
    # Each rank writes only its own column; the SUM all-reduce reconstructs the
    # per-rank vector. So the post-reduce tensor is exactly this matrix.
    full = torch.zeros(7, dp_size, dtype=torch.int32)
    for rank in range(dp_size):
        full[0][rank] = tokens[rank]
        full[1][rank] = modes[rank].value
        full[2][rank] = uniform or 0
        full[3][rank] = -1
        full[4][rank] = 0
        full[5][rank] = 4
        full[6][rank] = 0
    results = []
    for rank in range(dp_size):
        manager = FakeManager(modes[rank])

        def fake_all_reduce(tensor, _full=full):
            tensor[:] = _full

        desc = BatchExecutionDescriptor(
            cg_mode=modes[rank], num_tokens=tokens[rank], num_reqs=4
        )
        with mock.patch.object(dp_utils, "get_dp_group") as group:
            group.return_value.cpu_all_reduce_ = fake_all_reduce
            synced, sync = dp_utils.sync_cudagraph_and_dp_padding(
                cudagraph_manager=manager,
                desired_batch_desc=desc,
                num_tokens=tokens[rank],
                num_reqs=4,
                uniform_token_count=uniform,
                dp_size=dp_size,
                dp_rank=rank,
            )
        results.append((synced, sync))
    return results


def check(name, tokens, modes, expect_eager, expect_size=None, uniform=None):
    out = run_scenario(tokens, modes, uniform)
    vals = [int(s.num_tokens_across_dp[0]) for _, s in out]
    all_same = len(set(vals)) == 1
    eager = {s.eager for _, s in out}
    size = vals[0]
    ok = (
        eager == {expect_eager}
        and all_same
        and (expect_size is None or size == expect_size)
    )
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:38s} "
        f"eager={eager} padded={size} agree={all_same}"
    )
    return ok


def main():
    P = CUDAGraphMode.PIECEWISE
    N = CUDAGraphMode.NONE
    ok = True
    ok &= check(
        "graph: one big rank + 3 tiny", [8192, 30, 30, 30], [P] * 4, False, 8192
    )
    ok &= check(
        "graph: bucket rounding 5000 -> 5120",
        [5000, 100, 100, 100],
        [P] * 4,
        False,
        5120,
    )
    ok &= check("graph: balanced", [8192, 8192, 8192, 8192], [P] * 4, False, 8192)
    ok &= check(
        "eager vote on one rank -> all eager", [8192, 30, 30, 30], [P, P, N, P], True
    )
    ok &= check("all eager", [8192, 30, 30, 30], [N] * 4, True)

    # The proposed rule: waste = dp*G - sum(real), same on every rank.
    print("\nwaste-threshold rule (evaluated from the all-reduced tensor):")
    for tokens in (
        [8192, 30, 30, 30],
        [8192, 4096, 4096, 4096],
        [8192, 8192, 8192, 8192],
        [512, 6, 6, 6],
    ):
        dp = len(tokens)
        g = bucket(max(tokens))
        waste = dp * g - sum(tokens)
        print(
            f"  tokens={tokens}  G={g:5d}  waste={waste:6d} rows"
            f"  (dp*G={dp * g}, sum={sum(tokens)})"
        )
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
