# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-order MoE combine of the allgather-reducescatter backend (gloo, CPU).

A token's combined output must not depend on which rank owns it: with three or
more ranks, NCCL's reduce-scatter adds the partials in an order set by the
destination rank, so a token moved across a shard boundary (a sequence-parallel
split shifting with the batch) changed by one ulp and greedy decoding diverged.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.all2all import (
    _reduce_scatterv_fixed_order,
)
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.cpu_test

NUM_TOKENS = 48
HIDDEN = 64


def _partials(world_size: int) -> torch.Tensor:
    """`[rank, token, hidden]` bf16 partial sums spanning many binades, so the
    order of addition changes the rounded result for some tokens."""
    gen = torch.Generator().manual_seed(0)
    values = torch.randn(world_size, NUM_TOKENS, HIDDEN, generator=gen)
    scale = torch.exp2(
        torch.randint(-6, 7, (world_size, NUM_TOKENS, HIDDEN), generator=gen)
    )
    return (values * scale).to(torch.bfloat16)


def _rank_order_sum(partials: torch.Tensor) -> torch.Tensor:
    acc = partials[0].float()
    for rank in range(1, partials.shape[0]):
        acc += partials[rank].float()
    return acc.to(torch.bfloat16)


def _bf16_ring_sum(partials: torch.Tensor, first: int) -> torch.Tensor:
    """A bf16 ring reduction starting at rank `first`: rounded after each add."""
    world_size = partials.shape[0]
    acc = partials[first]
    for step in range(1, world_size):
        acc = (acc.float() + partials[(first + step) % world_size].float()).to(
            torch.bfloat16
        )
    return acc


# Owners of the tokens, in gathered order. The first two layouts are the
# sequence-parallel split before and after six decode rows push tokens 18-23
# onto the next rank; the others are ragged, with empty ranks.
def _layouts(world_size: int) -> list[list[int]]:
    even = [NUM_TOKENS // world_size] * world_size
    shifted = list(even)
    shifted[0] -= 6
    shifted[1] += 6
    ragged = [0] * world_size
    ragged[-1], ragged[0] = 30, NUM_TOKENS - 30
    return [even, shifted, ragged, [NUM_TOKENS] + [0] * (world_size - 1)]


def _worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="ag_rs_fixed_order_combine_test",
    )
    # The CPU path of every platform: torch.distributed all-to-all over gloo.
    group.device_communicator = DeviceCommunicatorBase(
        group.cpu_group, torch.device("cpu"), group.cpu_group
    )

    partials = _partials(world_size)
    expected = _rank_order_sum(partials)
    # A bf16 ring reduction, whose start depends on the owner, must be order
    # sensitive on these inputs, or the test proves nothing.
    assert not torch.equal(_bf16_ring_sum(partials, 0), _bf16_ring_sum(partials, 1))

    for sizes in _layouts(world_size):
        out = _reduce_scatterv_fixed_order(group, partials[rank], sizes)
        start = sum(sizes[:rank])
        assert out.shape == (sizes[rank], HIDDEN)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out, expected[start : start + sizes[rank]], rtol=0, atol=0
        )

    group.destroy()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [3, 4])
def test_fixed_order_combine_ignores_the_owning_rank(world_size: int) -> None:
    """Every layout gives each token the bits of one fixed-order sum."""
    mp.spawn(_worker, args=(world_size, get_open_port()), nprocs=world_size)
