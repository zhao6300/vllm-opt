# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refreshing the adaptive-verification cost tables off the step loop."""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed import parallel_state
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    build_cost_tables_from_curves,
)
from vllm.v1.worker.gpu.spec_decode.cost_refresh import (
    CostRefresher,
    curve_to_dense,
    dense_to_curve,
)

MAX_REQS = 8
MAX_TOKENS = 64
BASE_DRAFT = [(1, 1.0), (8, 1.0)]
BASE_VERIFY = [(8, 8.0), (64, 64.0)]
ALONE = SimpleNamespace(world_size=1, ranks=[0], cpu_group=None)


def build_tables(draft, verify):
    return build_cost_tables_from_curves(draft, verify, MAX_REQS, MAX_TOKENS)


def timing(tokens: int, forward_ms: float, reqs: int = 2, drafter_ms: float = 0.5):
    return StepTimingSample(forward_ms, drafter_ms, tokens, reqs, full_cudagraph=True)


def make_refresher(installs, step, interval=16, tp=ALONE, pp=ALONE, build=build_tables):
    return CostRefresher(
        BASE_DRAFT,
        BASE_VERIFY,
        MAX_REQS,
        MAX_TOKENS,
        interval,
        window=16,
        build_tables=build,
        install=lambda tables: installs.append((step[0], tables)),
        tp_group=tp,
        pp_group=pp,
    )


def run_steps(refresher, step, count):
    for _ in range(count):
        step[0] += 1
        refresher.observe([])
        refresher.tick()


def test_dense_curves_round_trip_and_drop_what_does_not_fit():
    dense = curve_to_dense([(1, 2.0), (5, 3.5), (99, 1.0)], 8)
    assert dense_to_curve(dense) == [(1, 2.0), (5, 3.5)]


def test_new_tables_are_installed_on_a_fixed_step_after_each_rebuild():
    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16)
    try:
        refresher.observe([timing(32, 20.0) for _ in range(4)])
        refresher.flush()
        run_steps(refresher, step, 40)
    finally:
        refresher.close()

    # With no collective to wait for, a rebuild that begins on step 16 installs
    # one lag (4 steps) later; the next begins on step 32.
    assert [at for at, _ in installs] == [20, 36]
    _, verify = installs[0][1]
    assert verify[8] == 8.0
    assert verify[32] == 20.0
    assert verify[64] == 64.0


def test_a_size_keeps_its_startup_cost_until_it_has_enough_timings():
    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16)
    try:
        refresher.observe([timing(32, 20.0) for _ in range(3)])
        refresher.flush()
        run_steps(refresher, step, 30)
    finally:
        refresher.close()

    _, verify = installs[0][1]
    startup = build_tables(BASE_DRAFT, BASE_VERIFY)[1]
    assert np.array_equal(verify, startup)


def test_the_median_of_the_window_is_the_cost():
    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16)
    try:
        refresher.observe([timing(32, ms) for ms in (100.0, 20.0, 21.0, 22.0, 23.0)])
        refresher.flush()
        run_steps(refresher, step, 30)
    finally:
        refresher.close()

    assert installs[0][1][1][32] == 22.0


def test_nothing_is_installed_before_the_first_rebuild_falls_due():
    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16)
    try:
        run_steps(refresher, step, 15)
        assert installs == []
    finally:
        refresher.close()


def test_a_failing_rebuild_stops_the_step_loop_with_the_cause():
    def broken(draft, verify):
        raise ValueError("no table")

    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16, build=broken)
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            run_steps(refresher, step, 60)
    finally:
        refresher.close()
    assert installs == []


def refresh_worker(rank: int, world_size: int, store: str, tp_size: int, pp_size: int):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    ranks = np.arange(world_size).reshape(pp_size, tp_size)
    tp_groups = [list(map(int, row)) for row in ranks]
    pp_groups = [list(map(int, col)) for col in ranks.T]

    def coordinator(groups, name):
        return parallel_state.GroupCoordinator(
            group_ranks=groups,
            local_rank=0,
            torch_distributed_backend="gloo",
            use_device_communicator=False,
            group_name=name,
        )

    tp = coordinator(tp_groups, "tp")
    pp = coordinator(pp_groups, "pp")
    stage = pp.rank_in_group
    tp_rank = tp.rank_in_group

    installs, step = [], [0]
    refresher = make_refresher(installs, step, interval=16, tp=tp, pp=pp)
    # Stages and TP ranks time different costs; the agreed cost is the slowest
    # stage's, from each stage's TP rank 0.
    refresher.observe(
        [
            StepTimingSample(
                20.0 + 10 * stage + tp_rank,
                1.0 + 3 * stage + tp_rank,
                32,
                4,
                full_cudagraph=True,
            )
            for _ in range(4)
        ]
    )
    refresher.flush()
    run_steps(refresher, step, 30)
    refresher.close()

    mine = (installs[0][0], installs[0][1][0].tolist(), installs[0][1][1].tolist())
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, mine)
    tp.destroy()
    pp.destroy()
    dist.destroy_process_group()

    assert len(installs) == 1
    assert all(entry == gathered[0] for entry in gathered), gathered
    install_step, draft, verify = gathered[0]
    # One lag per stage: the rebuild, a TP broadcast, a PP gather.
    assert install_step == 16 + 4 * (1 + (tp_size > 1) + (pp_size > 1))
    assert verify[32] == 20.0 + 10 * (pp_size - 1)
    assert draft[4] == 1.0 + 3 * (pp_size - 1)


@pytest.mark.parametrize(
    ("tp_size", "pp_size"), [(2, 1), (1, 2), (2, 2)], ids=["tp2", "pp2", "tp2-pp2"]
)
def test_every_rank_installs_the_same_tables_on_the_same_step(
    tp_size, pp_size, tmp_path
):
    """TP ranks take rank 0's timings and PP stages the slowest stage's, over
    asynchronous collectives, and all of them switch on the same served step."""
    world_size = tp_size * pp_size
    mp.spawn(
        refresh_worker,
        args=(world_size, str(tmp_path / "store"), tp_size, pp_size),
        nprocs=world_size,
    )
