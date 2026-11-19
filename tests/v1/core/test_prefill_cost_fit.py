# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The online fit of the prefill step cost model."""

import pickle
import sys
from datetime import timedelta

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.core.sched.cost_fit import StepCostFitter, StepCostSolver, StepSample
from vllm.v1.core.sched.prefill_budget import PrefillCostModel

PRIOR = (0.0, 0.021, 4e-6)
TRUE = (20.0, 0.021, 1.5e-6)
BUDGET_MS = 700.0
INTERVAL = 16


def make_steps(count: int, seed: int = 0, truth=TRUE, noise: float = 0.02):
    """Steps of a lockstep DP deployment: tokens and attention work as the
    slowest rank carried them, and the period they gave."""
    rng = np.random.default_rng(seed)
    steps = []
    for key in range(count):
        tokens = int(rng.choice([512, 1024, 2048, 4096, 8192]))
        context = float(rng.choice([2e3, 16e3, 100e3, 400e3]))
        chunk = float(rng.choice([256, 512, 1024, 2048]))
        attn = chunk * (context + chunk / 2)
        period = truth[0] + truth[1] * tokens + truth[2] * attn
        steps.append(
            StepSample(
                key, period * (1 + noise * rng.standard_normal()), tokens, attn, 1
            )
        )
    return steps


def make_solver(prior=PRIOR, **kwargs) -> StepCostSolver:
    return StepCostSolver(prior, max_fixed_ms=BUDGET_MS / 4, **kwargs)


def test_the_fit_recovers_coefficients_far_from_the_configured_ones():
    solver = make_solver()
    for step in make_steps(400):
        solver.add(step)
    fixed, per_token, per_context = solver.fit()
    assert per_context == pytest.approx(TRUE[2], rel=0.05)
    assert per_token == pytest.approx(TRUE[1], rel=0.1)
    assert fixed == pytest.approx(TRUE[0], abs=10.0)


def test_the_prior_holds_until_enough_steps_are_in():
    solver = make_solver()
    for step in make_steps(15):
        solver.add(step)
    assert solver.fit() == PRIOR


def test_a_term_the_data_does_not_vary_stays_at_its_prior():
    solver = make_solver()
    for step in make_steps(300):
        solver.add(StepSample(step.key, step.period_ms, step.tokens, 0.0, 1))
    _, _, per_context = solver.fit()
    assert per_context == PRIOR[2]


def test_a_fit_stays_within_a_factor_of_the_prior():
    solver = make_solver()
    for step in make_steps(400, truth=(0.0, 0.021, 1e-8), noise=0.0):
        solver.add(step)
    assert solver.fit()[2] == pytest.approx(PRIOR[2] / 4)


def test_a_term_configured_as_zero_can_be_fitted():
    solver = make_solver(prior=(0.0, 0.021, 0.0))
    for step in make_steps(400):
        solver.add(step)
    assert solver.fit()[2] == pytest.approx(TRUE[2], rel=0.1)


def test_recent_steps_outweigh_old_ones():
    solver = make_solver(half_life=32.0, window=512)
    for step in make_steps(300, seed=1, truth=(20.0, 0.021, 4e-6)):
        solver.add(step)
    for step in make_steps(300, seed=2, truth=(20.0, 0.021, 1.5e-6)):
        solver.add(step)
    assert solver.fit()[2] == pytest.approx(1.5e-6, rel=0.1)


def make_fitter(fits: list, interval: int = INTERVAL) -> StepCostFitter:
    return StepCostFitter(
        make_solver(),
        interval=interval,
        max_period_ms=4 * BUDGET_MS,
        on_fit=lambda theta, samples: fits.append((theta, samples)),
    )


def record_steps(fitter: StepCostFitter, steps) -> None:
    clock = 20.0
    fitter.record(clock, -1, 0, 0.0, 0)
    for step in steps:
        clock += step.period_ms / 1000
        fitter.record(clock, -1, step.tokens, step.attn_work, 1)


def test_without_peers_the_fitter_fits_every_interval():
    fits: list = []
    fitter = make_fitter(fits)
    record_steps(fitter, make_steps(3 * INTERVAL + 5))
    assert len(fits) == 3
    assert all(len(samples) == INTERVAL for _, samples in fits)
    keys = [s.key for _, samples in fits for s in samples]
    assert keys == sorted(set(keys))


def test_an_interval_without_steps_changes_nothing():
    fits: list = []
    fitter = make_fitter(fits)
    fitter.tick(None, INTERVAL)
    assert fits == [(PRIOR, [])]


def test_an_idle_gap_is_not_a_step():
    fits: list = []
    fitter = make_fitter(fits)
    fitter.record(0.0, -1, 0, 0.0, 0)
    fitter.record(10.0, -1, 4096, 1e6, 1)  # ten seconds: the engine sat idle
    record_steps(fitter, make_steps(INTERVAL))
    ((_, samples),) = fits
    assert len(samples) == INTERVAL
    assert max(s.period_ms for s in samples) < 4 * BUDGET_MS


def merged_expected(per_rank: list[list[StepSample]]):
    by_key: dict[int, list[StepSample]] = {}
    for rank in per_rank:
        for s in rank:
            by_key.setdefault(s.key, []).append(s)
    return [
        (
            key,
            max(s.tokens for s in rows),
            max(s.attn_work for s in rows),
        )
        for key, rows in sorted(by_key.items())
    ]


def rank_steps(rank: int, world_size: int, count: int):
    """Each rank's own tokens and attention work for the same lockstep steps; the
    period is the slowest rank's, so no rank's own figures explain it."""
    rng = np.random.default_rng(10)
    steps = []
    for key in range(count):
        tokens = rng.integers(256, 8192, size=world_size)
        attn = 10 ** rng.uniform(6, 9, size=world_size)
        period = TRUE[0] + TRUE[1] * tokens.max() + TRUE[2] * attn.max()
        steps.append(
            StepSample(key, float(period), int(tokens[rank]), float(attn[rank]), 1)
        )
    return steps


def exchange_worker(rank: int, world_size: int, store: str, result_path: str):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    fits: list = []
    fitter = make_fitter(fits)
    own = rank_steps(rank, world_size, 6 * INTERVAL)
    clock = 0.0
    fitter.record(clock, 0, 0, 0.0, 0)
    step_counter = 0
    for i, step in enumerate(own):
        clock += step.period_ms / 1000
        fitter.record(clock, 1 + i, step.tokens, step.attn_work, 1)
        step_counter += 1
        fitter.tick(dist.group.WORLD, step_counter)
        # The engine's own DP sync, between the issue and the wait of an exchange.
        dist.barrier()
        if i == 3 * INTERVAL + 4:
            # A wave ends with an exchange in flight; the counter starts over.
            step_counter = 0
    gathered: list = [None] * world_size
    mine = [
        (theta, [(s.key, s.tokens, s.attn_work) for s in samples])
        for theta, samples in fits
    ]
    dist.all_gather_object(gathered, mine)
    if rank == 0:
        with open(result_path, "wb") as f:
            pickle.dump(gathered, f)
    dist.destroy_process_group()


def test_every_rank_fits_the_same_steps_and_installs_the_same_coefficients(tmp_path):
    """Each rank sees its own tokens and attention work, the period is the
    slowest rank's, and the exchange over the DP group gives all of them the same
    merged steps, so the fit finds the real coefficients that no rank's own
    figures would."""
    world_size = 3
    result = tmp_path / "result.pkl"
    mp.spawn(
        exchange_worker,
        args=(world_size, str(tmp_path / "store"), str(result)),
        nprocs=world_size,
    )
    gathered = pickle.loads(result.read_bytes())

    assert all(entry == gathered[0] for entry in gathered)
    fits = gathered[0]
    # The exchange started at an interval completes at the next one, or when a
    # new wave starts, so none of the steps sent is lost.
    assert len(fits) == 4
    expected = merged_expected(
        [
            [
                StepSample(s.key + 1, s.period_ms, s.tokens, s.attn_work, 1)
                for s in rank_steps(r, world_size, 6 * INTERVAL)
            ]
            for r in range(world_size)
        ]
    )
    sent = [(k, t, a) for fit in fits for (k, t, a) in fit[1]]
    assert sent == [e for e in expected if e[0] in {k for k, _, _ in sent}]
    _, per_token, per_context = fits[-1][0]
    assert per_context == pytest.approx(TRUE[2], rel=0.1)
    assert per_token == pytest.approx(TRUE[1], rel=0.15)


def test_fitting_on_the_local_figures_alone_is_biased():
    """The measured period belongs to the slowest rank, so a rank that fits its
    own tokens and attention work gets the coefficients wrong. This is why the
    steps are exchanged."""
    world_size = 3
    local = make_solver()
    merged = make_solver()
    per_rank = [rank_steps(r, world_size, 400) for r in range(world_size)]
    for step in per_rank[0]:
        local.add(step)
    for i in range(400):
        rows = [per_rank[r][i] for r in range(world_size)]
        merged.add(
            StepSample(
                i,
                rows[0].period_ms,
                max(r.tokens for r in rows),
                max(r.attn_work for r in rows),
                1,
            )
        )
    assert merged.fit()[2] == pytest.approx(TRUE[2], rel=0.05)
    assert abs(local.fit()[2] - TRUE[2]) > 0.2 * TRUE[2]


def make_model(c: float = 1e-3, a: float = 0.01, budget: float = 600.0):
    return PrefillCostModel(budget, 0.0, a, c, min_chunk_tokens=16)


@pytest.mark.parametrize("context", [0, 1000, 10_000])
def test_the_attention_cap_fills_the_room_exactly(context):
    model = make_model()
    chunk = model.max_attention_tokens(500.0, context)
    cost = model.ms_per_token_context * chunk * (context + chunk / 2)
    assert cost <= 500.0
    assert (
        model.ms_per_token_context * (chunk + 1) * (context + (chunk + 1) / 2) > 500.0
    )


def test_the_attention_cap_shrinks_as_the_context_grows():
    model = make_model()
    caps = [model.max_attention_tokens(500.0, c, align=16) for c in (0, 500, 5000)]
    assert caps == sorted(caps, reverse=True)
    assert caps[-1] < caps[0]


def test_a_context_that_costs_nothing_has_no_attention_cap():
    assert make_model(c=0.0).max_attention_tokens(500.0, 400_000) == sys.maxsize


def test_no_room_left_keeps_the_minimum_chunk():
    assert make_model().max_attention_tokens(0.0, 1000) == 16
    assert make_model().max_attention_tokens(-5.0, 1000) == 16


def test_fitted_coefficients_keep_the_budget_and_the_minimum_chunk():
    model = make_model().with_coefficients(3.0, 0.02, 2e-6)
    assert (model.fixed_ms, model.ms_per_token, model.ms_per_token_context) == (
        3.0,
        0.02,
        2e-6,
    )
    assert (model.budget_ms, model.min_chunk_tokens) == (600.0, 16)
