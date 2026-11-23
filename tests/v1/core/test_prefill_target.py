# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared token target that sizes prefill chunks and follows the step period."""

import time

import pytest

from vllm.config.compilation import prefill_cudagraph_capture_grid
from vllm.v1.core.sched.prefill_target import PrefillTarget

BUDGET_MS = 600
MIN_TOKENS = 512
MAX_TOKENS = 8192


def make_target(initial: int = 4096, **kwargs) -> PrefillTarget:
    return PrefillTarget(BUDGET_MS, initial, MIN_TOKENS, MAX_TOKENS, **kwargs)


def feed(
    target: PrefillTarget, period_s: float, count: int, clock: float = 0.0
) -> float:
    """Record `count` steps that each took `period_s` and filled the target."""
    for _ in range(count):
        clock += period_s
        target.record_output(clock, target.tokens, prefill_tokens=target.tokens // 2)
    return clock


def feed_prefill_only(
    target: PrefillTarget, period_s: float, count: int, clock: float = 0.0
) -> float:
    """Record `count` prefill-only steps that each took `period_s`."""
    for _ in range(count):
        clock += period_s
        target.record_output(clock, target.tokens, prefill_tokens=target.tokens)
    return clock


def test_chunk_fills_the_step_to_the_target():
    target = make_target(4096)
    assert target.max_prefill_tokens(384, min_chunk_tokens=256, align=64) == 3712
    # Decodes alone reach the target: the minimum chunk keeps prefill moving.
    assert target.max_prefill_tokens(5000, min_chunk_tokens=256, align=64) == 256


def test_ranks_with_fewer_decodes_take_more_prefill_and_end_together():
    target = make_target(4096)
    decodes = [0, 100, 300, 384]
    totals = [
        d + target.max_prefill_tokens(d, min_chunk_tokens=256, align=64)
        for d in decodes
    ]
    chunks = [t - d for t, d in zip(totals, decodes, strict=True)]
    assert chunks == sorted(chunks, reverse=True)
    assert max(totals) - min(totals) < 64
    assert max(totals) <= target.tokens


def test_an_update_solves_for_the_target_that_fills_the_budget():
    """A step that took a twelfth of the budget needs twelve times the tokens
    to fill it; the update goes straight there, bounded by the ceiling."""
    target = make_target(2048)
    feed(target, 0.05, 6)
    assert target.update() == MAX_TOKENS


def test_target_shrinks_when_steps_are_slower_than_the_budget():
    target = make_target(4096)
    # 800 ms against a 600 ms budget: three quarters of the tokens fill it.
    feed(target, 0.8, 6)
    assert target.update() == 3072


def test_the_solve_leaves_out_the_fixed_cost_of_a_step():
    """A 200 ms step is 190 ms of tokens and 10 ms of fixed cost: filling 600 ms
    takes 590/190 times the tokens, not 3 times of the whole period."""
    target = PrefillTarget(BUDGET_MS, 2048, MIN_TOKENS, MAX_TOKENS, fixed_ms=10.0)
    feed(target, 0.2, 6)
    assert target.update() == 6400


def test_a_fitted_fixed_cost_beyond_the_period_cannot_invert_the_solve():
    """The fit can report a fixed cost as large as the period it was fitted
    from (176 ms of a 175 ms step). Bounded to half of each, the solve still
    moves the target by the ratio of what the tokens really cost."""
    target = PrefillTarget(250, 2048, MIN_TOKENS, MAX_TOKENS, fixed_ms=140.0)
    feed(target, 0.1745, 6)
    # fixed 87.25: (250 - 87.25) / (174.5 - 87.25) = 1.865 -> 3820 -> 3840.
    assert target.update() == 3840


def test_steps_that_carry_decodes_steer_to_the_decode_period():
    """A step with decodes makes them wait for it, so while the recent steps
    carry decodes the target fills the shorter period, not the whole budget."""
    target = PrefillTarget(BUDGET_MS, 2048, MIN_TOKENS, MAX_TOKENS, decode_step_ms=200)
    feed(target, 0.1, 6)
    assert target.update() == 4096  # 200 ms buys twice the tokens of 100


def test_decode_free_steps_use_the_whole_budget():
    target = PrefillTarget(BUDGET_MS, 2048, MIN_TOKENS, MAX_TOKENS, decode_step_ms=200)
    feed_prefill_only(target, 0.1, 6)
    assert target.update() == MAX_TOKENS  # 600 ms buys six times as many


def test_one_period_serves_steps_with_and_without_decodes():
    target = make_target(2048, decode_step_ms=0)
    feed_prefill_only(target, 0.3, 6)
    assert target.update() == 4096  # 600 ms of 300 ms steps


def test_target_stays_within_its_bounds():
    high = make_target(MAX_TOKENS - 100)
    assert high.tokens == MAX_TOKENS - high.quantum
    feed(high, 0.01, 6)
    assert high.update() == MAX_TOKENS
    low = make_target(MIN_TOKENS + 100)
    feed(low, 2.0, 6)
    assert low.update() == MIN_TOKENS


def test_steps_that_say_nothing_about_a_full_step_are_ignored():
    target = make_target(4096)
    clock = 0.0
    for total, prefill in [(4096, 0), (100, 50), (4000, 0), (50, 0), (3000, 2000)]:
        clock += 0.01
        target.record_output(clock, total, prefill)
    target.record_output(clock + 0.01, 4096, 2000)
    # One step filled the target and carried prefill: too few to act on.
    assert target.update() == 4096


def test_target_waits_for_enough_steps():
    target = make_target(4096, min_samples=4)
    feed(target, 0.1, 4)  # the first output has no period: three samples
    assert target.update() == 4096
    feed(target, 0.1, 1, clock=10.0)
    assert target.update() == MAX_TOKENS


def test_a_stall_cannot_collapse_the_target():
    target = make_target(4096)
    feed(target, 100.0, 6)
    # A stall counts as four budgets, so the target falls by the limit of one
    # update, not to its floor.
    assert target.update() == 1024


def test_the_target_moves_on_a_grid_that_repeats():
    """Chunk lengths become kernel shapes, so the target moves on a coarse grid:
    a handful of shapes repeat instead of a new one after every update."""
    target = make_target(MIN_TOKENS, quantum=MIN_TOKENS)
    clock = 0.0
    seen = []
    for _ in range(40):
        clock = feed(target, 0.05, 6, clock)
        seen.append(target.update())
    assert all(tokens % MIN_TOKENS == 0 for tokens in seen)
    # Cheap steps take it from the floor to the ceiling, one grid step at a
    # time: a snapped value equal to the target must still move it.
    assert seen[0] > MIN_TOKENS
    assert seen[-1] == MAX_TOKENS
    assert len(set(seen)) <= MAX_TOKENS // MIN_TOKENS


def test_the_target_moves_between_captured_prefill_graph_sizes():
    """A step replays a captured graph only at a captured size, so the target
    takes only the sizes the runner captured, and every captured size."""
    grid = prefill_cudagraph_capture_grid(MAX_TOKENS)
    target = make_target(grid=grid)
    clock = 0.0
    seen = [target.tokens]
    for _ in range(20):
        clock = feed(target, 0.05, 6, clock)
        seen.append(target.update())
    assert all(tokens in set(grid) for tokens in seen)
    assert seen[-1] == MAX_TOKENS


def test_the_target_stops_at_the_last_captured_graph():
    target = make_target(4096, grid=prefill_cudagraph_capture_grid(1024))
    assert target.tokens == 1024
    assert target.ceiling == 1024
    feed(target, 0.05, 6)
    assert target.update() == 1024


def test_a_change_smaller_than_the_grid_moves_one_point():
    target = make_target(1024, grid=[512, 1024, 2048])
    # 10% fast, a 5% gain: 1126 rounds back to 1024, which would freeze it.
    feed(target, 0.545, 6)
    assert target.update() == 2048
    # 8% slow: 1968 rounds back to 2048, so it moves down one point instead.
    feed(target, 0.65, 6, clock=10.0)
    assert target.update() == 1024


def test_an_update_covers_only_the_samples_it_is_given():
    """A cost fit round judges its own merged steps: the local ones a target
    already moved on say nothing about the period the group measured."""
    target = make_target(4096)
    feed(target, 0.1, 6)
    seqs = [target.ingest(0.8, 4096, 2048) for _ in range(4)]
    assert target.update(since=min(seqs)) == 3072


def test_steps_are_not_counted_twice():
    target = make_target(4096)
    feed(target, 0.1, 6)
    first = target.update()
    assert target.update() == first


def test_target_settles_where_the_step_period_meets_the_budget():
    """A step takes 20 ms plus 0.04 ms per token; 150 ms is about 3250 tokens."""
    target = PrefillTarget(150, 1024, MIN_TOKENS, MAX_TOKENS)
    clock = 0.0
    for _ in range(40):
        period = 0.020 + 0.00004 * target.tokens
        clock = feed(target, period, 6, clock)
        target.update()
    assert target.tokens == pytest.approx(3250, rel=0.05)


def test_target_settles_at_the_ceiling_when_steps_are_cheap():
    target = PrefillTarget(600, 1024, MIN_TOKENS, MAX_TOKENS)
    clock = 0.0
    for _ in range(40):
        clock = feed(target, 0.020 + 0.00004 * target.tokens, 6, clock)
        target.update()
    assert target.tokens == MAX_TOKENS


def test_background_thread_moves_the_target():
    target = make_target(4096)
    feed(target, 0.1, 6)
    target.start(interval_s=0.005)
    try:
        deadline = time.monotonic() + 10
        while target.tokens == 4096 and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        target.stop()
    assert target.tokens > 4096


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_step_ms": 0},
        {"min_tokens": 0},
        {"min_tokens": MAX_TOKENS + 1},
        {"gain": 0},
        {"max_factor": 1.0},
        {"fixed_ms": -1.0},
        {"deadband": 1.0},
        {"decode_step_ms": -1.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    settings = {
        "target_step_ms": BUDGET_MS,
        "initial_tokens": 4096,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
    } | kwargs
    with pytest.raises(ValueError):
        PrefillTarget(**settings)
