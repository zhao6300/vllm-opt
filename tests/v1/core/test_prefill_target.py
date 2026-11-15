# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared token target that sizes prefill chunks and follows the step period."""

import time

import pytest

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


def test_target_grows_when_steps_are_faster_than_the_budget():
    target = make_target(4096)
    feed(target, 0.1, 6)
    assert target.update() == round(4096 * 1.25)
    feed(target, 0.1, 6, clock=100.0)
    assert target.update() == round(round(4096 * 1.25) * 1.25)


def test_target_shrinks_when_steps_are_slower_than_the_budget():
    target = make_target(4096)
    # 800 ms against a 600 ms budget: the period is 4/3 of the budget, the
    # target falls by the gain of one half times the error, to 7/8.
    feed(target, 0.8, 6)
    assert target.update() == round(4096 * 0.875)


def test_target_stays_within_its_bounds():
    high = make_target(MAX_TOKENS - 10)
    feed(high, 0.01, 6)
    assert high.update() == MAX_TOKENS
    low = make_target(MIN_TOKENS + 10)
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
    assert target.update() == round(4096 * 1.25)


def test_a_stall_cannot_collapse_the_target():
    target = make_target(4096)
    feed(target, 100.0, 6)
    # A stall counts as four budgets, so the target falls by the limit of one
    # update, not to its floor.
    assert target.update() == round(4096 * 0.8)


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
