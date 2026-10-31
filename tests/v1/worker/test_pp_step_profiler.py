# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.worker.gpu.pp_step_profiler import PPStepProfiler, StepStats

pytestmark = pytest.mark.cpu_test


class _Clock:
    """Fake CUDA events: each record() takes the next timestamp."""

    def __init__(self) -> None:
        self.now = 0.0
        self.done_until = float("inf")

    def event(self):
        clock = self

        class _Event:
            def record(self):
                self.t = clock.now

            def query(self):
                return self.t <= clock.done_until

            def elapsed_time(self, other):
                return other.t - self.t

        return _Event()


def _profiler(clock: _Clock, interval: int, logs: list):
    return PPStepProfiler(
        interval,
        "pp_rank=1 dp_rank=0",
        event_factory=clock.event,
        log=lambda fmt, *args: logs.append(fmt % args),
    )


def _step(
    prof: PPStepProfiler, clock: _Clock, recv: float, execute: float, idle: float
):
    prof.begin_step()
    with prof.span("recv_wait"):
        clock.now += recv
    with prof.span("execute"):
        clock.now += execute
    clock.now += idle


def test_reports_per_step_averages_and_busy_ratio():
    clock = _Clock()
    logs: list[str] = []
    prof = _profiler(clock, interval=2, logs=logs)
    for _ in range(3):
        _step(prof, clock, recv=2.0, execute=6.0, idle=2.0)
    prof.begin_step()  # Lets the third step close its cadence interval.

    assert len(logs) == 1
    assert "over 2 steps" in logs[0]
    assert "step 10.00" in logs[0]
    assert "execute 6.00 (busy 60%)" in logs[0]
    assert "recv_wait 2.00" in logs[0]
    # Stats restart after a report; the third step is already counted.
    assert prof.stats.num_steps == 1


def test_unfinished_steps_are_not_read():
    clock = _Clock()
    logs: list[str] = []
    prof = _profiler(clock, interval=1, logs=logs)
    clock.done_until = -1.0  # Nothing has run on the GPU yet.
    _step(prof, clock, recv=1.0, execute=1.0, idle=0.0)
    _step(prof, clock, recv=1.0, execute=1.0, idle=0.0)
    assert logs == [] and prof.stats.num_steps == 0

    clock.done_until = float("inf")
    prof.begin_step()
    assert len(logs) == 2


def test_spans_outside_a_step_are_ignored():
    clock = _Clock()
    logs: list[str] = []
    prof = _profiler(clock, interval=1, logs=logs)
    with prof.span("draft"):
        clock.now += 5.0
    assert prof.stats.num_steps == 0


def test_repeated_spans_in_a_step_add_up():
    stats = StepStats()
    stats.add(10.0, {"execute": 4.0, "draft": 1.0})
    stats.add(None, {"execute": 6.0})
    s = stats.summary()
    assert s["execute"] == 5.0
    assert s["draft"] == 0.5
    assert s["step"] == 10.0
    assert s["busy"] == 0.5


def test_token_wait_inside_execute_is_not_busy_time():
    # Stage 0: 6 ms execute of which 2 ms wait for the tokens of an earlier
    # step; stage 1 style 3 ms sampling (1 ms of it drafting) after it.
    stats = StepStats()
    stats.add(
        10.0,
        {"execute": 6.0, "token_wait": 2.0, "sample": 3.0, "draft": 1.0},
    )
    s = stats.summary()
    assert s["token_wait"] == 2.0
    assert s["busy"] == 0.4  # (6 - 2) / 10
    # The draft is part of sample: it is not added again.
    assert s["gpu"] == 0.7  # (6 - 2 + 3) / 10


def test_token_wait_span_is_reported():
    clock = _Clock()
    logs: list[str] = []
    prof = _profiler(clock, interval=1, logs=logs)
    prof.begin_step()
    with prof.span("execute"):
        clock.now += 8.0
        with prof.span("token_wait"):
            clock.now += 2.0
    clock.now += 2.0
    prof.begin_step()
    assert "token_wait 2.00" in logs[0]
    # execute spans 10 ms including the wait; the step is 12 ms.
    assert "execute 10.00 (busy 67%)" in logs[0]
