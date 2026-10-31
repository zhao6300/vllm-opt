# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank PP step timings from CUDA events.

Enabled with ``VLLM_PP_STEP_PROFILE_INTERVAL=N``: every N steps each worker
logs the average GPU time of its steps, split into spans:

- ``step``: GPU time between the starts of consecutive steps (the cadence).
- ``execute``: the model runner's ``execute_model`` (input prep + forward),
  including ``token_wait``.
- ``recv_wait``: the stream waiting for the previous stage's activations.
- ``send_wait``: the stream waiting for this rank's earlier sends to drain
  (with ``VLLM_PP_SEND_BUFFERS``, also the copy into the send buffer).
- ``token_wait``: the stream waiting for the sampled tokens and drafts that
  the last stage broadcast back (non-last stages). A large share of the step
  means the round trip through the stages, not a stage's own work, limits the
  pipeline: the case ``VLLM_PP_NUM_MICROBATCHES`` > ``pp_size`` addresses.
- ``sample``: ``sample_tokens`` on the last stage, including ``draft``.
- ``draft``: the speculative proposer (DSpark) on the last stage.

``busy`` is ``(execute - token_wait) / step`` and ``gpu`` adds the last
stage's sampling and drafting; a stage that is mostly ``recv_wait`` is starved by its
predecessor. Events are recorded on the
current stream outside CUDA graph replay, so FULL graphs are timed as a
whole. Kernel-level splits (EP all-to-all, P2P transfer) come from torch
profiler traces; see ``examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120``.
Completed events are read with ``Event.query()``; the host never waits.
"""

from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SPANS = ("execute", "recv_wait", "send_wait", "token_wait", "sample", "draft")


class StepStats:
    """Running sums of span times (ms), reset after each report."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.num_steps = 0
        self.step_ms = 0.0
        self.num_intervals = 0
        self.span_ms: dict[str, float] = defaultdict(float)

    def add(self, step_interval_ms: float | None, spans: dict[str, float]) -> None:
        self.num_steps += 1
        if step_interval_ms is not None:
            self.step_ms += step_interval_ms
            self.num_intervals += 1
        for name, ms in spans.items():
            self.span_ms[name] += ms

    def summary(self) -> dict[str, float]:
        """Per-step averages; ``busy`` is execute time over the step cadence."""
        steps = max(self.num_steps, 1)
        out = {name: self.span_ms.get(name, 0.0) / steps for name in SPANS}
        out["step"] = self.step_ms / max(self.num_intervals, 1)
        # ``token_wait`` happens inside ``execute`` and is waiting, not work.
        work = max(out["execute"] - out["token_wait"], 0.0)
        out["busy"] = work / out["step"] if out["step"] > 0 else 0.0
        # ``sample`` includes the draft; the last stage's sampling and drafting
        # keep the GPU busy too, which ``busy`` leaves out.
        out["gpu"] = (work + out["sample"]) / out["step"] if out["step"] > 0 else 0.0
        return out


class _Step:
    def __init__(self, start: Any) -> None:
        self.start = start
        self.spans: list[tuple[str, Any, Any]] = []


class PPStepProfiler:
    def __init__(
        self,
        interval: int,
        label: str,
        event_factory: Callable[[], Any] | None = None,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.interval = interval
        self.label = label
        self._event = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self._log = log or logger.info
        self._current: _Step | None = None
        self._pending: deque[_Step] = deque()
        self._last_start: Any = None
        self.stats = StepStats()

    def _record(self) -> Any:
        event = self._event()
        event.record()
        return event

    def begin_step(self) -> None:
        if self._current is not None:
            self._pending.append(self._current)
        self._current = _Step(self._record())
        self._drain()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if self._current is None:
            yield
            return
        start = self._record()
        try:
            yield
        finally:
            assert self._current is not None
            self._current.spans.append((name, start, self._record()))

    def _drain(self) -> None:
        # Steps complete in order; stop at the first one still running. The
        # newest pending step needs its successor's start for the cadence.
        while len(self._pending) >= 1:
            step = self._pending[0]
            successor = self._pending[1] if len(self._pending) > 1 else self._current
            ends = [end for _, _, end in step.spans]
            if not all(end.query() for end in ends):
                return
            if successor is None or not successor.start.query():
                return
            self._pending.popleft()
            spans: dict[str, float] = defaultdict(float)
            for name, start, end in step.spans:
                spans[name] += start.elapsed_time(end)
            self.stats.add(step.start.elapsed_time(successor.start), spans)
            if self.stats.num_steps >= self.interval:
                self.report()

    def report(self) -> None:
        s = self.stats.summary()
        self._log(
            "PP step profile %s over %d steps (ms/step): step %.2f, execute "
            "%.2f (busy %.0f%%), recv_wait %.2f, send_wait %.2f, token_wait "
            "%.2f, sample %.2f, draft %.2f, gpu %.0f%%",
            self.label,
            self.stats.num_steps,
            s["step"],
            s["execute"],
            100 * s["busy"],
            s["recv_wait"],
            s["send_wait"],
            s["token_wait"],
            s["sample"],
            s["draft"],
            100 * s["gpu"],
        )
        self.stats.reset()


_profiler: PPStepProfiler | None = None


def init_pp_step_profiler(interval: int, label: str) -> PPStepProfiler | None:
    global _profiler
    _profiler = PPStepProfiler(interval, label) if interval > 0 else None
    return _profiler


def get_pp_step_profiler() -> PPStepProfiler | None:
    return _profiler


def pp_step_span(name: str):
    """Span on the active profiler, or a no-op; never inside graph capture."""
    if _profiler is None or torch.cuda.is_current_stream_capturing():
        return nullcontext()
    return _profiler.span(name)
