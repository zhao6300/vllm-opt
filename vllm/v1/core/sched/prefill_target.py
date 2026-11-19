# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A shared per-step token target that sizes the prefill chunks.

A data-parallel MoE step ends when its largest rank does, and the smaller ranks
pad up to it. `PrefillTarget` is the number of tokens every rank fills its step
to: a rank takes `target - its decode tokens` of prefill, so a rank with fewer
decodes fills the room it would otherwise pad, and ranks end their steps
together. The target follows the measured step period toward a time budget.

Only the scheduler's own thread touches the hot path: `record_output` is O(1)
and `max_prefill_tokens` reads one integer. The target is adjusted by a
background thread (or by calling `update` directly).
"""

import itertools
import threading
from collections import deque
from dataclasses import dataclass
from statistics import median

import vllm.envs as envs


@dataclass(frozen=True)
class _Sample:
    seq: int
    period_s: float
    total_tokens: int
    prefill_tokens: int


class PrefillTarget:
    """Token count each step is filled to, adapted to the measured step period.

    Attributes:
        target_step_ms: Step period the target steers toward.
        min_tokens: Smallest target.
        max_tokens: Largest target, normally `max_num_batched_tokens`.
        gain: Share of the relative error corrected per update, in (0, 1].
        max_factor: Largest relative change of the target per update, above 1.
        fill_fraction: A step informs the update only when it carried at least
            this share of the target and some prefill.
        min_samples: Informing steps needed before the target changes.
        stall_factor: A period longer than this many budgets counts as this
            long, so an idle gap or a one-off stall cannot collapse the target.

    """

    def __init__(
        self,
        target_step_ms: float,
        initial_tokens: int,
        min_tokens: int,
        max_tokens: int,
        gain: float = 0.5,
        max_factor: float = 1.25,
        fill_fraction: float = 0.8,
        min_samples: int = 4,
        stall_factor: float = 4.0,
        max_samples: int = 4096,
    ) -> None:
        if target_step_ms <= 0:
            raise ValueError(
                f"The step period target must be positive: {target_step_ms}"
            )
        if not 0 < min_tokens <= max_tokens:
            raise ValueError(
                f"Need 0 < min_tokens <= max_tokens: {min_tokens}, {max_tokens}"
            )
        if not 0 < gain <= 1 or max_factor <= 1:
            raise ValueError(
                f"Need 0 < gain <= 1 and max_factor > 1: {gain}, {max_factor}"
            )
        self.target_step_ms = target_step_ms
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.gain = gain
        self.max_factor = max_factor
        self.fill_fraction = fill_fraction
        self.min_samples = min_samples
        self.stall_factor = stall_factor
        self._target = min(max(initial_tokens, min_tokens), max_tokens)
        # `_last_output` and `_seq` belong to the recording thread.
        self._last_output: float | None = None
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._samples: deque[_Sample] = deque(maxlen=max_samples)
        # `_consumed` and `_target` are written by the updating thread only.
        self._consumed = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, target_step_ms: float, max_tokens: int) -> "PrefillTarget":
        """The target the environment configures.

        Args:
            target_step_ms: The step time budget to steer toward.
            max_tokens: The most tokens one step can carry.

        Returns:
            A target starting at `VLLM_PREFILL_TARGET_INIT_TOKENS`, or a quarter
            of `max_tokens` when that is 0.

        """
        min_tokens = min(max(2 * envs.VLLM_PREFILL_MIN_CHUNK_TOKENS, 512), max_tokens)
        initial = envs.VLLM_PREFILL_TARGET_INIT_TOKENS or max_tokens // 4
        return cls(target_step_ms, initial, min_tokens, max_tokens)

    @property
    def tokens(self) -> int:
        """The current target; a single read, safe from any thread."""
        return self._target

    def max_prefill_tokens(
        self, decode_tokens: int, min_chunk_tokens: int, align: int = 1
    ) -> int:
        """Prefill tokens that fill a step to the target.

        Args:
            decode_tokens: Decode tokens this rank's step already carries.
            min_chunk_tokens: Fewest prefill tokens a step still carries when
                the decodes alone reach the target.
            align: Round a chunk longer than this down to a multiple of it.

        Returns:
            The chunk length, at least `min_chunk_tokens`.

        """
        chunk = self._target - decode_tokens
        if align > 1 and chunk > align:
            chunk -= chunk % align
        return max(chunk, min_chunk_tokens)

    def record_output(self, now: float, total_tokens: int, prefill_tokens: int) -> None:
        """Note that a step's output arrived. O(1); call from the scheduler.

        The step period is the time since the previous output, which is the
        period of the pipeline in steady state, with async scheduling and PP.

        Args:
            now: Monotonic time of the arrival, in seconds.
            total_tokens: Tokens the step carried.
            prefill_tokens: How many of them were prefill.

        """
        last, self._last_output = self._last_output, now
        if last is None:
            return
        sample = _Sample(next(self._seq), now - last, total_tokens, prefill_tokens)
        with self._lock:
            self._samples.append(sample)

    def ingest(self, period_s: float, total_tokens: int, prefill_tokens: int) -> None:
        """Add a step whose period the caller measured, e.g. merged over the DP
        group so that every rank updates from the same steps."""
        sample = _Sample(next(self._seq), period_s, total_tokens, prefill_tokens)
        with self._lock:
            self._samples.append(sample)

    def update(self) -> int:
        """Move the target toward the step period target.

        Only steps that carried prefill and filled the target tell how long a
        full step takes, so the others are ignored. The target changes by at
        most `max_factor` per call and stays within its bounds. It keeps its
        value until `min_samples` informing steps are in.

        Returns:
            The target after the update.

        """
        with self._lock:
            samples = [s for s in self._samples if s.seq > self._consumed]
        target = self._target
        floor = self.fill_fraction * target
        cutoff = self.stall_factor * self.target_step_ms / 1000
        periods = [
            min(s.period_s, cutoff)
            for s in samples
            if s.prefill_tokens > 0 and s.total_tokens >= floor
        ]
        if len(periods) < self.min_samples:
            return target
        self._consumed = samples[-1].seq
        ratio = self.target_step_ms / 1000 / median(periods)
        factor = min(
            max(1 + self.gain * (ratio - 1), 1 / self.max_factor), self.max_factor
        )
        self._target = min(
            max(round(target * factor), self.min_tokens), self.max_tokens
        )
        return self._target

    def start(self, interval_s: float) -> None:
        """Run `update` every `interval_s` seconds on a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("The prefill target updater is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(interval_s,), name="prefill-target", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the updating thread, if it runs."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            self.update()
