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
background thread (or by calling `update` directly). It moves on a grid of
token counts, normally the captured prefill graph sizes, so every full step
replays a captured graph.
"""

import itertools
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


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
        fixed_ms: Step time the target does not buy, e.g. the fitted fixed cost
            of a step, so each update solves for the tokens that fill the rest.
        decode_step_ms: Step period used instead of `target_step_ms` while the
            recent steps carry decode tokens, so a long prefill does not hold
            them for a whole budget; 0 uses one period for both.
        gain: Share of the relative error corrected per update, in (0, 1]. At
            1 an update solves for the target directly, which a step cost that
            is linear in the tokens reaches in one move.
        max_factor: Largest relative change of the target per update, above 1.
        deadband: Relative distance from the budget within which an update
            leaves the target, so it does not chase measurement noise; a grid
            point the solve lands back on would otherwise flip every update.
        fill_fraction: A step informs the update only when it carried at least
            this share of the target and some prefill.
        min_samples: Informing steps needed before the target changes.
        stall_factor: A period longer than this many budgets counts as this
            long, so an idle gap or a one-off stall cannot collapse the target.
        quantum: Token grid the target moves on, so the chunk lengths the
            kernels see repeat; 0 picks a sixty-fourth of `max_tokens`.
        grid: Token counts the target may take, normally the captured prefill
            graph sizes so that a full step replays a captured graph; it moves
            between these points instead of on the `quantum` grid. Fewer than
            two points within the bounds leave the `quantum` grid.

    """

    def __init__(
        self,
        target_step_ms: float,
        initial_tokens: int,
        min_tokens: int,
        max_tokens: int,
        gain: float = 1.0,
        max_factor: float = 4.0,
        deadband: float = 0.05,
        fill_fraction: float = 0.8,
        min_samples: int = 4,
        stall_factor: float = 4.0,
        max_samples: int = 4096,
        quantum: int = 0,
        grid: Sequence[int] | None = None,
        fixed_ms: float = 0.0,
        decode_step_ms: float = 0.0,
    ) -> None:
        if target_step_ms <= 0:
            raise ValueError(
                f"The step period target must be positive: {target_step_ms}"
            )
        if fixed_ms < 0:
            raise ValueError(f"The fixed step cost must be >= 0 ms: {fixed_ms}")
        if decode_step_ms < 0:
            raise ValueError(
                f"The decode step period must be >= 0 ms: {decode_step_ms}"
            )
        if not 0 < min_tokens <= max_tokens:
            raise ValueError(
                f"Need 0 < min_tokens <= max_tokens: {min_tokens}, {max_tokens}"
            )
        if not 0 < gain <= 1 or max_factor <= 1:
            raise ValueError(
                f"Need 0 < gain <= 1 and max_factor > 1: {gain}, {max_factor}"
            )
        if not 0 <= deadband < 1:
            raise ValueError(f"The deadband must be in [0, 1): {deadband}")
        if quantum < 0:
            raise ValueError(f"The target grid must be >= 0 tokens: {quantum}")
        self.target_step_ms = target_step_ms
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fixed_ms = fixed_ms
        self.decode_step_ms = decode_step_ms
        self.gain = gain
        self.max_factor = max_factor
        self.deadband = deadband
        self.fill_fraction = fill_fraction
        self.min_samples = min_samples
        self.stall_factor = stall_factor
        self.quantum = quantum or max(1, max_tokens // 64)
        self.grid = tuple(
            sorted({t for t in grid or () if min_tokens <= t <= max_tokens})
        )
        if len(self.grid) < 2:
            self.grid = ()
        self._target = self._on_grid(initial_tokens)
        # `_last_output` and `_seq` belong to the recording thread.
        self._last_output: float | None = None
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._update_lock = threading.Lock()
        self._samples: deque[_Sample] = deque(maxlen=max_samples)
        # `_consumed` and `_target` are written by an updating thread only.
        self._consumed = 0
        self._step_ms: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(
        cls,
        target_step_ms: float,
        max_tokens: int,
        grid: Sequence[int] | None = None,
        fixed_ms: float = 0.0,
        decode_step_ms: float = 0.0,
    ) -> "PrefillTarget":
        """The target the environment configures.

        Args:
            target_step_ms: The step time budget to steer toward.
            max_tokens: The most tokens one step can carry.
            grid: Token counts the target may take, normally the captured
                prefill graph sizes.
            fixed_ms: Part of the step period the target does not buy.
            decode_step_ms: The period to steer toward while decodes are in the
                recent steps; 0 uses the budget.

        Returns:
            A target starting at `VLLM_PREFILL_TARGET_INIT_TOKENS`, or a quarter
            of `max_tokens` when that is 0, and kept at least
            `VLLM_PREFILL_FILL_MIN_TOKENS` when that is set.

        """
        min_tokens = envs.VLLM_PREFILL_FILL_MIN_TOKENS or max(
            2 * envs.VLLM_PREFILL_MIN_CHUNK_TOKENS, 512
        )
        min_tokens = min(min_tokens, max_tokens)
        initial = envs.VLLM_PREFILL_TARGET_INIT_TOKENS or max_tokens // 4
        return cls(
            target_step_ms,
            initial,
            min_tokens,
            max_tokens,
            grid=grid,
            fixed_ms=fixed_ms,
            decode_step_ms=decode_step_ms,
        )

    @property
    def tokens(self) -> int:
        """The current target; a single read, safe from any thread."""
        return self._target

    @property
    def floor(self) -> int:
        """Smallest target the grid allows."""
        return self.grid[0] if self.grid else self.min_tokens

    @property
    def ceiling(self) -> int:
        """Largest target the grid allows."""
        return self.grid[-1] if self.grid else self.max_tokens

    @property
    def step_ms(self) -> float | None:
        """Median period of the steps that last informed the target, in
        milliseconds, or None before any did. What a step that fills the target
        takes."""
        return self._step_ms

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

    def ingest(self, period_s: float, total_tokens: int, prefill_tokens: int) -> int:
        """Add a step whose period the caller measured, e.g. merged over the DP
        group so that every rank updates from the same steps.

        Returns:
            The sample's sequence number, to pass to `update`.

        """
        sample = _Sample(next(self._seq), period_s, total_tokens, prefill_tokens)
        with self._lock:
            self._samples.append(sample)
        return sample.seq

    def update(self, since: int | None = None) -> int:
        """Move the target toward the step period target.

        Only steps that carried prefill and filled the target tell how long a
        full step takes, so the others are ignored. An update solves for the
        target that brings their median period to `target_step_ms` (to
        `decode_step_ms` while the recent steps carry decodes) after the
        `fixed_ms` cost, so a step cost that is linear in the tokens lands in
        one move instead of approaching by a fixed share; `max_factor` bounds
        what one update may change. The result lands on a grid point (moving
        one point when it would otherwise stay). It keeps its value until
        `min_samples` informing steps are in and while their period is within
        `deadband` of the budget.

        Args:
            since: First sample to consider, for a caller that just ingested
                merged steps; None takes every sample not yet consumed.

        Returns:
            The target after the update.

        """
        with self._update_lock:
            return self._update(since)

    def _update(self, since: int | None) -> int:
        with self._lock:
            first = self._consumed if since is None else since - 1
            samples = [s for s in self._samples if s.seq > first]
        if not samples:
            return self._target
        target = self._target
        # Steps that carry decodes wait for the whole step, so while the recent
        # ones do, a longer prefill step is paid by them; with none, it is not.
        decodes = median(s.total_tokens - s.prefill_tokens for s in samples)
        period_target = self.target_step_ms
        if self.decode_step_ms > 0 and decodes > 0:
            period_target = self.decode_step_ms
        floor = self.fill_fraction * target
        cutoff = self.stall_factor * period_target / 1000
        periods = [
            min(s.period_s, cutoff)
            for s in samples
            if s.prefill_tokens > 0 and s.total_tokens >= floor
        ]
        if len(periods) < self.min_samples:
            return target
        with self._lock:
            self._consumed = max(self._consumed, samples[-1].seq)
        self._step_ms = median(periods) * 1000
        # The period is the fixed cost plus the target's tokens: solve for the
        # target that spends the rest of the period target. A fixed cost the
        # fit reports near or above the period it was fitted from would zero
        # what is left of it and invert the solve, so it is bounded by both.
        fixed_ms = min(self.fixed_ms, self._step_ms / 2, period_target / 2)
        floor_ms = 0.05 * period_target
        ratio = max(period_target - fixed_ms, floor_ms) / max(
            self._step_ms - fixed_ms, floor_ms
        )
        if abs(ratio - 1) <= self.deadband:
            return target
        factor = min(
            max(1 + self.gain * (ratio - 1), 1 / self.max_factor), self.max_factor
        )
        raw = min(max(round(target * factor), self.min_tokens), self.max_tokens)
        new = self._on_grid(raw)
        if new == target and raw != target:
            # Landing back on the target would freeze it, and the smallest one
            # always snaps back: move one grid point the way the ratio points.
            new = self._next_point(target, up=raw > target)
        if new != target:
            logger.info(
                "Prefill target %d -> %d tokens (step %.1f ms of %.1f ms, "
                "%d informing steps, %d decode tokens)",
                target,
                new,
                self._step_ms,
                period_target,
                len(periods),
                int(decodes),
            )
        self._target = new
        return new

    def _on_grid(self, tokens: int) -> int:
        """`tokens` moved to the nearest grid point, within the bounds."""
        tokens = min(max(tokens, self.min_tokens), self.max_tokens)
        if self.grid:
            return min(self.grid, key=lambda t: (abs(t - tokens), t))
        grid = self.quantum
        snapped = (tokens + grid // 2) // grid * grid
        return min(max(snapped, self.min_tokens), self.max_tokens)

    def _next_point(self, target: int, up: bool) -> int:
        """The nearest grid point above (`up`) or below `target`, else it."""
        if self.grid:
            if up:
                return next((t for t in self.grid if t > target), target)
            return next((t for t in reversed(self.grid) if t < target), target)
        step = self.quantum if up else -self.quantum
        return min(max(target + step, self.min_tokens), self.max_tokens)

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
