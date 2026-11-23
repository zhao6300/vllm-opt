# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online fit of the prefill step cost model.

A data-parallel MoE step ends with its slowest rank, so the period a rank
measures is explained by the largest token count and the largest attention work
across the ranks, not by its own. `StepCostFitter` records each step's local
figures, exchanges them with the DP group over an asynchronous CPU collective
on a fixed schedule, and fits `fixed + a * tokens + c * attention work` to the
merged steps. Every rank receives the same samples at the same step and so
installs the same coefficients.
"""

import math
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Columns of one step's row in the exchanged payload.
_KEY, _PERIOD_MS, _TOKENS, _ATTN, _PREFILL = range(5)
_COLUMNS = 5


@dataclass(frozen=True)
class StepSample:
    """A step as the DP group saw it.

    Attributes:
        key: Step identifier, equal on every rank for the same step.
        period_ms: Time since the previous step's output.
        tokens: Tokens the step carried, on the rank that carried most.
        attn_work: `sum(q * (context + q / 2))` over a rank's prefill chunks,
            on the rank with the most.
        prefill_tokens: Prefill tokens, on the rank with the most.

    """

    key: int
    period_ms: float
    tokens: int
    attn_work: float
    prefill_tokens: int


@dataclass(frozen=True)
class FitDiagnostics:
    """The window a fit used.

    Attributes:
        steps: Steps in the window.
        prefill_steps: How many carried prefill.
        tokens_std: Weighted standard deviation of the token counts; a
            coefficient on a figure that barely varies is not determined.
        attn_work_std: The same for the attention work.
        residual_ms: Weighted RMS of what the fit leaves unexplained.
        median_period_ms: Median step period.

    """

    steps: int
    prefill_steps: int
    tokens_std: float
    attn_work_std: float
    residual_ms: float
    median_period_ms: float


class StepCostSolver:
    """Ridge fit of `fixed_ms + ms_per_token * tokens + ms_per_token_context *
    attn_work` toward configured coefficients.

    The prior counts as `prior_samples` observations of typical size, enough to
    keep a term the data does not vary at its prior. A fitted term is only kept
    valid, so a cost model can be built from it: fixed and per-context time not
    below 0, per-token time above 0. Recent steps weigh more: a step's weight
    halves every `half_life` steps.
    """

    # Size of a parameter whose prior is 0, which only fixes the units.
    _FLOOR = (1.0, 1e-3, 1e-7)
    # What a coefficient may not fall below: the per-token time must stay positive.
    _LOWER = (0.0, 1e-9, 0.0)

    def __init__(
        self,
        prior: tuple[float, float, float],
        window: int = 512,
        half_life: float = 256.0,
        prior_samples: float = 1.0,
        min_samples: int = 16,
    ) -> None:
        self.prior = prior
        self.min_samples = min_samples
        self._scale = np.array([max(p, f) for p, f in zip(prior, self._FLOOR)])
        self._prior_u = np.array(prior) / self._scale
        self.prior_samples = prior_samples
        self.half_life = half_life
        # period, 1, tokens, attention work, whether the step carried prefill
        self._rows: deque[tuple[float, float, float, float, float]] = deque(
            maxlen=window
        )

    def add(self, sample: StepSample) -> None:
        """Add a step; one with a figure that is not a finite number is ignored."""
        if not all(
            math.isfinite(v)
            for v in (sample.period_ms, sample.tokens, sample.attn_work)
        ):
            return
        self._rows.append(
            (
                sample.period_ms,
                1.0,
                float(sample.tokens),
                float(sample.attn_work),
                float(sample.prefill_tokens > 0),
            )
        )

    def __len__(self) -> int:
        return len(self._rows)

    def diagnostics(self, theta: tuple[float, float, float]) -> "FitDiagnostics":
        """What the window holds and how well `theta` explains it, to tell a fit
        the data determines from one it does not."""
        rows = np.array(self._rows)
        if not len(rows):
            return FitDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0)
        age = np.arange(len(rows) - 1, -1, -1)
        w = 0.5 ** (age / self.half_life)
        w = w / w.sum()

        def spread(x: np.ndarray) -> float:
            return float(np.sqrt((w * (x - (w * x).sum()) ** 2).sum()))

        residual = rows[:, 0] - rows[:, 1:4] @ np.array(theta)
        return FitDiagnostics(
            steps=len(rows),
            prefill_steps=int(rows[:, 4].sum()),
            tokens_std=spread(rows[:, 2]),
            attn_work_std=spread(rows[:, 3]),
            residual_ms=float(np.sqrt((w * residual**2).sum())),
            median_period_ms=float(np.median(rows[:, 0])),
        )

    def fit(self) -> tuple[float, float, float]:
        """The coefficients, the prior until `min_samples` steps that carried
        prefill are in: the prefill terms are what this prices, and a window of
        decode-only steps would fit them to the decode step's own overhead."""
        rows = np.array([row for row in self._rows if row[4]])
        if len(rows) < self.min_samples:
            return self.prior
        y = rows[:, 0]
        z = rows[:, 1:4] * self._scale
        age = np.arange(len(rows) - 1, -1, -1)
        w = 0.5 ** (age / self.half_life)
        gram = (z * w[:, None]).T @ z
        typical = (w[:, None] * z**2).sum(axis=0) / w.sum()
        # The floor keeps a term the data never varies solvable: it then stays put.
        ridge = np.diag(
            np.maximum(self.prior_samples * typical, 1e-9 * np.diag(gram).max())
        )
        rhs = (z * w[:, None]).T @ y + ridge @ self._prior_u
        u = np.linalg.solve(gram + ridge, rhs)
        theta = np.maximum(u * self._scale, self._LOWER)
        return float(theta[0]), float(theta[1]), float(theta[2])


class StepCostFitter:
    """Records steps, shares them over the DP group and fits the cost model.

    The scheduler calls `record` for each step's output. With a DP group the
    engine core calls `tick` once per loop pass with its step counter, which is
    equal on every rank. Every `interval` steps `tick` finishes the exchange
    started at the previous interval, fits, and calls `on_fit`, then starts the
    next exchange, so the collective has a whole interval to complete. Without a
    group `tick` fits at once.
    """

    def __init__(
        self,
        solver: StepCostSolver,
        interval: int,
        max_period_ms: float,
        on_fit: Callable[[tuple[float, float, float], list[StepSample]], None],
    ) -> None:
        if interval < 1:
            raise ValueError(f"The fit interval must be >= 1 step: {interval}")
        self.solver = solver
        self.interval = interval
        self.max_period_ms = max_period_ms
        self.on_fit = on_fit
        self._max_rows = interval + 8
        self._local: deque[StepSample] = deque(maxlen=4 * self._max_rows)
        self._last_output: float | None = None
        self._local_key = 0
        self._unsent = 0
        self._sent_key = -1
        self._last_step = 0
        self._pending: tuple[dist.Work, list[torch.Tensor]] | None = None
        self.exchanging = False

    @classmethod
    def from_env(
        cls,
        prior: tuple[float, float, float],
        budget_ms: float,
        on_fit: Callable[[tuple[float, float, float], list[StepSample]], None],
    ) -> "StepCostFitter":
        return cls(
            StepCostSolver(prior),
            envs.VLLM_PREFILL_COST_FIT_INTERVAL,
            4 * budget_ms,
            on_fit,
        )

    def record(
        self,
        now: float,
        key: int,
        tokens: int,
        attn_work: float,
        prefill_tokens: int,
    ) -> None:
        """Note a step's output. O(1).

        A gap longer than `max_period_ms` is an idle engine, not a step, and is
        left out.

        Args:
            now: Monotonic time of the output, in seconds.
            key: The step's id, equal on every DP rank for the same step; negative
                when the engine has no DP peers to match it with.
            tokens: Tokens the step carried on this rank.
            attn_work: `sum(q * (context + q / 2))` over this rank's prefill chunks.
            prefill_tokens: Prefill tokens on this rank.

        """
        last, self._last_output = self._last_output, now
        if last is None:
            return
        period_ms = (now - last) * 1000
        if period_ms > self.max_period_ms:
            return
        if key < 0:
            key, self._local_key = self._local_key, self._local_key + 1
        self._local.append(
            StepSample(key, period_ms, tokens, attn_work, prefill_tokens)
        )
        self._unsent += 1
        if not self.exchanging and self._unsent >= self.interval:
            self.tick(None, self._last_step + self.interval)

    def tick(self, group: dist.ProcessGroup | None, step: int) -> None:
        """Advance the schedule; call on every rank at the same `step`.

        Args:
            group: The DP group, or None when this rank is alone.
            step: The engine core's step counter; a smaller value than the last
                call starts a new wave.

        """
        if group is not None and group.size() > 1:
            self.exchanging = True
        if step < self._last_step:
            self._complete_pending()
        self._last_step = step
        if step % self.interval != 0:
            return
        if group is None or group.size() == 1:
            self._finish([self._payload()])
            return
        self._complete_pending()
        payload = self._payload()
        gathered = [torch.empty_like(payload) for _ in range(group.size())]
        self._pending = (
            dist.all_gather(gathered, payload, group, async_op=True),
            gathered,
        )

    def _complete_pending(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            work, gathered = pending
            work.wait()
            self._finish(gathered)

    def _payload(self) -> torch.Tensor:
        fresh = [s for s in self._local if s.key > self._sent_key]
        fresh = fresh[-self._max_rows :]
        self._unsent = 0
        payload = np.full((self._max_rows, _COLUMNS), -1.0)
        if fresh:
            payload[: len(fresh)] = [
                (s.key, s.period_ms, s.tokens, s.attn_work, s.prefill_tokens)
                for s in fresh
            ]
            self._sent_key = fresh[-1].key
        return torch.from_numpy(payload)

    def _finish(self, gathered: list[torch.Tensor]) -> None:
        rows = np.concatenate([g.numpy() for g in gathered])
        rows = rows[rows[:, _KEY] >= 0]
        rows = rows[np.argsort(rows[:, _KEY], kind="stable")]
        if not len(rows):
            self.on_fit(self.solver.fit(), [])
            return
        keys, starts = np.unique(rows[:, _KEY], return_index=True)
        tokens = np.maximum.reduceat(rows[:, _TOKENS], starts)
        attn = np.maximum.reduceat(rows[:, _ATTN], starts)
        prefill = np.maximum.reduceat(rows[:, _PREFILL], starts)
        periods = np.split(rows[:, _PERIOD_MS], starts[1:])
        merged = [
            StepSample(
                int(key),
                statistics.median(period.tolist()),
                int(tokens[i]),
                float(attn[i]),
                int(prefill[i]),
            )
            for i, (key, period) in enumerate(zip(keys, periods, strict=True))
        ]
        for sample in merged:
            self.solver.add(sample)
        self.on_fit(self.solver.fit(), merged)
