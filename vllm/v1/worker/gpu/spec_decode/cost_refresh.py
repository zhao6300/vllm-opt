# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refreshing the adaptive-verification cost tables off the step loop.

Every TP rank and PP stage must pick the same budget for a step, so they must
hold the same cost tables. The tables are rebuilt from live step timings, which
differ per rank, so the ranks have to exchange them. `CostRefresher` does that
without ever stopping the step loop on a collective or on numeric work:

- The medians and the final table build run on a worker thread.
- The exchange uses asynchronous CPU collectives that the step loop only issues,
  on a fixed schedule counted in served steps, which every rank counts alike.
- The new tables are installed at a fixed step too, `3 * lag` steps after the
  rebuild began, so all ranks switch on the same step. The loop waits only when
  a stage is not done by its step, which a lag of several steps makes rare.
"""

import queue
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.v1.worker.gpu.async_utils import StepTimingSample

logger = init_logger(__name__)

# Waiting for the worker thread this long means it died.
_WORKER_TIMEOUT_S = 60.0

_Tables = tuple[np.ndarray, np.ndarray]


def curve_to_dense(curve: list[tuple[int, float]], size: int) -> np.ndarray:
    """Lay a sparse `(size, cost)` curve out on a grid; NaN where it has no point."""
    dense = np.full(size, np.nan, dtype=np.float32)
    for key, cost in curve:
        if 0 <= key < size:
            dense[key] = cost
    return dense


def dense_to_curve(dense: np.ndarray) -> list[tuple[int, float]]:
    """The points of a dense curve, in order."""
    return [(int(i), float(dense[i])) for i in np.flatnonzero(~np.isnan(dense))]


class CostRefresher:
    """Folds live step timings into the cost curves and installs the result.

    The step loop calls `observe` and `tick` once per served step. Everything
    else runs on a worker thread or inside the collectives' own threads.
    """

    def __init__(
        self,
        base_draft_curve: list[tuple[int, float]],
        base_verify_curve: list[tuple[int, float]],
        max_num_reqs: int,
        max_batch_tokens: int,
        interval: int,
        window: int,
        build_tables: Callable[
            [list[tuple[int, float]], list[tuple[int, float]]], _Tables
        ],
        install: Callable[[_Tables], None],
        tp_group: Any,
        pp_group: Any,
        min_samples: int = 4,
    ) -> None:
        """Args:
        base_draft_curve: This rank's startup draft curve, by request count.
        base_verify_curve: This rank's startup verify curve, by token count.
        max_num_reqs: Largest request count a curve covers.
        max_batch_tokens: Largest token count a curve covers.
        interval: Served steps between rebuilds.
        window: Timings kept per size; the median of these is the cost.
        build_tables: Turns the agreed curves into cost tables.
        install: Takes the new tables; called from the step loop.
        tp_group: Coordinator with `world_size`, `ranks` and `cpu_group`.
        pp_group: The same, for the pipeline stages.
        min_samples: Timings a size needs before it replaces its startup cost.

        """
        if interval < 1:
            raise ValueError(f"The rebuild interval must be positive: {interval}")
        self._draft_size = max_num_reqs + 1
        self._verify_size = max_batch_tokens + 1
        self._interval = interval
        # A rebuild takes three stages; they must finish before the next one.
        self._lag = max(1, min(8, interval // 4))
        self._min_samples = min_samples
        self._window = max(window, 1)
        self._build_tables = build_tables
        self._install = install
        self._tp = tp_group
        self._pp = pp_group

        self._online_draft: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._online_verify: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        # This rank's curves: the startup profile with live medians laid over
        # it. Replaced whole by the worker, never edited, so a reader holding
        # one sees a consistent pair.
        self._local: _Tables = (
            curve_to_dense(base_draft_curve, self._draft_size),
            curve_to_dense(base_verify_curve, self._verify_size),
        )
        self._lock = threading.Lock()
        self._inbox: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._merged = threading.Event()
        self._result: _Tables | None = None
        self._error: BaseException | None = None

        # The step loop's own state.
        self._step = 0
        self._due = interval
        self._stage = "idle"
        self._send: torch.Tensor | None = None
        self._gathered: list[torch.Tensor] = []
        self._work: Any = None

        self._thread = threading.Thread(
            target=self._run, name="adaptive-verification-costs", daemon=True
        )
        self._thread.start()

    def observe(self, samples: list[StepTimingSample]) -> None:
        """Hand the timings of finished steps to the worker. O(1) per sample."""
        if samples:
            self._inbox.put(("samples", samples))

    def tick(self) -> None:
        """Count one served step and run the stage that falls due on it."""
        if self._error is not None:
            raise RuntimeError("The cost table worker failed") from self._error
        self._step += 1
        if self._step >= self._due:
            self._advance()

    def flush(self) -> None:
        """Wait until the worker has folded every timing handed to `observe`."""
        done = threading.Event()
        self._inbox.put(("flush", done))
        if not done.wait(_WORKER_TIMEOUT_S):
            raise RuntimeError("The cost table worker did not finish its backlog")

    def close(self) -> None:
        """Stop the worker thread."""
        self._inbox.put(("stop", None))
        self._thread.join()

    def _advance(self) -> None:
        stage = self._stage
        if stage == "idle":
            self._begin()
        elif stage == "tp":
            self._work.wait()
            self._exchange_over_pp(self._send)
        elif stage == "pp":
            self._work.wait()
            self._merge(self._gathered)
        elif stage == "merge":
            self._install_result()

    def _begin(self) -> None:
        with self._lock:
            draft, verify = self._local
        send = torch.from_numpy(np.concatenate([draft, verify]))
        if self._tp.world_size > 1:
            # Every TP rank takes rank 0's curves, as the budget does.
            self._send = send
            self._work = dist.broadcast(
                send, src=self._tp.ranks[0], group=self._tp.cpu_group, async_op=True
            )
            self._schedule("tp")
        else:
            self._exchange_over_pp(send)

    def _exchange_over_pp(self, send: torch.Tensor) -> None:
        if self._pp.world_size > 1:
            # Every stage trims the same batch, so each needs the cost of the
            # slowest one.
            self._send = send
            self._gathered = [
                torch.empty_like(send) for _ in range(self._pp.world_size)
            ]
            self._work = dist.all_gather(
                self._gathered, send, group=self._pp.cpu_group, async_op=True
            )
            self._schedule("pp")
        else:
            self._merge([send])

    def _merge(self, parts: list[torch.Tensor]) -> None:
        self._merged.clear()
        self._inbox.put(("merge", [part.numpy().copy() for part in parts]))
        self._schedule("merge")

    def _install_result(self) -> None:
        if not self._merged.wait(_WORKER_TIMEOUT_S):
            raise RuntimeError("The cost table worker did not finish a rebuild")
        if self._error is not None:
            raise RuntimeError("The cost table worker failed") from self._error
        assert self._result is not None  # set before the event
        self._install(self._result)
        self._result = None
        self._stage = "idle"
        self._send = None
        self._gathered = []
        self._work = None
        # The next multiple of the interval after this step.
        self._due = (self._step // self._interval + 1) * self._interval

    def _schedule(self, stage: str) -> None:
        self._stage = stage
        self._due = self._step + self._lag

    def _run(self) -> None:
        while True:
            kind, payload = self._inbox.get()
            if kind == "stop":
                return
            try:
                if kind == "samples":
                    self._fold(payload)
                elif kind == "flush":
                    payload.set()
                else:
                    self._rebuild(payload)
            except Exception as error:  # reported to the step loop
                logger.exception("The cost table worker failed")
                self._error = error
                self._merged.set()
                return

    def _fold(self, samples: list[StepTimingSample]) -> None:
        draft, verify = self._local
        draft, verify = draft.copy(), verify.copy()
        for sample in samples:
            if 0 <= sample.num_target_tokens < self._verify_size:
                self._online_verify[sample.num_target_tokens].append(sample.forward_ms)
                self._replace(verify, sample.num_target_tokens, self._online_verify)
            if sample.full_cudagraph and 0 <= sample.num_reqs < self._draft_size:
                self._online_draft[sample.num_reqs].append(sample.drafter_ms)
                self._replace(draft, sample.num_reqs, self._online_draft)
        with self._lock:
            self._local = (draft, verify)

    def _replace(
        self, dense: np.ndarray, key: int, buckets: dict[int, deque[float]]
    ) -> None:
        """Lay a size's median over the startup cost once it has enough timings."""
        values = buckets[key]
        if len(values) >= self._min_samples:
            dense[key] = float(np.median(values))

    def _rebuild(self, parts: list[np.ndarray]) -> None:
        """Take the slowest stage's cost for every size, and build the tables."""
        # fmax skips a NaN, so a size no stage has a point for stays empty.
        merged = np.fmax.reduce(np.stack(parts), axis=0)
        draft = dense_to_curve(merged[: self._draft_size])
        verify = dense_to_curve(merged[self._draft_size :])
        if not draft or not verify:
            raise RuntimeError(
                "Adaptive verification could not rebuild its step costs from the "
                "ranks' curves"
            )
        self._result = self._build_tables(draft, verify)
        self._merged.set()
