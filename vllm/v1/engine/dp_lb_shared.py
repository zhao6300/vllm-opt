# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any, Protocol, cast


class CounterBuffer(Protocol):
    def __getitem__(self, index: int) -> int: ...

    def __setitem__(self, index: int, value: int) -> None: ...


class SharedDPLoadStats:
    """Per-engine load of every API server, shared across the API server processes.

    Each server publishes its in-flight requests and pending prefill per engine
    into a slot only it writes; a router adds the other servers' slots to its own
    figures, so it sees their dispatches at once instead of after an engine
    snapshot. Lock-free like `SharedAdmissionStats`: one writer per slot, readers
    sum the slots.

    A slot is a heartbeat followed, per engine, by in-flight requests, pending
    prefill tokens and pending prefill microseconds, padded to whole cache lines.
    A slot whose heartbeat is older than `stale_after_s` (a dead server) or was
    never written is ignored.
    """

    FIELDS_PER_ENGINE = 3
    COUNTER_BYTES = 8
    CACHE_LINE_BYTES = 64
    COUNTERS_PER_LINE = CACHE_LINE_BYTES // COUNTER_BYTES

    def __init__(
        self,
        counters: Any,
        client_count: int,
        client_index: int,
        num_engines: int,
        stale_after_s: float = 2.0,
    ):
        assert 0 <= client_index < client_count
        self._counters = cast(CounterBuffer, counters)
        self._client_count = client_count
        self._client_index = client_index
        self.num_engines = num_engines
        self._stride = self.slot_counters(num_engines)
        self._base = client_index * self._stride
        self._stale_after_ns = int(stale_after_s * 1e9)

    @classmethod
    def slot_counters(cls, num_engines: int) -> int:
        """Counters in one server's slot, a whole number of cache lines."""
        used = 1 + num_engines * cls.FIELDS_PER_ENGINE
        lines = -(-used // cls.COUNTERS_PER_LINE)
        return lines * cls.COUNTERS_PER_LINE

    @classmethod
    def num_counters(cls, client_count: int, num_engines: int) -> int:
        return client_count * cls.slot_counters(num_engines)

    def heartbeat(self) -> None:
        """Mark this server alive; an idle server calls it on a timer."""
        self._counters[self._base] = time.monotonic_ns()

    def publish(
        self, engine_index: int, inflight: int, pending_tokens: int, pending_ms: float
    ) -> None:
        """Write this server's own figures for one engine."""
        if not 0 <= engine_index < self.num_engines:
            return
        at = self._base + 1 + engine_index * self.FIELDS_PER_ENGINE
        self._counters[at] = inflight
        self._counters[at + 1] = pending_tokens
        self._counters[at + 2] = round(pending_ms * 1000)
        self.heartbeat()

    def others(self, num_engines: int) -> list[tuple[int, int, float]]:
        """Per engine, the other live servers' summed (in flight, pending prefill
        tokens, pending prefill ms); zeros past the array's capacity."""
        totals = [[0, 0, 0] for _ in range(num_engines)]
        now = time.monotonic_ns()
        counters = self._counters
        covered = min(num_engines, self.num_engines)
        for client in range(self._client_count):
            if client == self._client_index:
                continue
            base = client * self._stride
            beat = counters[base]
            if beat == 0 or now - beat > self._stale_after_ns:
                continue
            at = base + 1
            for engine in range(covered):
                total = totals[engine]
                total[0] += counters[at]
                total[1] += counters[at + 1]
                total[2] += counters[at + 2]
                at += self.FIELDS_PER_ENGINE
        return [(t[0], t[1], t[2] / 1000.0) for t in totals]
