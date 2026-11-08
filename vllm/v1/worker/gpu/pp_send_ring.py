# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring of send buffers for PP intermediate tensors.

A non-last PP rank sends its stage output to the next rank asynchronously.
The tensors a forward returns are often reused by the next forward: FULL
CUDA graphs replay into static output buffers, and the runner's persistent
input buffers are relayed as-is. Without a copy the next forward must wait
for the previous send before it can run.

``PPSendRing`` copies each step's payload into one of ``num_slots``
preallocated buffers and waits only for the send that last used that slot.
A stage can then run up to ``num_slots`` steps ahead of its successor's
receives. ``handle.wait()`` on an NCCL send makes the current stream wait
for it; it does not block the CPU.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SendFn = Callable[[dict[str, Any]], list[Any]]


class _Slot:
    def __init__(self, buffers: dict[str, torch.Tensor]) -> None:
        self.buffers = buffers
        self.handles: list[Any] = []


class PPSendRing:
    def __init__(
        self, num_slots: int, make_buffers: Callable[[], Mapping[str, torch.Tensor]]
    ) -> None:
        """``make_buffers`` returns one slot's buffers, keyed like the payload,
        each sized for the largest step on its first dimension."""
        assert num_slots >= 1
        self._slots = [_Slot(dict(make_buffers())) for _ in range(num_slots)]
        self._next = 0

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    @property
    def slot_nbytes(self) -> int:
        return sum(t.nbytes for t in self._slots[0].buffers.values())

    def send(self, tensors: Mapping[str, Any], send_fn: SendFn) -> list[Any]:
        """Copy ``tensors`` into the next slot and send them with ``send_fn``.

        ``send_fn`` takes the dict to send and returns its handles, the
        metadata handle first. Returns those handles.
        """
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        # The slot's previous send must finish reading before it is refilled.
        for handle in slot.handles:
            handle.wait()
        slot.handles = []

        staged: dict[str, Any] = {}
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                staged[key] = value
                continue
            buffer = slot.buffers.get(key)
            if (
                buffer is None
                or buffer.dtype != value.dtype
                or buffer.device != value.device
                or buffer.shape[1:] != value.shape[1:]
                or buffer.shape[0] < value.shape[0]
            ):
                # Not preallocated (an unexpected key or a larger step): grow
                # the slot once so later steps reuse it.
                logger.warning_once(
                    "PP send ring: allocating a buffer for %s %s at runtime.",
                    key,
                    tuple(value.shape),
                )
                buffer = torch.empty_like(value)
                slot.buffers[key] = buffer
            view = buffer[: value.shape[0]]
            view.copy_(value)
            staged[key] = view

        handles = send_fn(staged)
        # The metadata handle (first) is reaped by the GroupCoordinator; only
        # the device sends guard the slot's buffers.
        slot.handles = list(handles[1:])
        return handles

    def wait_all(self) -> None:
        """Wait for every outstanding send (e.g. before freeing memory)."""
        for slot in self._slots:
            for handle in slot.handles:
                handle.wait()
            slot.handles = []
