# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only dumps of a step's batch composition and PP row checksums.

Enable with ``VLLM_DEBUG_STEP_DUMP=1``.  Each dumped step emits one
``DSV41DEBUG {json}`` log line per worker with the per-request scheduled-token
counts, whether each request is prefilling, the step's input ids / positions,
and per-row sums of the PP intermediate tensors and the output hidden states.

The intended use is comparing one teacher-forced prefill alone against the same
prefill co-batched with a decode: with the step composition and the row values
in hand, "the decode never shared the step" can be told apart from "the rows
changed".  Debug only; not for production traffic.

Environment:
- ``VLLM_DEBUG_STEP_DUMP=1`` enable.
- ``VLLM_DEBUG_STEP_LIMIT`` tokens / rows to dump per step (default 64).
- ``VLLM_DEBUG_STEP_MAX`` maximum dumped steps per worker (default 200).
- ``VLLM_DEBUG_STEP_ONLY_MIXED`` 1 = only steps with both prefill and decode
  tokens (default 1); 0 = every step.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_STEPS = {"inputs": 0, "outputs": 0}
_DUMPS = 0


def _enabled() -> bool:
    return os.getenv("VLLM_DEBUG_STEP_DUMP", "0") == "1"


def _limit() -> int:
    return int(os.getenv("VLLM_DEBUG_STEP_LIMIT", "64"))


def _max_dumps() -> int:
    return int(os.getenv("VLLM_DEBUG_STEP_MAX", "200"))


def _only_mixed() -> bool:
    return os.getenv("VLLM_DEBUG_STEP_ONLY_MIXED", "1") == "1"


def _composition(input_batch: Any) -> tuple[int, int]:
    scheduled = input_batch.num_scheduled_tokens
    prefilling = input_batch.is_prefilling_np
    if len(scheduled) == 0:
        return 0, 0
    num_prefill = int(scheduled[prefilling].sum())
    num_decode = int(scheduled[~prefilling].sum())
    return num_prefill, num_decode


def _rows(tensor: Any, n: int) -> list[float] | None:
    if not torch.is_tensor(tensor) or tensor.dim() < 2:
        return None
    return [round(float(v), 4) for v in tensor[:n].float().sum(dim=-1).cpu()]


def _head(tensor: Any, n_rows: int, n_cols: int = 4) -> list[list[float]] | None:
    if not torch.is_tensor(tensor) or tensor.dim() < 2:
        return None
    head = tensor[:n_rows, :n_cols].float().cpu()
    return [[round(float(v), 5) for v in row] for row in head]


def _tensor_payload(obj: Any, n: int) -> dict[str, Any] | None:
    if obj is None:
        return None
    if hasattr(obj, "tensors"):
        items = obj.tensors.items()
    elif torch.is_tensor(obj):
        return {
            "output": {
                "shape": list(obj.shape),
                "sums": _rows(obj, n),
                "head": _head(obj, min(n, 8)),
            }
        }
    elif isinstance(obj, tuple):
        first = next((x for x in obj if torch.is_tensor(x)), None)
        return _tensor_payload(first, n)
    else:
        return None
    payload = {}
    for key, value in items:
        payload[str(key)] = {
            "shape": list(value.shape),
            "sums": _rows(value, n),
            "head": _head(value, min(n, 8)),
        }
    return payload


def _emit(payload: dict[str, Any]) -> None:
    global _DUMPS
    _DUMPS += 1
    logger.info("DSV41DEBUG %s", json.dumps(payload))


def dump_inputs(
    input_batch: Any,
    batch_desc: Any,
    model_inputs: dict[str, Any],
    dummy_run: bool = False,
) -> None:
    """Log the step's composition and the prefill rows before the forward."""
    if dummy_run or not _enabled() or _max_dumps() <= _DUMPS:
        return
    try:
        num_prefill, num_decode = _composition(input_batch)
        if _only_mixed() and not (num_prefill and num_decode):
            return
        n = _limit()
        num_reqs = int(input_batch.num_reqs)
        seq_lens = input_batch.seq_lens[:num_reqs].cpu().tolist()
        payload = {
            "phase": "inputs",
            "step": _STEPS["inputs"],
            "device": torch.cuda.current_device(),
            "cg_mode": str(batch_desc.cg_mode),
            "num_reqs": num_reqs,
            "num_tokens": int(input_batch.num_tokens),
            "padded": int(input_batch.num_tokens_after_padding),
            "num_prefill_tokens": num_prefill,
            "num_decode_tokens": num_decode,
            "scheduled": [int(x) for x in input_batch.num_scheduled_tokens],
            "is_prefilling": [bool(x) for x in input_batch.is_prefilling_np],
            "qsl": [int(x) for x in input_batch.query_start_loc_np],
            "seq_lens": [int(x) for x in seq_lens],
            "input_ids": [int(x) for x in input_batch.input_ids[:n].cpu().tolist()],
            "positions": [int(x) for x in input_batch.positions[:n].cpu().tolist()],
            "intermediate": _tensor_payload(
                model_inputs.get("intermediate_tensors"), n
            ),
        }
        _STEPS["inputs"] += 1
        _emit(payload)
    except Exception:  # noqa: BLE001
        logger.debug("DSV41DEBUG input dump failed", exc_info=True)


def dump_outputs(
    input_batch: Any,
    batch_desc: Any,
    model_output: Any,
    dummy_run: bool = False,
) -> None:
    """Log per-row checksums of the forward's output / PP payload."""
    if dummy_run or not _enabled() or _max_dumps() <= _DUMPS:
        return
    try:
        num_prefill, num_decode = _composition(input_batch)
        if _only_mixed() and not (num_prefill and num_decode):
            return
        payload = {
            "phase": "outputs",
            "step": _STEPS["outputs"],
            "device": torch.cuda.current_device(),
            "cg_mode": str(batch_desc.cg_mode),
            "num_tokens": int(input_batch.num_tokens),
            "padded": int(input_batch.num_tokens_after_padding),
            "num_prefill_tokens": num_prefill,
            "num_decode_tokens": num_decode,
            "output": _tensor_payload(model_output, _limit()),
        }
        _STEPS["outputs"] += 1
        _emit(payload)
    except Exception:  # noqa: BLE001
        logger.debug("DSV41DEBUG output dump failed", exc_info=True)
