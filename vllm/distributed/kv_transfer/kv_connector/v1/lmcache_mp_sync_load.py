# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LMCache MP connector that loads small external hits in the same step.

The stock connector reports every external hit as an asynchronous load, so the
scheduler parks the request in `WAITING_FOR_REMOTE_KVS` for at least one step.
For a small load that parking dominates the cost: the DeepSeek-V4.1 sliding
window tail (two 64-token blocks) copies in ~5 ms but waits ~40 ms, because the
request skips a scheduler step and is picked up on a later worker step.

This variant submits such loads from `start_load_kv` and makes the current
stream wait on the server's completion event, so the request is scheduled in
the step that allocated its blocks. Large loads stay asynchronous. A failed or
dropped same-step load is reported through `invalid_block_ids`, so the engine
recomputes those blocks; it is never reported as a finished receive, which the
scheduler would assert on for a request that is already running.

Which loads are same-step follows the scheduler's own admission decision (the
`async_load` flag `get_num_new_matched_tokens` returns), carried to the worker
on the metadata. The load's op span cannot stand in for it: a retrieve rounds
its start down to the LMCache chunk boundary, so a load the scheduler admits
synchronously can cover more than `lmcache.mp.sync_load_max_tokens` tokens.
Splitting on the span deferred such a load while the request was never parked
in `WAITING_FOR_REMOTE_KVS`, and the later finished receive for a running
request tripped the scheduler's assert.

Configure with:

- `lmcache.mp.sync_load_max_tokens`: load hits of at most this many tokens in
  the scheduler step. Unset or 0 (the default) keeps the stock behaviour.
- `lmcache.mp.sync_load_timeout_s`: server-response timeout for those loads,
  in seconds. Defaults to 60.

Enable the module with `kv_connector_module_path` pointing here and
`kv_connector` set to `LMCacheMPConnector`.
"""

from typing import TYPE_CHECKING, Any

import torch
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector as _LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import KVCacheConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.request import Request

logger = init_logger(__name__)


def current_stream() -> Any:
    """The stream the forward pass will run on."""
    return torch.cuda.current_stream()


def split_sync_retrieves(
    metas: list[LMCacheMPRequestMetadata],
) -> tuple[list[LMCacheMPRequestMetadata], list[LMCacheMPRequestMetadata]]:
    """Split retrieve metadata by the scheduler's admission decision.

    Stores and empty loads belong in neither list.

    Args:
        metas: The step's request metadata.

    Returns:
        The same-step and the asynchronous retrieve metadata, in input order.

    """
    sync: list[LMCacheMPRequestMetadata] = []
    deferred: list[LMCacheMPRequestMetadata] = []
    for meta in metas:
        if meta.direction != "RETRIEVE" or meta.op.end <= meta.op.start:
            continue
        if getattr(meta, "same_step_load", False):
            sync.append(meta)
        else:
            deferred.append(meta)
    return sync, deferred


class LMCacheMPConnector(_LMCacheMPConnector):
    """LMCache MP connector that loads small external hits in the same step."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config
        self.sync_load_max_tokens = int(
            extra.get_from_extra_config("lmcache.mp.sync_load_max_tokens", 0)
        )
        self.sync_load_timeout_s = float(
            extra.get_from_extra_config("lmcache.mp.sync_load_timeout_s", 60.0)
        )

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        need_to_load, async_load = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        same_step = (
            async_load
            and need_to_load is not None
            and self.sync_load_max_tokens > 0
            and need_to_load <= self.sync_load_max_tokens
        )
        self._record_load_mode(request, same_step)
        if same_step:
            return need_to_load, False
        return need_to_load, async_load

    def _record_load_mode(self, request: "Request", same_step: bool) -> None:
        """Remember this lookup's decision for the metadata built this step.

        The decision is later attached to the request's retrieve metadata,
        which is the only place the worker can read it: the op span the worker
        sees can exceed `sync_load_max_tokens` even for a same-step load.
        """
        tracker = self._request_tracker(getattr(request, "request_id", None))
        if tracker is not None:
            tracker.same_step_load = same_step

    def _request_tracker(self, request_id: str | None) -> Any:
        trackers = getattr(self, "request_trackers", None)
        if trackers is None or request_id is None:
            return None
        return trackers.get(request_id)

    def _process_retrieve_requests(self, metadata: LMCacheMPConnectorMetadata) -> None:
        start = len(metadata)
        super()._process_retrieve_requests(metadata)
        for meta in metadata.requests[start:]:
            tracker = self._request_tracker(meta.request_id)
            meta.same_step_load = bool(
                tracker is not None and getattr(tracker, "same_step_load", False)
            )
            if tracker is not None:
                tracker.same_step_load = False

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self.sync_load_max_tokens <= 0:
            return super().start_load_kv(forward_context, **kwargs)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        self._delay_free_req_ids.update(metadata.delay_free_req_ids)

        sync, deferred = split_sync_retrieves(metadata.requests)
        if deferred:
            self._submit_retrieves(deferred)
        if sync:
            self._submit_retrieves(sync)
            self._wait_for_retrieves(sync)

    def _submit_retrieves(self, metas: list[LMCacheMPRequestMetadata]) -> None:
        adapter = self.worker_adapter
        adapter.batched_submit_retrieve_requests(
            [meta.request_id for meta in metas],
            [meta.op for meta in metas],
            adapter.create_recorded_event(),
            cache_salts=[meta.cache_salt for meta in metas],
            request_configs_list=[meta.request_configs for meta in metas],
        )

    def _wait_for_retrieves(self, metas: list[LMCacheMPRequestMetadata]) -> None:
        """Order the current stream after the server's copies.

        Only the server's response waits on the host; the copy itself is
        ordered by a stream wait, so the forward pass starts right after the
        data lands. The futures are consumed here instead of in `get_finished`,
        which would report a finished receive for a request that is already
        running. Failures are reported as invalid blocks, the channel the
        connector reserves for loads detected in the forward pass.
        """
        adapter = self.worker_adapter
        stream = current_stream()
        for meta in metas:
            dropped = getattr(adapter, "_dropped_retrieves", None)
            if dropped is not None:
                dropped.discard(meta.request_id)
            entry = adapter.retrieve_futures.pop(meta.request_id, None)
            adapter.retrieve_events.pop(meta.request_id, None)
            if entry is None:
                self._mark_load_failed(meta, meta.op.flat_block_ids)
                logger.error(
                    "Same-step KV load for request %s was not submitted; its "
                    "blocks are reported invalid.",
                    meta.request_id,
                )
                continue
            future, block_ids = entry
            if not self._wait_for_copy(future, stream):
                self._mark_load_failed(meta, block_ids)
                logger.error(
                    "Same-step KV load for request %s failed; its blocks are "
                    "reported invalid and will be recomputed.",
                    meta.request_id,
                )

    def _wait_for_copy(self, future: Any, stream: Any) -> bool:
        """Wait until the server's copy into the engine's blocks has landed."""
        wait_on_stream = getattr(future, "wait_on_stream", None)
        if wait_on_stream is not None:
            return bool(wait_on_stream(stream, timeout=self.sync_load_timeout_s))
        # Engine-driven transports copy before returning the future.
        try:
            return future.wait(self.sync_load_timeout_s) and bool(future.result())
        except Exception:
            logger.exception("Same-step KV load failed")
            return False

    def _mark_load_failed(self, meta: LMCacheMPRequestMetadata, block_ids: Any) -> None:
        adapter = self.worker_adapter
        adapter.error_block_ids.update(block_ids)
        failed = getattr(adapter, "_failed_request_ids", None)
        if failed is not None:
            failed.add(meta.request_id)
