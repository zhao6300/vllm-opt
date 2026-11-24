# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step loads for small external hits in the LMCache MP connector."""

import pytest

pytest.importorskip("lmcache")

from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector as _BaseConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)

from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_mp_sync_load
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_sync_load import (
    LMCacheMPConnector,
    split_sync_retrieves,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.request import Request  # noqa: E402


def _meta(request_id: str, start: int, end: int, direction: str = "RETRIEVE"):
    return LMCacheMPRequestMetadata(
        request_id=request_id,
        direction=direction,
        op=LoadStoreOp(token_bytes=b"", block_ids=[[7]], start=start, end=end),
    )


def test_split_follows_the_scheduler_decision():
    # The retrieve rounds its start down to the LMCache chunk boundary, so a
    # load the scheduler admitted synchronously can span more tokens than the
    # threshold. Deferring it left a finished receive for a running request.
    tail = _meta("tail", 16256, 16383)
    tail.same_step_load = True
    chunked = _meta("chunked", 16128, 16383)
    chunked.same_step_load = True
    full = _meta("full", 0, 16256)

    sync, deferred = split_sync_retrieves([tail, full, chunked])

    assert [m.request_id for m in sync] == ["tail", "chunked"]
    assert [m.request_id for m in deferred] == ["full"]


def test_split_skips_stores_and_empty_loads():
    store = _meta("store", 0, 512, direction="STORE")
    store.same_step_load = True

    sync, deferred = split_sync_retrieves([store, _meta("empty", 8, 8)])

    assert sync == []
    assert deferred == []


def _connector(max_tokens: int, multi_group: bool = False) -> LMCacheMPConnector:
    connector = object.__new__(LMCacheMPConnector)
    connector.sync_load_max_tokens = max_tokens
    connector.sync_load_timeout_s = 60.0
    connector.request_trackers = {}
    connector._report_request_errors = multi_group
    return connector


def _tracked_connector(
    max_tokens: int,
) -> tuple[LMCacheMPConnector, Request, LMCacheMPRequestTracker]:
    request = Request(
        request_id="tail",
        prompt_token_ids=list(range(1024)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    connector = _connector(max_tokens)
    tracker = LMCacheMPRequestTracker(request)
    connector.request_trackers[request.request_id] = tracker
    return connector, request, tracker


def _stub_match(monkeypatch, need: int | None, async_load: bool):
    monkeypatch.setattr(
        _BaseConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (need, async_load),
    )


def test_small_hit_loads_in_its_step(monkeypatch):
    _stub_match(monkeypatch, 127, True)

    assert _connector(128).get_num_new_matched_tokens(None, 16256) == (127, False)


def test_large_hit_stays_asynchronous(monkeypatch):
    _stub_match(monkeypatch, 16256, True)

    assert _connector(128).get_num_new_matched_tokens(None, 0) == (16256, True)


def test_switch_off_keeps_stock_behaviour(monkeypatch):
    _stub_match(monkeypatch, 127, True)

    assert _connector(0).get_num_new_matched_tokens(None, 16256) == (127, True)


def test_pending_lookup_and_misses_pass_through(monkeypatch):
    _stub_match(monkeypatch, None, True)
    assert _connector(128).get_num_new_matched_tokens(None, 0) == (None, True)

    _stub_match(monkeypatch, 0, False)
    assert _connector(128).get_num_new_matched_tokens(None, 0) == (0, False)


class _Health:
    def __init__(self, healthy: bool) -> None:
        self.is_healthy = healthy


def test_unhealthy_server_keeps_the_load_asynchronous(monkeypatch):
    # A same-step load has no recovery path, so it must not be admitted while
    # the server behind the lookup is down.
    _stub_match(monkeypatch, 127, True)
    connector = _connector(128)
    connector.scheduler_adapter = _Health(healthy=False)

    assert connector.get_num_new_matched_tokens(None, 16256) == (127, True)


def test_lookup_records_its_decision_on_the_tracker(monkeypatch):
    connector, request, tracker = _tracked_connector(128)

    _stub_match(monkeypatch, 127, True)
    assert connector.get_num_new_matched_tokens(request, 16256) == (127, False)
    assert tracker.same_step_load is True

    _stub_match(monkeypatch, 129, True)
    assert connector.get_num_new_matched_tokens(request, 16256) == (129, True)
    assert tracker.same_step_load is False


def test_retrieve_metadata_carries_the_decision(monkeypatch):
    def emit(self, metadata):
        metadata.add_request_metadata(_meta("tail", 16128, 16383))

    monkeypatch.setattr(_BaseConnector, "_process_retrieve_requests", emit)
    connector, _, tracker = _tracked_connector(128)
    tracker.same_step_load = True
    metadata = LMCacheMPConnectorMetadata()

    connector._process_retrieve_requests(metadata)

    assert metadata.requests[0].same_step_load is True
    assert tracker.same_step_load is False


class _Future:
    def __init__(self, error: Exception | None = None, result=None) -> None:
        self.error = error
        self.result = result
        self.waited = False

    def wait_on_stream(self, stream, timeout):
        self.waited = True
        if self.error is not None:
            raise self.error
        return self.result


class _Adapter:
    def __init__(self, futures: dict) -> None:
        self.retrieve_futures = futures
        self.retrieve_events = {}
        self.error_block_ids = set()
        self._failed_request_ids = set()
        self._dropped_retrieves = set()


def _wait(monkeypatch, futures, multi_group: bool = False):
    monkeypatch.setattr(lmcache_mp_sync_load, "current_stream", lambda: None)
    connector = _connector(128, multi_group)
    connector.worker_adapter = _Adapter(futures)
    meta = _meta("tail", 16256, 16383)
    connector._wait_for_retrieves([meta])
    return connector.worker_adapter, meta


def test_wait_consumes_the_future_and_reports_success(monkeypatch):
    future = _Future(result=True)
    adapter, _ = _wait(monkeypatch, {"tail": (future, [7])})

    assert future.waited
    assert adapter.retrieve_futures == {}
    assert adapter.error_block_ids == set()
    assert adapter._failed_request_ids == set()


def test_false_server_result_reports_the_blocks_as_invalid(monkeypatch):
    # The server answers normally when it could not read every key; the
    # result flag, not the absence of an exception, is the success signal.
    adapter, meta = _wait(monkeypatch, {"tail": (_Future(result=False), [7, 8])})

    assert adapter.error_block_ids == {7, 8}
    assert adapter._failed_request_ids == {meta.request_id}


def test_failed_copy_reports_the_blocks_as_invalid(monkeypatch):
    adapter, meta = _wait(
        monkeypatch, {"tail": (_Future(error=TimeoutError("no response")), [7, 8])}
    )

    assert adapter.error_block_ids == {7, 8}
    assert adapter._failed_request_ids == {meta.request_id}


def test_missing_future_reports_the_blocks_as_invalid(monkeypatch):
    adapter, meta = _wait(monkeypatch, {})

    assert adapter.error_block_ids == {7}
    assert meta.request_id in adapter._failed_request_ids


def test_failed_copy_raises_when_blocks_cannot_be_reported(monkeypatch):
    # Multi-group layouts refuse flat block ids and cannot reschedule a running
    # request, so the only correct outcome is to stop before the forward pass
    # reads blocks that never loaded.
    with pytest.raises(RuntimeError, match="multiple KV cache groups"):
        _wait(
            monkeypatch,
            {"tail": (_Future(error=TimeoutError("no response")), [7, 8])},
            multi_group=True,
        )


def test_missing_future_raises_when_blocks_cannot_be_reported(monkeypatch):
    with pytest.raises(RuntimeError, match="server connection was lost"):
        _wait(monkeypatch, {}, multi_group=True)


def test_unscheduled_report_is_cleared_for_same_step_loads(monkeypatch):
    # An unhealthy adapter would report the id as a finished receive, which
    # vLLM asserts on for a request that is already running.
    monkeypatch.setattr(lmcache_mp_sync_load, "current_stream", lambda: None)
    connector = _connector(128)
    adapter = _Adapter({})
    adapter._dropped_retrieves.add("tail")
    connector.worker_adapter = adapter

    connector._wait_for_retrieves([_meta("tail", 16256, 16383)])

    assert adapter._dropped_retrieves == set()
