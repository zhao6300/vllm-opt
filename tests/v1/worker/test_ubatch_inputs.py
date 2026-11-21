# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.ubatch_inputs import (
    slice_lookback_token_ids,
    update_captured_lookback,
)


@pytest.mark.parametrize(
    "request_slice,token_slice,expected",
    [
        (slice(0, 1), slice(0, 2), [[9, 8, 7], [-1, -1, -1]]),
        (slice(0, 2), slice(2, 5), [[11, 10, 9], [29, 28, 27], [-1, -1, -1]]),
        (slice(1, 2), slice(3, 5), [[29, 28, 27], [-1, -1, -1]]),
        (slice(0, 1), slice(1, 3), [[10, 9, 8], [-1, -1, -1]]),
    ],
)
def test_lookbacks_follow_microbatch_chunk_start(request_slice, token_slice, expected):
    """A split request reads previous batch tokens before its older history."""
    actual = slice_lookback_token_ids(
        torch.tensor([[9, 8, 7], [29, 28, 27]], dtype=torch.int32),
        torch.tensor([10, 11, 12, 30, 31]),
        torch.tensor([0, 3, 5]),
        request_slice,
        token_slice,
    )
    assert actual.tolist() == expected


def test_lookback_padding_does_not_read_the_last_live_request():
    actual = slice_lookback_token_ids(
        torch.tensor([[5, -1, -1]]),
        torch.tensor([6, 0, 0]),
        torch.tensor([0, 1]),
        slice(0, 3),
        slice(0, 3),
    )
    assert actual.tolist() == [[5, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_replay_refreshes_history_and_clears_unused_request_rows():
    captured = torch.tensor([[1, 2, 3], [4, 5, 6]])
    address = captured.data_ptr()
    current = slice_lookback_token_ids(
        torch.tensor([[7, 8, 9]]),
        torch.tensor([10, 11]),
        torch.tensor([0, 2]),
        slice(0, 1),
        slice(0, 2),
    )
    update_captured_lookback(captured, current)
    assert captured.data_ptr() == address
    assert captured.tolist() == [[7, 8, 9], [-1, -1, -1]]


def test_replay_rejects_a_changed_history_signature():
    with pytest.raises(ValueError, match="graph signature"):
        update_captured_lookback(torch.empty(2, 3), torch.empty(3, 3))
