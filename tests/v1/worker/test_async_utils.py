# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accepted-token accounting between the worker and the scheduler.

The worker samples ``num_sampled + num_rejected`` tokens per request
(``num_speculative_steps + 1`` with drafts) and the scheduler reserves
placeholders for the ``num_sampled`` accepted ones. Handing it any other
number shifts every later position of that request.
"""

import pytest

from vllm.v1.worker.gpu.async_utils import trim_sampled_token_ids

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("num_speculative_steps", "num_sampled"),
    [
        (5, 6),  # every draft accepted, plus the bonus token
        (5, 3),  # the common case: a prefix accepted
        (5, 1),  # every draft rejected, only the bonus token
        (1, 2),
        (0, 1),  # no drafts: one sampled token
    ],
)
def test_rejected_slots_never_reach_the_scheduler(
    num_speculative_steps: int, num_sampled: int
):
    width = num_speculative_steps + 1
    num_rejected = width - num_sampled
    # The counts the last rank broadcasts must account for the whole row.
    assert num_sampled + num_rejected == width
    # Encode the row index so a request paired with another's tokens fails.
    rows = [[10 * req + i for i in range(width)] for req in range(4)]

    trimmed = trim_sampled_token_ids(rows, [num_sampled] * 4)

    assert trimmed == [row[:num_sampled] for row in rows]


def test_each_request_is_trimmed_by_its_own_count():
    """A batch mixes acceptance counts; one count must not apply to all rows."""
    width = 6
    rows = [[10 * req + i for i in range(width)] for req in range(3)]

    trimmed = trim_sampled_token_ids(rows, [6, 1, 4])

    assert trimmed == [rows[0], rows[1][:1], rows[2][:4]]


def test_chunked_prefill_row_keeps_no_token():
    """A request still prefilling samples nothing this step."""
    trimmed = trim_sampled_token_ids([[7, 8, 9, 10, 11, 12]], [0])

    assert trimmed == [[]]


def test_count_wider_than_the_row_is_rejected():
    """Otherwise the padding tail would pass as if it were sampled."""
    with pytest.raises(ValueError, match="num_sampled=7 does not fit"):
        trim_sampled_token_ids([[1, 2, 3, 4, 5, 6]], [7])


def test_negative_count_is_rejected():
    with pytest.raises(ValueError, match="num_sampled=-1 does not fit"):
        trim_sampled_token_ids([[1, 2, 3]], [-1])


def test_row_and_request_counts_must_agree():
    with pytest.raises(ValueError, match="2 sampled rows for 3 requests"):
        trim_sampled_token_ids([[1, 2], [3, 4]], [1, 1, 1])
