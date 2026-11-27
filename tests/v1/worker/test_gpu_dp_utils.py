# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
allow_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _run_sync(local_allow_skip: bool, other_allow_skip: bool) -> bool:
    """Run sync_cudagraph_and_dp_padding as dp rank 0 of 2, with rank 1's
    all_reduce contribution simulated."""

    def fake_all_reduce(tensor):
        tensor[0][1] = 8  # rank 1 num_tokens
        tensor[1][1] = CUDAGraphMode.NONE.value
        tensor[2][1] = 0  # uniform_token_count (0 means None)
        tensor[3][1] = -1  # max_query_len (-1 means None)
        tensor[4][1] = 0  # allow_ubatching
        tensor[5][1] = 2  # rank 1 num_reqs
        tensor[6][1] = int(other_allow_skip)

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=2
    )
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        _, sync = dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=None,
            desired_batch_desc=desc,
            num_tokens=4,
            num_reqs=2,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            allow_skip_drafts=local_allow_skip,
        )
    assert sync is not None
    return sync.skip_drafts


def test_skip_drafts_requires_all_ranks():
    assert _run_sync(True, True)
    assert not _run_sync(True, False)
    assert not _run_sync(False, True)
    assert not _run_sync(False, False)


def test_dispatch_dp1_returns_no_sync():
    # With a single rank there is no agreement to make; callers fall back to
    # their local allow_skip_drafts value.
    _, sync = dp_utils.dispatch_cg_and_sync_dp(
        cudagraph_manager=None,
        num_reqs=2,
        num_tokens=4,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        allow_skip_drafts=True,
    )
    assert sync is None


class _GridManager:
    """The two `CudaGraphManager` calls the DP sync makes, on a PIECEWISE grid."""

    def __init__(self, sizes: list[int]):
        self.sizes = sizes

    def piecewise_token_bucket(self, num_tokens: int) -> int | None:
        return next((s for s in self.sizes if s >= num_tokens), None)

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, **kwargs):
        size = self.piecewise_token_bucket(num_tokens)
        if size is None:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
            )
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=size, num_reqs=None
        )


_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _sync_as(
    dp_rank: int,
    tokens: list[int],
    modes: list[CUDAGraphMode],
    monkeypatch,
    threshold: int,
):
    """Run the DP sync as `dp_rank`, with the other ranks' counts simulated."""
    monkeypatch.setenv("VLLM_DP_EAGER_PAD_TOKENS", str(threshold))
    manager = _GridManager(_GRID)

    def fake_all_reduce(tensor):
        for rank, (n, mode) in enumerate(zip(tokens, modes)):
            tensor[0][rank] = n
            tensor[1][rank] = mode.value
            tensor[2][rank] = 0
            tensor[3][rank] = -1
            tensor[4][rank] = 0
            tensor[5][rank] = 1
            tensor[6][rank] = 0

    desired = manager.dispatch(1, tokens[dp_rank], None)
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        return dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=manager,
            desired_batch_desc=desired,
            num_tokens=tokens[dp_rank],
            num_reqs=1,
            uniform_token_count=None,
            dp_size=len(tokens),
            dp_rank=dp_rank,
        )


_PW = CUDAGraphMode.PIECEWISE


def test_padding_waste_counts_rows_over_grid_sized_eager():
    import torch

    bucket = _GridManager(_GRID).piecewise_token_bucket
    # One 8k prefill next to three ranks of 6 tokens: 4 * 8192 - (8192 + 3 * 8).
    assert dp_utils.dp_padding_waste(torch.tensor([8192, 6, 6, 6]), bucket) == 24552
    assert dp_utils.dp_padding_waste(torch.tensor([8192] * 4), bucket) == 0
    # Above the grid there is no graph to fall back from.
    assert dp_utils.dp_padding_waste(torch.tensor([9000, 6]), bucket) is None


def test_lopsided_piecewise_step_runs_eager_at_each_ranks_grid_size(monkeypatch):
    """A PIECEWISE step that would pad 24552 rows runs eager; every rank agrees,
    and each one runs its own grid size rather than the largest rank's."""
    tokens = [8192, 6, 5, 3]
    results = [
        _sync_as(rank, tokens, [_PW] * 4, monkeypatch, threshold=4096)
        for rank in range(4)
    ]
    for rank, (desc, sync) in enumerate(results):
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert desc.num_tokens == [8192, 8, 8, 4][rank]
        assert sync is not None and sync.eager
        assert sync.num_tokens_across_dp.tolist() == [8192, 8, 8, 4]


def test_padding_below_the_threshold_keeps_the_graph(monkeypatch):
    # 4 * 1024 - (1024 + 3 * 8) = 3048 padded rows, under 4096.
    desc, sync = _sync_as(0, [1024, 6, 6, 6], [_PW] * 4, monkeypatch, 4096)
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.num_tokens == 1024
    assert sync is not None and not sync.eager
    assert sync.num_tokens_across_dp.tolist() == [1024] * 4


def test_zero_threshold_disables_the_fallback(monkeypatch):
    desc, sync = _sync_as(0, [8192, 6, 6, 6], [_PW] * 4, monkeypatch, 0)
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert sync is not None and not sync.eager


def test_full_decode_steps_are_never_sent_eager(monkeypatch):
    """Only PIECEWISE steps fall back: a FULL decode graph's launch savings
    outweigh its padding."""
    full = CUDAGraphMode.FULL
    manager = _GridManager(_GRID)
    monkeypatch.setenv("VLLM_DP_EAGER_PAD_TOKENS", "1")
    tokens = [384, 6, 6, 6]

    def fake_all_reduce(tensor):
        for rank, n in enumerate(tokens):
            tensor[0][rank] = n
            tensor[1][rank] = full.value
            tensor[2][rank] = 6
            tensor[3][rank] = 6
            tensor[4][rank] = 0
            tensor[5][rank] = 1
            tensor[6][rank] = 0

    desired = BatchExecutionDescriptor(
        cg_mode=full, num_tokens=384, num_reqs=64, uniform_token_count=6
    )
    with (
        mock.patch.object(dp_utils, "get_dp_group") as mock_group,
        mock.patch.object(
            dp_utils,
            "_should_run_padding_eager",
            wraps=dp_utils._should_run_padding_eager,
        ) as decide,
    ):
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=manager,
            desired_batch_desc=desired,
            num_tokens=384,
            num_reqs=64,
            uniform_token_count=6,
            dp_size=4,
            dp_rank=0,
        )
    decide.assert_not_called()
