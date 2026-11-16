# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which CUDA graph buffers custom all-reduce may IPC-register."""

from types import SimpleNamespace

import pytest

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators.cuda_communicator import (
    registers_graph_buffers,
)


@pytest.fixture
def alloc_conf(monkeypatch):
    """Set the allocator config the communicator reads."""

    def set_conf(conf: str) -> None:
        monkeypatch.setattr(cuda_communicator, "current_alloc_conf", lambda: conf)

    return set_conf


def config(use_cumem_cudagraph_pool: bool = False):
    return SimpleNamespace(use_cumem_cudagraph_pool=use_cumem_cudagraph_pool)


def test_graph_buffers_register_with_the_default_allocator(alloc_conf):
    alloc_conf("")
    assert registers_graph_buffers(config())
    assert registers_graph_buffers(None)


@pytest.mark.parametrize(
    "conf",
    ["expandable_segments:True", "max_split_size_mb:512,expandable_segments:True"],
)
def test_expandable_segments_are_copied_not_registered(alloc_conf, conf):
    # cudaIpcGetMemHandle rejects the VMM memory the allocator hands out.
    alloc_conf(conf)
    assert not registers_graph_buffers(config())
    assert not registers_graph_buffers(None)


def test_expandable_segments_turned_off_register(alloc_conf):
    alloc_conf("expandable_segments:False")
    assert registers_graph_buffers(config())


def test_cumem_graph_pool_is_copied_not_registered(alloc_conf):
    alloc_conf("")
    assert not registers_graph_buffers(config(use_cumem_cudagraph_pool=True))
