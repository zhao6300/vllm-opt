# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.pp_send_ring import PPSendRing

pytestmark = pytest.mark.cpu_test


class _Handle:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def _sender(log: list[str], sent: list[dict]):
    def send(tensors: dict) -> list[_Handle]:
        step = len(sent)
        sent.append(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tensors.items()}
        )
        log.append(f"send:{step}")
        return [_Handle(log, f"meta{step}"), _Handle(log, f"t{step}")]

    return send


def _buffers(max_tokens: int = 8):
    return lambda: {
        "hidden_states": torch.zeros(max_tokens, 4, 3),
        "pre_mix": torch.zeros(max_tokens, 4),
    }


@pytest.mark.parametrize("num_slots", [1, 2, 3])
def test_slot_reuse_waits_only_for_that_slots_send(num_slots):
    log: list[str] = []
    sent: list[dict] = []
    ring = PPSendRing(num_slots, _buffers())
    send = _sender(log, sent)
    for step in range(num_slots + 2):
        ring.send({"hidden_states": torch.full((2, 4, 3), float(step))}, send)

    expected: list[str] = []
    for step in range(num_slots + 2):
        if step >= num_slots:
            # Never the metadata handle: the GroupCoordinator reaps it.
            expected.append(f"wait:t{step - num_slots}")
        expected.append(f"send:{step}")
    assert log == expected
    assert [float(s["hidden_states"][0, 0, 0]) for s in sent] == [
        float(step) for step in range(num_slots + 2)
    ]


def test_payload_is_copied_out_of_the_source():
    sent: list[dict] = []
    staged: list[dict] = []
    ring = PPSendRing(2, _buffers())

    def send(tensors):
        staged.append(tensors)
        return _sender([], sent)(tensors)

    source = torch.ones(3, 4, 3)
    ring.send({"hidden_states": source, "pre_mix": torch.ones(3, 4)}, send)
    source.zero_()  # The next forward overwrites its static output.

    out = staged[0]["hidden_states"]
    assert out.shape == (3, 4, 3)
    assert out.data_ptr() != source.data_ptr()
    assert torch.equal(out, torch.ones(3, 4, 3))
    assert out.is_contiguous()


def test_non_tensors_and_empty_tensors_pass_through():
    staged: list[dict] = []
    ring = PPSendRing(1, _buffers())
    empty = torch.empty(0, 4, 3)

    def send(tensors):
        staged.append(tensors)
        return [_Handle([], "meta")]

    ring.send({"hidden_states": empty, "flag": 7}, send)
    assert staged[0]["flag"] == 7
    assert staged[0]["hidden_states"] is empty


def test_unexpected_key_or_larger_step_grows_the_slot_once():
    staged: list[dict] = []
    ring = PPSendRing(1, _buffers(max_tokens=2))

    def send(tensors):
        staged.append(tensors)
        return [_Handle([], "meta")]

    big = torch.arange(5 * 4 * 3, dtype=torch.float32).reshape(5, 4, 3)
    extra = torch.ones(5, 7)
    ring.send({"hidden_states": big, "aux_hidden_states_0": extra}, send)
    ring.send({"hidden_states": big[:4], "aux_hidden_states_0": extra[:4]}, send)

    assert torch.equal(staged[0]["hidden_states"], big)
    assert torch.equal(staged[1]["aux_hidden_states_0"], extra[:4])
    # The grown buffer is reused by the next step.
    assert (
        staged[0]["hidden_states"].data_ptr() == staged[1]["hidden_states"].data_ptr()
    )


def test_wait_all_drains_every_slot():
    log: list[str] = []
    ring = PPSendRing(2, _buffers())
    send = _sender(log, [])
    ring.send({"pre_mix": torch.ones(1, 4)}, send)
    ring.send({"pre_mix": torch.ones(1, 4)}, send)
    ring.wait_all()
    assert log[-2:] == ["wait:t0", "wait:t1"]
    ring.wait_all()
    assert log[-2:] == ["wait:t0", "wait:t1"]
