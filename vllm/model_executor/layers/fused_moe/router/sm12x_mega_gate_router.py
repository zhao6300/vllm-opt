# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 sqrtsoftplus routing on FlashInfer's SM12x fused gate.

The fused gate computes the gate GEMM, the square-root-softplus score, the
correction bias, the top-k, and the normalized weights in one kernel. It is
opt-in (`VLLM_FLASHINFER_DSV41_SM120_MEGA_GATE=1`) because it replaces the
separate `GateLinear` launch; the router stays numerically identical to the
stock path by falling back to the stock `fused_topk_bias` whenever the fused
kernel's contract is not met.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    get_moe_padding_mask,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Plans bind operand addresses, so a captured step needs one per graph-pool
# operand. Graphs reuse their buffers; keep the recently used ones and rebuild
# past that.
_MAX_PLANS = 32

# Eager steps copy their hidden states into a plan-owned operand buffer, so they
# only get one while that copy stays cheaper than building a plan (~80 us), and
# few shapes keep the buffers small. Larger batches rebuild instead.
_EAGER_MAX_TOKENS = 1024
_MAX_EAGER_PLANS = 8


@dataclass
class _PlanEntry:
    plan: Any
    ids: torch.Tensor
    weights: torch.Tensor
    scratch: torch.Tensor
    barriers: torch.Tensor
    x: torch.Tensor
    mask: torch.Tensor | None


class Sm12xMegaGateRouter(FusedTopKBiasRouter):
    """Fused top-k router backed by `flashinfer.experimental.sm12x_mega_gate`."""

    def __init__(
        self,
        *,
        top_k: int,
        global_num_experts: int,
        e_score_correction_bias: torch.Tensor | None,
        renormalize: bool,
        routed_scaling_factor: float,
        scoring_func: str,
        eplb_state: Any = None,
    ) -> None:
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            e_score_correction_bias=e_score_correction_bias,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func=scoring_func,
            eplb_state=eplb_state,
        )
        # Held in a plain dict: assigning the module directly would register it
        # again and duplicate its parameters in the model.
        self._gate_ref: dict[str, torch.nn.Module] = {}
        self._gate_skipped = False
        self._unavailable: str | None = None
        self._plans: OrderedDict[tuple, _PlanEntry] = OrderedDict()
        self._eager_plans: OrderedDict[tuple, _PlanEntry] = OrderedDict()
        self._buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._route_masks: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def attach_gate(self, gate: torch.nn.Module, *, skipped: bool) -> None:
        """Bind the gate module; `skipped` means the runner no longer runs it."""
        self._gate_ref["gate"] = gate
        self._gate_skipped = skipped

    @property
    def uses_fused_gate(self) -> bool:
        return self._gate_skipped and self._unavailable is None

    def _gate(self) -> torch.nn.Module:
        return self._gate_ref["gate"]

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_fused_gate:
            routed = self._fused_routing(hidden_states, indices_type, input_ids)
            if routed is not None:
                return routed
        if self._gate_skipped:
            # The runner skipped the gate GEMM, so it must be run here, with
            # the module the stock path would have called.
            router_logits, _ = self._gate()(hidden_states)
        return super()._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )

    def _fused_routing(
        self,
        hidden_states: torch.Tensor,
        indices_type: torch.dtype | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if input_ids is not None and self._hash_indices_table is not None:
            return None
        if self.eplb_state is not None or self.bias_vl is not None:
            return None
        if self.num_fused_shared_experts > 0:
            return None
        # The kernel always normalizes over the top-k and always masks padded
        # tokens; the stock path does either only when asked to.
        if not self.renormalize or not self.masks_padding:
            return None
        x = hidden_states
        if (
            x.dim() != 2
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.shape[0] == 0
            or x.shape[1] % 64
            or self.global_num_experts % 4
            or not 0 < self.top_k <= min(self.global_num_experts, 32)
        ):
            return None
        if x.data_ptr() % 16:
            return None
        gate = self._gate()
        weight = getattr(gate, "weight", None)
        bias = self.e_score_correction_bias
        if (
            weight is None
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.shape != (self.global_num_experts, x.shape[1])
            or weight.device != x.device
            or weight.data_ptr() % 16
        ):
            return None
        if bias is not None and (
            bias.dtype != torch.float32
            or not bias.is_contiguous()
            or bias.numel() != self.global_num_experts
            or bias.data_ptr() % 16
        ):
            return None

        entry = self._entry(x, weight, bias, get_moe_padding_mask(x.shape[0]))
        if entry is None:
            return None
        ids, weights = entry.plan.run()
        out_type = torch.int32 if indices_type is None else indices_type
        return weights, ids.to(out_type)

    def _entry(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        padding: torch.Tensor | None,
    ) -> _PlanEntry | None:
        """The plan for this batch, bound to operands that outlive the call.

        A captured step hands in its graph-pool operands, which stay put for the
        life of the graph, so each graph keeps an entry of its own; preparation
        runs during capture and is the only way the fused gate reaches a replay.
        An eager step hands in a fresh hidden-states tensor every call, so small
        batches share one entry per shape and copy into the plan's own buffer;
        larger ones rebuild against the tensor they were given.
        """
        if not torch.cuda.is_current_stream_capturing() and (
            x.shape[0] <= _EAGER_MAX_TOKENS
        ):
            return self._eager_entry(x, weight, bias, padding)

        mask = None if padding is None else self._route_mask(padding)
        key = (
            x.shape[0],
            x.shape[1],
            x.data_ptr(),
            0 if mask is None else mask.data_ptr(),
        )
        entry = self._plans.get(key)
        if entry is not None:
            self._plans.move_to_end(key)
            return entry
        entry = self._build(x, weight, bias, mask)
        if entry is None:
            return None
        self._plans[key] = entry
        while len(self._plans) > _MAX_PLANS:
            self._plans.popitem(last=False)
        return entry

    def _eager_entry(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        padding: torch.Tensor | None,
    ) -> _PlanEntry | None:
        key = (x.shape[0], x.shape[1], padding is not None)
        entry = self._eager_plans.get(key)
        if entry is None:
            operand = torch.empty_like(x)
            mask = (
                torch.empty(x.shape[0], dtype=torch.bool, device=x.device)
                if padding is not None
                else None
            )
            entry = self._build(operand, weight, bias, mask)
            if entry is None:
                return None
            self._eager_plans[key] = entry
            while len(self._eager_plans) > _MAX_EAGER_PLANS:
                self._eager_plans.popitem(last=False)
        if entry.x.data_ptr() != x.data_ptr():
            entry.x.copy_(x)
        if padding is not None:
            # FlashInfer's mask marks the rows to route; vLLM's marks padding.
            torch.logical_not(padding, out=entry.mask)
        return entry

    def _build(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> _PlanEntry | None:
        from vllm.utils.flashinfer import flashinfer_sm12x_mega_gate_plan

        # The kernel state (k/sqrt(m) scaling) must match the stock gate's
        # sqrtsoftplus, and the output buffers are ours to reuse across plans.
        buffers = self._buffers.get(x.shape[0])
        if buffers is None:
            m = x.shape[0]
            ids = torch.empty((m, self.top_k), dtype=torch.int64, device=x.device)
            weights = torch.empty((m, self.top_k), dtype=torch.float32, device=x.device)
            scratch, barriers = self._kernel_buffers(m, x.shape[1])
            buffers = (ids, weights, scratch, barriers)
            self._buffers[m] = buffers
        ids, weights, scratch, barriers = buffers

        try:
            plan = flashinfer_sm12x_mega_gate_plan(
                x,
                weight,
                self.top_k,
                scoring_func=self.scoring_func,
                bias=bias,
                mask=mask,
                routed_scaling_factor=self.routed_scaling_factor,
                ep_rank=0,
                out=(ids, weights),
                scratch=scratch,
                score_barriers=barriers,
            )
        except (RuntimeError, ValueError, TypeError, NotImplementedError) as e:
            logger.warning_once(
                "FlashInfer SM12x Mega Gate declined this batch (%s); routing "
                "with the stock gate kernels.",
                e,
            )
            self._unavailable = str(e)
            return None

        return _PlanEntry(plan, ids, weights, scratch, barriers, x, mask)

    def _route_mask(self, padding: torch.Tensor) -> torch.Tensor:
        """Invert `padding` into a stable buffer the kernel can bind.

        The buffer is reused so the plan key — and the pointer a captured
        step writes to — stays put across calls.
        """
        key = (padding.shape[0], padding.data_ptr())
        mask = self._route_masks.get(key)
        if mask is None:
            mask = torch.empty(
                padding.shape[0], dtype=torch.bool, device=padding.device
            )
            self._route_masks[key] = mask
            while len(self._route_masks) > _MAX_PLANS:
                self._route_masks.popitem(last=False)
        else:
            self._route_masks.move_to_end(key)
        return torch.logical_not(padding, out=mask)

    def _kernel_buffers(
        self, num_tokens: int, hidden: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split-K scratch and score barriers sized by the kernel's heuristic.

        The plan picks the same schedule from the same inputs, so its shape
        checks accept these.
        """
        from flashinfer.experimental.sm12x_mega_gate.heuristics import (
            select_config,
        )

        device = torch.cuda.current_device()
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        config = select_config(num_tokens, hidden, self.global_num_experts, sms)
        scratch = torch.empty(
            config.scratch_shape(num_tokens, self.global_num_experts),
            dtype=torch.float32,
            device=torch.device("cuda", device),
        )
        barriers = torch.zeros(
            (config.num_token_blocks(num_tokens), 16),
            dtype=torch.uint64,
            device=torch.device("cuda", device),
        )
        return scratch, barriers

    def prepare_monolithic_logits(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self._gate_skipped:
            raise RuntimeError(
                "the fused SM12x gate router has no monolithic router logits"
            )
        return super().prepare_monolithic_logits(router_logits)


def sm12x_mega_gate_available() -> bool:
    """Whether this process can build the fused gate router at all."""
    if not envs.VLLM_FLASHINFER_DSV41_SM120_MEGA_GATE:
        return False
    if not current_platform.is_cuda() or not (
        current_platform.is_device_capability_family(120)
    ):
        return False
    from vllm.utils.flashinfer import (
        has_flashinfer_sm12x_mega_gate,
        sm12x_mega_gate_arch_error,
    )

    return has_flashinfer_sm12x_mega_gate() and sm12x_mega_gate_arch_error() is None
