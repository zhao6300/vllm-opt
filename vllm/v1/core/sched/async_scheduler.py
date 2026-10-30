# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size
        # Only XPU can disable PP decode microbatching via VLLM_XPU_PP_MICROBATCH.
        self.decode_stagger = self.pp_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            self.decode_stagger = 1
        # Keep the decode microbatches the same size (VLLM_PP_MICROBATCH_BALANCE).
        self.balance_decode_microbatches = (
            self.decode_stagger > 1 and envs.VLLM_PP_MICROBATCH_BALANCE
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        staggered: list[Request] = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = (
                    self.current_step + self.decode_stagger
                )
                staggered.append(request)

        if self.use_v2_model_runner and self.balance_decode_microbatches:
            self._balance_decode_microbatches(staggered)

    def _balance_decode_microbatches(self, scheduled: list[Request]) -> None:
        """Move decodes from this step's microbatch to the smallest one.

        With PP, a request decodes every ``decode_stagger`` steps, so the
        decoding requests fall into ``decode_stagger`` microbatches by the step
        they are next eligible in, and the stages work on them in turn. A burst
        of requests that start decoding together stays in one microbatch, and
        the other steps carry nearly nothing. This moves the lowest-priority
        requests just scheduled (``scheduled``, in scheduling order) into the
        smallest microbatch until the sizes differ by at most one.

        A moved request waits extra steps once, never fewer than
        ``decode_stagger``: the worker-side sampled-token ring hands its
        tokens back ``decode_stagger`` steps later and keeps them, so a longer
        gap stays valid.
        """
        stagger = self.decode_stagger
        base = self.current_step + 1  # The next step's microbatch is bucket 0.
        sizes = [0] * stagger
        for request in self.running:
            if request.is_prefill_chunk or request.is_finished():
                continue
            step = max(request.next_decode_eligible_step, base)
            sizes[(step - base) % stagger] += 1
        # Requests scheduled now are next eligible at current_step + stagger.
        this_bucket = stagger - 1
        movable = list(scheduled)
        while movable:
            target = min(range(stagger), key=lambda b: (sizes[b], b))
            if sizes[this_bucket] - sizes[target] <= 1:
                break
            request = movable.pop()
            # The first step of bucket `target` at least `stagger` steps away.
            request.next_decode_eligible_step = base + target + stagger
            sizes[this_bucket] -= 1
            sizes[target] += 1

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        # Tokens trimmed at a stop resolve their placeholders too.
        num_new_tokens = len(new_token_ids)
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids, is_stale
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= num_new_tokens
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
