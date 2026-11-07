# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pp_size = self.parallel_config.pipeline_parallel_size
        # Only XPU can disable PP decode microbatching via VLLM_XPU_PP_MICROBATCH.
        self.decode_stagger = self.vllm_config.pp_num_microbatches
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            self.decode_stagger = 1
        if self.decode_stagger > self.pp_size:
            logger.info_once(
                "PP decode microbatches: %d (pipeline_parallel_size=%d); each "
                "request decodes every %d steps.",
                self.decode_stagger,
                self.pp_size,
                self.decode_stagger,
            )
        # How decodes are spread over the microbatches
        # (VLLM_PP_MICROBATCH_BALANCE): equal sizes, or packed into the fullest.
        balance_mode = envs.VLLM_PP_MICROBATCH_BALANCE
        self.balance_decode_microbatches = (
            self.decode_stagger > 1 and balance_mode == "balance"
        )
        self.pack_decode_microbatches = (
            self.decode_stagger > 1 and balance_mode == "pack"
        )
        self.pack_max_seqs = envs.VLLM_PP_PACK_MAX_SEQS
        if self.pack_max_seqs < 0:
            raise ValueError("VLLM_PP_PACK_MAX_SEQS must be >= 0")
        # Prefill token caps of the steps with and without decodes (pack mode).
        budget = self.scheduler_config.max_num_batched_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        decode_step_cap = envs.VLLM_PP_PACK_DECODE_STEP_PREFILL_TOKENS
        free_step_cap = envs.VLLM_PP_PACK_FREE_STEP_PREFILL_TOKENS
        for name, cap in (
            ("VLLM_PP_PACK_DECODE_STEP_PREFILL_TOKENS", decode_step_cap),
            ("VLLM_PP_PACK_FREE_STEP_PREFILL_TOKENS", free_step_cap),
        ):
            if cap is not None and cap < 0:
                raise ValueError(f"{name} must be >= 0")
        self.pack_decode_step_prefill_tokens = (
            decode_step_cap if decode_step_cap is not None else threshold or budget
        )
        self.pack_free_step_prefill_tokens = (
            free_step_cap if free_step_cap is not None else budget
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
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
            request.spec_token_ids = spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = (
                    self.current_step + self.decode_stagger
                )
                staggered.append(request)

        if self.use_v2_model_runner:
            if self.balance_decode_microbatches:
                self._balance_decode_microbatches(staggered)
            elif self.pack_decode_microbatches:
                self._pack_decode_microbatches(staggered)

    def _prefill_token_cap(self) -> int | None:
        """Prefill token cap of the step being scheduled (pack mode).

        A step with decodes to run gets ``pack_decode_step_prefill_tokens``,
        any other step ``pack_free_step_prefill_tokens``. A cap of 0 for the
        decode steps is dropped when every microbatch has decodes, since no
        free step would be left to prefill in.
        """
        if not (self.use_v2_model_runner and self.pack_decode_microbatches):
            return None
        has_decodes = any(
            not r.is_prefill_chunk
            and not r.is_finished()
            and r.next_decode_eligible_step <= self.current_step
            for r in self.running
        )
        if not has_decodes:
            return self.pack_free_step_prefill_tokens
        cap = self.pack_decode_step_prefill_tokens
        if cap == 0 and all(self._decode_microbatch_sizes(self.current_step)):
            return None
        return cap

    def _decode_microbatch_sizes(self, first_step: int | None = None) -> list[int]:
        """Decoding requests per microbatch, indexed from ``first_step``.

        ``first_step`` defaults to the next step, once this one is scheduled:
        requests scheduled now (next eligible ``decode_stagger`` steps away)
        are then in the last microbatch.
        """
        stagger = self.decode_stagger
        # The microbatch of the first step is bucket 0.
        base = self.current_step + 1 if first_step is None else first_step
        sizes = [0] * stagger
        for request in self.running:
            if request.is_prefill_chunk or request.is_finished():
                continue
            step = max(request.next_decode_eligible_step, base)
            sizes[(step - base) % stagger] += 1
        return sizes

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
        base = self.current_step + 1
        sizes = self._decode_microbatch_sizes()
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

    def _pack_decode_microbatches(self, scheduled: list[Request]) -> None:
        """Move decodes from this step's microbatch into the fullest other one.

        Packing keeps the decodes in as few microbatches as possible: the steps
        of the other microbatches carry no decode, so a bounded prefill chunk
        there does not delay any token. A request just scheduled moves only
        while the fullest other microbatch is at least as big as its own, so
        a group that is already the fullest stays put. With
        ``VLLM_PP_PACK_MAX_SEQS`` set, a microbatch with that many decodes
        takes no more and the rest go to the next fullest one, including the
        excess of a burst that starts decoding together.

        A moved request waits extra steps once, never fewer than
        ``decode_stagger`` (see ``_balance_decode_microbatches``).
        """
        stagger = self.decode_stagger
        base = self.current_step + 1
        sizes = self._decode_microbatch_sizes()
        this_bucket = stagger - 1
        cap = self.pack_max_seqs
        movable = list(scheduled)
        while movable:
            targets = [b for b in range(this_bucket) if cap == 0 or sizes[b] < cap]
            if not targets:
                break
            target = max(targets, key=lambda b: (sizes[b], -b))
            over_cap = cap > 0 and sizes[this_bucket] > cap
            if sizes[target] < sizes[this_bucket] and not over_cap:
                break
            request = movable.pop()
            request.next_decode_eligible_step = base + target + stagger
            sizes[this_bucket] -= 1
            sizes[target] += 1
