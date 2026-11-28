# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-budgeted prefill chunking.

A step on a data-parallel MoE deployment ends when its slowest rank does, so a
prefill chunk sized by tokens alone makes steps of very different length.
`PrefillCostModel` prices a step and solves for the largest prefill chunk that
keeps it within a time budget.
"""

import math
import sys
from dataclasses import dataclass, replace

import vllm.envs as envs


@dataclass(frozen=True)
class PrefillCostModel:
    """Step time in milliseconds for `d` decode tokens and a prefill chunk of
    `q` tokens that attends over `c` tokens of earlier context::

        fixed_ms + ms_per_token * (d + q) + ms_per_token_context * q * (c + q / 2)

    Attributes:
        budget_ms: Time one step may take.
        fixed_ms: Step time independent of the token count.
        ms_per_token: Time one token adds to a step.
        ms_per_token_context: Time a prefill token adds per token of context.
        min_chunk_tokens: Fewest prefill tokens a step still carries when the
            decodes alone fill the budget, so a prefill always advances.
        min_attention_share: Share of the budget attention may still use when
            the token term alone prices a step above it, so the attention cap
            stays continuous in the fitted context cost.

    """

    budget_ms: float
    fixed_ms: float
    ms_per_token: float
    ms_per_token_context: float = 0.0
    min_chunk_tokens: int = 256
    min_attention_share: float = 0.1

    def __post_init__(self) -> None:
        if self.budget_ms <= 0:
            raise ValueError(f"The step time budget must be positive: {self}")
        if self.ms_per_token <= 0:
            raise ValueError(
                "A prefill step time budget needs VLLM_PREFILL_COST_MS_PER_TOKEN "
                f"> 0, fitted on the deployment: {self}"
            )
        if min(self.fixed_ms, self.ms_per_token_context) < 0:
            raise ValueError(f"The step cost coefficients must be >= 0: {self}")
        if self.min_chunk_tokens < 1:
            raise ValueError(f"The minimum chunk must be >= 1 token: {self}")
        if not 0 <= self.min_attention_share < 1:
            raise ValueError(f"The attention share must be in [0, 1): {self}")

    @classmethod
    def from_env(cls, budget_ms: float | None = None) -> "PrefillCostModel | None":
        """The model the environment configures, or None without a budget.

        Args:
            budget_ms: Step time budget to use instead of
                `VLLM_PREFILL_STEP_BUDGET_MS`, e.g. one derived from the TPOT
                target.

        """
        if budget_ms is None:
            budget_ms = envs.VLLM_PREFILL_STEP_BUDGET_MS
        if budget_ms <= 0:
            return None
        return cls(
            budget_ms=budget_ms,
            fixed_ms=envs.VLLM_PREFILL_COST_FIXED_MS,
            ms_per_token=envs.VLLM_PREFILL_COST_MS_PER_TOKEN,
            ms_per_token_context=envs.VLLM_PREFILL_COST_MS_PER_TOKEN_CONTEXT,
            min_chunk_tokens=envs.VLLM_PREFILL_MIN_CHUNK_TOKENS,
        )

    def with_coefficients(
        self, fixed_ms: float, ms_per_token: float, ms_per_token_context: float
    ) -> "PrefillCostModel":
        """The same budget with fitted coefficients."""
        return replace(
            self,
            fixed_ms=fixed_ms,
            ms_per_token=ms_per_token,
            ms_per_token_context=ms_per_token_context,
        )

    def step_ms(self, decode_tokens: int, prefill_tokens: int, context: int) -> float:
        """Modelled time of a step."""
        return (
            self.fixed_ms
            + self.ms_per_token * (decode_tokens + prefill_tokens)
            + self.ms_per_token_context
            * prefill_tokens
            * (context + prefill_tokens / 2)
        )

    def work_ms(self, prefill_tokens: int, context: int) -> float:
        """Modelled time of prefilling `prefill_tokens` over `context`.

        The per-step `fixed_ms` is left out: this prices work still owed by a
        request, which DP load balancing compares across engines however the
        engine batches it.
        """
        return self.ms_per_token * prefill_tokens + self.ms_per_token_context * (
            prefill_tokens * (context + prefill_tokens / 2)
        )

    def max_attention_tokens(self, room_ms: float, context: int, align: int = 1) -> int:
        """Largest prefill chunk whose attention over `context` fits `room_ms`.

        Only the context term is priced: the token term of a data-parallel step
        is shared by the ranks, so the caller accounts for it once.

        Args:
            room_ms: Time left for attention after the shared terms.
            context: Context the prefill chunk attends over.
            align: Round a chunk longer than this down to a multiple of it.

        Returns:
            The chunk length, at least `min_chunk_tokens`; `sys.maxsize` when the
            context costs nothing.

        """
        c = self.ms_per_token_context
        if c <= 0:
            return sys.maxsize
        # A token term fitted above the budget leaves no room for attention.
        # That fit contradicts the token target, which the measured step
        # periods chose; clamping the chunk to the floor there would cut it by
        # tens of times whenever the fit returns a meaningless positive cost
        # (~1e-7 ms/token/context), and steps that small carry too few tokens to
        # fill the target, so nothing could correct the target or the fit. Keep
        # a share of the budget for attention instead: the cap stays continuous
        # in the fitted cost, and an over-budget step informs both.
        room_ms = max(room_ms, self.min_attention_share * self.budget_ms)
        linear = c * context
        tokens = (2 * room_ms) / (linear + math.sqrt(linear * linear + 2 * c * room_ms))
        chunk = int(tokens)
        if align > 1 and chunk > align:
            chunk -= chunk % align
        return max(chunk, self.min_chunk_tokens)

    def max_prefill_tokens(
        self,
        decode_tokens: int,
        context: int,
        align: int = 1,
        budget_ms: float | None = None,
    ) -> int:
        """Largest prefill chunk whose step fits the budget.

        Args:
            decode_tokens: Decode tokens the step already carries.
            context: Context the prefill chunk attends over.
            align: Round a chunk longer than this down to a multiple of it.
            budget_ms: Step time to fit instead of `budget_ms`.

        Returns:
            The chunk length, at least `min_chunk_tokens`.

        """
        if budget_ms is None:
            budget_ms = self.budget_ms
        room_ms = budget_ms - self.fixed_ms - self.ms_per_token * decode_tokens
        if room_ms <= 0:
            return self.min_chunk_tokens
        # Solve ms_per_token_context / 2 * q^2 + (ms_per_token +
        # ms_per_token_context * context) * q = room_ms for q.
        linear = self.ms_per_token + self.ms_per_token_context * context
        # The positive root, written to avoid cancelling when the quadratic term
        # is small.
        tokens = (2 * room_ms) / (
            linear
            + math.sqrt(linear * linear + 2 * self.ms_per_token_context * room_ms)
        )
        chunk = int(tokens)
        if align > 1 and chunk > align:
            chunk -= chunk % align
        return max(chunk, self.min_chunk_tokens)
