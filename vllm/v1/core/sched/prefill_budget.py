# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-budgeted prefill chunking.

A step on a data-parallel MoE deployment ends when its slowest rank does, so a
prefill chunk sized by tokens alone makes steps of very different length.
`PrefillCostModel` prices a step and solves for the largest prefill chunk that
keeps it within a time budget.
"""

import math
from dataclasses import dataclass

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

    """

    budget_ms: float
    fixed_ms: float
    ms_per_token: float
    ms_per_token_context: float = 0.0
    min_chunk_tokens: int = 256

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

    @classmethod
    def from_env(cls) -> "PrefillCostModel | None":
        """The model the environment configures, or None without a budget."""
        if envs.VLLM_PREFILL_STEP_BUDGET_MS <= 0:
            return None
        return cls(
            budget_ms=envs.VLLM_PREFILL_STEP_BUDGET_MS,
            fixed_ms=envs.VLLM_PREFILL_COST_FIXED_MS,
            ms_per_token=envs.VLLM_PREFILL_COST_MS_PER_TOKEN,
            ms_per_token_context=envs.VLLM_PREFILL_COST_MS_PER_TOKEN_CONTEXT,
            min_chunk_tokens=envs.VLLM_PREFILL_MIN_CHUNK_TOKENS,
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

    def max_prefill_tokens(
        self, decode_tokens: int, context: int, align: int = 1
    ) -> int:
        """Largest prefill chunk whose step fits the budget.

        Args:
            decode_tokens: Decode tokens the step already carries.
            context: Context the prefill chunk attends over.
            align: Round a chunk longer than this down to a multiple of it.

        Returns:
            The chunk length, at least `min_chunk_tokens`.

        """
        room_ms = self.budget_ms - self.fixed_ms - self.ms_per_token * decode_tokens
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
