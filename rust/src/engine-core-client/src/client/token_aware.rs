// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Token-aware DP routing (`VLLM_DP_LB_TOKEN_AWARE`), ported from the Python
//! frontend's `DPLBAsyncMPClient.get_core_engine_for_request`.
//!
//! `VLLM_DP_LB_SHARED_LOAD` has no counterpart: it shares load between the
//! API server processes of one deployment, and the Rust frontend always runs
//! as a single API server process (`api_server_count` is forced to 1), the
//! case in which Python creates no shared slots either.

use tracing::warn;

/// Token-aware DP routing weighs an engine's unfinished prefill against its
/// request count. With a prefill step time budget configured, the engine
/// reports that backlog in modelled milliseconds and it is measured in steps
/// of that budget (`VLLM_PREFILL_STEP_BUDGET_MS`): a request that still has to
/// attend over a long context costs what it costs, not what it counts.
/// Without a budget, `VLLM_DP_LB_PREFILL_TOKENS_PER_REQUEST` tokens weigh as
/// one request. The weight is capped at this many requests: an engine already
/// prefilling several prompts' worth of tokens is simply busy, and scaling the
/// score with its whole backlog keeps it unroutable for the length of that
/// prefill, so every other engine piles up instead.
const MAX_PENDING_REQUESTS: f64 = 4.0;

/// A request this client dispatched floors its engine's pending prefill until
/// this many scheduler snapshots from that engine have arrived after it. The
/// first snapshot after a dispatch may have been computed before the engine
/// received the request, the second cannot have been.
pub(crate) const FLOOR_SNAPSHOTS: u64 = 2;

/// Snapshots count towards [`FLOOR_SNAPSHOTS`] at most this often per engine.
///
/// The Python frontend counts the coordinator's publications, which come at
/// most every 100 ms (`min_stats_update_interval_ms`). The Rust frontend sees
/// each engine's snapshots directly, and a busy engine sends one or two per
/// step whether or not a request still in flight to it has arrived, so
/// counting all of them would drop a floor within one step.
pub(crate) const SNAPSHOT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Prices an engine's unfinished prefill as a share of one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TokenAwareRouting {
    /// Pending prefill tokens that weigh as much as one request.
    prefill_tokens_per_request: f64,
    /// Step period the engines size prefill chunks for, in milliseconds; 0
    /// prices tokens instead of time.
    step_budget_ms: f64,
    /// Modelled cost of one prefill token, used to price a dispatch before its
    /// engine reports its own.
    default_ms_per_token: f64,
}

impl TokenAwareRouting {
    /// The routing the environment configures, or `None` when
    /// `VLLM_DP_LB_TOKEN_AWARE` is off.
    pub(crate) fn from_env() -> Option<Self> {
        Self::from_vars(|name| std::env::var(name).ok())
    }

    /// Like [`Self::from_env`], reading variables through `var`.
    pub(crate) fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let number = |name: &str, default: f64| -> f64 {
            let Some(value) = var(name) else {
                return default;
            };
            value.trim().parse().unwrap_or_else(|_| {
                warn!(name, value, "ignoring an unparsable value; using {default}");
                default
            })
        };
        if number("VLLM_DP_LB_TOKEN_AWARE", 0.0) == 0.0 {
            return None;
        }
        // A TPOT target takes the budget's place in the engines' chunk sizing.
        let tpot_ms = if number("VLLM_PREFILL_FILL_TARGET", 0.0) != 0.0 {
            number("VLLM_PREFILL_TPOT_TARGET_MS", 0.0)
        } else {
            0.0
        };
        let budget_ms = if tpot_ms != 0.0 {
            tpot_ms
        } else {
            number("VLLM_PREFILL_STEP_BUDGET_MS", 0.0)
        };
        Some(Self {
            prefill_tokens_per_request: number("VLLM_DP_LB_PREFILL_TOKENS_PER_REQUEST", 2048.0)
                .max(1.0),
            step_budget_ms: budget_ms.max(0.0),
            default_ms_per_token: number("VLLM_PREFILL_COST_MS_PER_TOKEN", 0.0),
        })
    }

    /// Requests' worth of load that `tokens` of unfinished prefill, modelled
    /// at `ms`, add to an engine's score.
    pub(crate) fn pending_load(&self, tokens: u64, ms: f64) -> f64 {
        let load = if self.step_budget_ms > 0.0 && ms > 0.0 {
            ms / self.step_budget_ms
        } else {
            tokens as f64 / self.prefill_tokens_per_request
        };
        load.min(MAX_PENDING_REQUESTS)
    }

    /// Modelled time of a dispatch of `tokens` prompt tokens to an engine
    /// that last reported `engine_ms_per_token` (0 if never).
    pub(crate) fn dispatch_ms(&self, tokens: u64, engine_ms_per_token: f64) -> f64 {
        let ms_per_token = if engine_ms_per_token > 0.0 {
            engine_ms_per_token
        } else {
            self.default_ms_per_token
        };
        tokens as f64 * ms_per_token
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::TokenAwareRouting;

    fn routing(vars: &[(&str, &str)]) -> Option<TokenAwareRouting> {
        let vars: HashMap<_, _> = vars.iter().copied().collect();
        TokenAwareRouting::from_vars(|name| vars.get(name).map(|v| (*v).to_string()))
    }

    #[test]
    fn off_unless_enabled() {
        assert_eq!(routing(&[]), None);
        assert_eq!(routing(&[("VLLM_DP_LB_TOKEN_AWARE", "0")]), None);
    }

    #[test]
    fn budget_follows_the_tpot_target_only_with_the_fill_target() {
        let base = [
            ("VLLM_DP_LB_TOKEN_AWARE", "1"),
            ("VLLM_PREFILL_STEP_BUDGET_MS", "700"),
            ("VLLM_PREFILL_TPOT_TARGET_MS", "200"),
        ];
        let without_fill = routing(&base).unwrap();
        let mut with_fill = base.to_vec();
        with_fill.push(("VLLM_PREFILL_FILL_TARGET", "1"));
        let with_fill = routing(&with_fill).unwrap();
        expect_test::expect!["(700.0, 200.0)"].assert_eq(&format!(
            "{:?}",
            (without_fill.step_budget_ms, with_fill.step_budget_ms)
        ));
    }

    #[test]
    fn pending_load_prices_time_when_budgeted_and_caps_at_four_requests() {
        let tokens = routing(&[("VLLM_DP_LB_TOKEN_AWARE", "1")]).unwrap();
        let timed = routing(&[
            ("VLLM_DP_LB_TOKEN_AWARE", "1"),
            ("VLLM_PREFILL_STEP_BUDGET_MS", "500"),
        ])
        .unwrap();
        expect_test::expect!["[1.0, 4.0, 0.5, 0.5, 4.0]"].assert_eq(&format!(
            "{:?}",
            [
                tokens.pending_load(2048, 0.0),
                tokens.pending_load(100_000, 0.0),
                timed.pending_load(4096, 250.0),
                // An engine that prices nothing falls back to tokens.
                timed.pending_load(1024, 0.0),
                timed.pending_load(0, 10_000.0),
            ]
        ));
    }

    #[test]
    fn dispatch_prefers_the_engines_own_price() {
        let routing = routing(&[
            ("VLLM_DP_LB_TOKEN_AWARE", "1"),
            ("VLLM_PREFILL_COST_MS_PER_TOKEN", "0.02"),
        ])
        .unwrap();
        expect_test::expect!["(20.0, 50.0)"].assert_eq(&format!(
            "{:?}",
            (
                routing.dispatch_ms(1000, 0.0),
                routing.dispatch_ms(1000, 0.05)
            )
        ));
    }
}
