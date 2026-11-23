# Data Parallel Deployment

vLLM supports Data Parallel deployment, where model weights are replicated across separate instances/GPUs to process independent batches of requests.

This will work with both dense and MoE models.

For MoE models, particularly those like DeepSeek that employ MLA (Multi-head Latent Attention), it can be advantageous to use data parallel for the attention layers and expert or tensor parallel (EP or TP) for the expert layers.

In these cases, the data parallel ranks are not completely independent. Forward passes must be aligned, and expert layers across all ranks are required to synchronize during every forward pass, even when there are fewer requests to be processed than DP ranks.

By default, expert layers form a tensor parallel group of size `DP × TP`. To use expert parallelism instead, include the `--enable-expert-parallel` CLI arg (on all nodes in the multi-node case). See [Expert Parallel Deployment](expert_parallel_deployment.md) for details on how attention and expert layers behave differently with EP enabled.

In vLLM, each DP rank is deployed as a separate "core engine" process that communicates with front-end process(es) via ZMQ sockets. Data Parallel attention can be combined with Tensor Parallel attention, in which case each DP engine owns a number of per-GPU worker processes equal to the configured TP size.

For MoE models, when any requests are in progress in any rank, we must ensure that empty "dummy" forward passes are performed in all ranks that don't currently have any requests scheduled. This is handled via a separate DP Coordinator process that communicates with all ranks, and a collective operation performed every N steps to determine when all ranks become idle and can be paused. When TP is used in conjunction with DP, expert layers form a group of size `DP × TP` (using either tensor parallelism by default, or expert parallelism if `--enable-expert-parallel` is set).

In all cases, it is beneficial to load-balance requests between DP ranks. For online deployments, this balancing can be optimized by taking into account the state of each DP engine - in particular its currently scheduled and waiting (queued) requests, and KV cache state. Each DP engine has an independent KV cache, and the benefit of prefix caching can be maximized by directing prompts intelligently.

This document focuses on online deployments (with the API server). DP + EP is also supported for offline usage (via the LLM class), for an example see [examples/features/data_parallel/data_parallel_offline.py](../../examples/features/data_parallel/data_parallel_offline.py).

There are two distinct modes supported for online deployments - self-contained with internal load balancing, or externally per-rank process deployment and load balancing.

## Internal Load Balancing

vLLM supports "self-contained" data parallel deployments that expose a single API endpoint.

It can be configured by simply including e.g. `--data-parallel-size=4` in the vllm serve command line arguments. This will require 4 GPUs. It can be combined with tensor parallel, for example `--data-parallel-size=4 --tensor-parallel-size=2`, which would require 8 GPUs. When sizing DP deployments, remember that `--max-num-seqs` applies per DP rank.

By contrast, the admission control limits `--max-num-queued-reqs` and `--max-num-queued-tokens` apply to the whole server, not per DP rank: each API server process counts in-flight requests and prefill backlog across all DP ranks it routes to. For example, with `--data-parallel-size=4 --max-num-seqs=256 --max-num-queued-reqs=256`, the deployment rejects new requests once 256 are in-flight in total, even though ranks could jointly run 1024. To keep ranks saturating, size the cap as roughly `data-parallel-size * max-num-seqs` plus the desired queue depth.

Running a single data parallel deployment across multiple nodes requires a different `vllm serve` to be run on each node, specifying which DP ranks should run on that node. In this case, there will still be a single HTTP entrypoint - the API server(s) will run only on one node, but it doesn't necessarily need to be co-located with the DP ranks.

This will run DP=4, TP=2 on a single 8-GPU node:

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

This will run DP=4 with DP ranks 0 and 1 on the head node and ranks 2 and 3 on the second node:

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

This will run DP=4 with only the API server on the first node and all engines on the second node:

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

This DP mode can also be used with Ray by specifying `--data-parallel-backend=ray`:

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

There are several notable differences when using Ray:

- A single launch command (on any node) is needed to start all local and remote DP ranks, therefore it is more convenient compared to launching on each node
- There is no need to specify `--data-parallel-address`, and the node where the command is run is used as `--data-parallel-address`
- There is no need to specify `--data-parallel-rpc-port`
- When a single DP group requires multiple nodes, *e.g.* in case a single model replica needs to run on at least two nodes, make sure to set `VLLM_RAY_DP_PACK_STRATEGY="span"` in which case `--data-parallel-size-local` is ignored and will be automatically determined
- Remote DP ranks will be allocated based on node resources of the Ray cluster

Currently, the internal DP load balancing is done within the API server process(es) and is based on the running and waiting queues in each of the engines. This could be made more sophisticated in future by incorporating KV cache aware logic.

When deploying large DP sizes using this method, the API server process can become a bottleneck. In this case, the orthogonal `--api-server-count` command line option can be used to scale this out (for example `--api-server-count=4`). This is transparent to users - a single HTTP endpoint / port is still exposed. Note that this API server scale-out is "internal" and still confined to the "head" node.

<figure markdown="1">
![DP Internal LB Diagram](../assets/deployment/dp_internal_lb.png)
</figure>

### Routing by pending prefill tokens

By default the API server sends a new request to the DP rank with the lowest load, counted in waiting and running requests. A long prompt weighs the same as a short one, although it keeps its rank busy for several steps, and every step of an MoE deployment waits for the slowest rank.

Set `VLLM_DP_LB_TOKEN_AWARE=1` to add the prompt tokens a rank has not finished prefilling to its score. Each engine reports its own remainder with the load snapshot it publishes every 100 ms, and until the second snapshot after a dispatch the API server floors it with the prefill it has just dispatched itself (its own requests' sizes are known before a snapshot has seen them; afterwards the engine's remainder decides), and that remainder already excludes the tokens a prefix cache or KV connector supplied, so a cached prompt weighs only what it still has to compute (a request that has not been scheduled yet counts in full). When a step time budget is configured (`VLLM_PREFILL_STEP_BUDGET_MS`), each engine reports the time its backlog takes to drain: the steps its chunks need (a request on a long context gets shorter chunks, so it needs more), each taking what a step with that chunk takes (with a token target, the period the steps that filled it measured; otherwise the cost model's estimate). The router measures it in steps of the budget, and prices a request it has just sent with the same time per token its engine reports, so the floor and the report compare. What is left of a long-context prompt weighs by its cost rather than its token count, and a fitted per-token coefficient that the data does not determine cannot reprice it. Otherwise `VLLM_DP_LB_PREFILL_TOKENS_PER_REQUEST` (default 2048) sets how many pending tokens weigh as much as one request. One rank's pending prefill counts for at most four requests, so a single long prompt cannot leave that rank unroutable while its prefill runs. It is off by default, and the default weight has not been tuned.

### Sharing load between API servers

With internal load balancing `--api-server-count` defaults to the data parallel size, and each API server routes alone: it knows its own dispatches exactly, guesses the other servers' by scaling its own by the server count, and otherwise relies on the engines' snapshot, which is up to 200 ms old. A burst that reaches the servers unevenly is then routed from a skewed view.

Set `VLLM_DP_LB_SHARED_LOAD=1` to let the API servers on one host publish their per-engine in-flight requests and pending prefill into shared memory. Each server sums the other servers' figures into its own, so it sees every dispatch as soon as it is made, and nothing is scaled by the server count. A server that stops heartbeating for 2 s (a crashed one) is ignored. It is off by default, does nothing with one API server, and cannot share across hosts, where the engines' snapshot remains the only common view.

### Sizing prefill chunks by a step time budget

The scheduler caps the prefill tokens of a step with `--max-num-batched-tokens`, whatever else the step carries. In a DP MoE deployment the chunks of a long prompt then make steps of very different length, and the decodes of every rank wait for them.

Set `VLLM_PREFILL_STEP_BUDGET_MS` to a positive number of milliseconds to cap the prefill tokens of each step so that a cost model keeps the step within it:

```text
fixed + per_token * (decode + prefill) + per_token_context * prefill * (context + prefill / 2)
```

with the coefficients `VLLM_PREFILL_COST_FIXED_MS`, `VLLM_PREFILL_COST_MS_PER_TOKEN` (required) and `VLLM_PREFILL_COST_MS_PER_TOKEN_CONTEXT` (default 0, a cost independent of the context). They have no defaults: fit them on the deployment from step times at several token counts. A step still carries at least `VLLM_PREFILL_MIN_CHUNK_TOKENS` (default 256) prefill tokens so a prefill advances under a heavy decode load, and chunks are rounded down to a block multiple. Ranks agree on the group's smallest chunk (re-agreed with the DP state sync, every `VLLM_DP_SYNC_INTERVAL` steps), so a step does not wait on one rank's longer prefill while the others idle; each rank still prices its own decodes and context. The budget combines with `--max-num-batched-tokens` and the other prefill caps by taking the tightest. It is off by default.

Set `VLLM_PREFILL_FILL_TARGET=1` (with the budget above) to size the chunk differently. The token-priced part of a data parallel MoE step is shared: a step ends when its largest rank does and the smaller ranks pad up to it. So each rank fills its step to a shared token target: it takes `target - its decode tokens` of prefill, a rank with few decodes takes a long chunk, a rank with many a short one, and the steps end together. The attention over the context is each rank's own, so a chunk is also capped at what its attention costs in the budget the target's tokens leave (`budget - fixed - per_token * target`, solved with `per_token_context` for this rank's context). The per-token term can make that budget negative when the fitted coefficients price the target above it; a tenth of the budget is then still kept for attention, so a fitted `per_token_context` that is merely a tiny positive number (a fit from idle steps returns ~1e-7) cannot cut the chunk towards `VLLM_PREFILL_MIN_CHUNK_TOKENS`, and a step that does overshoot carries enough tokens to fill the target, which is what moves it back. A rank with a long context takes a shorter chunk without holding the others to it, and the ranks do not agree on a minimum (the agreement above is skipped). `VLLM_PREFILL_STEP_BUDGET_MS` is also the step period the target steers toward: a background thread, every `VLLM_PREFILL_TARGET_UPDATE_MS` (default 200), takes the median period of the step outputs that carried prefill and filled the target and solves for the target that brings it to the budget (or to `VLLM_PREFILL_TARGET_DECODE_MS`, default 0 for the budget, while the recent steps carry decode tokens, which wait for the whole step) after the fixed cost of a step, so a step cost that is linear in the tokens lands in one move (the fixed cost it uses is bounded by half the measured period and half the budget, so a fit that reports a fixed cost as large as the period it was fitted from cannot invert the solve); one update changes the target at most four times either way, a median within 5% of the budget leaves it unchanged, and it is never above `--max-num-batched-tokens`. The target lands on the token counts of the captured prefill CUDA graphs (`prefill_cudagraph_capture_token_sizes`), so a step that fills it replays a captured graph instead of running eager above the largest one or padding far past its token count; a capture cap below `--max-num-batched-tokens` therefore also caps the target, and when no prefill graphs are captured the target moves on a grid of `--max-num-batched-tokens / 64` tokens (128 by default) instead. Every move is logged. The scheduler only reads the current value, so the update is off the scheduling path. The target starts at `VLLM_PREFILL_TARGET_INIT_TOKENS` (default a quarter of `--max-num-batched-tokens`); the cost coefficients also price a rank's pending prefill for routing. A rank without enough pending prefill cannot fill its room, which token-aware routing (above) addresses. It is off by default.

Set `VLLM_PREFILL_COST_FIT=1` to fit the coefficients to the served steps instead of trusting the configured ones. A step's period is its slowest rank's, so no rank can explain it from its own figures: every `VLLM_PREFILL_COST_FIT_INTERVAL` steps (default 64) the ranks exchange each step's token count, attention work (`sum(q * (context + q / 2))` over the prefill chunks) and period over an asynchronous CPU collective, issued at one interval and completed at the next, off the step loop. Every rank merges the same steps (the largest tokens and attention work of any rank, the median period), fits `fixed`, `per_token` and `per_token_context` by ridge regression toward the configured values, and installs the result at the same step. Recent steps weigh more, a term the served steps do not vary stays at its configured value, and a fitted value is only kept valid (`fixed` and `per_token_context` not below 0, `per_token` above 0), with no other limit, so it can end far from the configured one; a term configured as 0 can be fitted too. Only steps that carried prefill fit the coefficients: a window of decode-only steps would fit the prefill terms to the decode step's own overhead, so it keeps the configured ones. With this on, each fit round also moves the target from the merged steps of the group, so the ranks re-agree on one value; between rounds each rank moves its own target from its own steps, which the lockstep makes the group's. The fit counts the tokens the scheduler schedules, before adaptive verification trims drafts, as the chunk sizing does, so the two stay consistent. It is off by default.

## Hybrid Load Balancing

Hybrid load balancing sits between the internal and external approaches. Each node runs its own API server(s) that only queue requests to the data-parallel engines colocated on that node. An upstream load balancer (for example, an ingress controller or traffic router) spreads user requests across those per-node endpoints.

Enable this mode with `--data-parallel-hybrid-lb` while still launching every node with the global data-parallel size. The key differences from internal load balancing are:

- You must provide `--data-parallel-size-local` and `--data-parallel-start-rank` so each node knows which ranks it owns.
- Not compatible with `--headless` since every node exposes an API endpoint.
- Scale `--api-server-count` per node based on the number of local ranks

In this configuration, each node keeps scheduling decisions local, which reduces cross-node traffic and avoids single node bottlenecks at larger DP sizes.

## External Load Balancing

For larger scale deployments especially, it can make sense to handle the orchestration and load balancing of data parallel ranks externally.

In this case, it's more convenient to treat each DP rank like a separate vLLM deployment, with its own endpoint, and have an external router balance HTTP requests between them, making use of appropriate real-time telemetry from each server for routing decisions.

This can already be done trivially for non-MoE models, since each deployed server is fully independent. In that case, launch independent vLLM instances without any `--data-parallel-*` arguments; external DP CLI options are only supported for MoE deployments.

We support an equivalent topology for MoE DP+EP which can be configured via the following CLI arguments.

If DP ranks are co-located (same node / ip address), a default RPC port is used, but a different HTTP server port must be specified for each rank:

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

For multi-node cases, the address/port of rank 0 must also be specified:

```bash
# Rank 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

When using `--data-parallel-external-lb` with `--nnodes`, the data parallel rank
can be inferred from `--node-rank` only when `--nnodes` is evenly divisible by
`--data-parallel-size`. If it is not evenly divisible, automatic rank inference
is rejected and `--data-parallel-rank` must be set explicitly. When multiple
external-LB ranks are co-located on a node, assign each process a disjoint
device set with `CUDA_VISIBLE_DEVICES` or `--device-ids`; devices are not
partitioned automatically between processes.

The coordinator process also runs in this scenario, co-located with the DP rank 0 engine.

<figure markdown="1">
![DP External LB Diagram](../assets/deployment/dp_external_lb.png)
</figure>

In the above diagram, each of the dotted boxes corresponds to a separate launch of `vllm serve` - these could be separate Kubernetes pods, for example.
