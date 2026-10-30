# DeepSeek V4.1: PP=2 + attention DP=4 + EP=4 + DSpark on 8 SM120 GPUs

Target: 8x RTX PRO 6000 96GB (SM120, PCIe), FlashInfer sparse MLA
(`FLASHINFER_MLA_SPARSE_DSV41`), DSpark with 5 drafts per round.
This uses plain attention data parallelism. DCP, DBO and microbatching stay off.

## Layout

```
pipeline_parallel_size=2, tensor_parallel_size=1, data_parallel_size=4,
enable_expert_parallel=true, VLLM_PP_LAYER_PARTITION=20,20
```

vLLM orders ranks as `DP x PP x TP`, so global rank `r = 2 * dp + pp` runs on
GPU `r` on a single node:

| | GPUs (global ranks) | Layers | EP group |
|---|---|---|---|
| Stage 0 | 0, 2, 4, 6 | 0-19 (+ embedding, Engram layers 1 and 14) | {0, 2, 4, 6} |
| Stage 1 | 1, 3, 5, 7 | 20-39 (+ norm, LM head, DSpark) | {1, 3, 5, 7} |

The PP chains are (0,1), (2,3), (4,5) and (6,7), one per DP rank. Each
EP all-to-all stays inside one stage. Every worker logs a line like this at startup:

```
Parallel topology: global_rank=3 dp_rank=1 pp_rank=1 pp_ranks=[2, 3] tp_rank=0 ep_rank=1 ep_ranks=[1, 3, 5, 7] physical_gpu=3 device=cuda:3
```

The split at layer 20 matches the source-sharing boundary
(`kv_source_layer_ids=[2,8,14,20]`, candidate source 20). No KV, index or
candidate cache crosses the stage boundary, and `validate_local_sharing`
checks this when the model is built. The PP payload is `hidden_states [T, hc_mult, hidden]`
(BF16) plus `pre_mix [T, hc_mult]` (FP32).

DSpark target layers `[37, 38, 39]` are captured on stage 1. Aux id `L` is
the input of layer `L`, i.e. the output of layer `L-1`. Stage 1 runs DSpark
locally and nothing is relayed. The startup log confirms this:
`aux hidden states [37, 38, 39] are captured on PP stages [1, 1, 1]; relayed ... none`.

## Memory (rough, per GPU; measure on the real host)

| Stage | Routed experts (MXFP4, /4 by EP) + scales | Attention/dense (replicated per DP rank) | Embedding / head | DSpark |
|---|---|---|---|---|
| 0 | ~31 + ~3 GiB | ~3.5 GiB | embedding ~1.9 GiB | - |
| 1 | ~31 + ~3 GiB | ~3.5 GiB | head ~1.9 GiB + DSpark embedding ~1.9 GiB | draft layers |

The Engram tables (~183 GiB FP8 plus scales) stay in pinned host memory
(`--engram-config '{"cpu_offload":true}'`, the default). The four stage-0 DP replicas
share one copy through `/dev/shm`, so the host needs about 200 GiB free there.
Without EP each GPU would hold all of its stage's experts (~130 GiB), which does
not fit. The "PP=2, DP=4, EP=false" baseline therefore only works with a smaller
test checkpoint.

## SM120 path

What runs on SM120 for this configuration, and the settings it requires:

| Component | SM120 implementation | Requirement |
|---|---|---|
| Target and DSpark attention | `DeepseekV4FlashInferSM120Attention` (`FLASHINFER_MLA_SPARSE_DSV41`); the drafter inherits the target backend | `--kv-cache-dtype nvfp4_ds_mla`: FP8 MXFP8-record SWA cache plus NVFP4 compressed cache. `fp8` aliases to it. `fp8_ds_mla`, `auto` and `bfloat16` are rejected |
| FlashInfer decode dispatch | `SparseMLASm120Wrapper`, DSV4.1 table | entries for `(padded heads, SWA window)` for the target and `(padded heads, window + 5 padded to 64)` for the non-causal DSpark drafter. Layer construction raises without them |
| KV pages | 64 compressed states per kernel page | `--block-size 64`, the size the SM120 FP4 KV path was validated with |
| Indexer logits | DeepGEMM `fp8_fp4_(paged_)mqa_logits`. SM120 has no other backend; 5-token verify decode uses the flattened single-token path | DeepGEMM with sm_12x kernels. The indexer K cache stays FP8; `INDEXER_KV_DTYPE=mxfp4` is opt-in |
| Routed experts (MXFP4) | `DEEPGEMM_MXFP4` by default when DeepGEMM is installed, otherwise Marlin | `MOE_BACKEND=marlin` if the DeepGEMM FP4 MoE test fails on the host |
| Block-FP8 linears, mHC pre-norm GEMM | DeepGEMM, with a TileLang fallback for mHC | `VLLM_USE_DEEP_GEMM=0` disables DeepGEMM everywhere, including the indexer, which then has no backend. Do not set it here |
| EP all-to-all | `allgather_reducescatter` (NCCL) | works over PCIe. DeepEP, MoRI and the NVLink backends do not |
| CUDA graphs | `FULL_AND_PIECEWISE` (default); SM120 metadata builders report `ALWAYS` | none |

Not used on SM120: FlashMLA, FlashMLA mega attention, mega MoE and the mHC
stream overlap. PCP would force FlashMLA, so do not enable it.

The SM120 FP4 KV fix on the base branch (`a0e2c96938`, "address FlashInfer
sparse KV by cache row") was checked end to end with TP1 x DP4 + DSpark,
block size 64 and `FULL_AND_PIECEWISE`. This deployment keeps the same
attention, KV and DSpark settings and adds PP=2 and EP.

## Steps

1. **Phase 0, environment:**
   - `bash check_env.sh` checks for 8 SM120 GPUs, P2P within each PP pair and EP group,
     the FlashInfer build and the `/dev/shm` size, and prints `nvidia-smi topo -m`.
   - `python preflight.py /models/DeepSeek-V4.1-Flash` checks against the real checkpoint
     config. It confirms that the 20/20 split keeps every KV, index and candidate
     source local and that the DSpark target layers sit on stage 1. It also checks
     that FlashInfer ships the DSV4.1 SM120 dispatch entries for the target and
     DSpark shapes, and that the DeepGEMM symbols the SM120 path calls exist.
2. **Phase 0, SM120 kernel tests on one GPU:** these check that the kernels compute
   correct values. The preflight only checks that they exist.
   ```
   pytest tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py \
          tests/v1/attention/test_deepseek_v41_block_size.py \
          tests/v1/attention/test_dspark_noncausal_sparse_mla.py \
          tests/kernels/test_compressor_kv_cache.py \
          tests/kernels/attention/test_deepgemm_attention.py \
          tests/kernels/test_fused_indexer_q_rope_quant.py \
          "tests/kernels/moe/test_deepgemm.py::test_deepgemm_fp4_vs_triton"
   ```
   If only the last test fails, serve with `MOE_BACKEND=marlin`.
3. **Phase 1, serve:** `MODEL=/models/DeepSeek-V4.1-Flash bash serve.sh`.
   In the log, check:
   - the `Parallel topology` lines;
   - `aux hidden states [37, 38, 39] are captured on PP stages [1, 1, 1]`;
   - `Using DeepSeek's nvfp4_ds_mla KV cache format`;
   - the MoE backend line.
4. **Phase 1, tests:**
   `DSV41_MODEL=/models/DeepSeek-V4.1-Flash pytest tests/v1/distributed/test_dsv41_pp2_dp4_ep4_dspark.py`
   - Checks the rank and group mapping.
   - Runs single and concurrent requests with async scheduling on and off.
   - Checks DSpark against target-only greedy decoding by token match
     (`DSV41_MIN_TOKEN_MATCH_RATIO`, default 0.9).
   - Checks answers that catch SM120 KV addressing errors, which full-length
     outputs alone would hide: `17 * 23 = 391`, plus needle recall in a short
     prompt and in a ~20k-token prompt that spans several prefill chunks.
5. **Phase 2, measure:** `MODEL=... bash bench_matrix.sh` (see below).

## Phase 2: measurement

`bench_matrix.sh` runs these configurations, the plan's C, D and E:

| Name | DSpark | `PP_SEND_BUFFERS` | `PP_MICROBATCH_BALANCE` |
|---|---|---|---|
| `C_ep1_dspark0` | off | 0 | 0 |
| `D_ep1_dspark1` | on | 0 | 0 |
| `E1_dspark1_sendbuf2` | on | 2 | 0 |
| `E2_dspark1_balance` | on | 0 | 1 |
| `E3_dspark1_sendbuf2_balance` | on | 2 | 1 |

For each configuration, `vllm bench serve` runs at concurrency 1/8/32/64 with
2k-token inputs, plus a 32k-token prefill run. Use `ONLY="D_ep1_dspark1 ..."`
to run a subset. At the end, `summarize_results.py` prints:
- a table of TTFT, p99 TTFT, TPOT, p99 TPOT, output and total tok/s, and
  DSpark acceptance rate and length;
- the last `PP step profile` line of every rank.

**Concurrency sweep.** `CONCURRENCIES` is the total concurrency across the 4 DP
ranks. The default is `1 8 16 32 64 128 256 512 1024`. The matrix raises the
server limits so that admission does not cap the sweep first:
`MAX_NUM_SEQS=256` per rank (1024 total) and `MAX_CUDAGRAPH_CAPTURE_SIZE=1024`.
With `TPOT_SLO_MS` (default 50), the summary reports per configuration the
highest concurrency whose mean TPOT fits, plus the peak output throughput and
the concurrency where it occurs.

**Step profile** (`VLLM_PP_STEP_PROFILE_INTERVAL=N`, on by default in the
matrix). Every N steps each worker logs CUDA-event averages:
`step` (GPU cadence), `execute` and `busy` (execute / step),
`recv_wait`, `send_wait`, `sample` and `draft`.
- A low `busy` on stage 1 with a high `recv_wait` means stage 0 starves it.
- A high `send_wait` on stage 0 means stage 0 stalls on the 19 -> 20 send.
- `draft` is the DSpark proposer on stage 1.

The host never synchronizes; events are read once they complete.

**Kernel breakdown.** With `PROFILE_DIR=/tmp/traces` the matrix also captures
torch profiler traces, and `analyze_traces.py` splits each rank's GPU time
into these categories:
- PP P2P (NCCL SendRecv)
- EP all-to-all (NCCL AllGather/ReduceScatter of `allgather_reducescatter`)
- token/draft relay (NCCL Broadcast)
- indexer, attention, MoE, mHC, GEMM

It also reports the GPU busy fraction. This is how to get the EP all-to-all
and P2P shares: under FULL CUDA graphs, Python-level hooks around them never run.

## Maximum concurrency

`max_num_seqs` is per DP engine, so total concurrency is 4 × the per-rank
value. Per DP rank, the limits are, in the order they usually bind:

| Limit | Per DP rank | Source |
|---|---|---|
| TPOT target | measured: the knee of the sweep | each request decodes every 2 steps, so TPOT ≈ 2 × step time |
| SM120 decode plan | about 20 requests (10 per microbatch × 6 tokens = 64 rows) | `_FLASHINFER_DSV4_DECODE_MAX_TOKENS = 64`; wider steps use FlashInfer's prefill plan. A possible step change to look for in the sweep |
| CUDA graph coverage | about 170 requests with the default 512-token ceiling | `min(max_num_seqs × 6 × 2, 512)` on non-datacenter Blackwell; larger decode steps run eager. Raise with `MAX_CUDAGRAPH_CAPTURE_SIZE` |
| `MAX_NUM_SEQS` | 64 in `serve.sh`, 256 in the matrix | configuration |
| KV cache | ~4000 at 8k context, ~350 at 128k, ~45 at 1M (estimate) | stage 0 holds ~630 B/token of compressed KV + indexer K (FP8) and ~2 MB of SWA per request; stage 1 ~420 B/token + ~2.3 MB |
| Token budget | ~2700 (8192 × 2 microbatches / 6 tokens) | `max_num_batched_tokens` |

The KV numbers assume ~30 GiB of KV per GPU; the startup log's
`GPU KV cache size` and `Maximum concurrency for N tokens per request` lines
are authoritative. Set `MAX_NUM_SEQS` to the sweep's TPOT-SLO concurrency
divided by 4. If that lands above ~170 per rank, also raise
`MAX_CUDAGRAPH_CAPTURE_SIZE` to cover `requests per microbatch × 6` tokens.

## Phase 3: PP send buffers (`VLLM_PP_SEND_BUFFERS=N`, `PP_SEND_BUFFERS` in serve.sh)

By default, stage 0 waits at the top of every step for its previous send of
`hidden_states` + `pre_mix`. Its outputs are CUDA-graph static buffers, so the
next replay would otherwise overwrite data that is still being sent. With N ≥ 1:
- the worker (`vllm/v1/worker/gpu/pp_send_ring.py`) copies each step's payload
  into one of N buffers, preallocated at `max_num_batched_tokens` rows before
  memory profiling, so the KV cache budget accounts for them;
- each buffer is sent from its own copy, and only reusing a buffer waits for
  the send that last used it.

Stage 0 can then run up to N steps ahead of stage 1's receives. The waits are
`handle.wait()` on the NCCL work, which holds the GPU stream and never the CPU.
The ring is skipped when experimental pipeline-sharing payloads are on.

## Phase 4: multi-microbatch PP overlap

The step-level pipeline was already in place, and this deployment uses it as
is:
- With the V2 runner and async scheduling, the engine keeps `pp_size + 1 = 3`
  batches in flight (`max_concurrent_batches`, `EngineCore.step_with_batch_queue`).
- The scheduler lets each request decode only every `pp_size` steps
  (`AsyncScheduler.decode_stagger`), so consecutive steps carry disjoint
  request sets: microbatch A and microbatch B.
- Stage 0 runs B while stage 1 runs A.
- Sampled tokens and DSpark drafts return to stage 0 through `PPHandler`'s
  ring on a separate stream and communicator, `pp_size` steps later.
- KV slots, DP token sync and EP all-to-all are per step, in the same step
  order on every rank.

What was missing is that the two microbatches stay as unequal as they started.
A burst of requests that finish prefill in the same step all land in one
microbatch, so every other step reaches the stages nearly empty and each stage
idles about half the time. `VLLM_PP_MICROBATCH_BALANCE=1`
(`PP_MICROBATCH_BALANCE` in serve.sh) fixes this:
- after each schedule, it moves the lowest-priority decodes just scheduled into
  the smaller microbatch, until the sizes differ by at most one;
- a moved request waits one extra step once;
- no request decodes sooner than `pp_size` steps after its last decode, which
  is the gap the token ring requires.

DBO is a different mechanism: it splits one step into two ubatches to overlap
the EP all-to-all on one rank. It stays rejected with PP and with
speculative decoding (`vllm/config/vllm.py`) and is not used here.

Leave both options off for the first deployment, then compare D against
E1, E2 and E3. `test_pp_send_buffers_and_microbatch_balance_keep_outputs`
checks that enabling them keeps the greedy outputs.

## Code changes behind this deployment

- `vllm/models/deepseek_v41/nvidia/model.py`: V4.1 now declares
  `supports_aux_hidden_states_over_pp`. Without it the V2 runner refuses
  DSpark with PP. Non-last stages send their aux states in the PP payload,
  and the last stage prepends them. Decoder SWA bounded replay is turned off on a stage
  that starts exactly at the last KV source layer (layer 20 here), because it
  needs mHC state from the previous stage.
- `vllm/models/deepseek_v41/nvidia/dspark.py`: past the first PP stage the
  target has no input embedding to share, so DSpark loads `embed.weight` itself.
- `vllm/v1/worker/gpu_worker.py`: per-worker `Parallel topology` log and a
  `parallel_topology()` method reachable through `collective_rpc`.
- `vllm/models/deepseek_v41/common/pipeline.py`:
  `get_aux_hidden_state_stages()` maps each aux id to the stage that captures it.
- `vllm/v1/worker/gpu/pp_step_profiler.py`: the phase 2 step profiler. The
  worker and the V2 runner record spans for receive, execute, send, sample and
  draft.
- `vllm/v1/worker/gpu/pp_send_ring.py` and `Worker.execute_model`: the phase 3
  send buffers.
- `vllm/v1/core/sched/async_scheduler.py`: the phase 4 microbatch balancing.
- `vllm/envs.py`: `VLLM_PP_STEP_PROFILE_INTERVAL`, `VLLM_PP_SEND_BUFFERS` and
  `VLLM_PP_MICROBATCH_BALANCE`. All default to off and are left out of the
  compile cache key.

CPU unit tests (no GPU needed):
`tests/v1/worker/test_pp_send_ring.py`, `tests/v1/worker/test_pp_step_profiler.py`,
`tests/v1/worker/test_gpu_worker.py -k "pp_send or send_ring"`,
`tests/v1/core/test_async_scheduler.py -k microbatch`,
`tests/models/test_deepseek_v41_pipeline.py`.
