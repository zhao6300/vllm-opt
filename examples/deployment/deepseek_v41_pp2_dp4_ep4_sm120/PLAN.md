# DeepSeek-V4.1 SM120 优化计划（DeepInfer 对照）

**状态：** 草案，2026-11-11
**基线分支：** `port/dsv41-pp2-dp4-ep4-sm120-dspark-onto-20261108-main`
**基线部署：** `examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/README.md`
（PP2 × TP1 × DP4 × EP4，SM120 / PCIe，FlashInfer sparse MLA，DSpark 5，NVFP4 KV）

**对照对象：** `/workspace/deepinfer`，commit `a679bb8`（V4.1-Flash 开源版，
SM100 + Ascend 950，offline inference）。开源范围不含 KV cache 系统、
P/D 分离与负载均衡；这部分只参考 `doc/inference_deployment_qa.md` 的生产设计。

**使用方式：** 本文件是执行清单，不替代 README。每项任务独立可测，
先测量再改代码；完成后勾选并在此文件末尾的“验证记录”里填数字。

---

## 1. 结论

两边同源（DeepInfer 也是 vLLM 衍生）。本分支在 SM120、PP、FP4 KV、
DSpark 特殊解码上更深；DeepInfer 领先的是引擎工程：

1. 通信窗口内的 shared expert / combine 重叠与 SM 预算；
2. 动态投机验证的**在线**耗时估计与滞后一轮的置信度调度；
3. 单 kernel 融合采样与单 kernel decode 元数据；
4. SWA state 按 128 粒度持久化，前缀命中不再重放；
5. 端到端 LB（分层 prefill + 双维度 decode）与故障恢复；
6. 主机 / 3FS 分层 KV 与 Engram RDMA。

可移植到 SM120 + PCIe 8 卡的主要是 1–4 和部分 5/6，**不是** SM100 kernel。

---

## 2. 评测与判定口径

所有任务都必须给出同一口径的前后对比，否则不算完成。

**必测指标（decode）：**

- `avg_SPS`、`max_step`、decode tok/s/GPU、TPOT p50/p99；
- 接受长度（按 full-acceptance 记账：`num_reqs × (1 + num_speculative_tokens)`）；
- 每步 launch 数 / CPU 时间（`VLLM_PP_STEP_PROFILE_INTERVAL` 或 torch profiler）。

**必测指标（prefill / 端到端）：**

- TTFT p50/p99、prefill tok/s/GPU、前缀命中率；
- 峰值 HBM、主机 RSS、`/dev/shm` 占用。

**正确性门槛：**

- greedy 输出与基线逐 token 一致；
- 固定 seed 下 top-p 采样分布一致（统计检验，不要求逐 token 一致）；
- `tests/models/test_deepseek_v41_pipeline.py` 与相关 CPU 单测通过；
- 涉及 PP 的改动同时测 M=1（默认）与 M=2/3（`VLLM_PP_SEND_BUFFERS`）。

**测量纪律（借鉴 DeepInfer `examples/README.md`）：**

- 丢掉与 LB 统计 flush 重叠的区间，profiler 窗口不与测量窗口重叠；
- P/D 分开测；端到端另测，不混用；
- 固定输入长度分布、并发、`max_num_seqs`，记录每一步的配置。

---

## 3. 能力对照

| 能力 | DeepInfer | 本分支 | 结论 |
| --- | --- | --- | --- |
| shared expert 重叠 | combine 通信窗口内，SM 预算 | aux stream，≤256 token 时生效 | **部分**，需扩到 decode 全量 |
| 动态投机验证 | 有，在线估计 | 有，默认关；已在线化并支持 PP2 | **待 A/B** |
| 融合采样器 | 1 个 kernel，无 penalty | 多 kernel，flashinfer top-k/top-p | **可融合主路径** |
| decode 元数据 | 2 个 TileLang kernel | Triton + NumPy 记账 | **可单 kernel 化** |
| SWA 有界回放 | 默认有 | 默认有（128 window） | 已有 |
| SWA state 持久化 | 128 粒度、主机层 | 无；命中即重放 | **缺失** |
| 前缀缓存 | 开源版仅显式 fork | 完整 hash 前缀缓存 | 本分支更强 |
| 主机 / 3FS KV | 有 | HiSparse 主机层，但拒绝 PP | **受 PP 限制** |
| Engram | ENP + 主机 RDMA | UVA + Triton 查表 | **可减少 SM/PCIe 占用** |
| MoE 通信 | DeepEP-V2，异步 | NCCL allgather_reducescatter | SM120 不可用 DeepEP |
| Mega MoE | DeepGEMM，NVL 域 | 无 | SM120/PCIe 不适用 |
| EPLB | symm_mem + 冗余专家 | 有；symm_mem 需 NVLink | PCIe 上只可用静态/非 symm_mem |
| 全局 LB | 分层 prefill + 双维度 decode | 连接数最少 | **可加信号** |
| 故障恢复 | EP 重定尺寸、请求重路由 | 仅 stack dump watchdog，默认关 | **缺失** |
| 特殊解码 | — | SM103 + TP4，且禁止动态验证 | SM120 不适用 |

---

## 4. 任务清单

### P0-1 让 shared expert 重叠覆盖 decode 全量 token

- [x] 已测：无显著收益，保持默认关闭

**现状（已验证）：** 分支已有 aux stream 重叠：`SharedExperts.maybe_forward_async`
在 dispatch 前入队，MoE 结束后 `wait()`，覆盖 dispatch / GEMM / combine。
但触发条件是 `hidden_states.shape[0] <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`
（默认 256，`vllm/envs.py:328`）。64 请求 × 6 token = 384 行时退化为串行。
`allgather_reducescatter` 的 prepare/finalize 的 `supports_async()` 为 False
（`prepare_finalize/naive_dp_ep.py:146-151`），因此 MK 内部重叠也不可用。

**DeepInfer 做法：** shared expert 在 routed experts 之后、
`combine_pre_evt = ep_buffer.capture()` 之后启动，combine 通过
`previous_event=combine_pre_evt` 和 `defer_epilogue=True` 把自己的通信
与 shared 计算并行，最后再把 shared 结果作为 bias 合入；期间用
`limit_num_sms(DEEPINFER_USERBUFFER_COMM_SMS)` 把 DeepGEMM/TileLang 的
SM 数减掉通信配额，避免 shared GEMM 饿死通信。
（`deepinfer/model_executor/models/v4_modules/mlp.py:590-611`、
`deepinfer/model_executor/parallel_utils/userbuffer.py:40-51`）

**本分支落点：**

- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:117-160`
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:904-916`
- `vllm/envs.py:325-328`

**本次实现（`feat/dsv41-sm120-p0`）：** decode 步从 FULL graph 重放，MoE
all-to-all 也在关键路径上，overlap 的收益拐点与 prefill 不同。新增
`VLLM_SHARED_EXPERTS_STREAM_DECODE_TOKEN_THRESHOLD`（`vllm/envs.py`）：
记录 FULL graph 时用它替换 prefill 阈值（replay 不跑 Python，决定必须在
录制时做）；未设置时回落到 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`，
默认行为不变（`shared_experts.py:_token_threshold`）。`serve.sh` 暴露为
`SHARED_EXPERTS_STREAM_DECODE_TOKENS`。文档已补
（`docs/design/moe_kernel_features.md`）。

**更正（2026-11-12）：** 初版按 `CUDAGraphMode.FULL` 判断，但 V2 runner
录制 FULL graph 时 forward context 的 `cudagraph_runtime_mode` 是 `NONE`
（replay 则完全不跑 Python），所以 decode 阈值从未生效，下面的 A/B 实际
只测了 prefill 阈值。现改为用 `is_current_stream_capturing()` 识别录制中的
FULL graph；旧结论作废，需要重新 A/B。

**测量结果（`feat/dsv41-sm120-p0`，2026-11-11，阈值未生效，作废）：** 用
`SHARED_EXPERTS_STREAM_DECODE_TOKENS` 覆盖 decode 全量 token 后，
decode 吞吐与 TPOT 均无显著变化（未测到稳定收益，也未测到退化）。
推测 decode 步从 FULL graph 重放，关键路径由 MoE 通信与 attention 支配，
shared expert 的空隙已足够大，aux stream 再扩展也换不来关键路径的缩短。
**结论：** 保留 env 与实现（默认值不变），不再按步骤 2 改阈值策略，
也不重复 A/B。

**做法：**

1. 先用 `SHARED_EXPERTS_STREAM_DECODE_TOKENS=512/1024` 做 A/B，
   分别测 64/128/256 并发；对照组用 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`。
2. 若稳定为正收益，把阈值改成按 SM 预算判断（decode 时给通信预留
   `VLLM_EP_COMM_SMS` 个 SM），或对 decode 批次始终 overlap。
3. 不需要引入 DeepEP；NCCL 通信同样能受益。

**验收：** decode SPS 提升；profiler 中 MoE 段下降；greedy 输出一致；
CUDA graph 捕获下无 NCCL 死锁；M=1 与 M=2/3 都通过。

**风险：** 多流 fork/join 在 graph capture 中的顺序；PCIe NCCL 与 GEMM
抢 SM 可能反退化；prefill 大 batch 必须保留串行回退。

---

### P0-2 在 SM120 打开并在线化动态投机验证

- [x] PP2 支持已实现（A/B 待测）

**现状（已验证）：** `enable_adaptive_verification` 默认 False
（`vllm/config/speculative.py:628`）。Recipe 在 NVIDIA 预设里设为 true、
AMD 设为 false，但**没有 SM120 预设**；本分支 `serve.sh` 也没有传这个字段。
SM120 的 FlashInfer sparse metadata builder 是 `UNIFORM_BATCH`，但实现了
`get_varlen_cudagraph_max_query_len`（`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:266-300`），
工厂的 varlen 检查按该上界放行（`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:585-638`）。
注意 SM103 的 specialized decode 明确要求 `not enable_adaptive_verification`
（`vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py:96-100`），
但 SM120 本来就不满足 specialized decode，两者不冲突。

**DeepInfer 做法：**

- 在线估计 forward 斜率：5000 步窗口，按 graph size 两级中位数
  （`csrc/engine/mtp_varlen.hpp`，`ForwardRuntimeEstimator`）；
- 在线估计 sampler 耗时：按 `num_seqs` 的 9 样本中位数，不足回退全局；
- 用第 N−2 轮置信度选 batch size，第 N−1 轮置信度分配每请求长度，
  避免“本轮置信度未返回”的依赖；
- 文档给出的收益：60 并发 +4–7%，96 并发 +11%。

**原阻塞（已解除）：** `vllm/config/vllm.py` 曾在 `pipeline_parallel_size > 1`
时直接 raise `Adaptive verification is not currently compatible with
pipeline parallelism`（同处 TODO：把 cost curve 和置信度广播到所有 rank 即可
支持 PP），文档也写明 cost curve 与逐请求置信度只存在于最后一个 PP rank。
本次实现按下方步骤 1 的方案 A 补齐广播，并删掉该拒绝。

**本次实现（`feat/dsv41-sm120-p0`）：** 在线 cost table 刷新。
`StepTimingCollector` 新增非阻塞采样（每 N 个 forward 采一步，只
`query()` 不 `synchronize()`，见 `vllm/v1/worker/gpu/async_utils.py`）；
`AdaptiveVerificationManager.observe_timings` 把完成的样本按
`num_target_tokens` / `num_reqs` 分桶，桶内样本数 ≥4 时用中位数覆盖启动
profile，每 `VLLM_ADAPTIVE_VERIFICATION_ONLINE_INTERVAL` 步重建一次
（默认 0 = 关闭）。重建按 runner 步数计数而不是已排空的样本数，保证各 TP
rank 同步调用 `set_cost_curves` 的 broadcast。`model_runner` 在真实路径补上
`forward_end` / `drafter_start` / `drafter_end`（此前只在 capture 的 dummy
run 里记录，真实步没有结束事件）。文档已补
（`docs/features/speculative_decoding/adaptive_verification.md`）。

**PP2 支持（本次实现，方案 A）：** 每个 PP stage 都裁剪同一个 batch
（`query_start_loc` / `cu_num_logits` / `total_num_draft_tokens` 决定各
stage 的真实形状），所以两段的裁剪输入必须逐位一致。

- **cost curve**：`set_cost_curves` 在 TP `broadcast_object` 之后，用
  `all_gather_object`（`pp_group.cpu_group`）收集各 stage 的曲线并逐形状取
  **最大值**（`_merge_rank_curves`）。一步要等最慢的 stage 结束，所以形状的
  代价就是最慢 stage 的耗时，而不是各 stage 之和；非最后一段没有 drafter，
  上报 draft cost 0，取最大值不会覆盖最后一段的 draft 曲线。capture 阶段
  的 profile 尺寸由最后一段 `broadcast_object` 决定，保证各 stage 重放同一
  批 dummy batch。
- **置信度**：只有最后一段有 drafter。最后一段在 `broadcast_drafts` 之后用
  `pp_handler.broadcast_confidences`（`broadcast_stream` + sibling NCCL 组）
  把整个 `[max_num_reqs, num_steps]` 置信度缓冲发给前面的 stage；前面的
  stage 在 `pp_handler.receive` 里按同样顺序接收（`accept_remote_confidences`）。
  `execute_model` 开头的主流 `wait_event` 保证裁剪读到的 relay 已落地。
- **步计时**：非最后一段在 `sample_tokens` 里补
  `forward_end`/`drafter_start`/`drafter_end`（draft 段长度为 0），否则
  `StepTimingCollector` 不会解析这些步的样本，在线刷新会缺一半数据。
- **滞后槽位**：`_next_confidence_slot` 每步恰好调用一次（最后一段在
  `record_confidences`，其余在 `accept_remote_confidences`），各 stage 的
  双缓冲槽位同步推进，仍然读两步前写入的槽。

**本分支落点：**

- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`（预算 kernel 与 cost table）
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:build_cost_tables_from_curves`
- `vllm/v1/worker/gpu/pp_utils.py`（`broadcast_confidences` / `_relay_confidences`）
- `vllm/v1/worker/gpu/model_runner.py`（各 rank 创建 manager、`_end_stage_timing`、profile 尺寸广播）
- `vllm/config/vllm.py`（删除 PP 拒绝）
- `tests/v1/spec_decode/test_adaptive_verification.py`（`_merge_rank_curves`）
- `examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/serve.sh`（`ADAPTIVE_VERIFICATION` 开关）

**做法：**

0. 先在 PP=1 的 TP 部署（或单卡）上打开
   `"enable_adaptive_verification":true` + `ONLINE_INTERVAL=64`，验证
   在线刷新不改变 greedy 输出、预算 kernel 不退化。
1. ~~把 cost tables 与滞后一轮置信度广播到前面的 PP stage。~~
   已完成：曲线走 `all_gather_object` + 逐形状取最大，置信度每步在
   `broadcast_stream` 上 relay；`vllm/config/vllm.py` 的 PP 拒绝已删除。
2. 待做 A/B：固定 5 草案 vs 动态预算，测 32/64/96 并发
   （`ADAPTIVE_VERIFICATION=1`）。
3. ~~把启动时的静态 cost curve 换成在线中位数（forward slope + sampler），
   batch size 决策改用滞后一轮置信度。~~ 已完成。

**验收：** 启动成功；greedy 一致；接受长度与 SPS 提升；
p99 TPOT 不退化；预算 kernel 自身耗时小于省下的 decode 时间。

**风险：** varlen graph 的捕获内存；DSpark drafter graph 与目标 graph
尺寸的交互；低并发下动态预算可能不如固定 5。

---

### P0-3 融合采样主路径

- [x] 已量并关闭（2026-11-22）：整条采样链 16–20 µs = 0.2% 步长，上限 <1%，达不到 >5% 门槛

**现状（已验证）：** `Sampler` 是多段流水：penalties（独立）、min-p、
top-k/top-p（FlashInfer `flashinfer_sample`）、log-softmax / 采样、
grammar mask（独立）、logprob；`FusedMarkovSampler` 只用于 draft 侧
（`vllm/models/deepseek_v41/nvidia/dspark.py:332-346`）。

**DeepInfer 做法：** 一个 TileLang kernel 完成温度、top-k/top-p、
log-softmax、SplitMix64 采样、投机 rejection；grammar mask 单独；
**没有** repetition/frequency/presence penalty 和 min-p。
（`deepinfer/model_executor/layers/tilelang_ops/fused_sample.py`）

**本分支落点：**

- `vllm/v1/worker/gpu/sample/sampler.py:42-96, 264-344`
- `vllm/v1/worker/gpu/sample/penalties.py`
- `vllm/models/deepseek_v41/nvidia/ops/fused_markov.py`

**做法：**

1. ~~先量：每步采样 launch 数与耗时占比（固定 batch 与 DSpark 5）。~~
   已完成，见下。
2. 融合“温度 + top-k/top-p + log-softmax + 采样 + rejection”；
   penalty/min-p 保留为可选前置 kernel；mask 继续独立。
3. 只在无 penalty、无 min-p、无 logprob 的路径启用融合主路径，
   其余路径回退。

**实测（2026-11-22，`TIMING.md` §6.2）：** 离线微基准直接调用线上同一批
kernel，按 DSpark k=5（6 行/请求）用 CUDA graph 回放计时。temp=0 的整条
采样链（5×draft `greedy_argmax` + 1×verify `gumbel_sample`）是 16 µs（1 请求）/
20 µs（4 请求）；temperature>0 的完整链路（`apply_temperature` → 可选 min_p →
fused 或 `apply_top_k_top_p` + gumbel）是 6–222 µs。相对 9.5–20 ms 的步长，
采样段占 0.06–2.3%。fused 只在纯 top-p 批上明显更快（72 vs 222 µs），
`k+p` 与纯 `k` 反而更慢（SM120 上 `top_k_sampling_from_probs` 偏慢）。
**结论：上限 <1%，不满足 >5% 的门槛，第 2、3 步不做。**

**验收：** 固定 seed 下与现路径输出一致（greedy 与 top-p）；
launch 数下降；采样段耗时下降；带 penalty/min-p 的行为不变。

**风险：** 不能为了融合丢 penalty/min-p；mask 语义；
FlashInfer 采样器版本差异。

---

### P0-4 SWA state 持久化，前缀命中跳过重放

- [x] 完成（2026-11-22，由 LMCache 侧实现，未改 vLLM 代码）

**现状（已验证）：** `swa_bounded_replay=True`（`vllm/config/cache.py:371`），
`SlidingWindowManager.prefix_replay_tokens = sliding_window`，且该组不进 APC
（`vllm/v1/core/single_type_kv_cache_manager.py:1100-1109`）；前缀命中时
scheduler 把命中长度回退一个窗口重算（`_mark_prefix_replay`，
`vllm/v1/core/sched/scheduler.py:1445-1465`、`3972-3990`）。线上 layer 20 是
最后一个 KV source，stage 1 的 layer 21-39 只有 128-token SWA cache
（`--block_size 64` → 2 个 block），所以每次命中原本要重放 128 token
（README “Stage 1 in prefill”）。

**结论（2026-11-22）：现有实现已经覆盖，不需要新代码。** 启动配置里的
`lmcache.mp.store_replay_windows: true` 把 prefix-replay 组注册成可存组，connector
存/取窗口并经 `get_loaded_kv_cache_group_ids` 报告
（`lmcache/integration/vllm/lmcache_mp_connector.py:1768-1783`），于是
`_load_restores_replay_window()`（`scheduler.py:3963-3970`）为 True、
`window_loaded=True`，`_mark_prefix_replay` 返回 0。实测（`/tmp/probe_p04.py`，
地址与口径见 `TIMING.md` §5.6）：

- 满命中 computed tokens = **2**（1 个末 token 重算 + 1 个采样）；若有 replay 会是 129。
- 命中 TTFT **84 ms**（16k 本地）/ **103 ms**（64k 本地）/ 161、318 ms（跨引擎），
  比冷 128 token 的 **151 ms** 还低。
- 16k + 300 新 = 301、16k + 1200 新 = 1201，均无 replay（否则 429 / 1329）。

**残余缝隙：** 不配 LMCache、只用本地 APC 的部署；LMCache 窗口未存/被驱逐/重启；
本地命中与 LMCache 命中相距不足一个窗口（`num_computed_tokens <= ret - window`
不成立）。这些部署才需要下面的原方案。

**原方案（DeepInfer 做法）：** state block 分
`[stored: SWA][dropped: YOCO consumer SWA | compression state]`；
按 `KVSTORE_GRAIN=128` 只在最后一个 prefill chunk 落盘；
20 层 × 134 × 528 B = 1,415,040 B/seq；命中时直接从 state 恢复，
不再重放。（`deepinfer/worker/cache_engine.py:32, 224-306`、
`doc/inference_deployment_qa.md` 的 SWA 章节）

**若要做，落点：** `vllm/v1/kv_cache_interface.py:202`、
`single_type_kv_cache_manager.py:1100`、`scheduler.py:3963-3990`、
`vllm/models/deepseek_v41/decoder_replay_layers.py`、
`vllm/models/deepseek_v41/nvidia/model_state.py`。

**验收：** ✅ 2026-11-22 — 命中 computed = 2；命中 TTFT 低于冷 128 token；
输出一致；无新增主机内存（窗口本来就在 LMCache 里）；PP 两段边界无新增状态。
agent 场景的后续杠杆转向会话亲和（P1-8）与小 chunk 固定开销，见 `TIMING.md` §5.6。

---

### P1-5 decode 元数据单 kernel 化

- [x] 已量并关闭（2026-11-22）：关键路径上 ≤0.2–0.5 ms/步（1–2.4%，含 128k 上下文），达不到 >5% 门槛

**现状：** 仍有 NumPy 记账：`vllm/v1/worker/gpu/input_batch.py:173`、
`vllm/v1/worker/gpu/model_runner.py:1500, 1525, 1597` 的
`np.cumsum` / `np.add`，以及每步的 list/dict 组装。

**已量（2026-11-22，无重启；数字见 `TIMING.md` §7.6）：**

1. `/tmp/probe_cpu_step2.py`（/proc 口径）：EngineCore 主线程 user CPU/步 =
   15.6 / 11.4 / 9.3 ms（1 / 4 / 16 请求/engine），步长 19.4 / 14–16 / 12–13 ms；
   **CPU/步不随请求数增长** ⇒ 主体是等 stage 的忙轮询，不是元数据记账。
2. `VLLM_PP_STEP_PROFILE_INTERVAL=1` 的每步 span（`/tmp/probe_step_profile.py`
   等）：`execute` 的 CUDA 事件包住整个 `execute_model`，元数据只要拖慢首个
   kernel 就会计入 execute。实测 `step − execute − sample` = 0.20 ms（1 请求）、
   0.01 ms（16）、0.01 ms（32 请求/engine，打满 `max_num_seqs 64`）。
3. 加长上下文复核（32 请求/engine，64k 与 128k，`/tmp/probe_step_profile_long.py`）：
   pp0 的 step 18.2 → 19.8 → 21.6 ms，增量全在 execute（GPU attention/indexer +
   更大的 block table）；host 缝隙仍 0.00 / 0.00–0.47 ms。block table 从
   256 → 2048 块/seq 只换来 ≤0.5 ms。

**结论：关闭。** 调度与元数据真正加到关键路径上的部分 ≤0.5 ms，占 11–22 ms
步长的 1–2.4%。单 kernel 化省的是被 GPU 覆盖的 CPU（worker 1.0–1.45 核），
不是步长；改完最多赚 ~0.2 ms/步，达不到 5%。`sample` 里 `draft` 2.3–2.6 ms 是
3 层 draft + lm_head 的真算力（不随 batch/ctx 变），也不是 host 开销。

**DeepInfer 做法：** decode 元数据 2 个 TileLang kernel：
每序列一个 CTA 的主 kernel + block table 拷贝，无 CPU-GPU 同步
（`deepinfer/model_executor/layers/tilelang_ops/prepare_v4_decode_metadata.py`）。

**做法：** 先量每步 CPU 时间与 launch 数；把
positions / slot_mapping / context_lens / query_start_loc / block_table
展开合成一个 Triton 或 CUDA kernel；保留 host 侧只做 O(1) 参数更新。
PP 场景下两段各自维护自己的 metadata，不跨段传递。

**验收：** 每步 CPU 时间下降；graph 稳定；happy-path 输出一致。

---

### P1-6 图内计时（graph_timer）

- [ ] 完成

**现状：** 依赖 torch profiler 或 `VLLM_PP_STEP_PROFILE_INTERVAL`，
profiler 自身扰动 7–11%（DeepInfer 文档口径）。

**DeepInfer 做法：** `csrc/graph_timer` 把
`cudaGraphExecEventRecordNodeSetEvent` 重定向到目标事件，
不重捕获即可在 FULL graph 内测每个 kernel
（`deepinfer/csrc/graph_timer/`）。

**做法：** 借概念，不搬代码：在现有 FULL graph 的事件节点上做重定向，
输出每 kernel 的耗时；与 profiler 结果交叉校验。

**验收：** 计时残差低于 5%；不改变捕获与执行；默认关闭。

---

### P1-7 VE 独立池 + 按 patch 数路由（仅多模态部署）

- [ ] 完成（条件：启用视觉）

**现状：** 视觉与文本同池；SM120 示例是 `--language-model-only`。
已有 DP 分片 ViT、mm processor cache、encoder budget
（`vllm/models/deepseek_v41/common/vision.py`、
`vllm/models/deepseek_v41/nvidia/vl_model.py`）。

**DeepInfer 做法：** VE 独立池，通常 1 GPU/实例；按 patch 数
（`(w/patch)×(h/patch)`）路由；fork 预处理 worker + pinned 共享池；
图像 sentinel token 支持 chunk prefill
（`deepinfer/ve/multimodal_processor.py`）。

**做法：** 若要多模态：独立 VE 池、patch 数路由、预处理 worker 化；
否则保持现状，不投入。

**验收：** 图像 TTFT、VE GPU 利用率、文本 SPS 不受影响。

---

### P1-8 路由信号：token / KV / 前缀亲和

- [ ] 完成

**现状：** launcher 只在 API server 进程间按连接数分流
（`vllm/entrypoints/launchers/launcher.py:79-110`）；DP rank 的选择在
`DPLBAsyncMPClient.get_core_engine_for_request`（`vllm/v1/engine/core_client.py`），
只看 waiting + running 请求数。已加开关 `VLLM_DP_LB_TOKEN_AWARE`（待 A/B）。

**已定量（2026-11-22，`TIMING.md` §5.6.2）：** 引擎侧已经支持用
`X-data-parallel-rank` 钉住（`serving.py:196`），所以本项**不需要改 vLLM**。
第一次落到没有 device 副本的 rank，命中 TTFT 8k/16k/32k 从 96/96/100 涨到
134/178/229 ms（+38/+82/+129），第二轮就恢复本地速度。落地工作是路由器侧
的 session/前缀 → rank 粘滞 + 回退，以及按 token 而不是请求数分流（P1-8
原有的"按 token 亲和"结论不变）。

**DeepInfer 做法：** prefill 按 4 档上下文长度（<16K / 16–64K /
64–256K / ≥256K）分层，按上一轮算力配额拉取，组内按最低配额路由，
再用二次成本模型 `cost(q,c)` 解本步 chunk；decode 同时平衡请求数与
总 KV 长度（DualPath）；VE 按 patch 数。离线基准约达到完美均衡的 80%
（`doc/inference_deployment_qa.md`）。

**做法：** 不改引擎，先在网关/launcher 加三个信号：
pending tokens、running seqs、KV 占用；再加前缀 hash 亲和
（同一 system prompt 不跨 rank）。分层 prefill 与双维度 decode
放到网关侧做。

**验收：** 每 rank token 数方差下降；`max_step` 下降；
p99 TTFT 改善；不引入热点。

---

### P1-9 故障恢复：从 dump 到摘除/重启

- [ ] 完成

**现状：** 只有 stack-dump watchdog
（`vllm/utils/watch_dog.py`，由 `vllm/v1/executor/multiproc_executor.py:1081`
和 `vllm/v1/engine/core.py:1742` 启动），默认 `dump_dir=""` 即关闭，
且只 dump 不恢复。已观察到 PP1 worker 数小时一次静默退出
（见 `kvcache-perf/crash_analysis_20261108.md`）。

**DeepInfer 做法：** 单卡故障后按健康 GPU 重定 EP size；
CE 收到关闭通知后 1 分钟宽限，未完成请求连同 KV 转发；
自动扩缩容有速率限制，配额 >0.8 不缩容。

**做法：**

1. 默认打开 `watchdog_config.dump_dir`，拿退出时的栈；
2. 加 per-rank 心跳与请求超时，自动摘除/重启；
3. 网关侧对失败请求重试（幂等请求）或转发。

**验收：** 注入 kill / hang 后 N 秒内检测到；请求不丢或明确失败；
恢复后不产生脏 KV。

---

### P1-10 持久化前缀 KV（先主机层，不做 3FS）

- [ ] 完成

**现状：** hash 前缀缓存只在 HBM；HiSparse 主机层存在但拒绝 PP
（`vllm/config/vllm.py:1851`：“HiSparse does not support pipeline parallelism.”）。

**DeepInfer 做法：** GPU HBM（飞行中）→ 本地主机 SWA（400G VLAN）→
3FS block KV（14.4 TB 逻辑/GPU，10+2 EC，TTL 3/7 天，LRU）；
GDR swap-in；decode 不做 KV offload，也不做 HiSparse。

**做法：**

1. 先用 LMCache / offloading connector 的主机层验证跨请求复用收益；
2. 评估解除 HiSparse 的 PP 限制（工作量主要在 KV 布局与 PP 元数据）；
3. 不引入 3FS；除非主机层已被证明是瓶颈，再谈外部存储。

**验收：** 命中率、TTFT、主机带宽、连续运行稳定性。

---

### P2-11 多机 Wide-EP（暂缓）

- [ ] 暂缓

需要 64–128 卡超节点、DeepEP/NVSHMEM、EPLB symm_mem。8 卡 PCIe 上
不可达；仅作为未来机型的设计参考。

---

### P2-12 结构化输出离线编译（按需）

- [ ] 待定

DeepInfer 把 grammar 编译放到专用 SO 机器、与 decode 解耦。
本分支在请求路径内编译。先量大量 schema 的编译耗时占比，
超过 1% 再考虑进程外编译 + 缓存。

---

### P2-13 batch-invariant 覆盖（补测试）

- [ ] 待定

两边都有 batch-invariant 实现；本分支有 `VLLM_BATCH_INVARIANT`。
只需补“开/关 batch-invariant 输出一致”的测试与文档，
不需要借代码。

---

## 5. 不采纳 / 边界

- **SM100 kernel**：FlashMLA、DeepEP、Mega MoE、`fp8_fp4_mega_moe`
  依赖 SM100/NVLink/NVSHMEM，SM120 上不可用。
- **Wide-EP（EP64–128）**：需要超节点；本机 EP4 是现实选择。
- **“不要 PP”**：DeepInfer 在 64+ 卡 decode 上不用 PP，是因为复制的
  attention 权重相对 HBM 便宜。8 卡上 PP2 把每 GPU 复制的
  attention/shared 权重减半，并把 PCIe A2A 从 EP8 的约 35 层当量
  降到约 15 层当量；不要为了对齐 DeepInfer 而拆掉 PP。
- **DeepInfer 的采样器**：无 penalty/min-p，不能直接替换。
- **3FS / TTL / EC**：基础设施，先用主机层验证收益。
- **SWA 余量 128→140**：DeepInfer 需要 `window + max_backtracking`
  （`prepare_tensor_input_v4.cpp` 的断言），本分支 replay 发生在
  prefill 前缀命中、窗口之后仍有完整 suffix，大概率不需要；
  先在 DSpark 首步回退场景确认，再决定是否加。
- **EPLB symm_mem**：要求整个 EP 组在同一个 NVLink 域
  （`vllm/distributed/eplb/eplb_communicator.py:770-776`），
  RTX PRO 6000 是 PCIe，只能用非 symm_mem 后端。

---

## 6. 执行顺序

1. **测量**：用 `VLLM_PP_STEP_PROFILE_INTERVAL` + torch trace 出一步分解
   （attention / indexer / MoE GEMM / A2A / PP send / DSpark / Engram /
   sampler / metadata）。
2. **P0-1**：环境变量 A/B（`SHARED_EXPERTS_STREAM_DECODE_TOKENS`），
   再决定是否改默认。
3. **P0-2**：PP 支持已实现（曲线取最慢 stage、置信度每步 relay），
   下一步直接做 32/64/96 并发 A/B（`ADAPTIVE_VERIFICATION=1`）。
4. **P0-3**：已量（见 §7 与 `TIMING.md` §6.2），采样段 0.2–2.3%，关闭，不做。
5. **P0-4**：已关闭——LMCache `store_replay_windows` 已跳过 replay（`TIMING.md`
   §5.6）；agent 流量转做 P1-8 会话亲和。
6. **P1**：按 profile 结果排序，P1-8 / P1-9 可与 P0 并行。
7. 每完成一项，更新第 7 节。

---

## 7. 验证记录

每项记录：配置 / 指标（前 → 后）/ 结论 / 日期。

- P0-2：PP2 支持已实现（`feat/dsv41-sm120-p0` / 2026-11-11）；
  `test_adaptive_verification.py`、`test_gpu_model_runner_v2.py`、
  `test_pp_utils.py`、`test_config.py -k adaptive` 全通过；32/64/96 并发 A/B 待测
- P0-1（threshold 256→512，decode 全量 token）：2026-11-11 测得
  decode 吞吐与 TPOT 无显著变化，默认仍关闭，不再调阈值
- P0-3（采样段占比，2026-11-22）：离线微基准直接调用线上 kernel，CUDA graph
  回放（`/tmp/bench_sampler_ops.py`、`/tmp/bench_sampler_temp.py`，数字与
  方法见 `TIMING.md` §6.2）。temp=0 整条采样链 16 µs / 20 µs（1 / 4 请求），
  temperature>0 完整链路 6–222 µs，采样段占步长 0.06–2.3%；fused 只在纯
  top-p 批上更快（72 vs 222 µs），`k+p`、纯 `k` 更慢。**达不到 >5% 门槛，
  P0-3 关闭**；顺带记录 `apply_min_p` 单算子 71 µs 是最贵的一项
- P0-4（跳过命中时的 SWA replay，2026-11-22）：**已完成，未改 vLLM 代码**。线上
  `lmcache.mp.store_replay_windows: true` 让 connector 报告已恢复 replay 窗口，
  `_mark_prefix_replay` 返回 0（`/tmp/probe_p04.py`，数字见 `TIMING.md` §5.6）：
  满命中 computed = 2（有 replay 会是 129），16k/64k 本地命中 TTFT 84 / 103 ms，
  跨引擎 161 / 318 ms，冷 128 token 151 ms。残余只在"不配 LMCache"或"窗口没存住"
  的部署；agent 场景剩下的杠杆是会话亲和（P1-8，跨引擎命中 16k +77 ms、64k
  +215 ms）和小 chunk 的 ~50 ms 固定开销
- P1-5（decode 每步 CPU，2026-11-22）：两层证据（`TIMING.md` §7.6）。/proc：
  主线程 user CPU/步 15.6 / 11.4 / 9.3 ms（1 / 4 / 16 请求），不随请求数增长 ⇒
  忙轮询主导。每步 span：`step − execute − sample` = 0.20 / 0.01 / 0.01 ms
  （1 / 16 / 32 请求/engine），128k 上下文、32 请求下仍 0.00–0.47 ms ⇒
  元数据+launch+调度在关键路径上 ≤0.5 ms，占 11–22 ms 步长 1–2.4%。
  **关闭：达不到 5% 门槛**
- P1-8（会话亲和）：**已定量，不需要改 vLLM**（`TIMING.md` §5.6.2）。固定
  `X-data-parallel-rank` 时命中 TTFT 8k/16k/32k = 96/96/100 ms；没打过的 rank
  首轮 134/178/229 ms（+38/+82/+129），第二轮回到 96/115 ms——溢价是"该 rank
  没有 device 副本"而不是"引擎号不同"。agent 6 轮负载：不固定 rank 有 2 轮
  242/177 ms，固定后 145–150 ms。`serving.py:196` 已读该 header，剩下的是路由器
  侧 session→rank 粘滞 + 回退，属于网关工作
- 步级耗时拆分（§6 第 1 步）：线上 PP2 × TP2 × DP2 的实测已完成，数字见
  [`TIMING.md`](TIMING.md)（2026-11-21）：1 请求 decode 约 19–20 ms 一次更新且与
  context 无关；现网 iteration 口径实测 20.3 ms/iteration、2.13 token/iteration、
  TPOT 9.5 ms；PP1 比 PP0 多忙 10–25%。步内构成已用离线重建补上（同文件 §8）：
  专家权重流式 44–50%（1054 GB/s，已在带宽墙上）、注意力投影 28–32%、
  EP 集合通信 9–20%、mHC 5–6%、lm_head 2–3%，合计 12.0–13.7 ms——高于
  iteration/2，差额来自 MoE 的 L2 命中或投影与集合通信窗口的重叠，需一次带
  profiler 的重启定死。投影与集合通信这两行都是按线上 kernel、线上 shape 直接
  实测的（同文件 §8.1 / §8.2）：投影 194 µs/层（写死的 `backend="cutlass"`），
  换 `auto` 是 90 µs/层；集合通信等长 54 µs/层、不等长 135 µs/层，打包后 65 µs/层。
  两处已分别做成 `VLLM_MXFP8_GEMM_BACKEND` 与 `VLLM_MOE_AG_RS_PACKED_DISPATCH`
  （2026-11-21 落地），合计 −2.9 ms/步，可优化空间 ~3–3.5 ms
- 两个开关的实现与离线验证（2026-11-21）：`VLLM_MXFP8_GEMM_BACKEND` 经
  `FlashInferCutlassMxfp8LinearKernel.apply_weights` 验证三个线上 shape 逐位相同
  （`/tmp/verify_mxfp8_env.py`）；`VLLM_MOE_AG_RS_PACKED_DISPATCH` 在 4 rank NCCL 上
  验证行逐位相同（`/tmp/verify_packed_dispatch.py`）。单测：
  `tests/test_pcp_dp.py`、`tests/kernels/quantization/test_flashinfer_mxfp8_unaligned_n.py`。
  **14:45 重启已同时带上两个开关**（`VLLM_MXFP8_GEMM_BACKEND=auto`、
  `VLLM_MOE_AG_RS_PACKED_DISPATCH=1`）：重启后 1 请求/引擎 ITL p50 17.5 ms、
  均值 18.5 ms（之前 19–20 ms），方向对但两条同时开无法分项归因
- `VLLM_PP_MICROBATCH_BALANCE=balance` 的效果已实测（现役进程，两引擎 49,190 次
  ITL 更新）：`pack` 时期 31.5% 的 50–75 ms 峰**归零**，≤25 ms 占 98.9%，均值
  **22.7 ms**（TIMING.md §5.4）。调度附加基本清掉，但 TTFT/吞吐代价还没量
- 一轮不改代码的引擎侧实测（2026-11-21，`/tmp/probe_live.py`、`/tmp/bench_indexer_topk.py`，
  数字见 TIMING.md §5.3 / §5.5 / §6.1）：
    - 冷 prefill 常态 0.072–0.094 ms/token（16k→1.55 s、128k→10.3 s、256k→22.6 s），
      与引擎 cost fit 一致；旧记录「0.47–0.52 ms/token」是**未解的 5× 慢态**
      （6.8–8.1 s / 16k，出现在一串请求的头一两个冷 prefill），不是 LMCache 写入，
      排除项与下一步见 TIMING.md §5.3
    - decode 行数 ×4（1→4 请求/引擎）步长只 +56%（18.6→29 ms），验证专家流式假设，
      批大小与 k 都便宜
    - indexer top-k 换 `per_row`/`flashinfer` 在 128k 上省 0.30/0.32 ms/步（5 层），
      短 context 收益小；16:00/16:50 两次重启试过、17:12 回退：prefill +40 ms（2.7%），
      且与慢态无关（TIMING.md §5.3.1），净收益还没单独验证
- 16:00 → 17:12 的线上 A/B（k=7 + `per_row` → 两者都去掉，2026-11-21/22）：
  **k=7 在单请求探针上更快、在线上不一定**——1 请求 × 256 输出，k=7 客户端
  0.95 s、k=5 1.78 s，ITL p50 都是 17.5 ms，token/iteration 6.24 vs 2.94（1.87×），
  当时与 `per_row` 一起回退。**但线上开过 k=7 后整体变慢**：draft 链按 step
  串行（`speculator.py:334`），k=7 要跑 7 段 draft（每段 3 层 + 129k lm_head），
  而第 6/7 位的逐位接受率只有 ~0.19 及以下——多出的两段是"用整段 draft 换
  ~0.1-0.2 个 token"，只在 1 请求/引擎时因步长被权重流式主导而划算，批一大就亏
  （TIMING.md §5.3.1）。正确旋钮是 `enable_adaptive_verification`（按置信度
  在整个 batch 上分配 verify 槽位，缺省关闭），不是固定 k=7
- 一次带 profiler 的重启（2026-11-22 04:19，`VLLM_PP_STEP_PROFILE_INTERVAL=1` +
  `--profiler-config`；窗口方法、崩溃链与读数口径见 TIMING.md §7）：
    - prefill 16k（PP0）：attention 23.9%、gemm 18.2%、ep_all2all 16.6%、
      **other_nccl 13.4%**、moe 13.4%、other 7.0%、mhc 4.2%、pp_p2p 2.9%、
      indexer 0.4%。other_nccl 是 **TP=2 多出来的 all-reduce**，§8 的离线重建按
      TP1 做，原来没有这一项。旧脚本把专家 GEMM 误分到 attention（`flash` 匹配
      `flashinfergemm`），已修（TIMING.md §7.3）
    - decode 32k（1 请求/引擎）：真实 GPU **7.86 ms/step**——专家段 3.67 ms
      （170 µs/层，按 DRAM 墙反推 ≈10 个 distinct expert；§8 的 6.04 ms 按 18 个
      `(row, expert)` 对算，是上界）、EP 1.40 ms、gemm 1.06、mHC 0.42、
      **attention 只有 0.23 ms**（不是旧表的 2.3 ms）。**"12–13.7 ms 模型比步长
      高"定死了一半**：模型重复计入了 overlap 的部分；7.86 ms 与实测 9.5–10 ms
      之间的 ~2 ms 主要是 stage 0 等 stage 1（stage 1 折算 ~9.0 ms），只剩 ≤1 ms
      可能是 launch/同步/采样
    - **stage 1 补齐**（10 步窗口）：纯计算 9.45 vs stage 0 7.88 ms/step，多出的
      1.6 ms 是 lm_head ~1.0、多 2 层 indexer +0.35、DSpark draft ~0.5，**MoE 反
      而略少**（4.35 vs 4.83）→ stage 1 的优化空间在 indexer/draft/lm_head
    - **decode trace 在 `profiler_stop` 时可能 `double free` 打死整个服务**
      （rank 0 兼 TCPStore 宿主 → 其余 rank store unavailable → `EngineDeadError`，
      退出码 None），eager prefill trace 不会。40 步窗口挂了；10 步窗口（flush 在
      压测中完成）8 个 rank 全部干净写盘。**08:36 又挂一次，且是 2 步窗口**：
      `max_iterations=2` 在空闲时被 1.2 ms 的空转步立刻用完，窗口里只有 dummy
      decode 的图重放，teardown 照样段错（`~Result`/`~KinetoEvent`，见
      TIMING.md §7.2）。结论收紧成"只采 eager prefill 窗口"，并注意
      `max_iterations` 数的是 worker step，不是请求的 step。`nsys` 在本机不可用：
      2026.1 的 CLI 没有 attach，容器也没有 `CAP_SYS_PTRACE`
    - 跨引擎对照：DP0 挂 4 请求、DP1 挂 1 请求，ITL 均值 25.7 vs **28.2 ms**——
      孤请求不比 4 请求的引擎快，EP 组跨 DP 强制 lockstep，**P1-8 必须按 token
      亲和分流**，按请求数分没有意义。受控波次 5+5 vs 4+6 只差 2%，按行数准入
      的收益 ≤0.3 ms/步，不值得先做
    - DP4 推演（`/tmp/dp4_sim.py`，校准到实测偏斜分布）：4 引擎的偏斜明显更深
      （0 档 46% → 20%，2 档 11% → 29%/31%）而 step p50 17.3 → 17.0 ms、
      TPOT 10.9 → 10.5 ms——锁步时钟看组内总行数，DP4 的收益在 prefill 容量 /
      TTFT（K=16 完成数 +52%）。真 DP4 是 TP1，常量会变，只能读方向
    - 无 trace 基线（新进程）：1k TTFT 140 ms / ITL 21.5 ms，32k TTFT 3.06 s /
      ITL 20.3 ms（p50 17.5）；16k 前缀命中 TTFT 98–156 ms vs 冷 1451 ms ⇒ replay
      加装载 ≈50–120 ms/命中。**P0-4 的门槛口径要先定**：占命中 TTFT 远超 5%
      （命中流量里它就是主项），占冷 TTFT 只有 4–7%。
      **07:40 复测修正（TIMING.md §5.6）**：那 ≈50–120 ms 里 replay = 0——
      LMCache `store_replay_windows` 已经把窗口装回来了；本地命中 84 ms 是
      84 ms floor + 少量外部装载，跨引擎（块不在本 engine）才是 77–215 ms
- 两个未解释数的收口（2026-11-22 08:3x–12:3x，TIMING.md §5.2 / §5.3 / §5.6）：
    - **84 ms 命中 floor** 不是常数，是「排队等外部 KV（16k ≈ 90 ms，
      3–5 µs/token）+ 一次 decode 形态 step（26–41 ms）」。lookup 只占 6–20 ms。
      满命中的 1 个新 token 走 `prefill_runs_as_decode`（`model_runner.py:1497-1504`），
      所以便宜；冷 1 token 没有 KV，只能走真 prefill。
    - **小 prompt 的冷 prefill 是每级固定 ~65 ms**（1 token = 68 ms/级，
      1k = 78 ms/级，4k = 262 ms/级）：token 斜率 ~0.05–0.06 ms/token/级到 2k
      以上才显出来。重启后进程墙钟复核：64 = 159、1024 = 166、2048 = 288、
      4096 = 590、8192 = 853 ms。
    - **空闲 1.4 kW 是状态不是常数**：12:2x 采样空闲态已收敛到 pause——整机
      698 W、util 0.6–2.7%、单卡 70–104 W；1.2 kW 出现在"刚跑完一批"的头几秒
      （1206 W → 10 s 内 630 W）或没收敛到 pause 时。host 侧仍恒定 8.7 核忙轮询。
    - 另外两个读数坑已写进 TIMING.md：空闲 DP engine 的 `step 130 ms /
      execute 0.00` 是锁步等待，不是丢 span；`prompt_tokens_cached_total` 是裸
      counter（带 `_total`），早先按 `_sum` 过滤才读出 cached = 0
- 三项收口（2026-11-22 12:2x–13:0x，TIMING.md §5.6.1 / §5.6.2 / §5.7）：
    - **本地命中的 46–49 ms queue = 取 127 token 的外部尾巴**，不是空等。
      device 覆盖 `floor((n-1)/128)*128`（128 = KV 组 LCM，启动日志
      `kv lcm block sizes 128`），n 是 128 倍数时 LMCache 里还有 127 token：
      queue 42 / prefill 52 / TTFT 96；没有尾巴时 queue 5 / prefill 135 /
      TTFT 143。**取比不取省 47 ms**。尾巴的 39 ms `in_progress` 里只有
      4.55 ms 是 h2d 拷贝，其余是 MP 往返 + 跳过调度步
    - **跨引擎搬运只有 260–310 MB/s**（16k = 65 chunk/卡、10.1 MB、38.2 ms），
      比 PCIe 应有值低一个量级；这是跨引擎命中的真实成本，值得单查
      （`use_vmm_api` 路径 / NUMA）
    - **小 batch prefill 的 65 ms/级 = eager 逐层 launch**：`torch.compile`
      不支持该模型 + 无 breakable 图 → 日志 `Overriding to FULL_DECODE_ONLY`
      → `mixed_mode()` NONE → 不建任何 prefill 图 → `cg_mode=NONE` →
      `self.model(**model_inputs)`。1–768 token 墙钟平在 138–146 ms、512 网格
      边界无台阶；同样 1 token 图上 26 ms/级 vs eager 68 ms/级（20 层/级
      ⇒ ~2 ms/层）；功率停在空闲档。杠杆是 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
      （配置层唯一入口，需一次带重启的 A/B）。14:14 首跑崩在捕获里建 cuDNN plan
      （索引器 `wq_b` 短上下文早退在捕获时被绕过，eager warmup 从未跑过该 GEMM）；
      已加「按 desc 预 warm」钩子，15:16 重启后 27/27 + 41/41 档全部捕获。冷
      prefill 墙钟：1/128/512/768/1024/2048 token = 31/61/108/138/170/289 ms
      （基线 138/141/146/~140/166/286），8k 917 ms、16k 1444–1664 ms 与基线
      持平，decode 步长未退化——即小 prefill 明显变快、大 prefill 不吃亏
      （TIMING.md §5.7.1、§9.3）
- 小外部命中同 step 装载（2026-11-22 13:0x，**已实现，默认关**）：把
  ≤128 token 的命中从异步改成在分配块的同一个 step 里提交并让当前流等待
  （`DeviceMessagingFuture.wait_on_stream`），去掉"跳过调度步"的 ~35 ms。
  代码在 `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_sync_load.py`
  （外部 connector 的子类；`kv_connector_module_path` 指过去，类名仍是
  `LMCacheMPConnector`），旋钮 `lmcache.mp.sync_load_max_tokens`（0 = 原行为）
  与 `lmcache.mp.sync_load_timeout_s`。14:53 首次启用跑了 6 分钟流量后崩在
  scheduler 的 finished-recv assert：worker 按 op 跨度、scheduler 按
  `need_to_load` 划分同步/异步，APC 命中非 chunk 对齐时两者差最多 255 token。
  已改为把 scheduler 的判定随 metadata 传给 worker，单测 12 passed；
  线上 A/B 仍需重启，看 16384 TTFT 96 → 60–70 ms 与 decode 步长不回退
  （TIMING.md §5.7.2）

---

## 8. 参考索引

**DeepInfer（`/workspace/deepinfer`）**

- `doc/inference_deployment_qa.md`：生产架构、EPLB、LB、KV 分层、SWA、动态验证
- `examples/README.md`：基准口径
- `deepinfer/model_executor/models/v4_modules/mlp.py:560-611`：shared expert / combine 重叠
- `deepinfer/model_executor/parallel_utils/userbuffer.py:40-51`：`limit_num_sms`
- `csrc/engine/mtp_varlen.hpp`：动态验证与在线估计
- `deepinfer/model_executor/layers/tilelang_ops/fused_sample.py`：融合采样
- `deepinfer/model_executor/layers/tilelang_ops/prepare_v4_decode_metadata.py`：decode 元数据
- `deepinfer/worker/cache_engine.py:32, 224-306`：SWA state 布局
- `deepinfer/model_executor/eplb/__init__.py`：EPLB
- `deepinfer/model_executor/models/v4_modules/engram.py`：Engram
- `deepinfer/ve/multimodal_processor.py`：VE
- `csrc/engine/scheduler.hpp`：C++ 调度与块管理
- `csrc/graph_timer/`：图内计时

### 本分支

- `examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/README.md`：部署与阶段说明
- `examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/TIMING.md`：线上链路耗时
  实测（decode step、prefill、PP 平衡、索引器、逐 kernel trace 与 profiler 崩溃
  纪律）
- `vllm/config/cache.py:372`：`swa_bounded_replay`
- `vllm/config/compilation.py:652`：`decoder_replay_trim_threshold`
- `vllm/v1/kv_cache_interface.py:1048`：`prefix_replay_tokens`
- `vllm/v1/core/sched/scheduler.py:3649-3656`：前缀命中回退
- `vllm/models/deepseek_v41/decoder_replay_layers.py`：SWA 有界回放
- `vllm/models/deepseek_v41/nvidia/model_state.py`：replay 起点
- `vllm/config/speculative.py:628`：`enable_adaptive_verification`
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:585-638`：工厂检查
- `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:266-300`：varlen graph 上界
- `vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py:77-131`：SM103 约束
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:117-160`：shared expert 重叠
- `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:146-151`：A2A async 限制
- `vllm/v1/worker/gpu/sample/sampler.py:42-96, 264-344`：采样流水
- `vllm/v1/worker/gpu/input_batch.py:173`、`model_runner.py:1500, 1525, 1597`：NumPy 记账
- `vllm/config/vllm.py:1851`：HiSparse 拒绝 PP
- `vllm/distributed/eplb/eplb_communicator.py:770-776`：symm_mem NVLink 要求
- `vllm/entrypoints/launchers/launcher.py:79-110`：连接数负载均衡
- `vllm/utils/watch_dog.py`、`vllm/config/watchdog_config.py`：stack-dump watchdog
- `vllm/v1/worker/gpu/pp_step_profiler.py`：步级 profiler
