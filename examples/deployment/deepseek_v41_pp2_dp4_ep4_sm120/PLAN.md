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

- [ ] 完成

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

1. 先量：每步采样 launch 数与耗时占比（固定 batch 与 DSpark 5）。
2. 融合“温度 + top-k/top-p + log-softmax + 采样 + rejection”；
   penalty/min-p 保留为可选前置 kernel；mask 继续独立。
3. 只在无 penalty、无 min-p、无 logprob 的路径启用融合主路径，
   其余路径回退。

**验收：** 固定 seed 下与现路径输出一致（greedy 与 top-p）；
launch 数下降；采样段耗时下降；带 penalty/min-p 的行为不变。

**风险：** 不能为了融合丢 penalty/min-p；mask 语义；
FlashInfer 采样器版本差异。

---

### P0-4 SWA state 持久化，前缀命中跳过重放

- [ ] 完成

**现状（已验证）：** `swa_bounded_replay=True`（`vllm/config/cache.py:372`），
`prefix_replay_tokens = sliding_window`（`vllm/v1/kv_cache_interface.py:1048`），
前缀命中时 scheduler 把命中长度回退一个窗口重算
（`vllm/v1/core/sched/scheduler.py:3649-3656`）。PP 部署中 stage 1 的
layer 21-39 每次命中都要重放最后 128 token（README “Stage 1 in prefill”）。

**DeepInfer 做法：** state block 分
`[stored: SWA][dropped: YOCO consumer SWA | compression state]`；
按 `KVSTORE_GRAIN=128` 只在最后一个 prefill chunk 落盘；
20 层 × 134 × 528 B = 1,415,040 B/seq；命中时直接从 state 恢复，
不再重放。（`deepinfer/worker/cache_engine.py:32, 224-306`、
`doc/inference_deployment_qa.md` 的 SWA 章节）

**本分支落点：**

- `vllm/v1/kv_cache_interface.py:231, 1048`
- `vllm/v1/core/sched/scheduler.py:3649-3656`
- `vllm/models/deepseek_v41/decoder_replay_layers.py`
- `vllm/models/deepseek_v41/nvidia/model_state.py`

**做法：**

1. 先量：在目标命中率下，replay 占 TTFT 的比例。低于 5% 就先不做。
2. 若值得做：在主机层按 block hash 存 SWA state，128 粒度，
   只在最后一个 prefill chunk 写；命中时装载并跳过 replay。
3. 首版可只覆盖 stage 1 的 21-39 层，layer 20 仍然全量写 KV。

**验收：** 前缀命中 TTFT 改善；命中率不变；输出一致；
主机内存增量可接受；PP 两段的 state 边界正确。

**风险：** PP stage 边界与 mHC 状态；state 布局必须与 KV 精确匹配；
与 CUDA graph replay 尺寸的交互。

---

### P1-5 decode 元数据单 kernel 化

- [ ] 完成

**现状：** 仍有 NumPy 记账：`vllm/v1/worker/gpu/input_batch.py:173`、
`vllm/v1/worker/gpu/model_runner.py:1500, 1525, 1597` 的
`np.cumsum` / `np.add`，以及每步的 list/dict 组装。

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
4. **P0-3**：只有采样段占比 >5% 才做。
5. **P0-4**：只有 replay 占 TTFT >5% 才做。
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
- P0-3（固定 seed）：待测
- P0-4（目标命中率）：待测
- P1-5（decode 每步 CPU）：待测
- P1-8（多 rank 方差）：待测

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
