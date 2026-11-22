# DeepSeek-V4.1 线上链路耗时实测

**状态：** 实测记录，2026-11-21
**被测部署：** 线上 PP2 × TP2 × DP2 + EP（与本目录 README 的 PP2 × TP1 × DP4
不同），8 × RTX PRO 5000 72GB（SM120 / PCIe）

本文件记录"整条链路的时间花在哪"的实测过程与结论，是
[`PLAN.md`](PLAN.md) §6 第 1 步（步级耗时拆分）的现场数据。步内构成已用
离线重建补上（§8）；线上的逐 kernel trace 仍缺一次带 profiler 的重启（§7），
但已经不再是优化开工的前置。

## 1. 被测配置

```text
pipeline_parallel_size=2, tensor_parallel_size=2, data_parallel_size=2,
enable_expert_parallel=true, all2all_backend=allgather_reducescatter,
VLLM_PP_LAYER_PARTITION=20,20, VLLM_PP_MICROBATCH_BALANCE=pack,
--moe-backend flashinfer_cutlass, --kv-cache-dtype nvfp4_ds_mla,
attention FLASHINFER_MLA_SPARSE_DSV41 + indexer_kv_dtype=mxfp4,
--speculative-config {"method":"dspark","num_speculative_tokens":5},
KVConnector LMCacheMPConnector (kv_both, grpc, use_vmm_api, store_replay_windows),
VLLM_PREFILL_STEP_BUDGET_MS=700, VLLM_PREFILL_FILL_TARGET=1,
VLLM_PREFILL_COST_FIT=1, --async-scheduling,
--max-num-batched-tokens 8192, --max-num-seqs 64, --max-model-len 1048576
SM120 开关：MEGA_GATE=1, INDEXER_LOGITS=1, FP8_CACHE_WRITER=0, CAKE=1
其他 linux 侧：VLLM_MXFP8_GEMM_BACKEND=auto, VLLM_MOE_AG_RS_PACKED_DISPATCH=1
（`--kernel-config` 未设，即 `sparse_indexer_topk_backend=auto` → `persistent`）
```

rank 布局：DP0 = 全局 0–3（PP0: 0,1 → 物理 GPU0,1；PP1: 2,3 → 物理 GPU4,5），
DP1 = 全局 4–7（PP0: 4,5 → 物理 GPU2,3；PP1: 6,7 → 物理 GPU6,7）。

## 2. 测量口径

**只用引擎侧时间戳。** 客户端流式计时不可信：同一次请求，客户端测到 TTFT
1.68 s，引擎直方图记录 ≤0.25 s，差在 HTTP/urllib 的缓冲与合并。可用的三个来源：

| 来源 | 指标 / 位置 | 含义 |
| --- | --- | --- |
| 引擎直方图 | `vllm:inter_token_latency_seconds`（`vllm/v1/metrics/stats.py:599`） | 同一请求相邻两次 iteration 之间（每请求每步一个样本，不是每个 token） |
| 引擎直方图 | `vllm:time_to_first_token_seconds` | 到达 → 首 token |
| 在线 cost fit | 日志 `Prefill cost model fitted: ...`（`vllm/v1/core/sched/scheduler.py:849`） | 每步 fixed / ms per token / 每步周期中位数 |
| 进程级利用率 | NVML `nvmlDeviceGetProcessUtilization` | 每个 rank 的 SM 忙时间占比 |

注意：cost fit 只在系数变化 >5% 时打印，**日志里相邻两行的间隔不是步频**。
另外 GPU 利用率在"空闲"时也不可信，见 §5.2。

读取直方图：

```bash
KEY=$(tr '\0' '\n' < /proc/$(pgrep -f 'vllm serve' | head -1)/cmdline \
  | sed -n '/^--api-key$/{n;p}')
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/metrics \
  | grep 'vllm:inter_token_latency_seconds_bucket'
```

## 3. 整条链路的实测数字

| 环节 | 实测值 | 条件 |
| --- | --- | --- |
| **decode step** | **~19–20 ms/step**，3.0–3.5 token/step | 1 个请求/引擎，2k 与 100k context 相同 |
| decode step | 18.6 → 24–27 → 27–31 ms | 1 → 2 → 4 个请求/引擎（§5.5；50–75 ms 的峰是 pack，见 §5.4） |
| **现网 ITL**（engine 0，28,720 次更新） | ≤25 ms **66.6%**，50–75 ms **31.5%**，0.75–1 s 1.0%，均值 46.4 ms | 线上真实流量 |
| 投机解码 | 接受率 41.2%（2.06 + 1 = 3.06 token/step），逐位 0.73 / 0.52 / 0.37 / 0.26 / 0.19 | `vllm:spec_decode_*` |
| **prefill 冷启动** | **0.072–0.094 ms/token**，固定 ~85 ms（16k → 1.55 s，128k → 10.3 s，256k → 22.6 s） | 单请求，随机 token id 保证 miss（§5.3） |
| prefill 命中 LMCache | ~2–3 µs/token（64k → 0.12 s） | 同一 prompt 第二次 |
| **引擎 cost fit** | fixed **9.5–10 ms/step**，0.06–0.08 ms/token，6.7e-7 ms/token/context，残差 6–8 ms，步周期中位数 8.4–11 ms | 在线拟合，64 步一轮 |
| TTFT | 冷 500 token ≤0.25 s；现网均值 3.5 s（416 s / 119 请求） | 尾部来自 128k–512k 冷 prompt |
| 前缀缓存 | 内部命中 90.8% / 90.5%；LMCache 外部查询 116 万，命中 40.8% | `vllm:prefix_cache_*` |
| KV 占用 | 6.4% / 4.9% | 容量不是瓶颈 |

ITL 直方图（engine 0，n = 28,720）：

| 桶 | 次数 | 占比 |
| --- | --- | --- |
| ≤25 ms | 19,132 | 66.6% |
| 25–50 ms | 129 | 0.45% |
| 50–75 ms | 9,057 | 31.5% |
| 75–150 ms | 34 | 0.12% |
| 150–300 ms | 15 | 0.05% |
| 300–750 ms | 40 | 0.14% |
| 750 ms–1 s | 292 | 1.02% |
| 1–2.5 s | 21 | 0.07% |

双峰、25–50 ms 几乎为空，说明 50–75 ms 那个峰是"步长变了"，不是"kernel 变慢"。

**步长与 ITL 差 2 倍的原因：** `resolve_pp_num_microbatches`
（`vllm/config/vllm.py:139`）默认取 M = pp_size = 2，文档原话是 "a request decodes
once every M steps"。所以引擎 cost fit 的 9.5–10 ms 是**一个 step**，
而单个请求的 ITL = M × step ≈ 19–20 ms，两者一致。调度层任何按 step 计的成本
（每步集合通信、每步固定开销）都要乘 M 才等于用户看到的 TPOT。

## 4. 一个 decode step 里装了什么

PP0 是层 0–19，PP1 是层 20–39（20/20 切分）。每层
（`vllm/models/deepseek_v41/nvidia/model.py:599`）：

```text
mHC pre → [indexer?] → sparse MLA → KV/compress 写回 → MoE gate
        → SP all-gather (TP2) → EP all-to-all dispatch → 专家 GEMM
        → all-to-all combine → SP reduce-scatter → mHC post
```

- indexer 层 `[2,8,14]` 在 PP0、`[20,24,28,32,36]` 在 PP1（`index_topk=512`）；
- `kv_source_layer_ids=[2,8,14,20]`，`engram_layer_ids=[1,14]` 全在 PP0；
- DSpark 3 层 draft + lm_head + sampler + PP token relay 全在 PP1。

只有 6 个 query token 时，所有 GEMM 都是 6 × 5120 的瘦矩阵，模型权重流量占主导，
每层真正花掉的是**4–6 个小集合通信的延迟**（EP dispatch/combine 走 PCIe，
SP all-gather/reduce-scatter 走 TP）。20 ms ÷ 40 层 ≈ 0.5 ms/层，这个量级只可能是
通信与启动延迟。可验证的推论：context 从 2k 拉到 100k，ITL 桶一个没动（实测）。

## 5. 四个系统级发现

### 5.1 PP 两级不均衡约 20%

同一 step 内，PP1 的 SM 忙时间占比稳定高于 PP0（NVML 按进程采样）：

| 时间窗 | DP0 PP0 (GPU0,1) | DP0 PP1 (GPU4,5) | DP1 PP0 (GPU2,3) | DP1 PP1 (GPU6,7) |
| --- | --- | --- | --- | --- |
| 无请求（11:38–11:39） | 76.3 / 76.5 | 95.7 / 95.5 | 83.1 / 82.8 | 95.5 / 95.2 |
| 有请求（11:37:28–36） | 89.1 / 89.4 | 94.2 / 94.0 | 69.0 / 67.6 | 93.8 / 94.0 |

PP1 满载 (94–96%)，PP0 只有 68–89%，即 PP0 每步空转 10–25%。原因与层数无关（两边
都是 20 层），而是 PP1 多背了 2 个 indexer 层、3 层 DSpark draft、lm_head 与采样。

**切点不能挪。** 直觉上 `VLLM_PP_LAYER_PARTITION=22,18` 能把 PP0 加重，但切点
必须落在 KV source 层上：cut=22 时 `get_pipeline_boundaries`
（`vllm/models/deepseek_v41/nvidia/pipeline.py:95-120`）会判定 layer 20 的
kv source / index source / candidate source / indexer-K source 四个生命周期全部
跨过切点，`needs_relay=True`，随后 `vllm/models/deepseek_v41/nvidia/model.py:893`
直接抛 `NotImplementedError`——cross-stage cache relay 要求
"PCP=DCP=1, without microbatching, speculative decoding or KV transfer"，
而本部署同时有 DSpark 与 LMCache。

也就是说 **20/20 不是可调参数**，是被 `max(kv_source_layer_ids)=20` 钉死的：
切在 20 上 crossing 集合为空，relay 不启用。即便将来放开 relay，它还要把 source
层的 cache 复制进下一 stage（`model.py:1081` 的 `replica`），那是多一份 cache
内存加一次关键路径上的跨 socket 同步；相比之下 PP 每步只传 6 × 5120 的 hidden
states（约 60 KB），**挪切点本身并不增加 PP 的带宽**，代价在 relay 机制上。

不动切点可以试的（都需要重启 A/B）：

| 手段 | 预期 | 代价 |
| --- | --- | --- |
| `VLLM_PP_NUM_MICROBATCHES=3` | 不是治不平衡的手段：`SampledTokenRing` 深度 = M，sample 要等 M 步才回到 stage0，每个 step 每 stage 仍只推进一个 microbatch；实际效果是把请求按 3 组轮转，per-step token 降到约 2/3，而每层的集合通信与 mHC 是延迟主导、不随 batch 缩小，PP1 多出的 5 个 indexer + 3 层 draft + lm_head/sampler 也不缩小 → 不平衡比例反而变差 | ITL 上限变成 3 × step。它是吞吐/延迟旋钮，要 A/B 才能下结论 |
| k=5 → 4 或 3 | PP1 关键路径少 1–2 层 draft 与 verify token | 逐位接受率 0.26 / 0.19，token/step 3.06 → 约 2.6–2.8 |
| `VLLM_PP_MICROBATCH_BALANCE=balance` | 消掉 50–75 ms 峰（§5.4） | free step 不再塞 prefill，TTFT 与吞吐变差 |
| 切点 22/18 | 需先支持带 spec decode 的 relay | 目前起不来 |

### 5.2 空闲时引擎仍在整步空跑

0 请求、`num_requests_running/waiting` 全 0、`generation_tokens` 与
`iteration_tokens_total_count` 冻结时，8 个 rank 里 6 个仍然是 99% SM，
单卡 170–223 W，整机约 1.5 kW。

机制：`_has_global_unfinished_reqs`（`vllm/v1/engine/core.py:2920`）在非同步步
无条件 `return True`，而 `dp_sync_interval` 默认 16，于是每个同步步之间忙等循环都会
走 `execute_dummy_batch()`（`vllm/v1/worker/gpu_worker.py:1783` →
`_dummy_run(6, uniform_decode=True)` → 1 请求 × 6 token 的真实图重放）。
同步步本该收敛到 pause，实际没有收敛（`pending_pause` 共识或 connector 的
pending push work 之一），引擎就一直整步重放。

影响：任何"GPU 利用率 / 功率 / 温度"观测在空闲时测量的都是这条 dummy 循环；
它也是 §2 里所有 idle 数据的来源。是否属于 bug 需要单独确认，与 TPOT 无关。

### 5.3 冷 prefill：常态 0.072–0.094 ms/token，另有未解的 5× 慢态

引擎直方图逐请求取差（TTFT `count` 从 0→1，`sum` 即该请求的 TTFT；
`/tmp/probe_live.py a2`），随机 token id 保证 miss：

| prompt | 冷 TTFT | µs/token | 命中后 TTFT |
| --- | --- | --- | --- |
| 1k | 156 ms | 152 | 89 ms |
| 4k | 532 ms | 130 | 96 ms |
| 16k | 1.55 s | 95 | 98 ms |
| 64k | 5.07 s | 77 | 117 ms |
| 128k | 10.3 s | 78 | — |
| 256k | 22.6 s | 86 | — |

固定开销约 85 ms（队列/调度唤醒），斜率 72–94 µs/token，与引擎 cost fit
（`VLLM_PREFILL_COST_MS_PER_TOKEN=0.021`、拟合 0.06–0.08 ms/token）一致；
262k 起略超线性（attention 随 ctx 涨）。每个请求的 prefill 都在**一次
iteration** 内算完（`iteration_tokens_total` 单样本 = prompt 长度）。

命中路径约 2–3 µs/token（64k 只要 0.12 s），所以 90% 命中率下 TTFT p50 的
0.25–0.5 s 主要是固定开销 + 调度，不是搬运。

**TTFT 尾部不需要额外机制解释**：0.75–1 s 是 ~10k token 的冷 prefill，
10–40 s 是 128k–512k 的冷 prefill；1% 的长 prompt 自然落在这里。并发解码的
干扰也不大：16k 冷 prefill 在 4 个解码请求背景下 1.55 → 1.65–1.9 s（+10–25%）。

注意：早先记录的「0.47–0.52 ms/token、比 cost fit 高 6–7 倍、归因 LMCache 写入」
**不能稳定复现**，但也不是纯误差：冷 prefill 是**双峰**的
（2026-11-21 16:20 – 11-22 00:28，跨 4 个进程，同一个 16k shape，随机 token id
保证 miss）：

| 状态 | 16k 冷 TTFT | µs/token | 出现条件 |
| --- | --- | --- | --- |
| 快（常态） | 1.47–1.55 s | 90–95 | 连续打请求时稳定 10–13 次 |
| 慢 | 6.8–8.1 s | 415–494 | 一串请求里的**头一两个**冷 prefill，之后自己恢复 |

慢态期间：GPU 利用率 30–65%、单卡 130 W（不是算力打满）、NVMe 零读、
LMCache `l1_staging_bytes=0`、store/retrieve 无积压、`num_requests_running`
`waiting` 全 0、日志除 cost fit 外无任何输出；慢态下 4 个解码请求在跑时同一个
16k 反而只要 1.9 s。它**早于** `per_row` 与 k=7 就存在：11:17 与 14:45 两个
进程自己的 cost fit 日志里 p90 已达 0.62–0.66 ms/token（max 1.15/1.64）。
所以那组 0.47–0.52 ms/token 大概率就是在慢态下取的数，与 `--moe-backend
deep_gemm` 无关。

**未解决，暂停（2026-11-22）**：机制尚未定位。已排除：per_row（见下）、k、
外部请求、NVMe、LMCache 存储/查找积压（Lookup p50 10 ms）、显存/功率/温度限制、
prefill 被拆块（慢态下也是一次 iteration 装完 16385 token）。
120 s 空闲**不触发**（`/tmp/probe_idle.py 120`，两轮都是 1.47 s），10 分钟空闲
的测试在得出结果前按需求中止。下一步只有一条：带
`VLLM_PP_STEP_PROFILE_INTERVAL=1` 重启，把慢 prefill 的时间分到 `execute`
还是 `recv_wait`/`token_wait`（§7）；`py-spy` 附着被沙箱拒绝
（`Permission denied`），没有别的无重启手段。

#### 5.3.1 `per_row` 与 k=7 的线上 A/B（2026-11-21 16:00 – 11-22 00:28）

四次重启的其他 args 完全一致，只差这两个开关：

| 重启 | k | `sparse_indexer_topk_backend` | 现象 |
| --- | --- | --- | --- |
| 14:45 | 5 | 未设 | 基线，16k 冷 1.55 s |
| 16:00 | 7 | `per_row` | 用户报 TTFT 变慢；16k 冷快态 1.51 s |
| 16:50 | 5 | `per_row` | 只留 per_row |
| 17:12 / 00:19 | 5 | 未设 | 两个都去掉，用户观测「正常」 |

**`per_row` 不是慢态的原因**。在 00:19 那台（k=5、无 per_row）上打 12 个 16k 冷
prefill：`#0 7795 ms`、`#1 7196 ms`、`#2–#11 1465–1483 ms` —— 与带 per_row 时
的形态完全一样。`per_row` 对 prefill 的真实代价也测出来了：快态 16k
**1.47 s → 1.51 s（+40 ms，2.7%）**，其中可归因的只有 candidate indexer 的
decode 式 top-k（`[≤8192, 16384]` logits，2 块 × 20 层，0.82 vs 0.67 ms/块 ≈ 6 ms），
其余是状态噪声。prefill 的 indexer top-k 根本不走这个开关
（`forward_prefill` 一律 `ops.top_k_per_row_prefill`，§6.1）。

**被一起回退的 k=7 是真的赚**，且与 TTFT 无关。同一探针（1 请求 × 256 输出，
`/tmp/probe_live.py` 口径）：

| 配置 | 客户端 wall | ITL p50 | token/iteration |
| --- | --- | --- | --- |
| k=7（16:00，同时带 per_row） | **0.95 s** | 17.5 ms | 6.24 |
| k=5（00:19） | **1.78 s** | 17.5 ms | 2.94 |

（1k 上下文下 `per_row` 的收益 <2%，不足以解释这个差。）

每 iteration 的 ITL 一个没多，出的 token 翻倍，生成速度差 1.87×。k=7 的代价只在
KV 容量：可用显存同为 21.67 GiB，容量 7,921,701 → 7,419,949 token（−6.3%），
缓存早 6% 淘汰。**建议下一个窗口单独把 k=7 加回来（不要和 per_row 同一次上）。**

探针方法论：同一 seed 段的 prompt 会命中 LMCache（重跑 warm-up 得到 152 ms，
看起来像「prefill 变快了」），每次实验必须换新 seed 段；判断跑在快态还是慢态
只能看同 shape 冷 prefill 的绝对值，不能看 `/metrics` 累计直方图或 cost fit
（小 prompt 的 fit 恒在 0.07 左右，与状态无关）。

### 5.4 decode 的 50–75 ms 峰不是 kernel

单请求 19 ms、4 请求 25–50 ms，而现网 31% 的更新落在 50–75 ms。这是
`VLLM_PP_MICROBATCH_BALANCE=pack` 的直接后果：它把 decode 全集中到一个
microbatch，**另一个 step 没有 decode**，专门用来塞 prefill chunk
（`VLLM_PP_PACK_DECODE_STEP_PREFILL_TOKENS` / `..._FREE_STEP_PREFILL_TOKENS`，
预算由 `VLLM_PREFILL_FILL_TARGET=1` 与 700 ms 步时间定价）。

所以 decode 请求每 2 步更新一次是 19–20 ms；一旦中间那个 free step 装了 prefill
chunk，它的更新间隔就变成 解码步 + prefill 步 ≈ 50–75 ms。p99 的 1 s 是同一个机制
的尾部。换句话说，**50–75 ms 峰是调度策略换来的 prefill 吞吐，不是 decode 变慢**。

结论：decode 侧任何 kernel 优化最多影响那 66.6% 中的一部分；先解决调度层
（§5.1 的 microbatch 深度）收益更大。

### 5.5 步长 vs 解码行数：4 倍请求只贵 1.6×

`/tmp/probe_live.py a3`：1/4/8 个并发请求（DP2 下每引擎 1/2/4 个，即 6/12/24 行），
各自 1k prompt + 256 输出，逐引擎直方图取差：

| 请求/引擎 | 行数 | ITL 均值 | ITL p50 | token/iteration |
| --- | --- | --- | --- | --- |
| 1 | 6 | 18.5–18.8 ms | 17.5 | 2.4–5.5 |
| 2 | 12 | 24.0–26.8 ms | 27.6–34.8 | 2.7–4.7 |
| 4 | 24 | 26.6–30.6 ms | 28.5–37.2 | 5.4–6.4 |

（token/iteration 随随机 prompt 的接受率变，同一行内不可横向换算 TPOT；
可比的是 ITL。ITL 每请求每 iteration 记一个样本，不是每 token。）

行数 ×4 步长只 +56%（18.6 → 29 ms），不是 ×4。这验证了 §8 的专家流式假设：
每步的固定成本（读 ~18–20 个专家权重、PP 两端通信、mHC、投影）占大头，多一个
请求只多命中几个专家和一点 attention。所以**批大小便宜，k 也便宜**（§9.2 第 5 条）。

## 6. 索引器微基准（不是瓶颈）

单层耗时（µs），1 请求 × 6 行，背景有空闲 dummy 循环争用：

| ctx | DeepGEMM paged logits | FlashInfer SM120 logits | `persistent_topk` |
| --- | --- | --- | --- |
| 4k | 6.1 | 8.4 | 27.1 |
| 32k | 10.3 | 14.6 | 70.5 |
| 128k | 30.6 | 31.0 | 96.9 |
| 512k | 106.4 | 102.7 | 104.6 |

2 请求 × 12 行分别是 6.2 / 16.3 / 48.3 / 172.4（DeepGEMM）、
8.7 / 17.2 / 45.5 / 158.0（FlashInfer）、29.0 / 73.8 / 123.3 / 134.0（top-k）。

128k context 时 PP1 的 5 个 indexer 层约 5 × (31 + 97) = 0.64 ms，占 20 ms 步长的
3%。FlashInfer SM120 logits 在 ≥128k 只是打平、短 context 更慢，且
`indexer_kv_dtype=mxfp4` 下该路径本来就是 inert（`_sm120_paged_logits_width`
返回 None）。MEGA_GATE 的 −2 µs/层、FP8 writer 的 +130 µs/步，放在 20 ms 步长上
都是噪声，见 `../dsv41-sm120-knob` 相关记录（`vllm-dsv41-2026*.log` 同目录）。

### 6.1 decode top-k 也有 2–3× 的后端差（2026-11-21）

`sparse_indexer_topk_backend` 默认 `auto`，在 SM120 上解析成 `persistent`：
cooperative 限 SM90 且显式排除 SM11x/SM12x，deep_select / flashinfer_gvr2 要求
SM100 family（`indexer_topk.py:420-470`、`:253-260`）。可用的另外两个是
`per_row`（`top_k_per_row_decode`）和 `flashinfer`（`top_k_ragged_transform`）。
`/tmp/bench_indexer_topk.py`，同一 fp32 `[rows, max_model_len]` logits，graph replay：

| rows | ctx | `persistent`（现网） | `per_row` | `flashinfer` |
| --- | --- | --- | --- | --- |
| 6 | 1k | 21.4 | 21.9 | 11.8 |
| 6 | 4k | 25.3 | 21.9 | 13.5 |
| 6 | 32k | 61.4 | 33.4 | 29.4 |
| 6 | 128k | 97.9 | 37.5 | 33.5 |
| 12 | 32k | 64.1 | 41.5 | 41.4 |
| 12 | 128k | 120.3 | 46.4 | 47.6 |
| 24 | 32k | 122.3 | 50.4 | 70.9 |
| 24 | 128k | 236.7 | 54.6 | 78.3 |

`persistent` 随 ctx 线性涨（radix 要扫满 `max_seq_len`），`per_row` 基本不变。
三个后端选出的 key 集合逐值相同（`/tmp/verify_per_row_2d.py` 也验了线上
`seq_lens=(1,6)`、`next_n=6` 的 2D 形态），只是索引顺序不同——下游按集合做
sparse attention，结果不受影响，但不是逐位相同。

1 请求/引擎（6 行）下 PP1 的 5 层可省：`per_row` −0.02 / −0.14 / −0.30 ms
（4k / 32k / 128k），`flashinfer` −0.06 / −0.16 / −0.32 ms。4 请求/引擎（24 行）
时 `per_row` 仍然最好，`flashinfer` 反超不了。切换**不改代码**，但要走
`--kernel-config '{"sparse_indexer_topk_backend":"per_row"}'`，即下一次重启才生效。

**线上已经试过（16:00 与 16:50 两次重启，见 §5.3.1）**：`per_row` 只影响 decode，
prefill 侧的代价是 16k 冷 prefill **+40 ms（2.7%）**，而且它**与 §5.3 的 5× 慢
prefill 无关**（去掉它之后慢态照样复现）。收益（128k 下 −0.3 ms/步，TPOT 约
−5%）仍然成立，但每次重启只能单变量：16:00 那次它与 k=7 一起上、17:12 两个
一起下，所以这两个开关的净收益都还没被单独验证过。

## 7. 盲区：kernel 级拆分需要一次带 profiler 的重启

容器内无法附加 profiler（无 `CAP_SYS_PTRACE`、`ptrace_scope=1`、
`perf_event_paranoid=4`），线上进程 `VLLM_PP_STEP_PROFILE_INTERVAL=0` 且启动时没有
`--profiler-config`，因此 `vllm/v1/worker/gpu/pp_step_profiler.py` 的
`execute / recv_wait / send_wait / token_wait / sample / draft / busy / gpu`
拆分拿不到，torch trace 也拿不到。

补测方式（需要维护窗口）：

```bash
# 1) 启动时打开 profiler 与步级 profiler
VLLM_PP_STEP_PROFILE_INTERVAL=200 vllm serve ... \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}'

# 2) 带 --profile 压测
vllm bench serve --profile ...

# 3) 按 kernel 分类统计（pp_p2p / ep_all2all / pp_token_relay / indexer /
#    attention / moe / mhc / gemm）
python examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/analyze_traces.py \
  --dir /tmp/dsv41-traces
```

这一步能验证 §8 的重建：集合通信在步内的真实 overlap 情况。

## 8. 步内构成：离线重建（2026-11-21）

线上进程挂不了 profiler（§7），所以用 `/tmp/dsv41_step_parts.py` 在同样的 4 张卡上
按线上 shape 重建每一项：CUDA graph replay 计时（线上 decode 走 FULL graph；eager
计时会把每次 ~8 µs 的 launch 开销算进去，mHC 因此虚高 4 倍），fp8 权重按字节折半
换成 fp4。

EP 组是 4 个 rank（`ep_ranks=[0,1,4,5]`）；DP2 每 engine 1 个请求时 EP 组内共 24 行
（12 行唯一，TP2 重复），top-6 → 72 个 `(row, expert)` 对 ÷ 4 rank ≈
**18 个 expert/rank/layer**。

| 项 | 每次 | 每层 | ×20 层 | 占 12–13.7 ms |
| --- | --- | --- | --- | --- |
| 专家权重流式（18 experts × 17.7 MB fp4） | 302 µs | 302 | **6.04 ms** | 44–50% |
| 注意力投影（线上 kernel：cutlass MXFP8 + bf16 wo_a） | 194 µs | 194 | 3.87 ms | 28–32% |
| EP 集合通信（dispatch all-gather + combine reduce-scatter） | 54–135 µs | 54–135 | 1.08–2.70 ms | 9–20% |
| mHC（2 个 boundary，graph 内 18 µs/个） | 36 µs | 36 | 0.72 ms | 5–6% |
| lm_head（TP2 shard 64640×5120 fp8） | 317 µs | 1 次 | 0.32 ms | 2–3% |
| 合计 | | 586–667 | 12.0–13.7 ms | 100% |

集合通信这一行不是常数，取决于两个 DP engine 的 batch 是否等长（§8.2）：等长时
dispatch 走一次 `ncclAllGather`（30 µs/层），不等长时 `pynccl.all_gatherv` 用
每 rank 一次 `ncclBroadcast` 模拟（110 µs/层）。

投影这一行是按线上 kernel 与线上 shape 直接测出来的（§8.1），旧版的 58 µs / 1.16 ms
是用 `torch._scaled_mm` 代理、且 shape 取错（`o` 用了 `[5120,1024]`、漏了 wo_a）得到的
低估值。换成实测后总和 12.0–13.7 ms **超过**了原来「ITL / 2 ≈ 9.5 ms」的锚点，
锚点本身要重算：

- 非流式 128 token 请求实测 60 个 iteration / 1.20 s → **20.3 ms/iteration、
  2.13 token/iteration、TPOT 9.5 ms**（另一条 prompt 上 18.9 ms/iteration、2.95 token）。
- 若一个 iteration 里 PP 两级串行，单级通过 ≈ 10.2 ms，模型算的 12–13.7 ms 仍高 2–3.5 ms。
  只可能来自两处：18 个专家的流量被 L2 部分吸收，或投影跑在 aux stream 上、
  与集合通信窗口重叠。要一次带 profiler 的重启才能定死（§7）。

indexer（§6，0.64 ms）与 sparse MLA 未单列，填在上面 2.4 ms 的误差带里。

三条硬约束：

1. **MoE 已经在带宽墙上。** 纯 copy 128 MiB 是 1088 GB/s，专家 GEMM 是
   1054 GB/s（fp8 per-expert 循环）——6 ms 不是 kernel 慢，是要读的字节数。
   每层 18 个专家 × 17.7 MB 已经是 fp4 压缩后的量。唯一杠杆是摊薄：
   每 EP 组从 12 行增到 24 行，命中专家约 18 → 34（流量 +90%），而 token 是 4 倍。
2. **集合通信是延迟主导。** 4 B 的 all-gather 就要 33 µs（PCIe 固定延迟），
   30 KB 的 dispatch + 245 KB 的 combine 在 graph 内是 54（等长）/135（不等长）µs。
   20 层 = 1.1–2.7 ms，和 payload 大小基本无关，只和**通信次数**有关（§8.2）。
3. **可优化空间 ~3–3.5 ms。** 投影换 backend −2.07 ms/步（§8.1、§9.2 第 1 条），
   Ragged dispatch 打包 −0.9 ms（不等长时；§8.2、§9.2 第 2 条），mHC / indexer
   各还有 ~0.2–0.3 ms。MoE 与 lm_head 没有空间。

### 8.1 投影的实测：线上 kernel、M=6、冷权重（`/tmp/bench_attn_proj.py`）

shape 不是猜的，是从线上 FlashInfer autotune cache 反查确认的：
`/cache/jit/vllm/flashinfer_autotune_cache/0.7.5.dev20261121/120a/*/autotune_configs_dp*_rank*.json`
里正好有 `(1,5120)×(5120,1792)`、`(1,1280)×(1280,16384)`、`(1,4096)×(4096,5120)`
三条 `mxfp8_gemm`，runner 全是 `CutlassMxfp8GemmRunner`，且**只调了 M=1 这一档**。

单层四个投影（192 MiB flush 掉 L2 后的 cold 计时，CUDA graph replay）：

| 投影 | N × K | 权重 | 线上 kernel（cutlass） | `backend="auto"` |
| --- | --- | --- | --- | --- |
| `fused_wqa_wkv`（replicated） | 1792 × 5120 | 9.0 MiB | 26 µs | 17 µs |
| `wq_b`（TP2） | 16384 × 1280 | 20.6 MiB | 36 µs | 23 µs |
| `wo_a`（Emulation → bf16，`torch.bmm` 4 组） | 4 × 1024 × 4096 | 32.0 MiB | 41 µs | 41 µs |
| `wo_b`（TP2） | 5120 × 4096 | 20.6 MiB | 63 µs | 22 µs |
| 合计（整链一条 graph） | | 82 MiB | **194 µs** | **90 µs** |

roofline：82 MiB 按 1259 GB/s（同机 192 MiB 写的实测值）是 69 µs，所以线上配置只跑到
447 GB/s，换 `auto` 就能到 960 GB/s。逐 shape 扫 M=1…1024，`auto` **每一档都不慢于**
cutlass（M=6 时 2.6× / 6.3× / 6.9×），输出逐位相同（max abs diff = 0）。

原因在代码里：`FlashInferCutlassMxfp8LinearKernel.apply_weights` 写死了
`backend="cutlass"`（`vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:152`），
而 M=6 这个档不在 autotune 的 bucket 里，FlashInfer 会打印
`No tuned config covers mxfp8_gemm input_shapes=((6,5120),(5120,1792),…) …
falling back to runner=CudnnMmMxfp8GemmRunner tactic=-1` —— 这条「fallback」恰好比
被写死的 cutlass 快 2–7 倍。

wo_a 为什么是 bf16：`init_mxfp8_linear_kernel` 对 `bmm_batch_size` 非空的层只考虑
`DeepGemmMxfp8BmmLinearKernel` 与 `EmulationMxfp8LinearKernel`，前者要求
`is_device_capability_family(100)`，SM120 被挡在外面，于是落到 emulation，
`VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=True` 在加载时把权重反量化成 bf16（32 MiB/层）。
`dsv41_o_proj` 里那条 fp8 `fp8_einsum` 分支要求 `wo_a.weight.dtype == fp8`，同一条门也进不去。

两条已排除：

- **wo_a 改 fp8 走 4 次 `mm_mxfp8`**：实测 38 µs（`auto`）对今天 bf16 bmm 的 41 µs，
  省 3 µs/层（0.06 ms/步），不值得为它加代码。
- **放开 `DeepGemmMxfp8BmmLinearKernel` 的 arch 门**：DeepGEMM 的 grouped einsum 在
  SM120 上直接 `CUDA error: unspecified launch failure`（只编了 sm100 的 kernel），
  这条门不是保守，是必要的。

**已落地**：`VLLM_MXFP8_GEMM_BACKEND`（默认 `cutlass`，设成 `auto` 走 cudnn）。
经 `FlashInferCutlassMxfp8LinearKernel.apply_weights` 验证，三个线上 shape 在 M=6 上
两个 backend 输出逐位相同（`/tmp/verify_mxfp8_env.py`）。注意 cudnn 的 plan 要在
graph capture 之前建好——线上 warmup 跑过 decode 档才能安全。

### 8.2 EP 集合通信的两种形态：等长 / 不等长（`/tmp/verify_packed_dispatch.py`）

旧表里「3 × all-gather + 1 × reduce-scatter = 83 µs」是按**逐次 eager、未分组**的
`dist.all_gather_into_tensor` 测的，两处都不对：线上 `AgRsAll2AllManager.dispatch`
把三个张量放在同一个 `pynccl` group 里（`cuda_communicator.all_gatherv`），而且
decode 全程在 CUDA graph 内。4 rank NCCL、graph 内重测：

| 场景 | 3 个张量 | 打包成 1 个 | combine RS |
| --- | --- | --- | --- |
| 等长（`sizes` 一致 → 一次 `ncclAllGather`/张量） | **29.9 µs** | 30.6 µs | 24.2 µs |
| 不等长（`sizes` 不一致 → 每 rank 一次 `ncclBroadcast`/张量） | **110.5 µs** | **65.4 µs** | 24.2 µs |

不等长才是贵的那种：`pynccl.all_gatherv` 没有 `ncclAllGatherv` 可用
（NCCL 2.32.3 里没有这个符号），ragged 版本被模拟成「每 rank 一次 `ncclBroadcast`」，
三个张量就是 3 × 4 = 12 次广播。两个 DP engine 的 batch 长度不同时就走这条路径。

修法是把 dispatch 的张量按行拼进一个 `uint8` buffer 再做一次 `all_gatherv`
（`VLLM_MOE_AG_RS_PACKED_DISPATCH=1`），12 次广播变 4 次，回来再按切片
`.contiguous().view(dtype)` 拆开。4 rank 实测：**110.5 → 65.4 µs（graph）、
257 → 166 µs（eager，prefill 也受益）**，每个 rank 上的行与原路径逐位相同。
等长时几乎不变（30.6 对 29.9，噪声内），所以这个开关在不等长的步上省
**45 µs/层 = 0.9 ms/步**，在等长的步上什么也不变。

pack/unpack 的代价已经算在 65.4 µs 里：28 行 × 10288 B ≈ 288 KB 的拷贝，graph 内 4 个
kernel。替代方案 `ncclAllGatherv` 不存在，`all_gatherv` 的 padded 实现要重排输出，
都不如打包干净。

## 9. 优化方向

先固定换算关系，后面所有取舍都按它折算（用 2026-11-21 现网实测的 iteration 口径）：

- **一次 engine iteration = 20.3 ms**（非流式 128 token 请求：60 iteration / 1.20 s），
  一次产出 2.13 token → **单请求 TPOT 9.5 ms**；另一条 prompt 上是 18.9 ms/iteration、
  2.95 token/iteration。旧口径「步长 9.5 ms、3.06 token/step、TPOT 6.2 ms」对应的是
  acceptance 更高、且把 iteration 当成两次 PP 通过的情形，按今天的实测偏低。
- **TPOT 随输出变长而下降**（线上实测）：上面 20.3 ms/iteration 那组数字来自 128
  token 的请求，短请求里 prefill 之后头几次 iteration 还在填 draft 流水与 PP 气泡，
  固定开销摊到每个 token 上就把 TPOT 抬高；长输出跑到稳态后 TPOT 更低。所以
  9.5 ms 是**偏保守**的锚点，A/B 对比要用 ≥512 token 的输出，否则差异会被启动
  开销吃掉。
- **单级通过时间每减 1 ms ≈ iteration 减 2 ms ≈ TPOT 减 ~0.94 ms（约 −10%）。**
- 现网 ITL 均值 46 ms 与上面的下限之间的差额仍主要来自调度，不是算力：
  31.5% 是 pack 的 free-step prefill（§5.4），1.0% 是长 prefill（§5.3）。

所以是两条互不替代的线：**降步长**（§8 的 3–3.5 ms 可优化部分，§9.2）和
**降调度附加**（pack 与 free-step 分配，§9.3）。混在一起优化会互相掩盖。

### 9.1 立刻可做（不改代码、不重启）

| 事项 | 目的 | 预期 |
| --- | --- | --- |
| 离线重建已经完成，见 §8 / §8.1 / §8.2 | 步内拆分不再需要 profiler 才能开工 | 已定出 44–50/28–32/9–20/5–6/2–3 的排序 |
| indexer top-k 后端对比（`/tmp/bench_indexer_topk.py`） | 5 层 × 97 µs = 0.5 ms（§6） | **已完成**，见 §6.1：`per_row` −0.30 ms / `flashinfer` −0.32 ms（128k）。16:00/16:50 已上线试过，17:12 回退（§5.3.1）：prefill +40 ms，未单独归因 |
| TTFT 尾部归因：prompt 长度 × 冷/热矩阵（`/tmp/probe_live.py a2`） | 拆开「prefill 计算」与「LMCache 写入」 | **已完成**，见 §5.3：冷 prefill 0.072–0.094 ms/token，写缓存不在 TTFT 路径上；尾部就是 128k–512k 长 prompt |
| 单请求 vs 4/8 请求的步长斜率（`/tmp/probe_live.py a3`） | 验证专家流式假设（§8 约束 1） | **已完成**，见 §5.5：行数 ×4 步长只 +56%，批大小与 k 都便宜 |

### 9.2 代码级（按 §8 的占比排序）

1. **注意力投影：把写死的 `backend="cutlass"` 换成 `auto`**（3.87 ms，28–32%）：
   §8.1 的整链实测 194 µs/层 → 90 µs/层，**−2.07 ms/步**（ITL −2×2.07 ms，
   TPOT 约 −18%）。输出逐位相同，M=1…1024 全档不亏。**已实现**：
   `VLLM_MXFP8_GEMM_BACKEND=auto`（默认保持 `cutlass`，便于回滚与 A/B）。
   因为 cudnn 的 plan 必须在 graph capture 之前建好，重启后要确认线上 warmup
   覆盖到 decode 档。顺带把 autotune 的 tuning bucket 扩到 decode token 数
   （现在只调了 M=1），那条 fallback warning 就是 bucket 没覆盖。
   **14:45 重启已带上**（与第 2 条同时开），重启后单请求 ITL p50 17.5 ms
   （§5.5），比之前的 19–20 ms 低；但两条一起开无法单独归因，要分项需要
   再各重启一次。
2. **Ragged dispatch 打包成一次 all-gather**（1.08–2.70 ms，9–20%）：三个张量本来
   就在同一个 `pynccl` group 里，不是「三次独立 AG」；贵的是 `sizes` 不等长时
   `all_gatherv` 退化成 3 × 4 次 `ncclBroadcast`（§8.2）。按行打包后一次广播/rank，
   **110.5 → 65.4 µs/层 = −0.9 ms/步**（只在不等长的步上），数值逐位不变。
   **已实现**：`VLLM_MOE_AG_RS_PACKED_DISPATCH=1`（默认关），14:45 重启已带上；
   只在两个 DP 引擎的 token 数不等时才有收益。
3. **mHC**（0.72 ms，6%）：graph 内 18 µs/boundary 已经不高，`hc_sinkhorn_iters=20`
   是唯一旋钮（有精度风险）。注意下一层的 mHC pre 依赖 combine 之后的 MoE 输出，
   基本没有可重叠的窗口，不要按「藏进通信窗口」来设计。
4. **lm_head**（0.32 ms，仅 PP1）：331 MB fp8 一次读完，已在带宽墙，TP2 已切开，
   没有空间。
5. **MoE**（6 ms，48%）：无 kernel 级空间（§8 约束 1）。它只吃两件事：
   批大小（摊薄）与 tokens/step。由此得一个反直觉结论——**加大 k 是便宜的**：
   多 1 行 verify 只让命中专家 +2~4%（约 +0.1 ms/步），token/step 增加
   p6 ≈ 0.1 以上就划算；反之 k=5 → 4 省不下步长（token 项只有 0.07 ms/token），
   只降 token/step。

### 9.3 一次维护窗口（带 profiler 重启）

这不是开工前置了（§8 已经给出拆分），但 trace 仍能验证 collectives 与投影的
overlap 情况，见 §7。同窗口可做：

- `VLLM_PP_MICROBATCH_BALANCE=balance`：预期消掉 31.5% 的 50–75 ms 峰（ITL 全部
  回到 ~20 ms），代价是 free step 不再塞 prefill → TTFT 与吞吐变差。**纯延迟优先时选它。**
- k=5 → 6：按 §9.2 第 5 条，步长 +0.1 ms、token/step 若 >3.16 即 TPOT 净赚；
  用 `enable_adaptive_verification` 可以由接受率在线决定。**k=7 已经线上验证过**
  （§5.3.1）：ITL p50 不变、生成速度 1.87×，代价只是 KV 容量 −6.3%；建议重启用
  k=7 单独上，不要和 `per_row` 同一次。
- `VLLM_PP_STEP_PROFILE_INTERVAL=1`：唯一能定位 §5.3 那 5× 慢 prefill 的手段
  （`execute` vs `recv_wait`/`token_wait`），重启即生效，代价是每个 worker 每 N 步
  打一行。

### 9.4 调度与运维

- `VLLM_PP_MICROBATCH_BALANCE` 的两种取值是 §9.3 的 A/B；`VLLM_PP_NUM_MICROBATCHES`
  是吞吐/延迟旋钮，不是治不平衡的手段（§5.1 表）。
- 空闲 dummy 循环（§5.2）：不影响 TPOT，但让所有 GPU 利用率观测失真、整机待机
  ~1.5 kW。需要查 `pending_pause` 共识或 connector push work 为什么没让引擎静默。

### 9.5 已排除（不要再花时间）

- `VLLM_PP_LAYER_PARTITION=22,18`：relay 起不来（§5.1）。
- `VLLM_PP_NUM_MICROBATCHES=3` 当不平衡修复：见 §5.1 表。
- MoE kernel 级优化：已在 1054 GB/s，纯 copy 只有 1088 GB/s（§8）。
- indexer SM120 FlashInfer logits（`indexer_kv_dtype=mxfp4` 下 inert，≥128k 只打平）、
  FP8 cache writer（每档都更慢）、mega gate（−2 µs/层）：放在 20 ms 步长上都是噪声（§6）。
