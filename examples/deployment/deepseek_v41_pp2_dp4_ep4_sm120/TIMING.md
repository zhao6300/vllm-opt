# DeepSeek-V4.1 线上链路耗时实测

**状态：** 实测记录，2026-11-21
**被测部署：** 线上 PP2 × TP2 × DP2 + EP（与本目录 README 的 PP2 × TP1 × DP4
不同），8 × RTX PRO 5000 72GB（SM120 / PCIe）

本文件记录"整条链路的时间花在哪"的实测过程与结论，是
[`PLAN.md`](PLAN.md) §6 第 1 步（步级耗时拆分）的现场数据。步内构成先由离线重建
补上（§8），2026-11-22 又用一次带 profiler 的重启补了 PP0 的逐 kernel trace（§7）；
trace 会杀死服务、绝对时间失真，读数前先看 §7.2–7.3。

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

**进程版本**：本文数字来自两个进程，引用时看清出处——旧进程
`vllm-dsv41-20261122-041924.log`（步级 profiler 开、`VLLM_PP_MICROBATCH_BALANCE=1`）
和 2026-11-22 12:16 重启的进程 `vllm-dsv41-20261122-121614.log`（多三个 flag：
`--prefill-schedule-interval 1`、`--enable-cumem-allocator`、`--numa-bind`；
`VLLM_PP_STEP_PROFILE_INTERVAL=0`，即步级 profiler 关闭）。重启后没有重新做
步级拆分，两版在 TTFT 上同量级（冷 1 token 132 vs 150 ms、命中 16k 181 vs 91–131 ms）。

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

torch trace 只用来取**占比**：prefill trace 会把 TTFT 放大 25×，decode trace 会把
broadcast 等待放大 30×，且 decode trace 在 stop 时会打死服务（§7.2–7.3）。

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
| **现网 ITL**（engine 0，28,720 次更新） | ≤25 ms **66.6%**，50–75 ms **31.5%**，0.75–1 s 1.0%，均值 46.4 ms | 线上真实流量，`pack` 时期 |
| **现役 ITL**（两引擎，49,190 次更新） | ≤25 ms **98.9%**，50–75 ms **0**，>75 ms 16 次（0.03%），均值 **22.7 ms** | `balance`，2026-11-22 进程；流量是本机探测（1k–32k、256 输出），没有长 prompt 尾 |
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
2026-11-22 的 trace 把它量成 **+1.6 ms/step 的纯计算**（同窗口 stage 1 9.45 vs
stage 0 7.88 ms）：lm_head ~1.0、多 2 层 indexer +0.35、draft 相关（`gemvx`、
`_prepare_dflash_inputs`）~0.5；**MoE 两边相当**（4.35 vs 4.83），不是 stage 1 更重
的地方（§7.4）。

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
| `VLLM_PP_MICROBATCH_BALANCE=balance` | **已上线**：50–75 ms 峰为 0，ITL 均值 22.7 ms（§5.4） | free step 不再独占 prefill，TTFT 与吞吐变差（未量化） |
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

**2026-11-22 12:2x 复核（重启后的进程）**：这次空闲态**收敛到了 pause**——10 s
采样里 8 张卡 `utilization.gpu` 均值 0.6–2.7%（尖峰是采样窗噪声）、单卡
70–104 W、整机 **698 W**，`power.draw` 后半段读出 0 W。所以上面的 1.5 kW /
99% SM 不是稳态：同一次采样里，发完一个请求后的整机功耗是 **1206 W → 10 s 内
衰减到 630 W**，也就是"没收敛到 pause"和"刚跑完一批的头几秒"都会看到 ~1.2 kW。
别把 1.4–1.5 kW 当成恒定的空闲开销。一直没变的只有 host 侧：8 个 worker 在
空闲时仍各烧 ~1 核（合计 8.7 核），来自忙轮询本身，与 GPU 是否在工作无关。

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

**固定开销在步级是这样分布的**（2026-11-22 08:2x–08:4x，进程
`vllm-dsv41-20261122-041924.log`，`VLLM_PP_STEP_PROFILE_INTERVAL=1`，随机
token id 保证 miss，`/tmp/probe_floor3.py`）：

| 冷 prompt | TTFT | queue | prefill | pp0 execute | pp1 execute |
| --- | --- | --- | --- | --- | --- |
| 1 token | 150–153 | 7 | 141–143 | 67.9–69.7 | 63.7–65.0 |
| 64 | 155 | 7 | 147 | 135.3 step / exec 0.00（这段没记到） | 142.8 step / exec 0.00 |
| 512 | 152–155 | 5 | 144–155 | 77.6–77.9 | 66.5–68.0 |
| 1024 | 174 | 8 | 162 | 77.4 | 77.9 |
| 4096 | 551 | 5 | 533 | 262.2 | 同量级 |

1 token 与 1024 token 的每级 execute 几乎一样（68–78 ms）——小 prompt 的
prefill 是**每级固定 ~65 ms**，token 项斜率（~0.05–0.06 ms/token/级）要到
2k 以上才显出来。重启后的进程（12:16）用 client 墙钟复核，曲线一致：
64 = 159 ms、1024 = 166、2048 = 288、4096 = 590、8192 = 853，即固定 ~155 ms
（两级串行）+ 边际 0.06–0.12 ms/token。

这 65 ms 只在 **prefill 形态的 step** 上付：同样的 1 个新 token，只要它前面
已经有 KV（满命中），`model_runner` 会走
`prefill_runs_as_decode`（`model_runner.py:1497-1504`：
`one_token_ext = num_new_tokens == 1 且 num_computed_prefill_tokens > 0`），
整个 batch 就 decode-graph eligible，pipeline 只要 ~20–40 ms。冷 1 token 是
`num_computed_prefill_tokens == 0`，只能走真 prefill，于是付满 65 ms/级。
这 65 ms 的成因 2026-11-22 12:5x 已在代码侧定位：本进程的 prefill 形态 step
**没有 CUDA 图**，是 eager 逐层 launch 的 host 开销（§5.7）。

上面「固定 ~85 ms」是大 prompt 段线性拟合的截距，别把它当成小 prompt 的底价：
1k 以下实测的固定项是 **130–155 ms**（两级各付一次 65–78 ms 的 prefill 形态
step），1k token 的斜率根本摊不掉它，所以 1k 的 152 µs/token 看起来偏高。
斜率 72–94 µs/token 与引擎 cost fit
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
**该重启已于 2026-11-22 04:19 完成**（`vllm-dsv41-20261122-041924.log`，步级 profiler
与 torch profiler 同时开着），此后测到的冷 prefill 都在快态（16k 1.45 s、32k 3.06 s），
慢态没有复现，所以拆分手段已就绪但还没有慢态样本可拆（§7）。

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
缓存早 6% 淘汰。

**但这个探针只在 1 请求/引擎上成立，线上开 k=7 会整体变慢**（用户的实测），
原因是 k 的成本结构与探针的前提相反：

- draft 链是**逐位串行**的（`spec_decode/dspark/speculator.py:334` 的
  `for i in range(num_speculative_steps)`），每一位都要跑 3 层 draft + 一次
  129k 词表 lm_head，全在 PP1（本来就是关键路径）上；k=5 → 7 是 +2 段串行 draft。
- 第 6/7 位的逐位接受率是 0.19 往下（§3），也就是多出的两段"用一整段 draft +
  一行 verify 换 ~0.1–0.2 个 token"。1 请求时步长被专家权重流式主导，多几行几乎
  免费，所以探针赢；线上每引擎 3–4 个请求 ×8 行时，行数与 draft 段数都按批放大，
  而边际接受率不变 → 净亏。

**正确旋钮是 `enable_adaptive_verification`（当前 `--speculative-config` 里没开，
`VLLM_ADAPTIVE_VERIFICATION_*` 那几个 env 因此全是空转）**：它把整个 batch 的
verify 槽位按"存活概率 / 成本"排序后分配（`adaptive_verification.py:131` 的
`assign_draft_token_budget`，预算由 `:648` 的接受 token/ms 最大化选出），低置信度
请求只留短前缀，多出的槽位给高置信度请求——同样的行数花在更可能被接受的位置。
它要求 FULL CUDA graph（现配置 `FULL_AND_PIECEWISE` 满足）。要上就单变量重启，
用它替代 k=7。

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

**2026-11-22 的对照确认了这一条**：现役进程（`VLLM_PP_MICROBATCH_BALANCE=1`，
即 `balance`）两个引擎共 49,190 次 ITL 更新，**50–75 ms 档为 0**，≤25 ms 占
98.9%，均值 **22.7 ms**——已接近 2 × step 的结构下限（~19–20 ms）。剩下的
16 次 >75 ms 与 1.1% 的 25–50 ms 是探测流量的混入，不是新的模式。
`balance` 只是把 prefill 摊进两个 microbatch、不再独占 free step，所以峰消失
而 TTFT/吞吐的代价仍要单独 A/B；上面 46.4 ms 的数字属于 `pack` 时期，不要与
现役配置混引。

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

**跨引擎对照（2026-11-22，`ragged_plain`）**：DP0 挂 4 个请求、DP1 只挂 1 个，同为
1k prompt / 300–400 输出。ITL 均值 eng0 25.7 ms、eng1 **28.2 ms** —— **只挂一个请求
的引擎并不比挂 4 个的快**。原因是 EP 组 = TP×DP = 4 rank，跨两个引擎
（`parallel_state.py:2404` 按 `dp_size × tp_size` 切 EP 组）；每层 MoE 的
all-gather / reduce-scatter 以及 `sync_cudagraph_and_dp_padding` 的 gloo all-reduce
都要求两端对齐，孤请求替邻居的负载付钱（§8.2：不等长时 dispatch 从 30 µs 涨到
110 µs）。所以 DP 侧分流要按 token 亲和（§9.4），按请求数分没有意义。

**这不是"路由选错了引擎"，而是 EP 的耦合面选大了。** 这里的 DP 只对稠密权重
是数据并行；MoE 那几层里 DP 就是 EP 的一个维度，而 EP 是同步集合通信。后果有三条：

1. **成本是 max，不是 sum。** 一层的 dispatch 要等两个引擎的行都到齐，大的那个
   定时间；现代码对不等长走 `pynccl.all_gatherv`，就是按 root 逐次
   `ncclBroadcast`（`pynccl.py:368`，4 个 rank 就是 4 次串行 broadcast），
   等长时走一次 all-gather。所以 30 → 110 µs 是**形状代价**，不是数据量代价
   （离线微基准的数；线上步长里看不到它，见本节末的实测）。
2. **LB 只能管准入，管不到已经跑起来的 batch。** 现行打分
   （`core_client.py:1940`）= `max(inflight, waiting + running)` + prefill 积压项
   （`VLLM_DP_LB_TOKEN_AWARE=1` 已开）。它按请求数和 prefill token 打分，**不按
   decode 行数**（行数 = 请求数 × (draft+1)）；而且请求不能迁移，一次准入时刻的
   偏斜会按整段生成反复付钱。
3. **单请求 SLO 有地板。** 孤请求的最快一步仍被邻居引擎的批大小决定，所以"一个
   引擎只放一个请求"换不来低延迟。

**2026-11-22 的实测把它拆开了**（`/tmp/rowgap_probe.py` 混合负载 +
`/tmp/wave_probe.py` 受控波次，两个 probe 都从 `/proc` 读 key、只打聚合量）：

- **偏斜有多常见**：8→12 并发、1k–4k 冷 prompt、100 s、243 次采样，
  `|running_0 − running_1|` 的时间占比是 **0 档 50.6%、1 档 34.6%、2 档
  11.9%、3 档 2.5%、≥4 档 0.4%**，最长非零段只有 1.7 s。也就是说偏斜常见但
  很浅、且随请求周转自行消失。
- **偏斜的代价**：受控波次（10 个同 size 请求一起发，相等长度下 16/16 次都被
  LB 分成 5+5，`VLLM_DP_LB_TOKEN_AWARE=1` 在起作用）把"总行数固定、只差切分"
  做成了对照：**5+5 的步长中位数 10.5 ms，4+6 是 10.7 ms**（n=28 / n=6，
  同一 `iteration_tokens_total_count` 口径）——差 ~2%，在噪声内。**§8.2 那个
  30→110 µs/层的 ragged 代价没有出现在线上步长里**，说明它被 overlap 吸收或
  只在不均衡的极端形状下才显形。
- **真正的耦合**在低并发：4+1 时（§5.5 上一段）单请求引擎的 ITL 28.2 ms，
  而它单独跑是 19–20 ms；纯 decode 5+5 时两引擎 ITL 是 31.4 / 30.8 ms，
  步长两侧基本相等。所以代价不是"路由选错引擎"，而是**一个引擎的请求按
  另一个引擎的批来出 token**。

于是三条路的优先级要反过来：**(c)**（拓扑去掉 DP）是唯一能解耦的，代价要单独
评估；**(a)**（补齐 dispatch，1 次固定长度 all-gather 代替每 root 一次
broadcast）和 **(b)**（按行数准入）在 1–2 个请求的偏斜下预期收益 ≤0.3 ms/步，
**不值得先做**；如果将来要碰 (a)，先确认高偏斜（≥3 个请求差）在线上真实出现过
多久。

**DP=4 的推演（离散事件模拟，`/tmp/dp4_sim.py`，2026-11-22）**：这台机器上 DP4
只能是 TP1×DP4（要重启，未做），于是把本节两个常量（`step = max_i(own_i) +
4.6 + 0.6·总行数`，`own_i = 2.8 + 0.15·(n_i−1)`，prefill 0.085 ms/token、每引擎
每步 64 token）连同 `core_client` 的准入规则（取 running 最小、不迁移）写成模拟器，
闭环 K 个 worker 跑同样的负载。校准：LB 快照落后 ~150 ms 时，DP2 的偏斜分布
46.0 / 41.1 / 10.9 / 1.9 / 0.1 与上面实测的 50.6 / 34.6 / 11.9 / 2.5 / 0.4 最接近，
以下都按这个口径（两个 seed、150 s）：

| K=8 | 0 档 | 1 档 | 2 档 | 3 档 | step p50 | TPOT p50 | 完成数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DP2 | 46.0% | 41.1% | 10.9% | 1.9% | 17.3 ms | 10.9 ms | 412 |
| DP4 | 19.7% | 46.3% | 29.3% | 4.5% | 17.0 ms | 10.5 ms | 466 |

K=16 是同一方向且更大：0 档 41.2% → 13.9%，2 档 14.1% → 31.2%，而 step 22.4 →
22.1 ms、TPOT 14.7 → 14.3 ms，完成数 426 → 647（+52%）。K=1/2 两侧步长完全相同
（8.6 / 14.0 ms），只是 DP4 下 K=2 的偏斜恒为 1 档。

读法：**偏斜确实会变严重**（0 档腰斩、2 档三倍，快照越旧越明显：无 staleness 时
0 档 53.9% → 39.1%，400 ms 时 31.6% → 7.2%），**但步长和 TPOT 不跟着坏**。锁步
时钟由 `max_i(own_i) + 组内总行数` 决定，而 `own_i` 只占 17 ms 里的 3–4 ms；
偏斜浪费的是各引擎自己的并行容量，不是全局时钟。所以 DP4 的收益在 **prefill
容量（每引擎 chunk 减半）与 TTFT**，不在 ITL；代价是偏斜更深，§8.2 的 ragged
形状在 ≥3 档时占 5–8% 的时间（实测只覆盖到 2 档的 ~2%）。

口径警告：模拟假设 DP4 的每引擎成本与现在（TP2）相同。真换 TP1 后稠密权重不再
切分、TP all-reduce 也没了，`own_i` 与 prefill 常量都会变，所以上表只能读相对
方向，不能当 DP4 的绝对步长。

**行对齐（把各引擎 MoE 行数 pad 到组内 max）值多少**：把 ragged 罚项按
"只要不等长就固定加"写进模拟（罚 0.2 ms = 实测 5+5 vs 4+6 的差；罚 0.9 ms =
§8.2 离线估计的上界，两者都是二值的，不随 gap 增大），DP4 的 step p50
17.0 → 17.2 / 17.6 ms、TPOT p50 10.5 → 10.5 / 11.0 ms，p95 18.0 → 18.7 ms；
DP2 也同步变差（10.9 → 11.2）。也就是说**对齐最多买回 0.2–0.4 ms，且 DP4 与
DP2 的差距不变**。padding 行落在本来就在等锁步、不在关键路径的引擎上，所以
代价很小；但它拿不掉 `max_i` 本身——孤请求 28.2 ms 里只有 ~1 ms 是 ragged，
其余是邻居批大小的锁步时钟。

**改打分能不能"自然对齐"**：同一模拟器换了三种准入打分（K=8，staleness 150 ms，
3 seed）：

| 打分 | DP2 step / TPOT / 0 档 | DP4 step / TPOT / 0 档 | 完成数 DP2 / DP4 |
| --- | --- | --- | --- |
| `count`（线上规则） | 17.3 / 10.9 / 46.0% | 17.0 / 10.5 / 20.3% | 412 / 465 |
| `count` + 本步完成修正 | 17.3 / 10.9 / 47.6% | 17.0 / 10.4 / 21.2% | 410 / 457 |
| oracle 剩余工作量、0 延迟 | 17.3 / 11.1 / 22.3% | 17.0 / 10.5 / 4.9% | 386 / 434 |

- **按行数打分 ≈ 按请求数打分**：每步行数 = `(1 + k) × 请求数`——验证前向给每个
  decode 请求排 `1 + k` 个位置（`scheduler.py:1002` 的 `num_new_tokens =
  num_tokens_with_spec + placeholders − num_computed_tokens`），**与接受率无关**，
  所以按行数打分只是把请求数乘一个常数，引擎排序不变。线上
  `VLLM_DP_LB_TOKEN_AWARE=1` 本来就
  已经在打分里加了 prefill token 项——而一个 prefill 请求最多 64 行、一个 decode
  请求只有 6 行，**prefill 项才是一阶的，行数项是二阶的**。
- **加"本步将完成"修正**只把 0 档抬 1.6 个点，步长不动。
- **oracle 剩余工作量反而更差**：0 档掉到 22.3% / 4.9%，完成数 −6~7%。因为步长
  由**瞬时行数**决定，不由剩余工作量决定——least-work 路由会把新请求堆到
  "剩余少"的引擎上，制造瞬时行数偏斜；而 prefill chunk 是按引擎限额的，堆请求
  还会拖 TTFT。
- 偏斜的主要来源是**完成时刻的随机性**（同批请求长度、think time 不同），准入
  时刻对齐（实测 16/16 分成 5+5）之后还会重新长出来。要"自然对齐"只能改调度
  结构（同步批边界或请求迁移/抢占），不是打分能做的。

所以打分最多值 0.1–0.3 ms，不值得动。

**"DP 组内同步 dispatch"在代码里同步什么**（两层：每步 + 每层）：

1. **步级对齐行数**：`vllm/v1/worker/dp_utils.py:190 coordinate_batch_across_dp`
   → `_synchronize_dp_ranks`（:122）先做一次 DP 组 all-reduce（`_run_ar`；async
   调度下默认 CPU gloo，可用 NCCL），交换未 pad 行数、pad 后行数、cudagraph
   mode。cudagraph mode 取 min；只要不是 NONE（或开了 ubatching），就把每个
   rank 的 `num_tokens_after_padding` 置成组内 max（`_post_process_dp_padding`，
   :99）。补出来的行是**真实张量行**（`model_runner.py:1712` 用
   `input_ids[:num_tokens_after_padding]`），但被标记 `is_padding`，路由把它们设成
   expert id −1（`router/base_router.py:138`；`VLLM_MOE_SKIP_PADDING` 默认开），
   所以路由和专家 GEMM 会跳过它们——**pad 行不花专家计算，只花形状**。
2. **每层集合通信**：`all2all.py:89 _gather_rows` 沿 dim 0 `all_gatherv`，
   `sizes` = `dp_metadata.get_chunk_sizes_across_dp_rank()`，即上一步对齐后的行数。
   它是阻塞集合通信：每个 stage 20 层 MoE，就是每步 20 个必须同时到达的对齐点。

含义：步长 = 组内**真实**行数最多的引擎的计算 + 集合通信；padding 只把形状变等
（走一次 all-gather，而不是每 root 一次 broadcast），**不会缩短任何人的计算**。
live 是 `cudagraph_mode=FULL_AND_PIECEWISE`，graph 步都会 pad，所以 ragged 代价
在线上看不到（5+5 vs 4+6 只差 2%）；能露出它的只有 eager 步。另外 DSv4.1 还有
一步额外同步：`model_state.py:800 _agree_across_dp` 每步用 gloo 在 DP 组内商量
"是否有 rank 需要 replay batch、各自多少行"，因为 draft/replay 层必须全体一起跑
或一起不跑。

### 5.6 前缀命中的成本：replay 已被 LMCache 跳过（P0-4 已完成）

DSv4.1 的 layer 20 是最后一个 KV source，21–39 层只有 128 token 的 SWA cache
（`sliding_window=128`；线上 `block_size=64`，即窗口 2 个 block）。
`swa_bounded_replay=True` 让这 128 token **不进 APC**，命中后由调度器回退一个窗口
重算（`SlidingWindowManager.prefix_replay_tokens`，
`single_type_kv_cache_manager.py:1100-1103`；`scheduler.py:3963-3990`）。

本部署已经在 **LMCache 层**把这一步跳过了：启动配置带
`lmcache.mp.store_replay_windows: true`（第 1 节的现役配置），connector 把
prefix-replay 组注册成可存组、存/取窗口，并在 `get_loaded_kv_cache_group_ids`
里报给 vLLM（`lmcache_mp_connector.py:1768-1783`），于是
`_load_restores_replay_window()` 为 True，`_mark_prefix_replay` 返回 0。

测量（`/tmp/probe_p04.py`，2026-11-22 07:40 前后；engine 侧
`time_to_first_token_seconds` 单请求差分，`max_tokens=1`，每次新 seed）：

| 用例 | TTFT | computed tokens | 命中来源 |
| --- | --- | --- | --- |
| 冷 128 token ×5 | 151–155 ms | 129 | — |
| 冷 16k / 64k | 1575 / 5047 ms | 16385 / 65537 | 0.096 ms/token |
| 命中 16k，跨引擎 | 161 ms | 2 | device 16256 + external 127 |
| 命中 16k，本地 | 84 / 91 ms | 2 | device 16256 + external 127 |
| 命中 64k，跨引擎 | 318 ms | 2 | external 65535 |
| 命中 64k，本地 | 103 ms ×2 | 2 | device 65408 + external 127 |
| 命中 16k + 300 新 | 162–170 ms | 301 | 无 replay |
| 命中 16k + 1200 新 | 216 ms ×2 | 1201 | 无 replay |

**判定"没有 replay"的两条硬证据：**

1. `iteration_tokens_total` 观测的是 computed prompt + generated：满命中 = 2
   （1 个末 token 重算 + 1 个采样），16k+300 = 301（正好 300+1）。若有 replay，
   满命中会是 129、16k+300 会是 429。
2. device 命中恰好 16256 = 16384 − 128（窗口 2 个 64-token block 不在 APC 里），
   external 命中 127 = 窗口（128 减掉留给末 token 的 1 个）。

**命中后的成本结构：**

- **floor ≈ 84 ms**，与命中长度无关（16k 84 ms、64k 103 ms）。冷 128 token 是
  151 ms，所以 128 token 的小 chunk ≈ 65 ms——这就是 replay 真发生时的量级。
- **新增 token**：300 → +78–86 ms（≈0.26 ms/token）；1200 → +132 ms
  （≈0.11）；大 prefill 是 0.096。小 chunk 有明显的固定惩罚。
- **跨引擎命中**（块不在本 engine，只能从 LMCache 取）：16k +77 ms、64k
  +215 ms，即 3–5 µs/token。

**84 ms floor 的构成（2026-11-22 08:3x 步级 + 12:3x 复核）**：它不是常数，
而是「排队等外部 KV + 一次 decode 形态的 step」：

| 用例 | TTFT | queue | prefill | lookup | computed / cached |
| --- | --- | --- | --- | --- | --- |
| 命中 16k（本地/旧进程） | 91 | 49 | 40 | 7.7 | 1 / 16383 |
| 命中 16k（跨引擎/旧进程） | 131 | 98 | 26 | 20.5 | 1 / 16383 |
| 命中 16k（新进程 12:16） | 181 | 96 | 82 | 6.1 | 1 / 16383（external） |
| 命中 64k（新进程 12:16） | 164 | 13 | 145 | 12.7 | 128 / 65408（device） |

- queue 里 lookup 只占 6–20 ms，剩下是**等 KV 就绪**：本地命中也常有 42–49 ms，
  **跨引擎命中**则涨到 96–98 ms，多出来的正是 LMCache 搬 16k token 的 ~90 ms
  （3–5 µs/token）。本地命中这 42 ms 具体在等什么，见下面的新一节（结论：不是白等）。
- 本地命中的 prefill 只要 26–41 ms：满命中的 1 个新 token 走 decode 形态
  （见 §5.3），所以 floor 里真正算掉的只有这一小段。
- 64k 那条 computed = 128 是**解决了上一版记录的矛盾**：那 128 token 是
  device 命中的 SWA 窗口 replay（`window_loaded=False` 的分支），不是度量错。

#### 5.6.1 本地命中那 42 ms 在等什么：127 token 的外部尾巴

先纠正上一条的旧说法"本地命中没有搬运"。满命中时 device 覆盖 =
`floor((n-1)/128)*128`，剩下 ≤127 token 的尾巴仍从 LMCache 取。128 的粒度来自
启动日志：`kv cache group sizes (32×15, 128, 8, 8, 64)`、`kv lcm block sizes 128`
（32 与 128 两组的最小公倍数，与 `sliding_window=128` 同一来源）。
`prompt_tokens_cached_by_source_total` 七个长度的差分全部吻合：8192 = device
8064 + external 127；8320 = 8192 + 0；16384 = 16256 + 127；16256 = 16128 + 0；
16320 = 16256 + 0；16512 = 16384 + 0；16640 = 16512 + 127。即只有 n 是 128 的
倍数时才有外部尾巴。

于是"本地命中"分两种，成本相差 47 ms：

| 用例 | 尾巴 | queue | prefill | TTFT | 形态 |
| --- | --- | --- | --- | --- | --- |
| 16384（尾 127 在 LMCache） | 取尾 | 42 | 52 | 96 | 停一步等 KV，然后 decode 形态算 1 token |
| 16256（无尾） | 不取 | 5 | 135 | 143 | 127 token 走真 prefill（65 ms/级） |

所以那 42 ms 不是白等：它把 127 token 从"prefill 形态重算"换成"外部拷入"，
净省 **47 ms**；不取尾巴反而更慢。这也解释了此前 46–49 ms 的 queue 为何
时有时无——取决于 prompt 长度是否卡在 128 的倍数上。

LMCache 侧计时（`:8080/metrics` 的 `lmcache_mp_transfer_phase_elapsed_seconds_total`
逐请求差分）：尾 2 个 chunk 的 h2d kernel 4.55 ms/卡、staging 0.03 ms
（1.42–1.60 MB/卡）；而 vLLM 观测的 `in_progress` 约 39 ms、`completed_waiting`
再 10 ms。**约 35 ms 是 MP 往返 + 等调度步**（`WAITING_FOR_REMOTE_KVS` 至少
跳过当前 step，见 `scheduler.py` `load_kv_async`），不是拷贝本身。

跨引擎 16k 的大搬运是另一个问题：65 chunk/卡、10.1 MB/卡、h2d kernel
38.2 ms/卡，即 **≈260–310 MB/s**，比 PCIe Gen5 x16 应有的低一个量级
（`lmcache.mp.use_vmm_api=true` 路径或 NUMA 落位），是跨引擎命中（§5.6.2）
真正的成本项，值得单独查。

那 35 ms 里的"等调度步"可以直接去掉：connector 已经能在提交后拿到服务器的
完成句柄（`DeviceMessagingFuture.wait_on_stream`），把 ≤N token 的尾巴改成在
分配块的同一个 step 里提交、并让当前流排在拷贝之后，请求就不必进
`WAITING_FOR_REMOTE_KVS`。实现是本仓库里的外部 connector 子类（默认关）：

```yaml
kv_connector: LMCacheMPConnector
kv_connector_module_path: vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_sync_load
lmcache.mp.sync_load_max_tokens: 128   # 0 = 原行为
lmcache.mp.sync_load_timeout_s: 60
```

大搬运（跨引擎 16k 那 38 ms 的 h2d）仍走异步：让整个 step 等它不划算。A/B 看
三件事：16384 的 TTFT 是否从 96 ms 降到 60–70 ms；`in_progress` 是否归零；
decode 步长是否被拖长（host 侧只等响应句柄，拷贝本身是流等待，但同一 step
里的其他请求会跟着这条流一起等那 4.55 ms）。需要重启，配置细节见 §9.3。

#### 5.6.2 会话亲和（P1-8）的确定性测量

暖好一个 16k/32k 前缀后，同一个 prompt 分别固定打到持有 device 副本的 rank
和没有副本的 rank（`/tmp/probe_affinity3.py`，中间 sleep 1.5 s 让 LMCache
store 落盘，200 token/轮）：

| prompt | 本 rank（device + 尾） | 跨 rank 首次 | 跨 rank 第二次（已暖） |
| --- | --- | --- | --- |
| 8k | 96 ms | 134 ms（queue 60 / prefill 71） | — |
| 16k | 96 ms（queue 43 / prefill 51） | 178 ms（queue 93 / prefill 82） | 96 ms |
| 32k | 100 ms | 229 ms | 115 ms |

溢价全部来自"该 rank 上没有 device 副本"（queue 里多搬 3–5 µs/token，
prefill 也多 ~30 ms），不是"引擎号不同"：第二次打过去就恢复成本地速度。
按单轮算：**8k +38 ms、16k +82 ms、32k +129 ms**，且只有某 rank 首轮付
（此前"每轮省 77–215 ms"高估了，跨引擎溢价一次付清后，后续命中都已在该
rank 的 device 上）。

agent 逐轮复发的负载（16k 前缀 + 200 新 token，6 轮）也一致：不固定 rank 时
LB 交替，6 轮里出现 2 次跨引擎（242 / 177 ms，其余 149–155 ms）；固定
`X-data-parallel-rank: 0` 后 6 轮 145–150 ms（其中一轮 186 ms 是并发的
干扰，不是跨引擎）。

实现不需要改 vLLM：`vllm/entrypoints/generate/base/serving.py:196` 的
`_get_data_parallel_rank` 已优先读 `X-data-parallel-rank`。缺的是路由器侧的
session/前缀 → rank 粘滞（命中失败时回退 least-loaded），加上文档。

顺带修一个**探测脚本的读法错误**（不是引擎行为）：`vllm:prompt_tokens_cached_total`
和 `vllm:prompt_tokens_by_source_total` 是裸 counter（名字本身带 `_total`，
没有 `_sum`），早先的脚本按 `_sum` 过滤，于是每个请求都读成 cached = 0，
"命中却没算 cached"的假象来自这里。按名字直读后满命中就是 16383 / 65408。

**replay 会回来的条件**（这三条就是 P0-4 残余的缝隙）：LMCache 查不到窗口
（没存过 / 被驱逐 / LMCache 重启）；部署不配 LMCache、只用本地 APC；
vLLM 自己的命中离 LMCache 命中不足一个窗口（`num_computed_tokens <=
ret - window` 不成立）。

含义：P0-4 关闭。agent 这类"每轮重发整段对话"的负载每轮都命中（前提是前缀
字节稳定：时间戳/request id 不要放在提示头，工具 schema 不要重排），命中后
每轮 TTFT = 84 ms floor + 增量，**不含 replay**。agent TTFT 的剩余杠杆按大小是：
会话亲和（避免首次跨引擎取 KV，§5.6.2）> 小 chunk 的 ~50 ms 固定开销
（其中 42 ms 是换来省 47 ms 的外部尾巴，§5.6.1）> floor 本身（prefill 形态
step 的 eager launch，§5.7）。

### 5.7 小 batch prefill 为什么是 ~65 ms/级：eager 逐层 launch

启动日志直接写明了降级（`vllm-dsv41-20261122-121614.log:357`）：

```text
WARNING [compilation.py:1687] Cudagraph mode FULL_AND_PIECEWISE requires
piecewise capture, but the loaded model provides neither a compiled submodule
nor breakable CUDA graphs. Overriding to FULL_DECODE_ONLY.
```

同一份日志更早处还有 "`torch.compile` is turned on, but the model ... does not
support it"。降级后 `CudaGraphManager._init_candidates` 的
`if mixed_mode:` 为假，只建 decode 的 token-keyed 图，**一条 prefill 图也不建**；
prefill 形态的 batch 在 `dispatch` 里拿到 `BatchExecutionDescriptor(cg_mode=NONE)`，
`model_runner.py` 走 `self.model(**model_inputs)`——纯 Python 逐层调用，既无图
也无编译。于是"小 batch prefill 是 eager 逐层 launch"成立，而且不只是小 batch：
**所有** prefill 形态的 step 都是。配置里 `prefill_cudagraph_capture_token_sizes`
= 1…512 的网格确实存在（第 1 节的现役配置），但没有 piecewise capture 就一条也
用不上。

三条实测佐证：

1. 固定成本与 token 数无关（`/tmp/probe_boundary.py`，12:2x）：冷 prompt
   1/128/384/496/512/520/576/640/768 的墙钟是 138/141/146/141/139/144/142/141/140 ms。
   512 这条"捕获网格的边界"没有台阶，到 1024 才 166、2048 才 286。65 ms/级
   是每 step 的固定项，不是算力。
2. 同样的 1 个 token 走 decode 形态（满命中的延伸）每级约 26 ms（§5.6 的
   prefill 52 ms / 2 级），走 prefill 形态要 68 ms。PP2、40 层 → 20 层/级，
   差出的 ~42 ms/级 ≈ **2 ms/层**的 host 开销（launch + Python）。
3. 12 个连续冷 1-token 请求期间，8 张卡功率停在 75–110 W（空闲 70–104 W），
   这 130 ms 里 GPU 基本是空的（`/tmp/probe_power.py`）。并发放大同理：
   1/8/16/32/64 个并发 1-token 冷请求的墙钟是 151/269/494/508/543 ms
   （`/tmp/probe_batch.py`），固定 150 ms 的 step 成本 + 每请求 ~8 ms。

可行的杠杆只有一条：让 prefill 也进图。本模型没有 torch.compile 子模块，但
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`（或 `compilation_config.use_breakable_cudagraph`）
会打开 breakable piecewise 图（`vllm/config/vllm.py:960-985`，同时把
`CompilationMode` 置为 NONE），这是配置层唯一的入口；DSv4.1 的 SWA replay
机器已经会构造 `BreakableCUDAGraphWrapper`
（`vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py`）。它是否对
eager prefill 路径安全（attention 作为 break 点、Mega MoE 的静态缓冲）必须
一次带重启的 A/B 才能确认，未做。预期收益上限：agent 每轮多算的 200–400 新
token 现在付满 130 ms 固定价，若进图应压到 ~50–60 ms。

### 5.7.1 第一次 breakable A/B 崩在 cuDNN plan 构建，已加「按 desc 预 warm」

**14:14 重启的崩溃**：唯一的变量是 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（其余与
14:19 基线一致，含 `max_prefill_cudagraph_capture_tokens: 1024` → PIECEWISE
27 档 1…1024）。14:17:37 在捕获第 4 档（M=8）时，8 个 worker 同一秒、同一算子
挂掉：`capture_model → forward_fn(PIECEWISE) → run_pw_graph` →
`attention.py:1818 indexer.forward_q → _wq_b_proj → mm_mxfp8`（`VLLM_MXFP8_GEMM_BACKEND=auto`
选了 cuDNN runner）→ `build_cudnn_gemm_mxfp8_graph` 未命中 →
`CudnnPlanBuildInCaptureError`。M=1 那档还有 "The CUDA Graph is empty" 警告
（分段里没有可捕获 kernel），无害。

**根因是三件事叠加**：

1. 索引器的短上下文早退带 `not torch.cuda.is_current_stream_capturing()`
   （`attention.py:1856-1858`）：候选数 ≤ topk 时直接返回、**不执行 `wq_b`**。
   eager warmup 正好走这条早退，所以该 desc 的 warmup 里 `wq_b` 一次都没跑。
2. FlashInfer autotune 的 dummy run 用 `round_up=True` 把 M 向上取档
   （`kernel_warmup.py:552-562`，`autotune_indexer_wq_b` 只在取档后的
   `max_tuning_tokens` 上跑），所以 wq_b 的 cuDNN plan 只建在取档 M 上，
   精确 M=8 的 key 在进程里不存在。
3. 非 breakable 时这些算子在图外 replay，第一次建 plan 无害；breakable 把它们
   收进捕获区间，才暴露。

**离线复现与修法验证**（`/tmp/verify_cudnn_plan_prewarm.py`，GPU7 裸进程，
wq_b 的 N=16384/K=1280）：先在 M=64 建 plan，再捕获 M=8 → 抛同一异常；先在
M=8 跑一次 eager，再捕获 → `RESULT: capture OK`。

**已实现**：`ModelCudaGraphManager.capture` 在每个 desc 的 warmup pass 里调用
模型可选的 `warmup_capture_ops(num_tokens)`（`cudagraph_utils.py:747-757`，
在 `graph_capture` 内、任何 `torch.cuda.graph` 之前）。DSv4.1 两个入口类各自
转发到已有的 `autotune_indexer_wq_b(model, num_tokens)`
（`nvidia/vl_model.py`、`nvidia/model.py`）。成本是每档每 rank 每个 indexer
一次 ≤M 行的 GEMM（27 档 × 20 层 ≈ 几十 ms 启动时间）。

状态：单算子层面已证明「预 warm 后捕获安全」；15:16 重启的整机验证通过——
PIECEWISE 27/27 档、FULL 41/41 档全部捕获，日志里没有
`CudnnPlanBuildInCaptureError`（实测数字见 §9.3 的 breakable 条目）。其它
capture-gated 分支（`attention.py:1379` 的 `copy_q`、
`sm12x_mega_gate_router.py:191`、`indexer_topk.py:563`）不是 plan/JIT 构建类；
若再出现同型报错，按同一模式加进 hook。

#### 5.7.2 sync-load 连接器的崩溃：worker 与 scheduler 判据不一致

**14:53 那次启动**是唯一一次把 `kv_connector_module_path` 指到
`vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_sync_load`、并设
`lmcache.mp.sync_load_max_tokens: 128` 的运行。14:59:01，约 6 分钟真实流量后
`EngineCore_DP0` 死在 `scheduler.py:4107`
`assert RequestStatus.is_finished(req.status)`：有个请求被 worker 报成
`finished_recving`，但它既不在 `WAITING_FOR_REMOTE_KVS`（没被 park），也没结束。
只有 DP0 崩、DP1 正常——不是 DP/EP 并行的问题，是单引擎内 scheduler 与 worker
的契约不一致。

**两边判据不等价**：scheduler 是否 park 只看 `get_num_new_matched_tokens` 返回的
`async_load`（本模块在 `need_to_load <= 128` 时改成 False）；而模块的 worker 侧
原先按 op 跨度 `op.end - op.start <= 128` 划分。retrieve 的起点却向下取整到
LMCache chunk 边界（`GetRetrieveMetadata` 里
`start = num_vllm_hit_tokens // chunk * chunk`），scheduler 的
`need_to_load = ret - num_computed_tokens`，于是

```text
op 跨度 - need_to_load = num_computed_tokens % chunk ∈ [0, chunk)
```

APC 命中不是 chunk 对齐时（例如 200 token 命中、LMCache 命中 256）：
`need = 56` → 不 park，但 `op 跨度 = 256` → worker 判成异步，future 留在
adapter 里，下一步被 `get_finished` 报成 finished receive，而请求已在跑 →
assert。偏差上限是 chunk-1（本模型 255 token），触发不频繁但一定会出现。

**修法**：判据只留一份。`get_num_new_matched_tokens` 的结论记在请求 tracker
上，`_process_retrieve_requests` 把它写进该请求的 retrieve metadata
（`same_step_load` 字段），worker 的 `split_sync_retrieves(metas)` 只读这个字段
（metadata 经 pickle 到 worker，布尔属性会保留）。单测覆盖了「跨度 255 > 阈值
但标记为同 step」这条旧代码判错的用例。

**另两次崩溃与并行无关**：15:02 / 15:07 把 `lmcache.mp.host` 写成
`grpc://172.17.0.1`（应为 `127.0.0.1`），MQ 等满 300 s 后 `extend_kv_cache`
抛 `ConnectionError`；15:02 还带 `expandable_segments` 映射告警。现役进程
（15:16 起）用 stock 连接器，不触发本问题；重新启用要等一次重启。

#### 5.7.3 大尺寸 prefill 的不确定性：target 漂移 + attention cap 悬崖

**大部分不是 breakable，但 >1024 的 eager 是**：这类步命中不了任何图（`max_prefill_cudagraph_capture_tokens: 1024`），`dispatch()` 回落到 `cg_mode=NONE`；而 breakable 把 `compilation_config.mode` 置成 `NONE`，这条路径是**未编译**的 eager，breakable 之前同一大 chunk 走的是编译后的 piecewise。见 §5.7.4。大尺寸的不确定性本身来自部署里开着的 DP prefill 分块控制器（`VLLM_PREFILL_STEP_BUDGET_MS=700` + `FILL_TARGET=1` + `COST_FIT=1`）。

**表现**（15:56–15:59 同一会话、冷 seed、单请求、engine 0）：

```text
n= 2112  prefill=286 ms  / 582 ms      # 1 块 vs 2 块
n= 4224  prefill=535 ms  / 841 ms
n=12288  prefill=1216 ms / 1404 ms
n=16384  prefill=1485/1486/1489/1494/1610/1623 ms
n= 8192  prefill=963–970 ms（连续 6 次稳定，说明那段时间 target 没动）
```

多一块的代价是 ~250–320 ms 的阶梯：2048→2112 +305、6144→6208 +297、32768→32832 +262。所以 8k 是 917 还是 1242 ms，取决于当时 `target` 在 8192 还是 2048/4096。

**机制一（漂移）**：`chunk = target - decode_tokens`（`prefill_target.py`），`target` 初值 2048（`max_num_batched_tokens//4`）、范围 512–8192、每轮 ±25%，只在该轮有 ≥4 个「填满 80% target」的 prefill 步时动（`update()`），逼近 700 ms/步。突发大 prefill 自己会把 target 抬高（每轮 +25%），没有流量就冻结在上一轮——所以同尺寸重复测量天然不可重复。

**机制二（悬崖，能解释 §5.3 的 5× 慢态）**：`_prefill_chunk` 的 attention cap。只要在线拟合的 context 系数 > 0（不判量级；15:55–15:58 的 400 条拟合里 166 条是 1e-7~1e-6 的正数），而 `budget - fixed - ms_per_token*target <= 0`（target ≥ ~4k 时成立），`max_attention_tokens` 直接返回 `min_chunk_tokens=256`——单步 chunk 从 8192 掉到 256（32×）。而且 256-token 的步永远达不到 `0.8*target`，`update()` 学不到样本，`target` 卡在 8192，**慢态自锁**，只能等下一轮拟合恰好给出 0。

用真实类复现（`PrefillCostModel(700, 9.6, 0.116, c)` + `PrefillTarget(..., 8192)`）：c=0 → step 8192；c=8e-7 → step 256。

**修法（保留控制器）**：

- `PrefillCostModel.min_attention_share=0.1`：token 项定价超预算时，attention 仍保留 10% 预算，cap 对拟合系数连续；微小的 c 不再把 chunk 打到地板，超预算的步反而能填满 target 把它拉回来。
- `PrefillTarget` 落到 token 网格（`quantum = max_num_batched_tokens/64 = 128`）：target 只在少数固定档位间移动（2048→2560→3200→3968→4992→6272→7808→8192），chunk 形状重复，cuDNN plan / b12x JIT 可复用；`raw` 落回原值时就地走一格，避免地板自锁。已进一步改为直接落在捕捉尺寸上，见 §5.7.4。
- target 每次变化打一行 INFO（原先完全不可观测）。

单测：`test_prefill_cost_fit.py`（cap 保持 10% 预算、1e-7 系数不切 chunk）、`test_prefill_target.py`（网格 + 从地板爬到 8192 + 捕捉尺寸）、`test_scheduler.py`（target 512、`ms_per_token=1.2`、`c=1e-7` 时 chunks 仍是 `[512,512,512,464]`，旧代码是 16 token 一块）。改动要重启才生效。

#### 5.7.4 prefill 步与大尺寸 CUDA graph：target 直接对齐捕捉尺寸

**>1024 token 的 prefill 步不在任何图里**。`_init_candidates`（`cudagraph_utils.py:420-428`）把每个 `n` 映射到"最小的 ≥n 的已捕捉尺寸"，所以 ≤1024 总能命中；>1024 没有 key，`dispatch()` 返回 `cg_mode=NONE`，`NONE` 分支直接 `model(**model_inputs)`（`:862-863`）。breakable 又把 `compilation_config.mode` 置成 `CompilationMode.NONE`（`vllm/config/vllm.py:975-978`），所以这条 eager **未编译**；breakable 之前同样的大 chunk 走 `run_pw_graph` 的编译 piecewise。实测 8k 965 ms vs 基线 917–923 ms（+5%），而 ≤1024 因为进图反而快得多（1 token 138→31 ms）。

**target 阶梯和捕捉网格原本对不上**。`prefill_cudagraph_capture_grid(8192)` 是 54 个尺寸：1,2,4,8,16,24,…,1024（每 128/64 一档），之后每 256 到 7680，最后一跳 512 到 8192。原来的 128-token 网格阶梯只有 3 个点命中：

```text
target   图           浪费
2048     2048         0
2560     2560         0
3200     3328         3.8%
3968     4096         3.1%
4992     5120         2.5%
6272     6400         2.0%
7808     8192         4.7%
8192     8192         0
```

加上 chunk 向下对齐 block 64（step 总 token 比 target 低 0–63），最坏 ~5%。

**修法**：`PrefillTarget` 新增 `grid` 参数，scheduler 把解析后的 `compilation_config.prefill_cudagraph_capture_token_sizes` 传进来，target 只在捕捉尺寸之间移动；`ceiling` 就是最大捕捉尺寸（cap=1024 时 target 不会再爬到图外），没有捕捉图时退回 `quantum` 网格。吸附回原值时就走到相邻网格点，避免自锁。全速上行阶梯变为（全部是捕捉尺寸）：

```text
2048 → 2560 → 3072 → 3840 → 4864 → 6144 → 7680 → 8192
```

**还剩一步配置改动（需重启）**：把 launch 里的 `--compilation-config` 改成 `max_prefill_cudagraph_capture_tokens: 8192`（调度器本来就用 `max_num_batched_tokens=8192` 封顶 step，8192 覆盖全部 token-keyed 路径）。启动多捕捉 27 张图、共享 graph pool（`breakable_cudagraph.py:296,395`）；mega-gate plan 已按 eager batch shape 缓存、`warmup_capture_ops` 已在捕获前预 warm。改完 A/B 1k/2k/4k/8k/16k/32k。

单测：`test_prefill_target.py` 新增 `test_the_target_moves_between_captured_prefill_graph_sizes`、`test_the_target_stops_at_the_last_captured_graph`、`test_a_change_smaller_than_the_grid_moves_one_point`。

#### 5.7.5 target 控制器太慢：一次求解取代 ±25% 爬行

**live 证据**（16:55 重启后的 target 日志）：

```text
16:59:40  2048 -> 2560  (171.9 ms / 700 ms, 58 个 informing 步)
16:59:59  2560 -> 3072  (305.2 ms, 45 步)
17:01:06  3072 -> 3840  (274.2 ms,  4 步)
17:03:00  3840 -> 4864  (451.3 ms, 32 步)
之后 3.5 小时不动
```

5 分钟才到 4864，且每步只在一次 64 步拟合轮里动一次。慢在四处：

1. **更新只在拟合轮**（`VLLM_PREFILL_COST_FIT_INTERVAL=64` 步，约 20 s 一次）；开 `COST_FIT` 时后台线程根本没启动，两轮之间 target 冻结。
2. **每轮 ±25%**（`max_factor=1.25`）且 `gain=0.5` 只修一半误差：2048→8192 至少 7 轮，几何收敛。
3. **拟合窗口没有 prefill 步时照样安装系数**：00:53 那次 `fixed 8.36 ms, 0.4964 ms/token, 0 步带 prefill`。0.4964 ms/token 让 `_prefill_chunk` 的 `room_ms = 700 - fixed - per_token*target` 在 target ≥ 1400 时变负，只剩 10% 预算（70 ms）给 attention cap；一旦 context 系数被拟合成正，长上下文 chunk 会被腰斩——突发 prefill 时 TTFT 变慢正是这条路。
4. **死区缺失**：直接求解下 median 抖动会让 target 在两个相邻捕捉尺寸间来回跳。

**修法**（`prefill_target.py` / `cost_fit.py` / `scheduler.py`）：

- 直接求解：`ratio = (budget - fixed) / (period - fixed)`，`gain=1` → 一次更新落到位；`max_factor=4` 仅作安全上限。
- `deadband=0.05`：median 在预算 ±5% 内不动（模拟里 7168↔7424 每个更新翻一次，加死区后稳定）。
- 后台线程两种模式都跑（200 ms），`_update_lock` 串行化线程与拟合轮；拟合轮 `update(since=...)` 只判本轮合并样本，并把各 rank 拉回同一值。
- `fixed_ms` 从拟合结果同步进 target，求解时先扣掉固定成本。
- 拟合只用带 prefill 的步（窗口内不足 `min_samples=16` 步则保持配置值）。

**模拟**（budget 700、真实 0.093 ms/token、fixed 10 ms）：

```text
2048 -> 7424（一次更新）；之后不动
上下文成本出现（+4 ms、0.15 ms/token）：7424 -> 4608（一次）
恢复：4608 -> 7424（一次）
```

**同期冷 prefill 实测**（target 4864、图命中、按尺寸递增的唯一 prompt，client 侧）：

```text
n= 1024  170 ms      n= 4096  576 ms      n= 8192   921 ms（4864+3328）
n= 2048  288 ms      n= 4864  646 ms      n=16384  1512 ms（4 块）
n= 2560  351 ms      n= 5120  710 ms
n= 3328  483 ms
```

8k 回到基线水平（917–923 ms），16k 因为 PP 流水，每多一块约 +290 ms 而不是 +646 ms。注意长时间空闲后的**第一发**带冷 shape（mega-gate plan、b12x JIT）约 +200–300 ms，属于首请求成本。

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

### 6.2 采样算子微基准：同样是噪声（2026-11-22）

目的：回答 P0-3「融合采样」值不值得做，即采样段在步长里占多少。
`/tmp/bench_sampler_ops.py`（temp=0）与 `/tmp/bench_sampler_temp.py`（temp>0），
直接调用引擎里的同一批函数（`gumbel_sample`、`greedy_argmax`、
`apply_top_k_top_p`、`flashinfer_sample`、`apply_temperature`、`apply_min_p`、
`apply_penalties`），合成 fp32 logits，vocab 129280，行数按 DSpark k=5
（每请求每步 k+1 = 6 行）。链路就是这个部署的真实形态：`draft_sample_method`
默认 `greedy` → draft 侧走 `speculator.py:659` 的 `greedy_argmax`（SM120 既不满足
`specialized_decode` 也不满足 `dspark_fused_markov`/`is_ampere()` 的条件，所以不走
`FusedMarkovSampler`）；verify 侧 `Sampler.sample` 因批内有贪心请求回退到
`gumbel_sample`。全部用 CUDA graph 回放计时——线上 decode 就是
graph 形态，单发 `cuEvent` 会把 7.7 µs 的 kernel 测成 35 µs（纯 launch 开销）。
背景：其余 7 张卡正在跑压测，本基准在 0% util 的 3 号卡上做，单核数字有
±10–20% 邻卡噪声。

temp=0（现网流量）与常驻 processor，graph 回放中位数（µs）：

| 路径 | 6 行（1 请求） | 24 行（4 请求） |
| --- | --- | --- |
| verify `gumbel_sample`（temp=0，纯 argmax） | 7.7 | 7.6 |
| **整条 temp=0 采样链**（5×draft `greedy_argmax` + 1×verify） | **16.0** | **20.1** |
| `apply_penalties`（rep=1.05） | 21 | 22 |

temperature=0.7 的完整链路（按线上顺序串起来，已扣 7.5–9.5 µs 回放底噪）：

| 链路 | 6 行 | 24 行 |
| --- | --- | --- |
| T only（温度 + gumbel，无 top-k/p） | 6 | 19 |
| T + min_p | 74 | 88 |
| T + p=0.9（fused 生效） | 64 | 72 |
| T + p=0.9 + seed（fallback 排序） | 78 | **222** |
| T + k=100（fused） | 176 | 190 |
| T + k=100 p=0.9（fused） | 99 | 125 |
| T + k=100 p=0.9 + seed（fallback） | 84 | 99 |

单算子：`apply_temperature` 7.5 / 9.3，`apply_min_p` 71（两档相同）。

结论：

- 现网全是 temp=0：整条采样链 **16–20 µs**，是 20.3 ms iteration 的 0.1%、
  9.5 ms 单级步长的 0.2%，比 lm_head（0.32 ms）小一个数量级。采样段没有
  >5% 可挖，**P0-3 关闭**。
- fused 只在**纯 top-p** 批上明显划算（72 vs 222 µs）；`k+p` 反而略慢
  （125 vs 99），纯 `k` 更慢（190 量级）——SM120 上
  `top_k_sampling_from_probs` 本身偏慢。且 fused 只在批内无贪心、无 seed、
  无 processed-logprobs 时启用；任何一条不满足，整批退回排序路径，那一格
  最坏 222 µs（2.3%）。
- 单个最贵的采样算子是 `apply_min_p`（71 µs，每 token 扫全词表），比整条
  贪心链贵 3.5 倍；若线上有人用 min_p，值得单独看，但同样不到 5%。
- 线上 `VLLM_USE_FLASHINFER_SAMPLER` 未设置（默认 1），SM120 过
  `supports_compute_capability`（≤12.1）且 SM 数 >16，本机 JIT 可用
  （基准里直接跑通了 FlashInfer 采样 kernel），所以合格的采样流量本来就会
  自动走 fused，P0-3 的增量空间更小。
- 方法学坑：graph 回放里 `apply_temperature` 是原地除，同一份 logits 会被
  反复缩放，几轮后数值失真、排序变慢（未修正时 fallback 行虚高约 1.7×）。
  每次回放前必须把原始 logits 拷回，并扣掉拷贝/回放底噪。

## 7. kernel 级拆分：一次带 profiler 的重启（2026-11-22 已做）

容器内无法附加 profiler（无 `CAP_SYS_PTRACE`、`ptrace_scope=1`、
`perf_event_paranoid=4`），只能在启动时带 `--profiler-config`。2026-11-22 这样重启
一次，用运行时窗口采到了 prefill、decode 和 **stage 1** 的 trace。**但 decode trace
可能在 `profiler_stop` 时把服务打挂**，所以先写怎么安全地采（§7.1–7.2），再写读到
的数和读数必须带上的三条口径修正（§7.3）。

### 7.1 采集方式：启动一次，运行时开多个命名窗口

```bash
# 1) 启动时打开 profiler 与步级 profiler（serve.sh 160-164 行拼 --profiler-config；
#    VLLM_PP_STEP_PROFILE_INTERVAL 默认 0）
PROFILE_DIR=/tmp/dsv41-traces VLLM_PP_STEP_PROFILE_INTERVAL=1 \
  bash examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/serve.sh

# 2) 运行时按窗口采集，不动默认参数
#    POST /start_profile {"profile_prefix":"w3_dec","max_iterations":40}
#    发请求 → POST /stop_profile
```

- 默认参数就够用：`torch_profiler_activities=None` 在 NVIDIA 上等于 CPU+CUDA，
  `with_stack=True`、`use_gzip=True`、`delay_iterations/max_iterations=0`（不自动停）。
- 每个窗口一个 `profile_prefix`，文件名 `{prefix}_{rank_suffix}.<session>.pt.trace.json.gz`，
  **一次重启可以采很多片**；`stop_profile` 只做 flush（大 trace 要等几分钟）。
- `max_iterations` 只在 `WorkerProfiler.step()` 里计数，而它只由 `execute_model →
  annotate_profile()` 调用，所以 **idle dummy 循环不推进计数**，`max_iterations=40`
  就是 40 个真实 decode 步。
- 体量：32k ctx 的 decode 每 rank 约 28–40 MB / 40 步、3.5–5.7 MB / 10 步；
  prefill 9–10 MB。小窗口导得快，也更不容易挂（§7.2）。
- `analyze_traces.py` 收位置参数（目录会 rglob `*.pt.trace.json*`），不是 `--dir`：

```bash
.venv/bin/python examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/analyze_traces.py \
  /tmp/dsv41-traces
```

`bench_matrix.sh` 也串了这条链（`PROFILE_DIR=... bash bench_matrix.sh` 会在每个配置上
多跑一轮 8 并发 `--profile`）——但见 §7.2，**它跑的是 decode，会触发同一个崩溃**。
只看 `execute / token_wait / recv_wait` 拆分时不需要 `--profiler-config`，只开
`VLLM_PP_STEP_PROFILE_INTERVAL=N`（每 N 步每 rank 打一行）。

### 7.2 崩溃：decode trace 在 `profiler_stop` 时 double free，会带走整个服务

实测（2026-11-22 02:51，`vllm-dsv41-20261122-024354.log`）：

```text
02:51:26 ...SyncActivityProfilerHandler.cpp:46] profiler_stop
02:51:26 double free or corruption (!prev)      ← glibc 堆破坏，进程直接 abort
02:52:34 Worker proc VllmWorker-0 died unexpectedly (exit code: None)
```

- 出事的是 `Worker_DP0_PP0_TP0_EP0`（全局 rank 0），它同时是全局 PG 的 **TCPStore 宿主**。
- 它一死，其余 rank 全部 "store unavailable / connection reset"；DP1 卡在
  `dp_utils.sync_cudagraph_and_dp_padding → get_dp_group().cpu_all_reduce_`（gloo）；
  两个引擎的 shm broadcast 同时报 60 s 无可用 block；随后 `EngineDeadError` →
  `ApiServer_0 died` → 主进程 `RuntimeError` → 全体退出。**没有 Python traceback，
  退出码是 None**，所以从日志上看起来像"无声猝死"。
- `POST /stop_profile` 返回的 HTTP 500 是最后一环，不是原因。
- 触发条件是 **CUDA-graph 回放的 decode trace**：同一进程稍早的 eager prefill 窗口
  （`w2_cold`）在 02:49:40 干净 stop 并完整写盘。这是 kineto/CUPTI teardown 侧的问题，
  vLLM 参数规避不了。
- 副作用：崩溃窗口里 stage 1 的 trace 只写出一部分，`dp0_pp1` 那份是被截断的 gzip，
  `analyze_traces.py` 读它会直接抛 `json` 解析错误，要剔除。stage 1 的数据后来用
  10 步窗口补齐了（§7.4）。

**不是每次都会挂**：2026-11-22 04:48 用 `max_iterations=10` 再采一次，8 个 rank
全部干净写盘、gzip 完整、服务存活。两次的差别在 flush 时机：10 步窗口在压测进行中
就 auto-stop 并导完，`/stop_profile` 到达时 worker 侧已经收尾（8 个 worker 的
`profiler_stop` 日志错开 28 s）；40 步窗口是压测刚结束时 flush 大 trace。所以可用
的规则是 **窗口小 + 等 auto-stop 导完再调 `stop_profile`**，仍不要指望它一定不挂。

**第二次崩溃（2026-11-22 08:36，`vllm-dsv41-20261122-041924.log`）——小窗口
也会挂，而且不是 double free 的 glibc 报法**：

```text
08:36:27 POST /start_profile {"profile_prefix":"coldpref","delay_iterations":0,"max_iterations":2}
08:36:27 (8 个 worker) profiler_start
08:36:27 (8 个 worker) profiler_stop          ← 2 个 worker step 就 auto-stop（空闲时 1.2 ms/步）
08:36:27 !!!!!!! Segfault encountered !!!!!!!
         ~Result → ~ExtraFields → vector<KinetoEvent>::~vector()
         → pybind ProfilerResult::dealloc_without_manipulating_gil → 进程死
```

这次窗口里**没有任何真实工作**：engine 空闲，`max_iterations=2` 被 1.2 ms 的
空转步立刻消耗掉，trace 里只有 dummy decode 的图重放——仍然是 graph replay，
照样在 teardown 段错。与 02:51 那次同一类（kineto/CUPTI 析构阶段的内存错误），
只是这次报 segfault 而不是 glibc double free。后果一样：rank 0 死 → TCPStore
不可用 → 另一个 engine 卡在 `cpu_all_reduce_` → `EngineDeadError` → 全体退出。

两条新教训：

1. **`max_iterations` 数的是 worker step，不是"请求的 step"**。空闲时 worker
   以 ~1.2 ms 的节奏空转，`max_iterations=2` 在请求到达前就用完了；想采 prefill
   窗口，必须先有请求在飞（或用 `delay_iterations`），不能靠小 `max_iterations`
   去"卡"到某个迭代。
2. **窗口小不是保护**。只要窗口里有 CUDA-graph 回放，2 步窗口一样会挂；
   唯一干净收尾过的窗口是 02:49 的 **eager prefill**（`w2_cold`）。所以规则要
   收紧成：**只采 eager prefill 窗口，任何含图重放的窗口都按"必挂"对待**，
   并且不要在活跃服务上采——先算清后果再动手。

**纪律**：cuda-graph decode trace 都按"可能挂"对待——小窗口、一次一个、
`stop_profile` 之前留几分钟；挂了就重启。也可以重启后用
`torch_profiler_activities: ["CPU"]` 只拿 host 侧。`nsys` 这条路在本机不可用：
2026.1 的 CLI 没有 attach（只有 `launch`），且容器无 `CAP_SYS_PTRACE`、
`ptrace_scope=1`。

### 7.3 三条口径修正（不修正会得出错误结论）

1. **`ncclDevKernel_Broadcast_RING` 6.34 ms/step 是采样噪声**，不是 token relay 的
   真实成本。decode trace 里每步 1.5 次 broadcast，是因为被 trace 拖慢的 stage 1
   （draft/sample）把 stage 0 卡在 token 回传上；同窗口 `token_wait` 只有 0.20 ms。
   绝对时间不能读，只有**非阻塞 kernel 的相对占比**可信。
2. **专家 GEMM 曾被算成 attention**（`analyze_traces.py` 已修）：旧的正则让
   `attention` 里的 `flash` 匹配到
   `...flashinfergemm...dense_blockscaled_gemm_sm120...`，于是专家 GEMM 全进了
   attention 桶，`moe` 只剩 0.2%。修正后（moe 先于 attention、去掉裸 `flash`、
   显式匹配 `dense_blockscaled` / `GroupProblem` / `flashinfergemm` /
   `mxfp8_quantize`）decode 的 attention 从 2.3 掉到 **0.23 ms/step**，专家段是
   **3.67**。注意 `blockscaled` 不能当判据——稠密 MXFP8 投影
   （`flashinfer::gemm::DeviceGemmMxfp8GemmSm120`）也是块缩放的。§8 的重建不受
   影响，但凡是引用旧 trace 类别表的地方都要按修正后的读。
3. **prefill trace 的绝对时间是 25× 失真**：16k 冷 TTFT 带 trace 40.6 s、不带
   1451 ms。prefill trace 只取占比，不取耗时；trace 里 stage 1 的 `pp_p2p 35.5% /
   busy 6%` 同理是 PP 气泡，不是它的真实负载。

### 7.4 采到的数（stage 0 与 stage 1）

prefill（`w2_cold`，16k，GPU 804 ms，按修正后的分类）：

| 类别 | 占比 | 备注 |
| --- | --- | --- |
| attention | 23.9% | `sparse_mla_prefill_kernel` + cake |
| gemm | 18.2% | 投影（flashinfer MXFP8 + trtllm cutlass） |
| ep_all2all | 16.6% | |
| **other_nccl** | **13.4%** | **TP=2 的 all-reduce**（`Reduce_bf16_RING` / `AllReduce_bf16_RING`） |
| moe | 13.4% | 专家 GEMM + 输入量化 |
| other | 7.0% | |
| mhc | 4.2% | |
| pp_p2p | 2.9% | |
| indexer | 0.4% | |

**other_nccl 是 TP=2 多出来的 all-reduce**：§8 的离线重建按 TP1 做，没有这一项，
所以 TP2×DP2 拓扑下 prefill 每层投影要多付一次 all-reduce，约占 1/8 GPU 时间。
decode 里同一笔开销是 `vllm::cross_device_reduce_1stage<bf16, 2>`，每步 1 次
（只在 TP0 上跑，trace 里 17.3 ms/step 是被拉长的，绝对时间不可读）。

decode（`w3_dec`，32k，两引擎各 1 请求，40 步，每步均值）：

| 类别 | ms/step | 内容 |
| --- | --- | --- |
| `pp_token_relay` | 6.34 | **采样噪声**（§7.3 第 1 条），不计入 |
| moe | 3.67 | blockscaled 专家 GEMM 2.11 + grouped 1.30 + 输入量化 0.21 |
| ep_all2all | 1.40 | AllGather 0.57 + ReduceScatter 0.83 |
| gemm | 1.06 | lm_head / 稠密投影（wmma + trtllm cutlass） |
| other | 0.90 | elementwise 记账 + memcpy |
| mhc | 0.42 | mHC |
| attention | **0.23** | cake sparse MLA |
| indexer | 0.19 | top-k |
| pp_p2p | 0.01 | |

去掉被污染的 broadcast 后，**每步真实 GPU 时间 7.86 ms**（314.5 ms / 40），其中
专家段 3.67 ms、EP 1.40（与 §8.2 的 1.08–2.70 同量级）。注意 **attention 只有
0.23 ms**：decode 的稀疏 MLA 已经很小，旧表里的 2.3 ms 是专家 GEMM 被误分类
（§7.3 第 2 条）。

stage 1 在同一次窗口补齐（`v1_dec`，10 步，**11.5 步**——trace 起止错开半步）：

| 类别 | stage 1 (PP1) | stage 0 (PP0) | 差 |
| --- | --- | --- | --- |
| moe | 4.35 | 4.83 | −0.48 |
| gemm | 2.33 | 1.15 | **+1.18**（lm_head 0.97 + 投影） |
| ep_all2all | 1.69 | 1.81 | −0.12 |
| other | 1.41 | 0.88 | +0.53（draft 的 `gemvx`、`_prepare_dflash_inputs`） |
| indexer | 0.62 | 0.27 | +0.35（5 层 vs 3 层） |
| mhc | 0.42 | 0.43 | ≈ |
| attention | 0.32 | 0.32 | ≈ |
| pp_p2p | 2.82 | 0.02 | stage 1 在 recv 上等 stage 0（trace 放大） |
| 纯计算合计 | **9.45** | **7.88** | **+1.6** |

**stage 1 的关键路径多 1.6 ms/步，但不在 MoE 上**（MoE 反而略少）——是 lm_head、
多 2 层 indexer、DSpark draft 的固定开销。这解释了 §5.1 的 20% 不均衡，也说明
再优化 stage 1 的方向是 indexer/draft/lm_head，不是专家 GEMM。

两点口径：这次窗口的 stage 0 是 **9.71 ms/step**，比 40 步窗口的 7.86 高 24%。
差异集中在 grouped GEMM（1.30 → 2.49）和 EP reduce-scatter（0.83 → 1.34），而
blockscaled 专家 GEMM 两次完全相同（2.11）——不是整体变慢，具体原因（两请求收尾
不同步、capture 起点不同）没有定死。所以**跨窗口不要混用**：上表两列都取自
`v1_dec`，40 步窗口的 7.86 ms 只用于 stage 0 的绝对值。

**与 §8 的 6.04 ms 专家流式模型不一致**（实测 3.4 ms，即 170 µs/层）。按 DRAM 墙
（1054 GB/s）反推，170 µs 对应每层约 179 MB ≈ **10 个不同专家**，而 §8 的模型是按
**18 个 `(row, expert)` 对** 算的。两者差在"对"和"不同专家"：同一层里重复命中的
专家只读一次权重，所以 302 µs/层是**上界**，6.04 ms 也是上界。要定死得在 2 请求窗口
里打开 `aux_output_config.enable_return_routed_experts` 数实际 distinct expert；
在那之前，**既不要说"专家 GEMM 已优化"（3.4 是这次窗口的命中数），也不要把
6.04 ms 当实测**。

同 shape 的无 trace 对照（新进程，32k ctx，两引擎各 1 请求）：TTFT 3.06 s，引擎
ITL 均值 20.3 ms / p50 17.5 ms；1k ctx 是 TTFT 140 ms、ITL 21.5 ms。trace 把步长
放大数倍，所以绝对时间只能横向比类别，不能和这张表对齐。

### 7.5 仍然缺的

stage 1 的 decode 分解已经补齐（§7.4，`v1_dec` 的 10 步窗口没有崩）。还缺三样：

1. **非 kernel 时间与 PP 等待没有分开**：stage 0 的 7.86 ms GPU 对 9.5–10 ms 的
   步长，差的 ~2 ms 里大部分是它在等 stage 1（stage 1 的 GPU 总量按同窗口比例折算
   约 9.0 ms，见 §8 末），只剩 ≤1 ms 可能是 launch、同步与采样。要分开需要
   `torch_profiler_activities: ["CPU"]` 或 P1-6 的图内计时。
2. **overlap 关系**：投影是否与集合通信重叠（§8 的 12–13.7 ms 模型 vs 7.86 ms 实测
   已经暗示有重叠），需要 CPU timeline 或 NVTX。
3. **每层实际 distinct 专家数**：决定 3.67 ms 与 §8 的 6.04 ms 上界之间落在哪
   （`aux_output_config.enable_return_routed_experts`，2 请求窗口即可）。

### 7.6 host CPU 每步（/proc 口径，2026-11-22）

P1-5 的"先量"做过一次，方法是 `/proc/<tid>/stat` 的 utime/stime 加
`vllm:iteration_tokens_total_count` 的精确步数（`/tmp/probe_cpu_step2.py`，
脚本里写死了当次进程的 PID；重跑要先填新 PID）：

| 请求数（每 engine） | EngineCore 主线程 user CPU/步 | 步长 |
| --- | --- | --- |
| 1 | 15.6 ms | 19.4 ms |
| 4 | 11.4 ms | 14–16 ms |
| 16 | 9.3 ms | 12–13 ms |

主线程 91–99% 时间在 R 状态，user ≫ sys；4 个 worker 进程合计
58–84 ms CPU/步（NCCL 忙轮询）；空闲 dummy 循环只有 ~14%。

**结论：CPU/步 ≈0.75–0.8 × 步长，且不随请求数增长**——主体是等 stage 时的忙轮询，
不是元数据记账（记账应随行数增长）。所以 /proc 口径**无法**拆出 P1-5 的成本，
能得到的上界只有：元数据 ≤15.6 ms 且完全被 GPU 执行覆盖（1 请求时 15.6 < 19.4，
4/16 请求时余量更大）；叠加 §7.4 的 ≤1 ms 非 kernel 缝隙，P1-5 的天花板不大。

**步内口径确认（2026-11-22 08:30，`VLLM_PP_STEP_PROFILE_INTERVAL=1`，
`/tmp/probe_step_profile.py`、`/tmp/probe_step_profile16.py`，读日志里的每步
span，无重启、无 trace）：** `execute` 的 CUDA 事件包住整个
`model_runner.execute_model`，所以 host 侧的输入元数据组装与 launch 只要拖慢
首个 kernel，就落在 execute 里；`step − execute − sample` 就是调度与元数据真正
加到关键路径上的部分。

| ctx | 请求/engine | pp | step | execute | token_wait | sample（含 draft） | step − execute − sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16k | 1 | 0 | 11.07 | 10.87 | 1.59 | — | **0.20** |
| 16k | 1 | 1 | 14.02 | 8.89 | 0.00 | 2.98（2.31） | ~0.00（recv 2.15） |
| 16k | 16 | 0 | 15.42 | 15.41 | 2.05 | — | **0.01** |
| 16k | 16 | 1 | 16.19 | 13.01 | 0.00 | 3.01（2.29） | ~0.10 |
| 16k | 32 | 0 | 18.23 | 18.22 | 2.08 | — | **0.01** |
| 16k | 32 | 1 | 19.70 | 16.11 | 0.00 | 3.28（2.55） | ~0.22 |
| 64k | 32 | 0 | 19.75 | 19.74 | 3.01 | — | **0.00** |
| 64k | 32 | 1 | 21.01 | 17.18 | 0.00 | 3.36（2.57） | ~0.47 |
| 128k | 32 | 0 | 21.60 | 21.59 | 2.42 | — | **0.00** |
| 128k | 32 | 1 | 22.39 | 18.97 | 0.00 | 3.38（2.60） | ~0.04 |

（32 请求 = `--max_num_seqs 64` 打满；日志各 rank 98–216 个忙步的中位数。
64k/128k 用同一前缀 + 4 个不同后缀的并发请求，`/tmp/probe_step_profile_long.py`；
128k 冷填充 5.76 s，即 0.044 ms/token，比旧进程的 0.08 快约 1.8×。）
同一窗口的 worker CPU：pp0 每个进程 2.65–2.80 s / 2.83 s ≈ 1 核，pp1 4.05–4.16 s
≈ 1.45 核——host 一直在烧 CPU，但 GPU 流是 84–99% busy，CPU 完全被覆盖。

**P1-5 判定：** 元数据 + launch + 调度在关键路径上 ≤0.2–0.5 ms，占 11–22 ms 的
1–2.4%；换成单 kernel 也动不了这个量级（省的是被覆盖的 CPU，不是步长）。
**达不到 >5% 门槛，P1-5 关闭。** 加长上下文只改 GPU 侧：32 请求下 16k → 128k
让 pp0 的 step 从 18.2 → 21.6 ms，增量全落在 execute 里，host 缝隙仍在
0–0.5 ms（block table 从 256 → 2048 块/seq 只带来这点增量）。顺带确认两点：
`sample` 里的 `draft` = 2.3–2.6 ms 且不随 batch/ctx 变（3 层 draft + 129k
lm_head，按 0.8 ms/层算是真算力，不是 host 开销）；batch 从 1 → 32 只让 pp0 的
execute 从 10.9 → 18.2 ms（×1.7），加行的边际成本仍是亚线性的。

**读表时的一个坑（2026-11-22 08:4x）**：只看一个 rank 会看到
`step 129–138 ms、execute 0.00` 的怪行——那不是丢 span，而是**没接到请求的
DP engine** 在等另一个 engine 的批（`coordinate_batch_across_dp` 的锁步）。
把 4 个 dp rank 一起打出来就清楚了：请求落在 dp1 时，dp1 pp0 = 67.9 ms
execute、dp1 pp1 = 68.0 ms，同时 dp0 的 pp0/pp1 是 `step 129.8 / 138.5,
execute 0.00`。所以任何"某步 step 很大但 execute 为 0"的样本，先确认它是不是
空闲 engine 的等待步，再当成时间数据用。

## 8. 步内构成：离线重建（2026-11-21）

线上进程当时挂不了 profiler（§7），所以用 `/tmp/dsv41_step_parts.py` 在同样的 4 张卡上
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

专家那一行按 18 个 `(row, expert)` 对算，**是上界**：同层重复命中的专家只读一次
权重。2026-11-22 的 trace 实测专家段 170 µs/层（≈10 个 distinct expert，§7.4），
所以真实的专家流式是 3.4–6.04 ms 之间的某个值，取决于路由命中分布。

投影这一行是按线上 kernel 与线上 shape 直接测出来的（§8.1），旧版的 58 µs / 1.16 ms
是用 `torch._scaled_mm` 代理、且 shape 取错（`o` 用了 `[5120,1024]`、漏了 wo_a）得到的
低估值。换成实测后总和 12.0–13.7 ms **超过**了原来「ITL / 2 ≈ 9.5 ms」的锚点，
锚点本身要重算：

- 非流式 128 token 请求实测 60 个 iteration / 1.20 s → **20.3 ms/iteration、
  2.13 token/iteration、TPOT 9.5 ms**（另一条 prompt 上 18.9 ms/iteration、2.95 token）。
- 若一个 iteration 里 PP 两级串行，单级通过 ≈ 10.2 ms，模型算的 12–13.7 ms 仍高 2–3.5 ms。
  只可能来自两处：18 个专家的流量被 L2 部分吸收，或投影跑在 aux stream 上、
  与集合通信窗口重叠。2026-11-22 的 trace 给了半个答案（§7.4）：一个 decode 步的
  **真实 GPU 时间只有 7.86 ms**，低于模型的 12–13.7 ms，说明逐项相加重复计入了被
  overlap / L2 吸收的部分。多出来的 ~2 ms 则主要不是 launch，而是 **stage 0 在等
  stage 1**：stage 1 的 GPU 总量按同窗口比例（11.14 / 9.69）折算约 9.0 ms/步，比
  stage 0 多 ~1.2 ms，因此 `step ≈ stage 1 的 GPU 时间 + ≤1 ms 的 launch/同步/采样`
  ——后者只是余量，还没有测到。

indexer（§6，0.64 ms）与 sparse MLA 未单列，填在上面 2.4 ms 的误差带里。

三条硬约束：

1. **MoE 已经在带宽墙上。** 纯 copy 128 MiB 是 1088 GB/s，专家 GEMM 是
   1054 GB/s（fp8 per-expert 循环）——6 ms 不是 kernel 慢，是要读的字节数。
   每层 18 个 `(row, expert)` 对 × 17.7 MB 是**上界**（fp4 压缩后的量）：同层重复
   命中的专家只读一次权重，trace 实测 170 µs/层 ≈ 10 个 distinct（§7.4），所以
   真实值在 3.4–6.04 ms。唯一杠杆仍是摊薄：每 EP 组从 12 行增到 24 行，命中专家
   约 18 → 34（流量 +90%），而 token 是 4 倍。
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
三条 `mxfp8_gemm`。2026-11-22 复查（全量读了 60 个 json）：cache 覆盖的是 vLLM
自己的 hybrid 档位 M ∈ {1,2,4,…,…,2048, 2560,…,4096, 8192}（21 档 × 7 个
shape），runner 只有 `CutlassMxfp8GemmRunner` 和 `B12xMxfp8GemmRunner`，
**没有 cudnn 条目**。运行时 M 会先**向上取档**再查 cache
（`DynamicTensorSpec.map_to_hybrid_bucket_uncapped`：6→8、36→64），所以
「M=6 不在档位里」不影响命中：现役服务器整份 15 GB 日志里 `No tuned config
covers` **0 条**、`Config cache hit for mxfp8_gemm` 16 条（每 op×runner 只打
一次），即 decode 走的是**调过的 tactic**，不是默认 tactic。

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
而 `auto` 在 SM12x 上按 shape 把合适后端排成 `[b12x, cutlass, cudnn]` 的子集
（`_heuristic_func_mm_mxfp8`），tiny-M 上不同 runner 差 2–7 倍。**benchmark
是在裸进程里跑的**（没有加载 vLLM 的 autotune 文件），所以它打印过
`No tuned config covers mxfp8_gemm input_shapes=((6,5120),…) … falling back to
runner=CudnnMmMxfp8GemmRunner tactic=-1`，而 wq_b/wo_b 那两条落到
`… runner=CutlassMxfp8GemmRunner tactic=-1` —— fallback 走哪个 runner 取决于
该 shape 的 suitable 列表。**线上不会走 fallback**（见上：0 条 warning），所以
90 µs 这组数说明的是「auto 选到的执行路径」比 pinned cutlass 快，而不是
「线上在用默认 tactic」。runner vs tactic 的归因还没做，不要据此断定线上是哪个
kernel。

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

**逐 backend 实测（2026-11-22，`/tmp/bench_mxfp8_backends.py`）**：三个线上 shape
（TP2 后）× M（表里只列 1/8/64 与 512/1024/8192，6 和 16 也跑过，结论不变），
裸进程、CUDA graph replay、权重常驻 L2，
绝对值偏乐观，只看相对排名；四个 backend 的输出与 cutlass 逐位相同（maxdiff = 0）。
单位 µs，每行最快者加粗。

decode 档：

| shape (N×K) | M | cutlass | cudnn | auto | b12x |
| --- | --- | --- | --- | --- | --- |
| wq_b 16384×1280 | 1 | 80.6 | 32.7 | **11.3** | **11.3** |
| | 8 | 75.8 | 31.7 | 9.3 | **9.2** |
| | 64 | 53.8 | 26.7 | 14.3 | **14.3** |
| wo_b 5120×4096 | 1 | 80.6 | 30.8 | **10.3** | **10.3** |
| | 8 | 71.4 | 30.8 | **10.2** | **10.2** |
| | 64 | 28.4 | 30.8 | **12.3** | **12.3** |
| fused_wqa_wkv 1792×5120 | 1 | 33.8 | 37.5 | **7.2** | **7.2** |
| | 8 | 31.8 | 37.5 | **12.2** | 12.3 |
| | 64 | **14.4** | 37.3 | 13.3 | 13.3 |

prefill 档：

| shape | M | cutlass | cudnn | auto | b12x |
| --- | --- | --- | --- | --- | --- |
| wq_b | 512 | 78.8 | **53.6** | 59.3 | 60.6 |
| | 1024 | 151.3 | **111.2** | 117.4 | 120.3 |
| | 8192 | 1369.7 | **923.2** | 1021.6 | 1022.7 |
| wo_b | 512 | 80.3 | 61.5 | **55.4** | 57.3 |
| | 1024 | 151.6 | **102.7** | 116.6 | 118.9 |
| | 8192 | 1335.4 | **851.0** | 993.6 | 994.1 |
| fused_wqa_wkv | 512 | 42.1 | **37.9** | 41.0 | 40.9 |
| | 1024 | 70.6 | 76.7 | **61.2** | 61.1 |
| | 8192 | 579.4 | **368.8** | 427.5 | 427.6 |

读法：

- **decode（M ≤ 64）：`b12x`（及与之同路径的 `auto`）最快**，wq_b/wo_b 是 pinned
  cutlass 的 0.12–0.43×，fused_wqa_wkv 是 0.21–0.92×。唯一例外是 fused_wqa_wkv
  M=64，cutlass 反超（14.4 对 13.3）。cudnn 在 M≤64 有 ~31–38 µs 的固定底
  （plan/launch），M 再小也不降，decode 档不可用。
- **prefill（M ≥ 512）：`cudnn` 在 9 格中赢 7 格**（cutlass 的 0.64–0.90×），
  两格外包：wo_b M=512 的 `auto`（55.4）和 fused_wqa_wkv M=1024 的 `auto`/`b12x`
  （61.1–61.2）。b12x 在 prefill 多为 cutlass 的 0.71–0.97×。
- `auto` 在裸进程里与 `b12x` 数值同步（每格差 ≤ 3 µs，decode 档几乎重合），但这
  不能推广成「auto = b12x」：§8.1 那次裸进程 benchmark 的 fallback 日志里就有一条
  shape 落到 `CudnnMmMxfp8GemmRunner`。没有任何 tuning 命中时 `choose_one` 退回
  suitable 列表第一个，归因随 shape 变；线上有 tuned cache（且 cache 里只有
  Cutlass/B12x 两个 runner），auto 的 runner 要按 shape 查 cache 才能定。
- `b12x` 的资格在这台机器上全部满足：CUDA 13.2 ≥ 13、`nvidia-cutlass-dsl` 4.8.0
  ≥ 4.6.0、1D 128x4 尺度、K%128==0（5120/1280/4096）。它按 shape 做 CuTe-DSL JIT，
  **首次调用必须落在捕获之外**——与 cuDNN plan 同类，见 §5.7.1。

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
- `pack` 时期的现网 ITL 均值 46 ms 与下限之间的差额主要来自调度，不是算力：
  31.5% 是 free-step prefill（§5.4），1.0% 是长 prefill（§5.3）。换成
  `balance` 后现役均值 **22.7 ms**、50–75 ms 档为 0（§5.4），**调度附加基本
  清掉了**，剩下的是 2 × step 的结构下限本身。

所以是两条互不替代的线：**降步长**（§8 的 3–3.5 ms 可优化部分，§9.2）和
**降调度附加**（pack 与 free-step 分配，§9.3）。混在一起优化会互相掩盖。

### 9.1 立刻可做（不改代码、不重启）

| 事项 | 目的 | 预期 |
| --- | --- | --- |
| 离线重建已经完成，见 §8 / §8.1 / §8.2 | 步内拆分不再需要 profiler 才能开工 | 已定出 44–50/28–32/9–20/5–6/2–3 的排序 |
| indexer top-k 后端对比（`/tmp/bench_indexer_topk.py`） | 5 层 × 97 µs = 0.5 ms（§6） | **已完成**，见 §6.1：`per_row` −0.30 ms / `flashinfer` −0.32 ms（128k）。16:00/16:50 已上线试过，17:12 回退（§5.3.1）：prefill +40 ms，未单独归因 |
| TTFT 尾部归因：prompt 长度 × 冷/热矩阵（`/tmp/probe_live.py a2`） | 拆开「prefill 计算」与「LMCache 写入」 | **已完成**，见 §5.3：冷 prefill 0.072–0.094 ms/token，写缓存不在 TTFT 路径上；尾部就是 128k–512k 长 prompt |
| 单请求 vs 4/8 请求的步长斜率（`/tmp/probe_live.py a3`） | 验证专家流式假设（§8 约束 1） | **已完成**，见 §5.5：行数 ×4 步长只 +56%，批大小与 k 都便宜 |
| 采样段占比（P0-3 的前置问题，`/tmp/bench_sampler_ops.py`、`bench_sampler_temp.py`） | 融合采样值不值得做 | **已完成**，见 §6.2：整条 temp=0 链 0.2%、temp>0 最坏 2.3%，**P0-3 关闭** |
| 路由器会话亲和（P1-8）：session/前缀 → DP rank 粘滞，命中失败回退 least-loaded | agent 每轮省首次跨引擎取 KV | 8k/16k/32k 分别 −38/−82/−129 ms（首轮一次），引擎侧已支持 `X-data-parallel-rank`，见 §5.6.2 |

### 9.2 代码级（按 §8 的占比排序）

1. **注意力投影：把写死的 `backend="cutlass"` 换成 `auto`**（3.87 ms，28–32%）：
   §8.1 的整链实测 194 µs/层 → 90 µs/层，**−2.07 ms/步**（ITL −2×2.07 ms，
   TPOT 约 −18%）。输出逐位相同，M=1…1024 全档不亏。**已实现**：
   `VLLM_MXFP8_GEMM_BACKEND=auto`（默认保持 `cutlass`，便于回滚与 A/B）。
   因为所选 runner 的 plan 必须在 graph capture 之前建好，重启后要确认
   autotune cache 文件被加载、decode 档命中——2026-11-22 全量日志复查：命中
   （`No tuned config covers` 0 条），运行时 M 向上取档后覆盖 decode 行数，
   这一条**不需要新建 tuning bucket**。剩下的只是次级空间：取档会把 36 行按
   64 行的点调优、以及 cudnn runner 一条 tuned 条目都没有（一旦某个 shape 的
   suitable 列表把 cudnn 排第一就退回默认 tactic）。要动先离线量。
   **14:45 重启已带上**（与第 2 条同时开），重启后单请求 ITL p50 17.5 ms
   （§5.5），比之前的 19–20 ms 低；但两条一起开无法单独归因，要分项需要
   再各重启一次。
2. **Ragged dispatch 打包成一次 all-gather**（1.08–2.70 ms，9–20%）：三个张量本来
   就在同一个 `pynccl` group 里，不是「三次独立 AG」；贵的是 `sizes` 不等长时
   `all_gatherv` 退化成 3 × 4 次 `ncclBroadcast`（§8.2）。按行打包后一次广播/rank，
   **110.5 → 65.4 µs/层 = −0.9 ms/步**（只在不等长的步上），数值逐位不变。
   **已实现**：`VLLM_MOE_AG_RS_PACKED_DISPATCH=1`（默认关），14:45 重启已带上；
   只在两个 DP 引擎的 token 数不等时才有收益。**线上步长里这部分看不到**：
   受控波次 5+5 vs 4+6 只差 ~2%（§5.5 末），所以不要按这 0.9 ms 去规划。
3. **mHC**（0.72 ms，6%）：graph 内 18 µs/boundary 已经不高，`hc_sinkhorn_iters=20`
   是唯一旋钮（有精度风险）。注意下一层的 mHC pre 依赖 combine 之后的 MoE 输出，
   基本没有可重叠的窗口，不要按「藏进通信窗口」来设计。
4. **lm_head**（0.32 ms，仅 PP1）：331 MB fp8 一次读完，已在带宽墙，TP2 已切开，
   没有空间。
5. **MoE**（6 ms，48%）：无 kernel 级空间（§8 约束 1）。它只吃两件事：
   批大小（摊薄）与 tokens/step。多 1 行 verify 只让命中专家 +2~4%（约
   +0.1 ms/步），所以**在行数不变的前提下**加 token/step 是便宜的；但这不等于
   "把固定 k 调大"——k 每加 1 还要多一段**串行** draft（3 层 + lm_head，全在
   PP1），见 §5.3.1，线上 k=7 就是这么变慢的。要花 verify 预算就用
   `enable_adaptive_verification` 按位置分配。反之 k=5 → 4 省不下步长
   （token 项只有 0.07 ms/token），只降 token/step。
6. **采样段**（见 §6.2）：整条 temp=0 采样链 16–20 µs，temperature>0 的完整
   链路 6–222 µs，即步长的 0.06–2.3%；P0-3 融合采样的上限 <1%，已关闭。
   其中最大的一项是 `apply_min_p` 单算子 71 µs，若将来有人用 min_p 再单独看。

### 9.3 一次维护窗口（带 profiler 重启）

这不是开工前置了（§8 已经给出拆分），trace 主要用于验证 collectives 与投影的
overlap。**这个窗口 2026-11-22 已经用掉**（§7）：prefill、decode、stage 1 都采到了，
其中 decode 的 40 步窗口在 stop 时 double free 打死过服务（§7.2），stage 1 是用
10 步窗口补齐的。若还要再采，按 §7.2 的纪律来。下一次重启时顺手要做的：

- `VLLM_PP_MICROBATCH_BALANCE=balance`：**现役进程已经开着**，实测 50–75 ms 峰
  为 0、均值 22.7 ms（§5.4）。代价是 free step 不再独占 prefill → TTFT 与吞吐
  变差，这部分还没量化，需要和 `pack` 做一次带 TTFT/吞吐的 A/B。
- **`enable_adaptive_verification: true`（加进 `--speculative-config`）**：k 保持 5，
  由置信度在 batch 内分配 verify 槽位（§5.3.1），低置信度请求少花行、高置信度
  请求多探一格。**不要再用固定 k=7**：线上开过，批一大就慢（draft 逐位串行，
  第 6/7 位接受率 ~0.19）。槽位数上限仍是 k=5，所以只会变便宜；风险是置信头
  标定不准时会错砍本可接受的槽位，A/B 要看 token/iteration 与 TPOT 两个数。
- `VLLM_PP_STEP_PROFILE_INTERVAL=1`：唯一能定位 §5.3 那 5× 慢 prefill 的手段
  （`execute` vs `recv_wait`/`token_wait`），重启即生效，代价是每个 worker 每 N 步
  打一行。**当前进程（2026-11-22 04:19 起）已经开着**，慢态没再复现，下次出现时
  先 `grep "PP step profile"` 取那几行。
- **`VLLM_USE_BREAKABLE_CUDAGRAPH=1`（A/B，卡在等重启）**：prefill 形态 step 现在没有
  CUDA 图（§5.7），每级固定 ~65 ms。这个开关会打开 breakable piecewise 图
  （同时把 `CompilationMode` 置 NONE），是本模型唯一的配置层入口；DSv4.1 的
  SWA replay 机器已经会构造 `BreakableCUDAGraphWrapper`。要看三件事：
  a) 启动是否还降级为 `FULL_DECODE_ONLY`；b) 冷 1-token / 512 / 1024 的墙钟
  是否从 138–166 ms 掉下来；c) decode 的 20 ms 步长与 ITL 是否退化。预期上限：
  每轮 200–400 新 token 的 130 ms 固定价 → ~50–60 ms。
  **14:14 那次 A/B 崩在 capture 里建 cuDNN plan（§5.7.1）**，已实现「按 desc 预 warm」
  （`cudagraph_utils.py:747-762` 调模型侧的 `warmup_capture_ops`，DSv4.1 两个
  入口类都已实现）并在 op 级验证（无预 warm → `CudnnPlanBuildInCaptureError`，
  有预 warm → capture 成功）。**15:16 起现役进程就是 breakable + `auto`**：
  PIECEWISE 27/27、FULL 41/41 全捕获，无 `CudnnPlanBuildInCaptureError`。
  冷 prefill 实测（现役进程、全新 seed，墙钟/`request_prefill_time_seconds`）：
  1 token 31/21 ms（基线 138），128 61/52（141），512 108/99（146），
  768 138/129（基线 ~140），1024 170/160（166），2048 289/278（286），
  8k 917–923 ms（0.112 ms/token），16k 1444–1664 ms（基线 1450–1575）。
  即 **≤768 token 明显变快，≥1024 基本持平**；同窗口另有一次 8k 落到 1242 ms
  的离群值，属 §5.3 的间歇慢态（重测 5 次稳定在 917–923）。decode 中位步长
  8.6–10.1 ms（基线 9.9–10.0），未退化。
  另需留意：`auto` 在无 tuning cache 时退到 b12x，而 b12x 也是按 shape JIT——
  首次调用同样必须落在捕获之外。常规 op 已被逐 desc 的 eager warmup 覆盖，
  捕获才独占的 op 现在由预 warm 钩子覆盖（`autotune_indexer_wq_b` 走的是同一条
  `mm_mxfp8` 分派，`auto` 选到谁就在那里建谁的 kernel）。
- **小外部命中同 step 装载（A/B，已修一个契约 bug，等重启）**：
  `lmcache.mp.sync_load_max_tokens`（§5.6.1），实现在
  `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_sync_load.py`。
  启用方式是把 `kv_connector_module_path` 指到该模块（类名仍是
  `LMCacheMPConnector`），加 `lmcache.mp.sync_load_max_tokens: 128`、
  `lmcache.mp.sync_load_timeout_s: 60`。默认 0 = 原行为，可随时回退。
  **14:53 首次启用崩在 scheduler 的 finished-recv assert（§5.7.2）**：
  worker 按 op 跨度、scheduler 按 `need_to_load` 划分，二者在 APC 命中非
  chunk 对齐时差最多 255 token。已改为把 scheduler 的判定随 metadata 传给
  worker，单测通过；重新启用仍需重启。测试：
  `python -m pytest tests/v1/kv_connector/unit/test_lmcache_mp_sync_load.py -q`
  （本机是 `/opt/venv/bin/python`，12 passed）。

### 9.4 调度与运维

- `VLLM_PP_MICROBATCH_BALANCE` 的两种取值是 §9.3 的 A/B；`VLLM_PP_NUM_MICROBATCHES`
  是吞吐/延迟旋钮，不是治不平衡的手段（§5.1 表）。
- 空闲 dummy 循环（§5.2）：不影响 TPOT，但让所有 GPU 利用率观测失真、整机待机
  ~1.5 kW。需要查 `pending_pause` 共识或 connector push work 为什么没让引擎静默。
- 会话亲和（P1-8，DP2 负载均衡按会话粘住）：agent 类负载每轮重发整段对话，
  命中本身已不花钱（§5.6），但块落在另一台 engine 时要从 LMCache 取，16k +77 ms、
  64k +215 ms，比本地命中（84–103 ms）贵一倍以上。这是 agent TTFT 现在的头号
  杠杆，排在 §5.6 的小 chunk 固定开销（~50 ms）之前。**2026-11-22 已定量
  （§5.6.2）**：按单轮算 8k/16k/32k = +38/+82/+129 ms，只有落到新 rank 的首轮
  付，第二轮就恢复本地速度；引擎侧 `X-data-parallel-rank` 已经可用，剩下的是
  路由器粘滞（§9.1）。

### 9.5 已排除（不要再花时间）

- `VLLM_PP_LAYER_PARTITION=22,18`：relay 起不来（§5.1）。
- `VLLM_PP_NUM_MICROBATCHES=3` 当不平衡修复：见 §5.1 表。
- MoE kernel 级优化：已在 1054 GB/s，纯 copy 只有 1088 GB/s（§8）。
- indexer SM120 FlashInfer logits（`indexer_kv_dtype=mxfp4` 下 inert，≥128k 只打平）、
  FP8 cache writer（每档都更慢）、mega gate（−2 µs/层）：放在 20 ms 步长上都是噪声（§6）。
- 融合采样（曾记为 P0-3）：整条采样链 ≤2.3% 步长，fused 只在纯 top-p 批上划算，
  见 §6.2。
- 自建 SWA 窗口 replay 的 host 侧状态块（曾记为 P0-4）：本部署里 LMCache
  `store_replay_windows` 已经把命中时的 128-token replay 消掉了，实测 computed
  = 2、命中 TTFT 84 ms 比冷 128 的 151 ms 还低（§5.6）。只剩"不配 LMCache"或
  "窗口没存住"这两种部署才需要它。
