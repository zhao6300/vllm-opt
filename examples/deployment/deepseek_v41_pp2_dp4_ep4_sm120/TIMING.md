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
完成事件与成功标志（`DeviceMessagingFuture.wait_on_stream`），把 ≤N token 的尾巴改成在
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

#### 5.7.6 一次求解上线后：6000+ 的 target 把长 prefill 期间的 ITL 拖到 1–2.5 s

**01:06 重启**（含 §5.7.4/§5.7.5 全部改动）后的 target 轨迹：

```text
01:09:39  2048 -> 8192  (174.5 ms / 700 ms, 57 个 informing 步)   # ×4 上限，一步打满
01:12:03  8192 -> 5120  (949.4 ms, 15 步)
01:13:40  5120 -> 6144  (589.1 ms, 23 步)
之后停住：589 ms 距 700 ms 只差 16%，但窗口里够不到 min_samples=4 个 informing 步
```

**冷 TTFT**（02:05，单请求、孤立、client 侧）：

```text
n= 1024 167   n= 2048 295   n= 2560 362   n= 3328 467   n= 4096 584
n= 4864 680   n= 5120 691   n= 8192 989   n=16384 1584
```

对比 target 4864 那组（8192 921、16384 1512）：**8k +7%、16k +5%**，就是「略高于之前」的量级。

**ITL**（engine 侧 `vllm:inter_token_latency_seconds`，取前后差值）：

```text
4 路 decode 独占              ：p50/90/95 ≤25 ms，98.2% ≤25 ms，均值 21.5–25.7 ms
4 路 decode + 1 个 16k prefill：p50 ≤25 ms，p95 ≤1.0 s，均值 ~90 ms，
                                1–2.5 s 档 4.5%，500–750 ms 档 0.65%
进程生命周期（47,213 样本）   ：p99 ≤50 ms，p99.9 ≤2.5 s
```

参照 11-22 的 `balance` 基线（98.9% ≤25 ms、均值 22.7 ms）：**decode 本身没退化，退化的是长 prefill 期间**。

**机制**：target 收敛到「一步 700 ms」。一个 16k prompt 是 3 块（6144+6144+4096），每步 589–949 ms，同 rank 的 decode 只能等这一步结束，于是一次 16k prefill 期间出现 3 个 1–2.5 s 的间隔。prefill 的总工作量没变，用户看到的是 ITL 尖峰。

**同一条曲线的另一种读法**（孤立冷 prefill 每块 `a + b·n`，用 1024→4096 拟合 a≈20 ms、b≈0.135 ms/token；PP2 下 K 块的总时间 ≈ `t(n/K)·(K+1)/2`）：

```text
target 6144：8k = 2 块 -> ~990（实测 989）；16k = 3 块 -> ~1510（实测 1584）
target 1500：8k = 6 块 -> ~740；      16k = 11 块 -> ~1330
```

即**这台机器上小块并不亏 TTFT**：每块固定成本只有 ~20–40 ms，PP2 又把块间重叠起来；而 ITL 尖峰会从 1–2.5 s 降到 ~250–500 ms。700 ms 的 budget 是 TPOT 的直接来源，不是吞吐来源。要 A/B 的：`VLLM_PREFILL_STEP_BUDGET_MS=700` vs `250`，测同一组数字（冷 TTFT 1k/8k/16k + 4 路 decode 伴 1 个 16k prefill 的 ITL 直方图）。

**顺带修的鲁棒性**：真实流量下拟合会给出 `fixed 131–176 ms`（01:31、01:32 两轮），而 target 2048 的实测步长只有 174.5 ms。budget 降到 250 时 `period - fixed` 会被压到 5% 下限，比值 ×5.9，一步又跳到 8192。求解现在用 `fixed = min(fit, period/2, budget/2)`，有单测覆盖。

#### 5.7.7 按负载选两种步长：没有 decode 时全速 prefill，有 decode 时压低步长

设计的本意是平衡 prefill 与 decode，但 §5.7.6 说明了单一 budget 做不到：target 就是「一步多久」，而这个「多久」对两种负载的意义相反——

- **没有 decode**：一步里只有 prefill，步长只由吞吐决定，越大越省每块固定成本（~20–40 ms/块，拟合高估到 130–176 ms 时更明显）。
- **有 decode**：同 rank 的 decode 要等整步结束，步长直接就是 ITL 尖峰。

所以 `PrefillTarget` 现在有两个周期：`target_step_ms`（= budget，decode-free 时用）和 `decode_step_ms`（`VLLM_PREFILL_TARGET_DECODE_MS`，默认 0 = 关）。每次更新用窗口内所有步的 decode 数中位数判模式：中位数 > 0 就解到 `decode_step_ms`，否则解到 budget。判据只看最近的记录步，不引入额外状态；切换由 200 ms 的线程在一次更新内完成（±4× 上限），拟合轮再把这批样本合并给全组对齐。

两个周期都走同一条一次求解公式，所以不动点分别是「decode 步 250 ms」和「无 decode 步 700 ms」。代价是切换时的一步：target 还停在 6144 时来了 decode，那一块仍按 6144 切（~900 ms），下一次更新就缩到 ~1500。

A/B 计划（需重启）：

```text
A: VLLM_PREFILL_STEP_BUDGET_MS=700                       （现状，已测：见 §5.7.6）
B: VLLM_PREFILL_STEP_BUDGET_MS=700 VLLM_PREFILL_TARGET_DECODE_MS=250
C: 同上但 150（若 B 的 ITL 仍然偏高）
```

测同一组：冷 TTFT 1k/8k/16k（孤立）+ 4 路 decode 伴 1 个 16k prefill 的 engine 侧 ITL 直方图（`/tmp/probe_itl_hist.py`）+ 纯 prefill 突发时的 TTFT（验证「没有 decode 时全速」还在）。

### 5.8 LMCache 恢复的 KV 是否正确（2026-11-23 审计：内容对，失败通道有洞）

起因是 §6.3 的 T=0 不逐位复现被怀疑成「LMCache 把别的请求的 KV 读进来了」。
结论分两句：**恢复的 KV 内容正确**，可以排掉；但自定义的 same-step 载入
（`lmcache.mp.sync_load_max_tokens`，§5.6.1）在这个模型上的失败路径是空的，
一旦失败会静默用没载入的块继续算。代码已改，**重启后生效**（§9.3）。

**为什么 19 个 group 决定了失败通道。** 启动日志：

```text
Resolved LMCache MP geometry: group_tokens_per_block=[32,32,32,32,32,32,32,32,
32,32,32,32,32,32,32,128,0,0,64], scheduler_block_size=128
```

（15 个 32 的窗口组 + 一个 128 + 两个 0 的 scratch 组 + 一个 64。）

block-outermost 打包后是 **19 个 KV cache group**。这种布局下 LMCache 连接器
把 `_report_request_errors` 置真，于是失败只能走请求级通道：

- `get_block_ids_with_load_errors()` 恒返回空集（块号只在组内唯一，vLLM 不允许
  扁平上报），`_handle_invalid_blocks` 对 >1 组直接 `RuntimeError`；
- `KVConnectorTransferResults.failed_recving` → `_handle_failed_recving` 只恢复
  状态是 `WAITING_FOR_REMOTE_KVS` 的请求。

**怎么排掉「读到错误 KV」**，三条独立证据：

1. 漂移复现本身不碰 LMCache：38-token prompt 每次 `cached_tokens: 0`，
   `vllm:prompt_tokens_by_source_total` 只有 `local_compute` 在动。
2. 语义召回：587-token prompt 第 ~110 token 处埋访问码（`ZQX-…`），dp0 写入后
   dp1 从 LMCache 恢复（`cached=512`）回答 **2/2 正确**；全程冷算 **3/3 正确**。
   字节错位会先毁掉这种召回。
3. 分歧幅度 A/B：restored-vs-cold 与 cold-vs-cold 的首 token top-5 偏差
   0.25–1.375 vs 0.5–1.125，32 token 续写的首次分歧位置同量级；最扎眼的
   pos-3 近似并列翻转（`" continue"` −0.93…−1.39 vs `"."` −1.21…−1.68）在一次
   **完全冷**的运行里（`cold3`，`cached=0`）同样出现。分歧来自 §6.3 的 top-k
   顺序不唯一，不是恢复的数据。
4. same-step 路径在活体上验过：dp1 先用 640-token 前缀焐热 APC，再发 970 token
   全文 → `cached=768 dt=442 ms`（640 APC + **128 same-step 载入**），冷算均值
   437 ms，服务端日志无失败。即常态下这条路径没有额外代价（这一次也没省出
   可见的墙钟，省掉的是一个 parked step ≈40 ms 量级）。

**洞在哪**（结构上可达，本次未触发）：

- 返回值其实是服务端的**成功标志**（`DeviceMessagingFuture.result_` 就是
  `retrieve() -> (event, retrieve_succeeded)` 里的布尔），按真假判成败本身没错；
  真正的洞是 `LMCacheTimeoutError` 等任何异常都直接从 `start_load_kv` 抛出去，
  既没走失败上报，也绕过了 `--kv-load-failure-policy`。
- `submit_retrieve_request` 在 adapter 不健康时直接丢弃；旧 `_mark_load_failed`
  写 `error_block_ids`（这个模型上恒被丢弃）+ `_failed_request_ids`（只对 parked
  请求有意义）。请求此时已经在跑，`_handle_failed_recving` 不看它，
  `--kv-load-failure-policy fail` 在这条路径上不生效。
- 后果是这一步的 attention 读到没载入的块（复用下来的旧数据），答案静默变坏，
  这正是「reasoning 出问题」最难查的一种形态。

**修法**（`lmcache_mp_sync_load.py`）：

1. 准入加健康门：`scheduler_adapter.is_healthy` 为假时退回异步载入。请求回到
   parked 路径，`failed_recving` 对它是有效的；lookup 刚成功时这条门不改变行为。
2. 失败即 fail-closed：多组布局上抛 `RuntimeError`（`_report_request_errors`
   为真时），单组布局仍走 `invalid_block_ids` 让 vLLM 重算。不再「记一笔然后
   继续」。
3. 成功判据是服务端返回的布尔（**不抛异常但返回 False**——例如个别 key 没读到
   ——也是失败），等待中的异常一律捕获后按失败上报，不再穿透 `start_load_kv`。
4. 模块 docstring 里「失败通过 `invalid_block_ids` 重算」的说法删掉——那在这台
   模型上不成立。

单测：`/opt/venv/bin/python -m pytest
tests/v1/kv_connector/unit/test_lmcache_mp_sync_load.py -q` → **16 passed**，
新增健康门、多组抛错、`False` 返回值算失败三例。

#### 5.8.1 `store_replay_windows` 的 SWA 窗口恢复：不会读错，也不会错位

2026-11-23 补审（起因：窗口恢复是上面唯一没单独验的部分）。现役传输是服务端
驱动的 `lmcache_driven`（`/config`：`supported_transfer_mode=lmcache_driven`、
`chunk_size=256`、`separate_object_groups=true`、
`window_commit_policy=turn_end/anchor=generation_end`、`server_modules=[]`），
所以走的是 `v1/multiprocess/modules/lmcache_driven_transfer.py` +
`object_group_transfer.py` 的对象组拷贝，不是 `vllm/.../lmcache_integration/`
里那份旧副本。

几何：**LMCache chunk 256 > SWA 窗口 128**，于是 `sw_size_chunks=1`、每 chunk 只
保留尾部 128 token（32-token 块里是 8 取 4）、H2D 只搬最后一个对象。逐条链路：

- **范围对齐**：`save_unfull_chunk` 未开、`mamba_cache_mode=none`，lookup 的
  `end` 只取整 chunk，所以 `ret` 恒为 256 的倍数；retrieve 又要求
  `key.end // chunk == prefetch_hit_chunks` 且该组被 lookup 锁过，才恢复
  optional window。窗口对象覆盖的位置因此恒为 `[ret-128, ret)`。
- **目标块对齐**：APC 命中是 128 对齐的（`scheduler_block_size=128`），
  `skip_first_n_tokens = num_vllm_hit - chunk_floor(num_vllm_hit) ∈ {0,128}`。
  `skip=128` 只落在 retrieve 首 chunk 的非窗口半边，
  `recalculate_blocks_to_skip(8, 4, 4) = 0`，4 个窗口块全部写；目标位置
  `start_object_idx * blocks_per_window` 指向末 chunk 的 4 个尾块，
  与对象内容逐 32-token 块对齐。`num_computed = ret-128` 的单 chunk 尾巴
  （`sync_load_max_tokens=128` 的常态）走的就是这条。
- **内容对齐**：store 时 `GetStoreMetadata` 把所有 `chunk_end <
  num_prompt_tokens` 的窗口块换成 null block，服务端 `all_null_chunk_masks`
  把全 null 的 chunk 整块跳过不提交（注释写明：键是前缀哈希，提交了会被
  更短前缀的命中读到）。所以任何存在的窗口对象都属于「以该 chunk 结尾」的
  某个请求——同样的前缀 token 决定同样的窗口内容，对任何命中同一前缀的请求
  都成立。
- **不重放/重放二选一**：只有 `num_computed <= ret-128` 才 `window_loaded`，
  并由 `get_loaded_kv_cache_group_ids` 把 15 个窗口组加进「已恢复组」，
  scheduler 于是不做 replay、也不清这些块；否则 `_mark_prefix_replay` 重算
  128 token，`sparse_swa.py` 用 `replay_start` 把可见前缀截在重放边界之后
  （`start_pos = max(start_pos, replay_start)`）。两条路互斥，不会一边恢复
  一边读到「恢复窗口之前的空块」。
- **缺数据 fail-closed**：retrieve 先在原始块号上查 underflow，再要求每个
  保留 key 都读到（`found_keys != in_window_keys` ⇒ 整次失败），没被 lookup
  锁住的 optional window 组直接跳过、留给引擎重放。

**限定条件（性能，不是正确性）**：`save_decode_cache` 未开时，只有 prompt
token 数恰为 256 的整数倍，末尾 chunk 的窗口才不会被 null 掉；其余情况
`store_replay_windows` 等于没存窗口，外部命中照走 128-token replay。窗口对象
也只进 L1（`swa_l1` store policy），被驱逐后同样退回复算。这也解释了实测中
窗口恢复很少出现。

**顺带修掉的一处真洞**：`wait_on_stream` 的返回值是服务端的成功标志，不是
空句柄。上一轮的「不抛异常即成功」把 `retrieve_succeeded=False`（个别 key
没读到、实例未注册、块号 underflow、拷贝异常）当成成功，会让这一步读到没
写入的块——正是「读到错误 KV」的形态。已改回按布尔判成败，并把等待异常也
收敛为失败上报（多组布局仍抛错），单测 16 passed。

`/tmp` 下的探针（`lmcache_recall_probe.py`、`lmcache_apc_control.py`、
`lmcache_sync_load_probe.py`）未改动，可原地复跑。

### 5.9 长上下文（160K+）的两条正确性假设：mxfp4 indexer 与高位 block（2026-11-23 已排除）

线上反馈 160K 上下文的思考段会突然变成多语种乱码，给出过两个解释：indexer 在
mxfp4 下选错 top-k；打包缓存的高位 block 字节偏移在某个 CUDA kernel 里用 int32
算而溢出。两条都定量排除了。本机复现到的同类乱码是**高温度下的采样塌缩**，
不是上下文丢失，也不是地址错误（证据在最后）。

**候选规模先纠正。** config 的 `index_source_layer_ids = [2, 8, 14, 20, 24, 28,
32, 36]`：PP0 的 [2, 8, 14] 是 `compress_ratio=2`，PP1 的 [20, 24, 28, 32, 36] 是
`compress_ratio=1`。160K token 时前者参与排序的压缩状态约 **8 万**，后者约
**16 万**——不是"4 倍压缩约 4 万"。每层取 `index_topk=512`。indexer K 是 mxfp4
（68 B/行：64 字节 e2m1 对 + 4 字节 scale），但 SM120 上
`indexer_logits_dtype=auto` 落到 float32 累加，只有 K 被量化。能压候选数的两级
筛选（`indexer_sparse_logits`，要求 SM90 + fp8 缓存）在 SM120 + mxfp4 下直接
raise，本部署用不了。

**检索实测：264/264。** T=0、`reasoning_effort=none`，在资料里均匀埋 32 条带
唯一编码的记录，问其中 12 条（早/中/晚各 4），要求固定格式逐条回答：

| 运行 | prompt tokens | 召回 |
| --- | --- | --- |
| 32K / 80K / 160K / 192K，填充偏移 250k | 34,046 / 83,199 / 165,119 / 197,887 | 各 12/12 |
| 160K，填充偏移 0 | 165,119 | 12/12 |
| 160K，偏移 250k，复跑 | 165,119 | 12/12 |
| 16 路并发 160K（偏移 250k+3000i，互不共享前缀） | 16 × 165,118–165,119 | 192/192 |

并发那 16 条全部 `finish=stop`，输出长度都是 251 token。之前 20 次同输入的离线
测里，top-k 选中的**集合**逐次相同、只有顺序不同（§6.3）。即候选从 8 万涨到
16 万，没有观察到检索丢失。

严格说这只证明"这套配置下检索没失败"，不能证明 mxfp4 与 fp8 精度等价——上游
确实缺长上下文评测（`serve.sh:52`）。要 A/B 只能重启换 `indexer_kv_dtype=fp8`
（§9.3，需批准），用同一套题对照；fp8 行是 132 B，缓存占用和容量要一起重算。

**高位 block：布局是真的，溢出路径不存在。** 打包后每个物理 block 的字节跨度
（`extensible_kv_cache.py` 启动日志口径）：

| stage | 每块字节 | 块数 | 缓存 | 2^31 边界 |
| --- | --- | --- | --- | --- |
| PP0 | 69,696（242 × 288） | 323,437 | 20.99 GiB | 30,813 号块 |
| PP1 | 34,848（121 × 288） | 552,331 | 17.93 GiB | 61,625 号块 |

两个 stride 都是 288 B 压缩记录的整数倍，所以 2^31 边界确实在 3 万/6 万号块。

但"低并发用不到高位 block"在本部署不成立：APC 让 `vllm:kv_cache_usage_perc`
长期停在 0.51/0.54，即约 16.4 万 / 29.8 万块常驻；block 池从 0 号块开始
`popleft` 分配，使用中的 id 前沿对应 **11.4 GB / 10.4 GB，约 5.3×/4.8× 2^31**。
上面 264 次检索就是在高位块上跑的。

源码层面，"看不到的三处"全部按 64 位算字节地址：

- FlashInfer FP4 写入 `data/csrc/sparse_mla_sm120/dsv41_fp4_cache_ops.cu`：
  `PagedCacheInfo` 的 `page_stride_bytes`/`rows_per_page`/`num_rows` 是 `size_t`；
  `token_rows()` 是 `cache + page * g.page_stride_bytes`（:196）；slot 用 `size_t`
  除模拆成 page/entry
- cake（`cake_sparse_mla_dsv41_mixed_dual_*`、`cake_fused_qk_rope_*_append`）：
  `long long page_stride_bytes`、`unsigned long long stride_u64`
- sparse MLA gather（`sparse_mla_sm120/kernels/dsv4_nvfp4/gather.cuh`）：
  `cache + size_t(page) * page_stride`
- DeepGEMM SM120 paged logits：block 号是 TMA 坐标（uint32 索引，不是字节偏移），
  `block_table` 步长用 `uint64_t`；host 只把 stride 本身（34–70 KB）放进 int
- vLLM 自己的 Triton 写入：`cache_utils.py:93` 的注释就是为这个边界写的
  （"block_idx * block_stride can exceed 2^31"），`indexer_k_store.py`、
  `fused_compress_quant_cache.py` 全用 `.to(tl.int64)`

**内核级复现**（空闲 GPU，32,000 页 × 69,696 B = 2.23 GB 的页跨步视图，跨过 2^31）：

```python
# 写入：page 31,999 与 page 1 写同一行 latent，应与 pack 参考逐字节相同
cache = torch.as_strided(
    torch.zeros(N * 69_696, dtype=torch.uint8, device="cuda"),
    (N, 64, 288), (69_696, 288, 1))
flashinfer_dsv41_fp4_quantize_append(
    latent.expand(2, 512),
    torch.tensor([1 * 242 + 3, 31_999 * 242 + 3],
                 dtype=torch.int32, device="cuda"),
    cache, inline_scale=True)

# 读取：page 1 / 31,000 / 31,999 放同一份 68 B indexer 页，logits 应逐位相同
view = kv_cache_as_quant_view(cache_68, 128, use_fp4_cache=True)
logits = fp8_fp4_paged_mqa_logits(
    (q, q_scale), view, weights, context_lens, block_table, schedule, 64)
```

```text
writer: high==low=True  low==pack_ref=True  nonzero_bytes_total=576 (=2×288)
logits: edge_bits_equal=True  high_bits_equal=True  high_diff_cols=0
```

换成不同页内容 logits 立即不同，说明读的确实是那一页，不是"都读到零"。注意
append 的 slot 是 `page * (block_stride / row_pitch) + entry`（这里是
`page * 242 + entry`），不是 `page * 64 + entry`；vLLM 用
`paged_slots_to_cache_rows()` 做这个换算。cake 的 sparse MLA 只在源码层核过，
没有单独做高位页的数值 A/B。

**那 160K 乱码是什么。** 本机在 T=1.3 复现到同一类输出，逐 64-token 窗口统计
生成 token 的 logprob：

| 条件 | 运行数 | 最差窗口 | 窗口 < −5 的次数 |
| --- | --- | --- | --- |
| 160K, T=1.0 | 4 | −1.1 … −1.4 | 0 |
| 160K, T=1.1 | 2 | −1.4 / −1.7 | 0 |
| 160K, T=1.2 | 2 | −2.2 / −2.3 | 0 |
| 160K, T=1.3，偏移 250k | 2 | −3.0 / −3.6 | 0 |
| 160K, T=1.3，偏移 0 | 4 | −2.6 / −4.4 / **−11.2** / **−11.5** | 0 / 0 / 5 / 16 |
| T=1.3，长度 8K→131K 扫描 | 12 | −2.6 … −3.8（随长度缓慢变差） | 0 |

特征与两条假设都不符：只在高温度出现（T≤1.2 的 8 次全干净）；与长度无台阶
（8K 到 131K 最差窗口只从 −2.6 缓变到 −3.8，160K 偏移 250k 也只有 −3.6）；
与内容和位置有关（同长度、换填充偏移即从崩到干净）；塌陷都起于长段回溯/
判断环节（16 个坏窗口那次在第 3 项判断，5 个坏窗口那次在核对第 1 项文档顺
序），且塌陷中三条针仍被正确取回。日志里的 logprob 是 vLLM 默认的
`raw_logprobs`（温度之前算）：塌陷窗口均值 −10.7，等于模型自己给采样到的
token 约 e^−10.7≈2×10⁻⁵（全词表均匀是 −11.8），所以是前向分布变平，不是采
样器抽错。这是 logits 分布塌缩 + 采样漂移，不是注意力漏上下文，也不是地址算
错（地址错会给稳定、可重复的错值）。定位用户侧现象前
先对齐采样参数——T=1.3 本身就会塌。

**塌陷是前向行为，不是 decode 路径的产物。** 把 16 个坏窗口那次（2048 起，
最差 −11.53@2880）的 164K prompt 与当时生成的 3072 个 token 一次性重新
prefill，用 `prompt_logprobs` 重算逐 token raw logprob：塌陷窗口与在线 decode
值逐窗口吻合（−10.53/−10.56、−11.22/−11.31、−11.53/−11.58），坏窗口平均差
+0.08 nats，整段 48 个窗口平均 |差| 0.30；5 个坏窗口那次坏窗口平均差 0.014；
干净对照平均 |差| 0.56 且无坏窗口。重放没有投机解码、没有增量 KV 状态、没有
graph，所以塌陷是"给定 token 序列的前向计算结果"，不是 DSpark、批处理或采样
器的产物。同日再跑 6 次同条件（T=1.3、偏移 0、3072 token）全部干净（最差
−2.6…−3.1），说明是否进入坏盆地取决于采样轨迹，T=1.3 只是把概率调高。

再做 `indexer_kv_dtype` 的 A/B：切到 fp8 重启后，用同一套重编码 token ids 做
teacher-forced 重放（两次重放只差 indexer dtype），7 段轨迹 287 个窗口的逐窗口
差平均 0.015–0.020 nats、最大 0.09，坏窗口集合与数值原样复现。indexer dtype 在
这些 OOD 状态上数值上是 inert 的，塌陷不来自它。重放请求带 `prompt_logprobs`
时 `skip_reading_external_kv` 为真，本地 APC 与 LMCache 命中都在
`vllm/v1/core/sched/scheduler.py:1378` 被清零，所以重放是纯全量 prefill——这也
证明在线 decode 用到的（可能由 LMCache 恢复的）KV 与从零重算等价，LMCache
没有注入错误；同期 metrics 里外部命中持续发生而所有请求都干净。

采样层面，mxfp4 的 3 次塌陷全部发生在跑了 10 小时的旧进程服务的那一批（3/4，
同日后续 0/6）；换新进程后 mxfp4 0/26（两批各 10 次 + 6 次）、fp8 0/18，共 0/44。
按 ~10% 的速率看，3/4 是一个罕见聚集，但旧进程状态无法排除；要定论只能复现
"长时运行后的批次"。

**同一请求也不逐次 bit 可复现。** 同一 prompt 的全量 `prompt_logprobs` prefill 连跑
两次（无投机、无采样）：top-1 token 在 99.5% 的位置相同，同位置 top-1 logprob 平均
只差 0.0026 nats；但约 0.5% 的位置会出现 0.1–1 nats 的跳变（近似平局翻转），另一次
重复则 0 跳变。逐位置全量看均值 0.35 nats、最大 20，那是长尾 token 的相对误差，不
影响采样。来源是 prefill chunk 大小自适应浮动（日志里 8192/5888/7680/5120 来回切）
加共批，改变了 FP8 GEMM 的累加顺序。T=1.3 下几次翻转就足以让轨迹在 2–7 个 token
内分叉，所以塌陷要按"轨迹系综"理解：重放窗口差 0.03–0.09 落在这个噪声底附近，
而坏窗口是 7 nats 的平台，结论不受影响。DSpark 没有可累积状态（自适应验证默认
关、采样 RNG 为 `murmur3(seed,pos)` 无状态、草稿 KV 按请求且 128 窗口），且上述
抖动在纯 prefill 上就存在，位于投机之前。

### 5.10 chat 模式（thinking=false）短答复损坏，与 DP 副本 logits 不一致（2026-11-24）

外部反馈：`thinking=false` 时短问答大量答错（自称 5/10），且 `repetition_penalty>1.05`
会放大成日语乱码，怀疑跨会话内容泄漏。按反馈的六道探针（pong / 2+2 / 3*7 / Paris /
Orwell / 29 天）在本机跑 `repro.py`，**一条都没错**：thinking off 10 轮 0/10，
`--thinking auto` 0/10，`mt=256` 24 轮 0/24，`reasoning_effort=none` 0/12，
`--repetition-penalty 1.15` 20 轮 0/20。六道题的答案都是单 token，这个 checkpoint
在 chat 模式下也稳，反馈里的 "pong→naar"、"2+2→I'm sorry 循环" 无法用这套题复现。

**换逐字复述探针就复现了。** system 给一个唯一口令（如 `cobalt-otter-1462`），
要求原样复述，每次 96 条（`leak_test.py`）：

| 配置 | 错答 |
| --- | --- |
| thinking=off, mt=24, 16 路并发 | 11–12/96（三轮） |
| thinking=off, mt=256, 16 路并发 | 6/9/13/96（三轮） |
| thinking=off, 串行, T=0 | 19/96、20/96 |
| thinking=off, 串行, T=0.5 | 19/96 |
| thinking=off, T=0, `repetition_penalty=1.15` / `1.3` | 18/96 / 21/96 |
| thinking=on / auto, mt=256 | **0/96**（各两次） |
| 无 system prompt，同一任务（user-only） | 0/64 |

典型坏答（`finish_reason=stop`，`reasoning=null`）：

```text
cobalt-saddle-6281 -> 'cobalt-saddle-\n6281'
cobalt-otter-8682  -> 'cobalt-\notter-8682'        # 原口令 cobalt-otter-8682
brisk-walnut-8017  -> 'brisk-wal<|endoftext|>'     # 停在词中间并 EOS
cobalt-otter-5617  -> 'cobalt-<ds_safety>[用户未成年]否\n[分类]其他\n[判定]…'
quartz-nzapsatv    -> 'quartz-nzapsat<|endoftext|>'
```

`<ip>`、`<point>`、`<ds_safety>`、`<|endoftext|>` 都查过 token ids（`return_token_ids`）：
全是普通文本 token 拼出的字面量，词表里只有 BOS/EOS 两个 special token；也不是
reasoning parser 切分（`reasoning=null`、content 就是全部）。即 chat 模式下模型自己
写 markup 并提前 EOS，不是模板、解析器或 detokenizer 的问题（渲染出的 prompt 也检查过：
thinking=false 只去掉 "Reasoning Effort" 前缀和结尾的 thinking 标记，其余结构相同）。
`repetition_penalty` 不改变错误率（18–21/96），只改变坏答的内容（更多 meta 文本）。

**跨会话泄漏：没有。** 96 并发 + 96 串行，每条带不同口令，检查是否出现别人的口令：
0 次。反馈里的 `/Users/dev/...` 更像客户端/代理侧内容，不是这个服务的 KV 串请求。

**顺带测出一个更值得追的数值事实：同一输入、greedy，logits 不唯一。** 用
`/v1/completions` 的 `prompt_logprobs`（`get_prefix_cache_hit_limit` 返回
`prompt_logprob_start or 0`，即 APC 与 LMCache 都不读，纯全量 prefill，同 §5.9 口径），
同一 prompt、`temperature=0`：

35 token 与 53 token 的 prompt，最后一个 prompt token 的 logprob **只有两个值**，
14/14 与 `/metrics` 的 `vllm:request_success_total{engine=…}` 一一对应：

| engine | logprob（35 tok prompt） |
| --- | --- |
| 0 | −13.961 |
| 1 | −13.625 |

两个 DP 副本各自确定，但彼此不同，差 0.34 nats。

13 token 的 prompt 更差：**同一个 engine 内也漂**（engine 0 五次：
−13.582 / −12.755 / −13.29 / −13.711 / −14.009），greedy 下一 token 在 `M`/`Ki`
之间翻转；请求间隔 1.5–2 s 也一样，排除 co-batch。22 token 的 thinking prompt
又回到"两个 engine 两个值"。

口径对照：§5.9 在 164K 长上下文上测到同请求重复 prefill 的 top-1 logprob 平均只差
0.0026 nats（0.5% 位置有 0.1–1 nat 跳变）。这里 13-token prompt 的逐次漂移
（~1.5 nats）比那个噪声底大三个数量级，又不是 chunk 大小浮动（13 token 只有一个
chunk）。它和 7 nats 的塌陷平台还差得远，但足以把 chat 模式下本来就很弱的答案
推过边界；melt 与 chat 坏答可能是同一个数值不稳定性的两种表现（另见 §6.3 的
top-k 顺序）。未归因。候选：prefill graph 的 padding 桶（13 token 落在最小的
桶里，22/35/53 token 用更大的桶）与 engram lookback/掩码、sparse indexer 的
top-k 顺序、DP 间 EP 分片的归约顺序。

**与上游 issue 的关系（2026-11-24 核对）。** 上游 vllm-project/vllm#62811
「DeepSeek-V4.1: wrong output when the last chunk of a chunked prefill runs as a
piecewise CUDA graph」不是这个现象：它要求 prompt 超过 `max_num_batched_tokens`
被切块，最后一块（2..capture size token）落进 PIECEWISE graph 时从第一个生成
token 起就错，**每次都是同一串错 token**，`enforce_eager` 正确；根因是
`eager_break_during_capture` 在 import 时解析开关，由 #59941 修复（2026-10-21
合入），issue 作者自己 bisect 后确认 main 已修。本分支基于 2026-11-21 main，
`git merge-base --is-ancestor 0eea5aca8 HEAD` 为真；同族另一个 replay 打分 bug
（#62651）的修复 #62652（`model_state.add_request` 里 `prompt_logprob_token_ids is
not None` 也保留全部行）也在树里。所以上面 `prompt_logprobs` 的测量不受 replay
行裁剪污染——`prompt_logprobs is not None` 本来就走保留全部 prefill 行的路径。

我们测到的与它正相反：prompt 只有 12–56 token，单 chunk，从不触发 chunked prefill
的尾块；错误不确定（两个 DP 副本各是一个固定值、同一副本内还会漂、greedy 下一
token 翻转），而 #62811 是确定性的固定错 token；坏答依赖 chat 模板（thinking off
坏、on 全对），#62811 与模板无关。更接近的上游 issue 是 #53257（DeepSeek-V4-Flash，
T=0 输出随并发变化，`enforce_eager` 下仍复现，仍未修）：它报告同 prompt 在并发下
出现少数派输出、分歧位置 logits 从 18 nats 的优势塌成近均匀；我们测到的是
"副本间固定分叉 + 短 prompt 副本内漂移"，可作为它缺的归因线索。

仍有一处重叠要记住：本部署的触发面和 #62811 一样（`deepseek_v41` decoder replay、
默认 breakable CUDA graph、`FULL_AND_PIECEWISE`、`max_num_batched_tokens=8192`、
prefill graph 捕捉到 8192、replay 切点在 layer 20 即 20–39 层，与 issue 指的是同一
段），而 issue 明确没测"整个 prompt 落在单 chunk、被 padding 到捕捉尺寸"这一档，
我们的短 prompt 恰好在这一档。以后若看到**确定性**的首 token 错误（尤其 ≥8192
token 的 chunk 边界），先查 #59941/#62652 这一族；这次"不确定 + 与 chat 模板
相关"的坏答不是它。

本次没有重启、没有改配置。复现命令：

```bash
cd /workspace/debug-vllm
/opt/venv/bin/python repro.py --pid <pid>             # 六道题，全对时退出码 1
/opt/venv/bin/python leak_test.py 6 16 <pid> off 24   # 逐字复述，能复现
/opt/venv/bin/python leak_test.py 6 16 <pid> on 256   # 对照，0 错
```

下一步（按"engine 归因"这条线）：去掉 DSpark 或 `--enable-expert-parallel`、
或把 DP 降到 1 重启重测；若单 DP 后两值消失即确认副本归因，再二分副本内部的
prefill graph / engram / indexer。**（2026-11-24 补测：副本归因已推翻，见 §5.11。）**

### 5.11 T=0 不可复现的根因：输出是 batch 形状桶的确定性函数（2026-11-24 补测）

§5.10 的两个前提都不成立。第一，**prefill 没有 CUDA graph**：日志里
`Cudagraph mode FULL_AND_PIECEWISE requires piecewise capture, but the loaded model
provides neither a compiled submodule nor breakable CUDA graphs. Overriding to
FULL_DECODE_ONLY.`，只捕捉了 41 个 FULL decode graph，所以"prefill graph 的 padding
桶"这条候选排除。第二，**"两个 DP 副本各一个值"是路由巧合**：加
`X-data-parallel-rank` 头固定副本后，两个副本都能产出两个值，§5.10 表格里的 1:1
映射不成立。

测量口径不变：`/v1/completions` + `prompt_logprobs=0`（`get_prefix_cache_hit_limit`
返回 `prompt_logprob_start or 0`，APC 与 LMCache 都不读，`cached_tokens=0` 且 prefix
cache 计数不变），`temperature=0`，读最后一个 prompt token 的 logprob 与 greedy 下一个
token。

**观察一：同引擎只要有一个 decode 请求在跑，prefill 结果就换到另一个固定值。**

| 同引擎背景 | 探针结果（L=32，重复 4 次） |
| --- | --- |
| 无 | `(−12.364833, 201)`，逐次逐位相同 |
| 1 个 decode 请求 | `(−16.383139, 22241)`，4/4 |
| 2 个 / 3 个 / 4 个 decode | `(−16.383139, …)` / `(−18.011234, …)` / `(−18.011234, …)` |
| 8 个 decode | `(−16.296143, …)` |
| 12 个 decode | 每次不同（`−11.72 / −12.19 / −14.54`，gen 在 1495/444 间翻） |
| 1 个 **prefill** 请求（6000 token，`mt=1`） | `(−12.364833, 201)`，8/8 |

混合值与该 decode 请求的**内容、长度、上下文都无关**：中文 64、英文 64、随机 id 64、
中文 200、随机 1200 token 全给同一个 `−16.383139`。背景刚发出 30 ms（其 seq_len 只有
几个 token）就已经切换，所以不是 `model_state` 的 SWA replay 裁剪（`window=128`）或
`_agree_across_dp` 的 replay 批；1200 token 的上下文也不改变值，所以也不是
`DeepseekV4Indexer.forward` 的短上下文快路径（`fill_short_context_topk_indices`，
阈值 `index_topk*compress_ratio` = 512/1024）。两个并发的 prefill 探针互不影响。

**观察二：同一条 prompt 尾部加 token，会改掉前面所有位置——违反因果性。**

L=32 与 L=40（同一前缀）逐位置 logprob 差 0.1–8.5 nats，**位置 0 差 −6.131**；位置 0
只看到 token 0..1，不可能依赖第 33 个 token。把长度从 32 加到 33/35/38/40/44/48，
**开头一段位置的 delta 完全相同**（`+6.131, −2.754, +1.946, −0.933, …`，正好是
"混入 decode 行"那组 delta 的相反数），之后才因离散翻转逐长度分叉：mxfp4 服务器上前
13 位一致，FP8 indexer 服务器上前 17 位一致（33/40/48 在第 17 或 20 位之后分开）。
56 起连开头几位也变，63/64/65/72/80/96/128 各不相同。

结论：**整条前向是 `(tokens, 补齐桶)` 的确定性函数**（前提是服务状态不变；
rank 副本的值会随服务长期状态漂一次，见 §5.12 末尾），桶边界在 32/33 之间（之后还有
16 或 32 的粒度）；同一桶内输入的扰动逐位相同，但下游的离散翻转是逐 token 的，
所以只有开头的连续段能逐位对上。混入 decode 行等价于把 batch 推过 32/33 这个边界。
这不是 indexer（位置 0 没有稀疏候选也偏 6 nats），也不是 GEMM 重结合（那只产生 1e-3
量级）。

**幅值来自离散翻转，不是浮点噪声。** 1–8 nats、逐位置符号和大小都不一样的跳变，最
符合"数值在舍入/并列边界上翻转"这一类事件：FP8 动态激活量化、NVFP4 expert 权重
（`expert_dtype: fp4`，每 16 元素一个 scale）、`noaux_tc` 的 384 选 6 路由，任何一处
在边界上翻一格都会改变 residual，再经 40 层 hyper-connection（`hc_mult=4`）逐层放大。
固定形状下也有逐次不确定性：L=13 连续 20 次给 20 个值、greedy token 在
16/33/201/305/1148 间翻，L=64 每次不同，而 L=32 逐位稳定——说明某些形状的 kernel 本身
还有非确定性归约（split-k/atomic），喂进同一个放大器。

**与 chat 坏答的关系。** thinking=false 的逐字复述任务本来就贴着决策边界，任何 batch
形状带来的 1–8 nats 扰动都会推过边界一部分请求；thinking 打开后输出分布更尖，所以
0/96。这把 §5.10 里"随并发、随模板"的表象统一成一个机制，也正好补上上游 #53257
（T=0 随并发变化、`enforce_eager` 仍复现）缺的归因：不是图捕捉，是形状相关的数值路径
加离散放大。

**换 FP8 indexer 后复测（2026-11-24 05:39 重启，`indexer_kv_dtype=fp8`）。** 现象逐位
不变：rank0 单独 L=32 仍是 `(−12.364833, 201)` 4/4，同引擎 1 个 decode 背景仍是
`(−16.383139, 22241)` 4/4，背景停掉后恢复；L=13 仍 8/8 各不同，L=64 6/6 各不同；
L=33/40/48 的开头 delta 仍是同一组。rank1 单独是 `(−14.49286, 22241)` 4/4，与 rank0
不同——说明这个形状下两个副本仍各有一个固定值，但 §5.11 的"副本差异"结论不变：
只要有混批/长度变化，同一个副本自己就会切到别的值。结论：**indexer 的 KV 精度不是
这个现象的变量**（短 prompt 本来走全候选快路径），换回 mxfp4 或继续用 fp8 都不影响。
基线已固化成一条命令（`/workspace/debug-vllm/det_battery.py --pid <pid>`，输出
JSON 逐段可 diff；这次是 `det_battery_fp8_indexer.txt`），切换 batch-invariant 后
重跑同一命令即可对照。

**可验证的下一步（需批准重启）。** `VLLM_BATCH_INVARIANT=1` 就是为这件事做的：matmul
按 M 桶固定 kernel 配置、attention 固定 tile、采样强制 seed 口径（见
`vllm/model_executor/determinism/`）。预期：同一 prompt 换长度/混批后逐位一致；若 chat
坏答率随之明显下降，根因闭合。两个注意点：该开关会自动打开
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` 并禁用 one-sided all2all（`config/vllm.py:1188`），
而 DSv4.1 只有 `sparse_mla`、`specialized_decode` 的部分路径读它，mHC / engram / FP4
expert GEMM 这些自研 op 要逐个确认是否还在桶外。
（2026-11-24 已把桶内那层"地板"定位到一处集合通信，见 §5.13。）

复现脚本（不改服务、不重启）：

```bash
cd /workspace/debug-vllm
# batch 组成：固定 rank 0，先测单独，再在同引擎起/停一个 decode 背景
/opt/venv/bin/python det_engine.py --lengths 32 --rank 0 --reps 4 \
    --bg-rank 0 --bg-tokens 1200 --bg-max-tokens 400
# 因果性：同一前缀加长，比较前 31 位
/opt/venv/bin/python det_engine.py --lengths 32,33,40,48 --rank 0 --reps 2 --print-all
```

### 5.12 从坏答反推：坏答是"低 margin token + 形状扰动"的确定性结果（2026-11-24 补测）

2026-11-24 06:06 重启，MoE 后端换成 `moe_backend=deep_gemm`（日志确认
`DeepGEMM PDL/E8M0 enabled`，`Detected quantization_config.scale_fmt=ue8m0`），
indexer 回到 `mxfp4`，其余不变；线上同款后端。仍然从 chat 坏答入手。

**DeepGEMM 去掉了固定形状的逐次不确定性，但没去掉形状依赖。** 同一组 96 个字母数字
密码、`temperature=0`、`thinking=false`、`max_tokens=24`、每次换一个 `cache_salt`
（强制整段 prefill，`cached_tokens` 全 0）逐条串行发：

| 量 | FlashInfer/FP8-indexer 服务器 | DeepGEMM 服务器 |
| --- | --- | --- |
| L=13 连续 8 次 | 8 个不同值 | 8/8 逐位相同 `(−14.877853, 16)` |
| L=64 连续 6 次 | 6 个不同值 | 6/6 逐位相同 `(−10.300628, 2418)` |
| rank0 单独 L=32 | `(−12.364833, 201)` | `(−11.969986, 291)` |
| 同 rank 1 个 decode 背景 | `(−16.383139, 22241)` | `(−11.432167, 22241)` |
| 背景停掉后 | 恢复 | 恢复 |

即：逐次非确定性归约（split-k/atomic）来自此前那条执行路径，DeepGEMM 的 GEMM 是
确定性的；§5.11 的混批/长度依赖原样存在（首 token 尾 logprob 仍被重写，greedy token
201→22241 的翻转仍是同一现象）。

**坏答实验（串行、并发 1、T=0、thinking=false、96 个密码 × 3 轮）。** 42/288 =
14.6% 坏答，但**全是固定子集**：14 个密码 3/3 全坏，82 个 0/3，没有一个是时好时坏
（sticky fraction = 1.0）。坏答内容不是"乱码"，而是把密码拆坏成标记或换行：

```text
cobalt-saddle-2194 -> cobalt-saddle-<ip>
cobalt-saddle-1730 -> cobalt-saddle-173\n0
amber-raven-4855   -> amber-ra<ref>4855</ref>
eager-pepper-2696  -> eager-pepper-<truncated>
brisk-otter-8731   -> brisk-otter-\n8731
```

`<ip>`/`<ref>`/`<truncated>`/`<refusal>`/`<final>` 都不是 tokenizer 的特殊 token
（`added_tokens` 和 base vocab 里都没有），是模型自己生成的普通文本标记。

**同一批密码在"有 decode 混批"时坏答集合会换。** 全程起一个 64-token、
`max_tokens=1000` 的 decode 背景再跑同一矩阵：12 个坏（11 个 3/3、1 个 1/3），与单独
时只有 8 个重合；6 个只在单独时坏（如 `cobalt-saddle-2194`、`amber-raven-4855`），
4 个只在混批时坏（`amber-saddle-1352`、`brisk-otter-0740`、`dusty-raven-2871`、
`eager-pepper-9720`），两种形状的并集 18/96。同一条 `cobalt-saddle-1730` 仍然是坏，
但内容从 `173␊0` 变成 `173`——翻转点也随形状移动。

**margin 完全分开两类。** 用 `logprobs=True, top_logprobs=5` 重放 96 条：14 个坏答在
某个位置的 top-1/top-2 差只有 0.0–0.875 nats（多为 0–0.375），82 个好答最小也有
5.5 nats（中位 10.9）。零重叠。典型片段（`cobalt-saddle-2194`，第 5 步）：

```text
'<':-2.660  '\n':-2.847  '#':-3.472  '##':-3.597      # 差 0.187，选了 '<'
```

而 §5.11 量到的形状扰动是 1–8 nats，是这类 tie 的 10 倍以上，所以"哪些 prompt 会坏"
本质上是"哪些 prompt 在某一位置恰好贴着决策边界"。

**这不是小扰动，是整条分支被换掉。** 同一 32-token prompt 单独与混批各测 5 次，
top-5 集合完全不同且分别稳定：

```text
lone: {' b':-2.674, 'ще':-3.236, '.':-3.361, 'not':-3.486, 'to':-3.486}
bg  : {'some':-2.122, 'to':-2.997, 'not':-3.122, '.':-3.747, '\n\n':-3.809}
```

扰动一直作用到**第 1 个位置**上：2-token prompt 的 token-1 logprob 单独
−21.702671、混批 −23.032379（1.33 nats）；8-token prompt 的位置 0 从 −22.792494
变到 −19.963120（2.8 nats）。2 个 token 的上下文里 attention 只有 2 个候选，所以不是
长程 attention，而是逐行的计算路径（GEMM/量化/路由）随 M 变化。幅值又远超浮点重结合
（1e-3），与 §5.11 的"离散翻转 + 40 层放大"一致；最可能是 `noaux_tc` 384 选 6 的
第 6/7 名在 1e-3 量级的形状差下互换，或 FP8 激活量化在舍入边界翻一格。

**thinking 打开后消失。** 同 14 个坏密码、`thinking=true`、`max_tokens=512`、
2 轮 28/28 全对（此前 96 条 0/96）。原因不是 thinking 让数值更稳，而是答复 token
被推到推理段之后生成，那里分布更尖（margin 远超扰动）。温度和 thinking 的 6 格扫描
见 §5.15。

**结论。** 坏答 = "低 margin 的复述决策" × "随 batch 形状变化的数值路径"。给定形状，
它是完全确定性的（3/3 sticky）；生产上"同一个问题有时好有时坏"就是并发/批次组成在
换形状。DeepGEMM 后端照样复现，所以这是线上现象的同款机制，不是某条 kernel 的偶发
bug。要根治只能让前向对形状不敏感（batch-invariant/确定性 kernel 或消除路由边界
翻转）；短期能确认的工程缓解是这类逐字复述任务走 thinking。

**还有一个随服务长期状态变化的量。** 06:14（刚启动、无缓存）时 rank1 单独 L=32 是
`(−13.702787, 1495)` 4/4；跑过约一千条 chat 请求后（06:22 起）固定变成
`(−11.634571, 201)`，之后 11/11 都不变；rank0 的 `(−11.969986, 291)` 两次
完全一致。`cached_tokens` 都是 0，换 `cache_salt` 也不变，所以不是前缀缓存命中。
说明除 batch 形状外还有块分配/LMCache 状态一类的长期变量会改数值路径。这只会加强
"输出由执行状态决定"的结论；坏答实验（矩阵 + margin 映射）全部在状态稳定后的一小时
内完成，margin 对坏答集合的 14/14 预测可以证明期间没有漂。

复现（不改服务、不重启；pid 从 `ss -ltnp 'sport=:8000'` 取）：

```bash
cd /workspace/debug-vllm
/opt/venv/bin/python det_battery.py --pid <pid>          # 形状基线（DeepGEMM: det_battery_deepgemm.txt）
/opt/venv/bin/python badanswer_matrix.py --pid <pid> --codes 96 --rounds 3        # 单独：14/96 坏，3/3 sticky
/opt/venv/bin/python badanswer_matrix.py --pid <pid> --codes 96 --rounds 3 --bg   # 混批：坏答集合换掉 1/3
/opt/venv/bin/python badanswer_margin.py --pid <pid> --codes 96                   # margin 与坏答一一对应
```

### 5.13 桶内地板定位到一处集合通信：EP combine 的按 root 分块归约（2026-11-24）

§5.11/§5.12 把现象收敛到 `(tokens, 补齐桶)`，但没有落到具体 kernel。这一节给出更干净
的测量律、代码级仿真和一次 4-rank 复现，把"同一个桶内还会随 token 数漂"的那层定位到
**MoE 的 EP combine**，并修正 §5.11 的一条表述。

**测量律：只有分片边界之后的后缀会变。** 4 s 空闲间隔、每点重复两轮（重复逐位相同），
单请求 L→L+1（`prompt_logprobs=0`，全量重算）：

| 对比 | 变的行 |
| --- | --- |
| 40 → 41 | `first=20 last=40 n=20/41`，连续 |
| 41 → 42 | 无 |
| 42 → 43 | `first=21 last=40 n=20/41` |
| 43 → 44 | 无 |
| 44 → 45 | `first=22` |

即：偶数 L 加一个 token 才变，变的正好是 `r ≥ ceil(padded_L/2)`，也就是 TP 序列分片的
**第二片**；奇数 L 补齐到同一个偶数后几何不变，逐位相同。42|43 的幅值：0..20 行连 10 位
小数都一样，21..40 行 +0.66 … +1.1 nats（行 27 +1.167、行 40 +1.096、行 24 −0.718）。

混批（B 排在 A 前面）的定量预测是 `first = ceil((A+B)/2) − B`。A=42 时：
B=1 不变、2→20、3→20、4→19、6→18、8→17、12→15，全部命中；B≥16 落到另一个 regime
（步被再切分，first=2）。注意混批本身是竞态：同一输入有一次给"不变"、另一次给
`first=17`，只有配空闲间隔重复才可读。

**代码上的那一处。** `AgRsAll2AllManager.combine`（`all2all.py:198`）发
`reduce_scatterv(hidden_states, dim=0, sizes)`，`sizes` 是按 rank 的 token 数
（`_get_sizes` ← `dp_metadata.get_chunk_sizes_across_dp_rank()`）。非均匀时
`_reduce_scatterv`（`cuda_communicator.py:628`）**逐个 root 各发一次 `ncclReduce`**
（`pynccl.py:410-445`），也就是每个 token 的 4 项专家部分和在"它所属 chunk 的 root"上
按该 root 的归约顺序累加。步的 token 数一变，分片边界移动，边界处的 token 换 chunk
（换 root），舍入就变；内容与位置都没变的其它 token 不变。

为什么只有后缀变：`sp_all_gather(x)[:positions.shape[0]]` 把分片拼回原来的平铺顺序，
attention 的输入对每一行逐位相同；`sp_reduce_scatter` 是 TP=2 的两项求和（精确，与顺序
无关）；而边界之前的位置在因果 attention 里永远不读边界那一行。所以唯一的种子是边界行，
经 40 层 hyper-connection + 路由放大到后继全部位置——这正是"非法因果性"看起来像
"整段被重写"的原因，其实它只是"从某一行起的后缀"。

**4-rank 复现（只跑这一个集合通信，不需要模型）。** 同机 GPU4-7（与 EP 组同为单 NUMA
quad），每个 rank 持 4 项部分和中的一项，行内容在两种几何下完全相同，只改 `sizes`：

```text
repeat c=21: bit-identical
  row 21: chunk 1->0  1201/5120 cols differ, ulp 2
c=21 vs c=22: 41/42 rows bit-identical, changed [21]
batch-invariant reduce (root 0 + broadcast): all rows bit-identical
```

两种几何只有跨界那一行变（2 ulp），其余 41 行逐位一致；换成
"固定 root 0 归约 + scatter"（`cuda_communicator.py:723-731`，`VLLM_BATCH_INVARIANT`
且 world>2 时启用，判断在 uniform/symm-mem 分支之前）后全部逐位一致。

**被排除的（真实权重逐个仿真过）。** 分片侧逐行 kernel 在"行数变化""一行存储偏移
（10240 B）""局部下标平移"三种扰动下全部逐位不变：mHC 的
`mhc_fused_post_pre_delayed_tilelang`、bf16 head（N=64640）、router gate（N=384）、
b12x MXFP8 linear（M=2..80）、DeepGEMM grouped expert GEMM + permute、SM120 sparse MLA
（M≤64）。它们只在各自的家族切换点整批跳变：gate 4|5 与 16|17、head 16|17 与 22|23、
mHC 32|33、MLA 64|65。另外 `_all_reduce_mhc` 根本没跑：`supports_mhc_all_reduce` 要求
SM90/TP8 或 SM100-family/TP4 且禁 EP，SM120 的 family 是 12；`supports_mhc_overlap`
同样 False，所以 `fuse_mhc_all_reduce=False`、`mhc_stream=None`。GEMM-RS 也没开
（`VLLM_ENABLE_GEMM_RS` 默认 0，环境里没有）。

**代价（同机 4 GPU、同 NUMA、bf16 `[rows, 5120]`，每档 60 层）。**

| gathered 行 | 每层缓冲 | ring RS | det（reduce+scatter） | 4×reduce（现状非均匀） |
| --- | --- | --- | --- | --- |
| 768 | 7.9 MB | 0.205 ms | 0.482 ms | 0.475 ms |
| 2048 | 21.0 MB | 0.476 ms | 1.023 ms | 0.828 ms |
| 8192 | 83.9 MB | 1.832 ms | 3.562 ms | 2.472 ms |

只要两个 DP rank 的 prompt 长度不同（本批实验里几乎总是），现网走的就是 4×reduce，
所以确定性版本相对"现在实际跑的"只有 1.23–1.44×，不是 2×。按 8192 token/DP 的步、
PP2 每级约 30 个 MoE 层估算：combine 74 ms → 107 ms，每步 +33 ms，
`VLLM_PREFILL_STEP_BUDGET_MS=700` 的自适应目标会把它从步长里扣掉，体感约
prefill 吞吐 −2~5%。decode 不必付：捕获图的 `sizes` 固定，同一张图内不存在形状漂移。

**修正 §5.11 的一条。** "位置 0 也变"不能用分片边界解释——那是 32/33 这类**粗桶**的
家族切换（上表那四个切换点），切换时整批所有行都变；分片边界是更细的一层，每"偶数→
奇数"一次就动一次后缀。§5.11 的"同桶内扰动逐位相同、跨桶全变"要读成这两层叠加，结论
"输出是执行几何的函数"不变。

**修法。** 根治只能让跨 rank 求和的顺序与 token 位置无关。两条路：

1. `VLLM_BATCH_INVARIANT=1`（需批准重启）。本机上它基本只改这一处：`VLLM_USE_NCCL_SYMM_MEM`
   默认 0（symm-mem 本来就没开），`flash_mla_mega_attn`/`supports_mhc_*`/
   `specialized_decode` 的 batch-invariant 分支都是 SM90/SM100 专属，SM120 不适用。
   代价就是上面的表。
2. 窄化补丁（上线用，未做）：只把 combine 的归约换成 det 路径，且只在 prefill 步生效，
   decode 保持 ring RS。代价按上表约 prefill −2~5%、decode 0。

**复现（不改服务、不重启）。**

```bash
cd /workspace/debug-vllm
/opt/venv/bin/python -I sim_rs_combine.py    # 4 rank，只跑 combine：只变边界行
/opt/venv/bin/python -I bench_combine.py     # 上面那张代价表（GPU4-7）
```

### 5.14 FlashInfer 两处共享状态的审计：sparse wrapper 计划缓存、mxfp8 workspace（2026-11-24）

起因是两条针对 SM120 部署的外部风险 review。逐条对照本机安装的
`flashinfer 0.7.5.dev20261121` 与线上进程后，结论是：**sparse wrapper 的计划缓存
确实存在「图写到已释放内存」的机制，但被当前 CAKE=1 + FULL_DECODE_ONLY 挡住；
mxfp8 的并发 workspace 在本机已经按 stream 隔离，风险不成立。**

| 风险 | 机制 | 当前状态 |
| --- | --- | --- |
| sparse wrapper 计划被替换 | 成立（代码链路完整） | 潜伏：decode 走 cake，prefill 未捕捉 |
| mxfp8 多 stream 共用 workspace | 不成立（per-stream buffer） | 已修复 |

#### A. `SparseMLASm120Wrapper` 的计划缓存

**机制（全部对得上）。** vLLM 的 wrapper 按
`(device, extra_kv_flag, kv_scale_format)` 跨层共享
（`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:70,95-105`）；压缩层和
SWA-only 层因为 `extra_kv_cache` 的有无各用一个，但各自都是所有层共用。
`_prepared_calls` 的 key 只有 `(pref, impl, shape, stride, dtype, device)`，
**不含数据指针**（`flashinfer/mla/_sparse_mla_sm120/_prepared.py:60-63`），
所以 capture 时的调用和之后同形状的 eager 调用命中同一条目。

每次 eager 调用都会 `refresh_store()`（`_prepared.py:475-476`，capture 时跳过）；
该校准 JSON 的 `(mtime_ns, size, inode)` 一变，`_constants_version` 就 +1
（`_calibration.py:1344-1355,1358-1382`），下一次 eager 调用重新 `prepare`
（`_prepared.py:489-514`）。这里有个关键限定：**只有新计划的 workspace tuple 变了
才会换 buffer**；workspace 相同走 `replace(current, plan=...)`，`lse/mid/mlse`
原样保留（`_prepared.py:338-345`）。真的变了才会分配新 tensor 并丢掉旧
`PreparedCall`（`_prepared.py:346-363`）。

vLLM 不传 `out_lse`/`mid_out`/`mid_lse`（`flashinfer_sparse.py:1181-1194`），
所以 `lse` 是 warmup 时 eager 分配的；decode 计划的 `mid/mlse` 在 capture 时
以 pinned 形式分配（`_prepared.py:373-394`）。这三个指针被录进 captured graph。
条目被替换后旧对象掉引用，内存回到 allocator 并可被复用，而图每次 replay 仍往
旧地址写 LSE——这就是文本说的静默改写。FlashInfer 自己在公开契约里写明了这一点：
“A changed calibration profile may replace resources during eager warmup;
recapture affected graphs after that update.”（`_api.py:1269-1273`）。vLLM 侧
没有任何 recapture 逻辑。

**真触发时的后果。** 旧 `lse`（每次 launch 都写）、旧 split-K scratch
`mid/mlse`（同一次 launch 内先写后读）都还在被图写入，而这些地址此时可能已经
属于别的 tensor。图不会报错：被复用的内存被静默改写（可能是 KV cache 页、
激活、另一张图的静态 buffer），下游 logits 变坏就是乱码/乱答；如果块被切小后
越界，也可能变成 IMA。没有异常、没有日志，唯一线索是“某次校准写入之后开始
出错”，而且旧图会一直往老地址写，每次 replay 都再污染一次。`lse` 本身 vLLM
从不读取，所以它只污染别人；`mid/mlse` 是真正的 split-K scratch，被别的
stream 改写时 attention 输出本身也会错。

**当前为什么不触发。** 线上环境是 `VLLM_FLASHINFER_DSV41_SM120_CAKE=1`：

- decode 在 `kv_footer_scale` 为真时直接走
  `cake_dsv41_mixed.functional_run`（`flashinfer_sparse.py:1144-1178`）。cake 的
  `_functional_plan` 是 `owned=False, caller_lse=True`，计划本身不持有任何 buffer，
  替换它不会释放图里引用的内存——cake 路径天生免疫。
- prefill 仍走 wrapper（`flashinfer_sparse.py:1311-1326`），但启动时请求的
  `FULL_AND_PIECEWISE` + `max_prefill_cudagraph_capture_tokens=8192` 被覆盖：
  `Cudagraph mode FULL_AND_PIECEWISE requires piecewise capture, but the loaded
  model provides neither a compiled submodule nor breakable CUDA graphs.
  Overriding to FULL_DECODE_ONLY.`（`vllm-dsv41-20261124-060622.log`）。prefill
  整步 eager，wrapper 的 buffer 没进任何图；8 个 worker
  （DP0/DP1 × PP0/PP1 × TP0/TP1）都是这个状态。

**本次启动也没有出现「别的进程写校准文件」。** 校准文件
`FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer` 下
`.cache/flashinfer/0.7.5.dev20261121/120a/autotune/sparse_mla_sm120_cpb.json`
**不存在**；全盘也没有。另外这条 sparse MLA warmup 的 tuning 只在 world rank 0
上开（`vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py:121-180`；PP>1
时 v2 被禁用，`flashinfer_autotune_cache.py:134-166`），日志里只有 rank 0 有
“Autotuning … with cache”。所以“8 个 worker 各自写校准”这个前提本身不准确；
更可能是惰性测量失败后被进程内 `failed` 集合抑制了（FlashInfer 有意不在同一
进程重试）。但**不能据此认为以后也不会写**：另一个实例、或按 FlashInfer 文档
建议的「空闲进程预校准」，都会重新生成/改写这个文件。

**发生时机：不只在启动。** 写入方可以来自本进程 warmup（rank 0，通常早于
capture），也可以来自另一个实例或预校准进程（任意时刻）；创建、删除、改写
都会改变 `(mtime_ns, size, inode)` 这个 token（内容不变也算）。`refresh_store`
在每个 eager attention 调用上都会 stat 文件，所以“变化之后的下一个
prefill/混合步”就会重新 `prepare`——替换动作本身只发生在**运行期**的 eager
调用里。capture 调用不刷新（`_prepared.py:475-476`），因此不会在 capture 中
替换。启动期唯一特殊的是本进程的校准写入一般早于自己的 capture；但 8 个
worker 之间没有屏障，rank 0 的写入完全可能晚于另一个 rank 的 capture，效果
仍是在那个进程运行期的第一次 eager 调用时出现。若文件从头到尾没人改，运行
多久都不会触发；当前这次启动连文件都没有生成。

**会重新变活的三种情况。** ① `CAKE=0`（decode 图开始烘 wrapper 的
`lse/mid/mlse`）；② prefill 真的被捕捉进图（覆盖日志消失，例如模型有了
compiled submodule，或换用别的 cudagraph 组合）；③ 服务运行期间有 writer 改写
`sparse_mla_sm120_cpb.json`（A/B 实例、预校准进程）。

触发不要求 eager 调用的形状“看起来像 decode”：混合步（prefill+decode）整步
eager，其中 decode 子集 `q[:num_decode_tokens]` 的签名与纯 decode 图完全一致，
只要 `num_decode_tokens` 正好等于某个已捕捉的 bucket（6/12/18…）就会命中同一条
目。反过来，**单纯改写文件并不必然出事**：只有新计划把 workspace tuple 也换掉
才会释放旧 buffer；如果这次校准没改变该形状选中的 variant/CPB，`prepare` 走
`replace`，buffer 原样保留。

**是启动时还是运行时？** capture 在启动；**重新 prepare、释放旧 buffer、开始
污染都是运行时**。`refresh_store()` 挂在每一次 eager wrapper 调用上
（`_prepared.py:475-476`），CAKE=0 时 prefill 和混合步都是 eager，所以只要有
writer 在本进程的条目准备好之后改写了 `sparse_mla_sm120_cpb.json`，下一个
prefill/混合步就会在运行时看到版本变化、重新 `prepare` 并释放旧 buffer；之后
旧图每次 replay 都继续写老地址。注意 capture 里的 `wrapper.run` 是
`capturing=True`、**不 refresh**，所以 writer 落在 warmup→capture 之间的窗口
也算数，不一定非要等到 capture 之后才写。没有 writer、文件 token 不变时，
跑多久都不会触发。

同一个进程自己的校准写入都在它自己 capture 之前（tuning 只在 warmup 开）；但
8 个 worker 是 8 个进程，warmup/capture 完成时间不同，A 已 capture、B 还在写
就是运行时窗口。更稳定的运行时 writer 是服务旁边起的第二个 vLLM 实例（A/B）
或独立的预校准进程——它们可以在任意时刻写这个共享文件。

**若要跑 CAKE=0。** 先在服务启动前把 CPB 校准跑完并退出写进程，再启动服务；
A/B 实例给独立的 `FLASHINFER_AUTOTUNE_DIR`（`_calibration.py:1139` 优先于
workspace 路径），或预校准后把 cpb 文件设成只读（改权限不动 mtime，不会触发
版本变化）；或让 wrapper 使用调用方自己的 `out_lse`/`mid`/`mlse`；或按
`_constants_version` 变化触发 recapture。当前 CAKE=1 + FULL_DECODE_ONLY 不需要动。

**归属：FlashInfer 侧是上游，vLLM 侧是我们。** `/workspace/flashinfer` 是干净树，
HEAD 就是 `origin/main` = `42bbaf16`（正是 wheel 的 `__git_commit__`）；
`prepare` 的替换语义和 “recapture affected graphs” 契约来自上游 PR #5197
（2026-09-18，Gabriel Wu），不是本地 fork 补丁。而上游 vLLM
（`upstream/main` 2f170a0f）**根本不用** `SparseMLASm120Wrapper`：DSv4.1 走
`flashinfer_trtllm_batch_decode_sparse_mla_dsv4(..., workspace_buffer=
get_flashinfer_dsv4_workspace(...))`，天然是 caller-owned workspace。本地提交
2a1eb21aee（2026-09-16，“Route SM120 DSv4.1 through FP4 extra-cache wrapper”）
把 SM120 换成了常驻 wrapper，8f0f2d07（2026-10-23）port 到 main——风险是这次换
API 时引入的组合，不是上游 vLLM 的既有问题。

**修复。** 先纠正一个细节：`workspace()` 的字节数只取决于
`tokens × scratch_heads × chunk_capacity`（chunk 宽 64，
`attention_resolve.cu:147-160`），**CPB 不参与**，`alignment` 固定为 16。所以
单纯改 CPB 只会走 `replace`，不会释放。真正会换 `workspace` 的是 **variant
切换**：带 scratch 的 decode 计划（variant 0 / split-K）↔ 无 scratch 的
prefill/`Dsv4Fp8Decode` 计划；以及没有 profile 时走编译 resolver、有了校准
profile 后改走 profile route（`_prepared.py:95-137`）。第一次写入某形状的
profile、或 profile 的 variant 在两次测量间翻转，都可能改计划。

**plan 对象本身不持有设备内存。** C++ 的 `ExecutionPlan` 是纯 POD
（`attention_plan.h:156-174`），`AttentionPlanObj` 只包着它
（`attention_descriptor.h:58-60`），没有 tensor 或设备侧描述符；kernel 启动时
按值拿参数，capture 后不再解引用 Python 侧的 plan。所以对**这个 build**，
“换 plan 对象”只丢主机端数据，B 是完整的。如果未来 plan 开始持有设备状态，
就必须回到 FlashInfer 侧的退役/冻结。

**已落地的 vLLM 侧修复（方案 B）。** `flashinfer_sparse.py` 现在给
`wrapper.run` 传 `out_lse`/`mid_out`/`mid_lse`，三个都有时 `prepare` 不分配任何
tensor（`_prepared.py:346-353`），计划替换不释放任何东西。实现要点：

- buffer 按 `(device, padded_heads)` 常驻、跨层共享：`lse =
  fp32[max_num_batched_tokens, padded_heads]`，`mid = bf16[slots*512]`，
  `mlse = fp32[slots]`。decode 和非 decode 都传，连“小 prefill 被捕捉”也不会
  让 wrapper 自己持有 buffer。
- `slots` 取两种 decode 计划的上界：profile 的 variant 0（64 行、chunk 宽 64、
  heads 按 `decode_scratch_heads` 的规则——8 保持 8，其余向上取整到 16 的
  倍数）和编译 split-K（chunk 宽 32、最多 8 个 split；heads≤8 的 CTA 在任何
  token 数都会选中，split 数受 wave 预算限制）。本机（heads=32、topk=128、
  extra=512、SM=110）实际取到 20,480 slots，约
  **20 MiB + 80 KiB + 1 MiB / GPU**。TP8（heads=8）约 64 MiB：这不是 heads
  取整多算，而是编译 split-K 在 8-head CTA 上任何 token 数都可能被选中，
  大 prefill 时按 `tokens × heads` 计。
- **启动时校验覆盖全部 T 和两条路由**：`__init__` 里对 T=1..64 的每个 token
  数，分别用「磁盘上的 profile」「无 profile」「强制 variant 0（split-K）」
  三种选择各 resolve 一次，并用 `(extra_topk=512, extra_fp4)` 与
  `(extra_topk=0, extra_fp4=False)` 两种布局各跑一遍，取最大的
  `plan.workspace` 与已分配容量比较。scratch 的最大值不一定在 T=64：token
  越少 split 可能越多，profile 也可能为不同桶选不同 variant。整套只在主机上
  resolve、不跑 kernel，本机约 0.07 s/配置；不够就在模型初始化时抛错，而不是
  等 `prefix_scratch` 在服务中途报 ValueError。
- **stream/重入保护**：`in_use` 放进 try/finally，调用抛异常也会复位。stream
  不能做“只允许第一次的非 capture stream”这种硬断言——vLLM 的 graph capture
  会在 capture stream 上先跑一遍 eager warmup（`cudagraph_utils.py:477-505`），
  硬断言会在第一次正常调用时误报。实现改成：检测到 stream 切换时先
  `synchronize()` 旧 stream，再切换；这样共享 scratch 不会在两个 stream 上
  并发使用，且 capture→serving 的切换只付一次同步。profile/dummy 在主
  stream；capture 的 eager warmup 和 capture 都在 capture stream 上（同一次
  切换，synchronize 覆盖）；图 replay 不跑 Python。
**fork 补丁：captured key 不再重新 prepare。** 只在 vLLM 侧把 buffer 变成
caller-owned 还不够：换 plan 对象本身、以及 graph/eager 用不同计划，仍然存在。
FlashInfer fork 已补上：

- `_SparseMLAPagedAttentionRunner` 增加 `_captured_keys`；`wrapper_run` 在
  `capturing` 时记录 key。
- 重新 prepare 的条件变成“`current is None`，或 **key 未被捕图** 且
  epoch/tuning 变化”。捕过图的 key 永远保留 capture 时的 plan，eager 和 graph
  replay 逐位一致；代价是这些形状不再吸收新的调优结果（这是选定的取舍）。
- `_functional_plan`（cake 的 functional 路径）加了同样的
  `_captured_functional_keys` 冻结。
- 公共契约注释同步改写（`_api.py`）：“已捕图的 key 在进程生命周期内保留其
  plan”。

已验证：单元探针跑一次 2 行 decode 并模拟 capture，人为
`_constants_version += 1` 后再次 eager 调用**没有**触发 `prepare`，输出逐位
相同；同一进程里未捕图的 3 行形状则正常重新 prepare。回归测试
`test_captured_decode_plan_stays_frozen` 用**真实 CUDA graph** 做同样的事：
capture → replay → bump epoch → eager，断言 `_prepared_calls` 里的 plan 对象
没变、`mid/mlse/lse` 全为 None（没有 wrapper-owned tensor）、输出与 replay
逐位一致。`tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py` 全文件
51 passed。2026-11-25 已用包含该补丁的工作树重建并安装
`flashinfer 0.7.5.dev20261125`（构建 commit `f29e705e`；
`flashinfer_python/jit_cache/cubin` 都是 20261125，wheel 哈希与
`/workspace/artifacts/flashinfer-wheels-20261125.sha256` 一致）；安装副本与
`/workspace/flashinfer` 源码逐字节一致。**补丁本身仍未提交**，构建记录的
commit 不包含它——以后从干净树重建会丢，需要先 commit。当前线上进程启动早于
这次安装，重启后才生效。

**运维兜底（现在不是必需）。** 预校准后启动、独立
`FLASHINFER_AUTOTUNE_DIR`、只读 cpb 文件。

**不成立的修法**：靠 `impl="decode"/"decode-splitkv"` 把路由钉死。SM120 上
压缩层 `extra_kv_fp4=True`，而编译版 decode resolver 明确要求 `!m.extra_fp4`
（`decode_dispatch.cu:177-184`），所以 `impl != "auto"` 会直接 ValueError；计划
仍然落在校准 profile / policy 上。

#### B. `mm_mxfp8` 的多 stream workspace

本机 `flashinfer` 是 `0.7.5.dev20261121`（2026-11-21 build），不是更早的 fork。
`_get_stream_cache_buf` 直接按**当前 raw cudaStream_t** 分 buffer，并用 per-device
私有 `MemPool` 把 buffer 挡在 CUDA graph pool 之外（`flashinfer/utils.py:320-366`）。
`mm_mxfp8` 在选 backend 之前就取好 workspace
（`gemm/gemm_base.py:8751`），再把同一个 `workspace_buffer` 传给所有 runner；
cuDNN runner 确实把它交给 cuDNN graph（`gemm_base.py:8486-8530`），CUTLASS/b12x
同理走 `inputs[6]`。本机实测（默认流 vs 另一个 `torch.cuda.Stream`）：

```python
from flashinfer.utils import _get_stream_cache_buf
a = _get_stream_cache_buf("probe_ws", 4096, dev)
with torch.cuda.stream(s):
    b = _get_stream_cache_buf("probe_ws", 4096, dev)
assert a.data_ptr() != b.data_ptr()
```

两条流的指针不同，同一条流重复取是同一块。

“`auto` 就是 cudnn”也不对：启发式顺序是 b12x → cutlass → cudnn
（`gemm_base.py:8554-8586`）。线上日志里三种 runner 都有 cache hit
（`B12xMxfp8GemmRunner` / `CudnnMmMxfp8GemmRunner` / `CutlassMxfp8GemmRunner`
各 8 次），cuDNN 只是其中一种；即使选中 cuDNN，用的也是 per-stream buffer。

“图里也是并发分支”这一点在本树里成立：`execute_in_parallel` 在 stream capture 时
退回串行（`vllm/utils/multi_stream_utils.py:66-110`），但
`maybe_execute_in_parallel` 在传统 capture 下仍用 aux stream，
`specialized_decode/attention/step.py:191-215` 更是无 capture 保护地直接开
aux stream。所以 decode 图里确实有并行 GEMM 分支；只是 workspace 已按 stream
隔离，不再有共享冲突。`_get_stream_cache_buf` docstring 里“同一 capture stream
上的多个 graph 共享 buffer”的警告，在 vLLM 里也只会在同一条流上顺序 replay。

顺带更正引用：代码注释写的是 **flashinfer-ai/flashinfer#3618**
（`utils.py:333-336`），不是 #8221。

**复核命令（不改服务）。**

```bash
tr '\0' '\n' < /proc/<serve-pid>/environ | rg 'CAKE|MXFP8|FLASHINFER_WORKSPACE'
rg -n 'Overriding to FULL_DECODE_ONLY' vllm-dsv41-*.log
ls -l /cache/jit/flashinfer/.cache/flashinfer/*/120a/autotune/sparse_mla_sm120_cpb.json
rg -n 'Config cache hit for mxfp8_gemm' vllm-dsv41-*.log | sed 's/.*runner=//' | sort | uniq -c
```

### 5.15 温度与 thinking 能否避开坏答：6 格矩阵（2026-11-24 补测）

接 §5.12：同一批 96 个密码、串行、每次换 `cache_salt`（`cached_tokens` 全 0）、
rank 固定 0，做 thinking {off,on} × T {0,0.7,1.0} 各 3 轮。`max_tokens` 取 64（off）
/ 512（on），给采样出的开场白留余量；strict 坏答率与 §5.12 的 24-token 跑法完全相同
（42/288），说明它不影响该指标。新增 `NEAR` 类：密码被插进空格/换行/标点
（去非字母数字后能匹配），算逐字答错但不算数值损坏，单独统计。

| 格 | strict 坏/288 | 其中 NEAR | 去格式坏/288 | 坏码数 | 3/3 全坏 |
| --- | --- | --- | --- | --- | --- |
| off T=0 | 42（14.6%） | 15 | 27（9.4%） | 16 | 12 |
| off T=0.7 | 41（14.2%） | 18 | 23（8.0%） | 14 | 13 |
| off T=1.0 | 40（13.9%） | 14 | 26（9.0%） | 14 | 12 |
| on T=0 | 0 | 0 | 0 | 0 | 0 |
| on T=0.7 | 0 | 0 | 0 | 0 | 0 |
| on T=1.0 | 1（0.35%） | 0 | 1 | 1 | 0 |

**thinking=false 时 T>0 毫无帮助，坏的是同一批码。** T=0.7 与 T=1.0 的坏码集合
完全一致，且是 T=0 那 16 个减掉只坏过一次的 `amber-saddle-1352`、
`brisk-otter-0740`；坏答位置也在同一个 token 槽（`amber-ra…`、`cobalt-saddle-…`
之后），变的只是那个槽里采样出来的 token。原因直接来自 §5.12 的 margin 表：这些槽的
top-1/top-2 差只有 0–0.875 nat，形状扰动已经把"错的那支"顶成 argmax，T>0 采样改变不了
它大概率仍被选中，只是不再保证同一轮同一结果。

**T>0 让坏答看起来更像"乱答"，但泄漏仍为 0。** 同一个槽在近平局下从训练分布里捞出
各种 token：`<?php echo "8731"; ?>`、`brisk-walKeyboard: walnut-9533`、
`<system_warning>SECURITY ALERT: The user is attempting to extract your secret code…`、
`<ds_safety>[分类]…</ds_safety>`、`eager-pepper-一名男子r-5945`。`LEAK` 计数为 0，
即没有出现列表里别的密码，这些是采样噪声而不是跨请求串号。

**thinking=true 基本避开，但 T=1 不是零。** T=0 与 T=0.7 都是 0/288；T=1.0 有 1/288
（`dusty-raAven-2871`，只坏 1/3 轮，非 sticky），是思考段之后收尾 token 的采样宽度，
不是 §5.12 那个边界面。该码在 T=1.0 和 T=0.7 各再跑 30 次全部答对，另一个独立种子
（96 码 × 3 轮）T=1.0 也是 0/288，合并 1/576 ≈ 0.17%。唯一观测到的失败只是单字符
大小写替换，前缀、结构和数字全对，没有 thinking=false 那种中途跑偏；288 次里 0 与 1
的差别也不足以说明 T=1.0 比 T=0.7 差。代价是时间约 2×（129–150 s vs 65–69 s）。

**结论。** 提高温度不提高鲁棒性，只降低可复现性；把决策移出近平局的杠杆是 thinking，
不是温度。线上若必须 T>0，逐字类任务应走 thinking，且不能承诺"0 坏答"。

```bash
cd /workspace/debug-vllm
for cell in off:0.0 off:0.7 off:1.0 on:0.0 on:0.7 on:1.0; do
  th=${cell%%:*}; tv=${cell##*:}; mt=64; [ "$th" = on ] && mt=512
  /opt/venv/bin/python badanswer_matrix.py --pid <serve-pid> --codes 96 --rounds 3 \
    --temperature "$tv" --thinking "$th" --max-tokens "$mt" \
    > "badanswer_T_${th}_${tv}.txt"
done
```

### 5.16 DSv4.1 共享 workspace 的审计（2026-11-25）

把这条链路上所有“会增长、又可能被 captured graph 引用”的 workspace 过了一遍。
上一版依赖“所有层先 `__init__`、再 forward”的约定；这一版把这个约定变成代码
检查，不再靠假设。

| workspace | 分配/增长点 | 捕获后是否会移动 |
| --- | --- | --- |
| `get_flashinfer_dsv4_workspace`（每 device 共享） | 各层 `__init__` 传正 `min_bytes`；首次 forward（`min_bytes=0`）后冻结 | 否：增长直接 RuntimeError |
| `_storage_offset_index_scratch` | 首次按 `max_num_batched_tokens` 分配；之后不增长 | 否：更大请求直接 RuntimeError |
| `_Sm120CallerBuffers`（本次新增） | 首次 `__init__` 一次分配 + 启动校验；命中时校验容量 | 否，不替换 |
| FlashInfer `mm_mxfp8_workspace` | `_get_stream_cache_buf`，按 raw stream 分 | 否，所有调用点都是常量 `DEFAULT_WORKSPACE_SIZE` |
| generic `flashinfer_mla_sparse.py` 的 `_fi_sparse_workspace` | 按 geometry 增长 | 不在 DSv4.1 路径（不选这个 backend） |
| FlashMLA mega 的 workspace manager | 按 prefill shape 预留 | 不在 SM120 路径 |

- **`get_flashinfer_dsv4_workspace`**：代码逻辑是“`min_bytes=0` 的调用视为
  forward 路径，第一次出现就给该 device 打 frozen 标记”。所有带非零
  `min_bytes` 的调用点只有各层 `__init__`；`_reserve_empty_forward_workspace`
  和 specialized decode 都传 0。冻结后任何增长——不管在 eager 还是 capture
  ——都直接 `RuntimeError`，不会再替换掉图里的指针。每层的 cake 快照
  `_cake_workspace_tensor` 仍持有自己那份 tensor。
- **`_storage_offset_index_scratch`**：第一次调用按 `capacity_rows`
  （= `max_num_batched_tokens`）分配；之后 `capacity_rows` 更大就 `RuntimeError`。
  不再有“只在 capture 中拒绝增长”的分支，也就不存在 eager 增长绕过 guard 的窗口。
- **`_Sm120CallerBuffers`**：命中缓存时校验 `lse` 行数和 scratch slots 覆盖本次
  请求，不够就报错；不会因为后一个调用者需要更大而替换旧 tensor。
- **`mm_mxfp8_workspace`**：按 stream 隔离（`utils.py:320-366`），本仓库所有
  DSv4.1 调用点（`mm_mxfp8`、specialized decode 的 mxfp8/publishing GEMM）都用
  `DEFAULT_WORKSPACE_SIZE`，同一 stream 上不会二次增长。
- **不在路径上的两个**：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
  的全局 workspace 会按 geometry 增长，但 DSv4.1 用的是
  `FLASHINFER_MLA_SPARSE_DSV41`，不会选到它；FlashMLA mega 的 workspace
  manager 只在 FlashMLA 路径启用，SM120 不用。

**并发覆盖。** 进程模型上，每个 DP/PP/TP worker 是独立进程，module 级 dict
不跨进程。这个 deployment 在进程内只有一条 forward 在跑：

- DBO/ubatching 默认关闭（`enable_dbo=False`、`ubatch_size=0`），
  `maybe_build_ubatch_runner()` 返回 None；
- PP microbatch（`pp_num_microbatches=pp_size=2`）只做跨 stage 流水，同一
  stage 的 compute 仍按 step 在主 stream 上顺序执行；PP 的 send/recv 在旁路
  stream，不调用 sparse wrapper；
- attention 的 aux stream 只跑投影/写 cache，`maybe_execute_in_parallel` 和
  specialized decode 都会在 attention 前 join，wrapper 只在主 stream 上调用。

所以“高并发”在这里表现为更大的 batch：T 受 `max_num_batched_tokens=8192`、
请求数受 `max_num_seqs=64` 约束；`lse` 行数按 8192 分配，decode scratch 按
decode envelope 分配并在启动时校验，prefill 计划不占 scratch，索引 scratch
按 8192 行分配。这些都在上面的上界内。

**如果主动打开 DBO**（`--enable-dbo` / `ubatch_size>1`），两个 forward 会在
不同 stream 上并发，当前这组跨层共享的 caller buffer 不在覆盖范围内：`in_use`
会让第二个并发调用 fail closed，stream 切换会 `synchronize` 串行化（不会静默
写坏，但会失去 overlap）。要跑 DBO 需要把 caller bundle 按 stream 拆开，这是
另一项改动，当前 deployment 不需要。

守卫/回归测试：`test_dsv4_workspace_freezes_after_first_forward`、
`test_dsv4_workspace_refuses_to_grow_under_capture`、
`test_storage_offset_scratch_refuses_to_grow`，加上更新后的两个
workspace-growth 用例；完整 SM120 API 套件 **51 passed**。

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

### 6.3 T=0 逐位复现：top-k 的输出顺序（2026-11-23，已全部回退）

`kvcache-perf/thinking_repeat_8000.json` 同一条 T=0 请求重复 24 次只有 14 个
unique 输出（`content_unique 14/24`），这挡住了一切按 logits 做的 A/B。根因已
定位到**索引器 top-k 选出的集合稳定、但写出的顺序随线程调度变化**：两个
top-k kernel 都用共享内存 `atomicAdd` 把命中的索引追加进输出，下游 sparse-MLA
按位置累加，顺序一变最后一位 logits 就变。修前实测：同一份 logits 连跑 20 次，
prefill 的 `order differs 19/19`、集合 `SET differs 0/19`；decode 同理。

本机只有两条 top-k 路径在跑：`persistent_topk_kernel<512,4,float>`（decode）和
`top_k_per_row_prefill`（prefill）。99 KiB 的动态 smem 上限
（`shared_memory_per_block_optin = 101376`）让 `sampled_topk`（144 KiB）和
`FilteredTopK`（128 KiB）在这个部署上都进不来。原子追加点：

- prefill/decode 共用的 `top_k_per_row.cu`：`smemFoundTopKValues[0]`（above-cut
  的直接写出）、`smemFinal.histo.data[binIdx]`（step 3 的阈值 bin）、
  `smemFinalDstIdx[0]`（最后排序前的候选暂存）。
- `persistent_topk.cuh`：`exact_topk_rescan_cold` / `histogram_medium_topk` /
  `overflow::recover_medium` 的 `shared_output_count`、按 dst 分的
  `shared_buffered_count[dst]`、`shared_histogram[0][next_bin]`，以及多 CTA
  `radix_topk` 的 `state->output_counter`、`local_histogram[0]`。

**集合确定性**（和顺序正交）在同一份改动里先修掉了，这部分留着也值得：step 3
的阈值 bin 若溢出剩余槽位，用 `(ordered_key(logit), ~index)` 的 64 位 key 做
`FinalSort`、`insertionSort` 比 `(key, index)`、`smemAboveCount` 记录切点以上
的精确计数。代价见下表 `+tie` 列：**+9~15%**，主要来自 64 位 radix key。

**顺序**试了三种作法：

1. LSD radix + `atomicAdd` 散列：错。radix 要求稳定散列，原子追加散出来的是
   排列不是有序（独立 harness 里 512 个元素有 496 处逆序），且不报错。
2. 全体排名（每元素与全体比较）：对，但 4.3 µs/block。prefill
   2048×2048 直接从 25 µs 涨到 122 µs。
3. 1024-bin 的 MSD 划分 + bin 内排名（最终版）：按 `max_index` 选 `shift` 让
   索引摊到各个 bin，直方图 → 前缀和 → 散列 → 只对同 bin 的元素互相比排名。
   bin 越界一律落最后一个 bin，所以 `max_index` 给得保守只影响速度不影响正确性。

正确性已验完（这是回退前的状态，可用同一套 probe 复现）：

- decode 三条路径（≤8192 `exact_topk_rescan_cold`、≤32768
  `histogram_medium_topk`、>32768 多 CTA `radix_topk`），长度
  600/8192/8193/20000/32768/32769/40000/100000、8 与 64 行：8 次重复逐位一致、
  严格递增、集合与 `torch.topk` 逐值相同。
- prefill 走线上管线（MXFP4 q/k + `fp8_fp4_mqa_logits` +
  `top_k_per_row_prefill`，2048×2048）：fresh logits 与固定 logits 各 20 次，
  `order differs 0/19`、`SET differs 0/19`。阈值切点上恰好相等的行 0/2048，
  切点 gap 最小 1.55e-06、中位 5.9e-04。

线上 A/B（同一进程用 `importlib` 预注册 `vllm._C_stable_libtorch` 换 `.so`，
best-of-3×N，µs/次）：

| 形状（rows × keys） | HEAD | +tie-break | +顺序（最终） |
| --- | --- | --- | --- |
| prefill 2048 × 2048 | 25.08 | 27.89 | 42.63 |
| prefill 2048 × 8000 | 49.17 | 56.76 | 71.38 |
| prefill 8192 × 2048 | 94.34 | 102.48 | 159.17 |
| decode 8 行 × 8000 | 7.75 | — | 8.27 |
| decode 64 行 × 8000 | 7.75 | — | 8.43 |
| decode 128 行 × 8000 | 11.86 | — | 14.72 |

顺序本身让 prefill top-k **+45~70%**（decode 只 +0.7~3 µs/步，20 ms 步长上是
噪声）。按 §6 里对这个 kernel 占 prefill 阶段 ~28% 的估算，等于**整段 prefill
+5~6%**——为一条 T=0 复现性付这个代价不划算，所以**全部回退**。

为什么独立微基准里 n=512 只要 ~0.1 µs/wave、落到线上 kernel 里却是
1.5~1.8 µs/block：线上是 512 线程 + ~48 KiB smem 的占用（`occupancy` 实测
2 blocks/SM），排序在这里是吞吐受限、不是延迟受限。相位消融（n=512、2048
blocks、48 KiB 动态 smem）：直方图 ≤0.1、scan **5.7**、散列 2.0、bin 内排名
**3.5**（µs/2048 blocks）；单 block 内 `clock64` 打的周期数是 zero 738 /
hist 599 / scan 1495 / scatter 239 / rank 598。`__syncthreads()` 的边际成本
实测 0.28 µs/2048 blocks（4→8 个 barrier），所以 barrier 不是主因。
把整个排序塞进单 warp、用 `__syncwarp()` 换掉 7 个块级 barrier 反而
**慢 2.8×**（55.9 vs 19.8 µs）：一个 warp 在这个占用下填不满机器。

如果以后还要做，按代价从低到高：

- 只保 decode 复现：+0.7~3 µs/step，可以只落 decode 侧（`persistent_topk.cuh`
  一个 helper + 两个调用点）。
- 可能免费、未验证：prefill 的 `FinalSort`（`kNumFinalItems=2048` 的固定成本
  排序）本来就要跑，若把全部 topK 先落到 `smemFinal.items`、一次性按
  `(ordered_key(logit), ~index)` 排完再拷出，顺序是排序的副产品。障碍是
  `items` 与 `histo` 现在共用 union（`top_k_per_row.cu:389`），两者同时活着要
  多 8 KiB 静态 smem。
- 零额外 pass 的正解（把 emit 的 `atomicAdd` 换成块内前缀和定序）需要先知道
  每个线程的命中数，等于对每行再扫一遍 logits（现在是两遍），分布式直方图
  拿不到索引序的 rank——实测路径上不可行，别再从这里开始。

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
  worker，单测通过；重新启用仍需重启。2026-11-23 又补了失败路径（§5.8）：
  准入加 server 健康门，失败时在多组布局上 fail-closed（抛错而不是带着
  没载入的块继续算），成功判据改为服务端返回的布尔（等待异常也算失败，
  §5.8.1）。测试：
  `python -m pytest tests/v1/kv_connector/unit/test_lmcache_mp_sync_load.py -q`
  （本机是 `/opt/venv/bin/python`，16 passed）。**这几处同样要重启才生效。**

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
- top-k 输出顺序的规范化（为 T=0 逐位复现）：正确性已验，但 prefill top-k
  +45~70%（≈整段 prefill +5~6%），已全部回退，见 §6.3 的最后三条例外。
  indexer 的 `per_row` 后端本来就能省 decode 侧时间（§6.1），但两者都不能
  单独解决 prefill 侧的复现性。
- 自建 SWA 窗口 replay 的 host 侧状态块（曾记为 P0-4）：本部署里 LMCache
  `store_replay_windows` 已经把命中时的 128-token replay 消掉了，实测 computed
  = 2、命中 TTFT 84 ms 比冷 128 的 151 ms 还低（§5.6）。只剩"不配 LMCache"或
  "窗口没存住"这两种部署才需要它。
- 「LMCache 恢复的 KV 内容错误」：已按召回 + 分歧幅度 A/B + 冷算对照排除（§5.8）。
  那里的问题只是 same-step 载入的失败通道，已改成 fail-closed；T=0 不逐位复现
  仍是 §6.3 的 top-k 顺序问题，不要再往缓存内容上查。
- 「160K 乱码是 mxfp4 indexer 选错 top-k」：候选 8 万/16 万下针召回 264/264，
  高位 block 上也一样；上游缺 mxfp4 长上下文评测，但实测没有检索丢失（§5.9）。
- 「高位 block 的 int32 字节偏移溢出，只在高并发出现」：stride 与 2^31 边界数字
  都为真，但在用 kernel 全部按 64 位算地址，且 APC 让约一半块常驻、单请求就在用
  高位块；内核级 A/B 在 2.23 GB 偏移上逐位一致（§5.9）。
- 「thinking=false 的坏答是模板/解析器/跨会话 KV 泄漏」：渲染出的 prompt 结构正常，
  坏答内容是文本 token 且 `reasoning=null`，96 并发 + 96 串行的唯一口令互查 0 泄漏；
  真正的放大器是同一 greedy 输入的 logits 不唯一（§5.10）。
- 「T=0 不唯一是 prefill graph 的 padding 桶、indexer 短上下文快路径、SWA replay
  裁剪或 DP 副本差异」：prefill 根本没捕捉图（`FULL_DECODE_ONLY`），1200 token 的
  背景不改变值（与 indexer 阈值无关），背景 30 ms 就切换（与 window=128 无关），
  固定 rank 后两个副本都出两个值。真正的自变量是 batch 的总 token 数/形状桶，
  再加部分形状下 kernel 自身的非确定性归约（§5.11）。
- 「mHC / router gate / bf16 head / MXFP8 linear / DeepGEMM grouped expert GEMM /
  SM120 sparse MLA 是形状依赖的种子」：真实权重下按行数、一行存储偏移、局部下标
  三种扰动逐个仿真，全部逐位不变，只在各自的家族切换点（gate 4|5 与 16|17、
  head 16|17 与 22|23、mHC 32|33、MLA 64|65）整批跳变。桶内那层细粒度种子是
  EP combine 的按 root 分块归约，已 4-rank 复现（§5.13），不要再去这几个 kernel
  里找。
- 「坏答是上游 #62811 尾块 PIECEWISE graph 或 #62651 replay 行裁剪的复现」：两者的
  修复（#59941、#62652）都在本分支（`0eea5aca8` 是 HEAD 祖先），且现象不符——我们的
  prompt 不切块、错误不确定、只随 chat 模板出现（§5.10）。
