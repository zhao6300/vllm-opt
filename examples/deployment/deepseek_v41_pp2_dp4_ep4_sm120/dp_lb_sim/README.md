# DP LB 路由模拟（VLLM_DP_LB_TOKEN_AWARE / VLLM_DP_LB_SHARED_LOAD）

支撑 `VLLM_DP_LB_TOKEN_AWARE` 与 `VLLM_DP_LB_SHARED_LOAD` 的离线验证与调参。
模拟器逐条复刻 `vllm/v1/engine/core_client.py` 的 `DPLBAsyncMPClient`：

- 打分 `score = max(scale*(本机 inflight + 他机 inflight), snap_waiting + snap_running)
  + min(scale*pending_load, cap) + snap_waiting*6*max(0, snap_kv-0.5)`；
- `waiting/running/kv` 取自 100 ms 快照，不是引擎实时状态；
- `pending_load = pending_ms/budget`（有预算且 ms>0）否则 `pending_tokens/T`；
- 引擎上报未完成 prefill token（已扣 cache 命中）与（有预算时）按 chunk 估的 ms；
- dispatch floor：本 server 刚发出的请求按**完整 prompt** 给引擎兜底，直到该请求
  首个输出或其后第 2 个快照；
- shared load：每个 server 把自己对每个引擎的 (inflight, pending tokens,
  pending ms) 写进共享槽，路由时把其他存活 server 的槽**不缩放**地相加；
- DP 锁步：一步耗时 = 各 rank 的 max。

## 校准（来自 TIMING.md §5.3 / §5.5，快态）

| 量 | 取值 |
| --- | --- |
| prefill 形态一步 | 130 ms 固定 + 0.087 ms/token（16K 冷 = 1.55 s） |
| 纯 decode 一步 | 20 ms + 每多一个请求 0.15 ms |
| max_num_batched_tokens | 8192 |
| DSpark | 6 行/请求，约 2 token/步 |
| 输出长度 | 均值 300 token |

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `sim.py` | 核心模拟器 + 单工作负载扫描（`--wl`，默认全部） |
| `focus.py` | 固定几组配置、多 seed，按短/长请求拆 TTFT 尾 |
| `shared_skew.py` | API server 到达偏斜下 shared load 的收益 |

工作负载：`uniform_long`（32×16K 均匀）、`staggered`（同上但到达分散）、
`burst`（1K/8K/32K 同时到）、`single_long`（1 个 128K + 47 个短请求）、
`online`（85% 命中、2% 长冷）、`online_hot`（85% 命中、8% 长冷、5 req/s）。

运行（仓库根目录，用仓库 venv）：

```bash
/opt/venv/bin/python examples/deployment/deepseek_v41_pp2_dp4_ep4_sm120/dp_lb_sim/focus.py
```

## 结论（6–8 seed 均值，短请求 TTFT p99，ms）

| 配置 | staggered | burst | single_long | online | online_hot |
| --- | ---: | ---: | ---: | ---: | ---: |
| off（只按请求数） | 5408 | 6411 | 12641 | 4527 | 11304 |
| 默认 T=2048 cap=4 | 5408 | 6668 | 10955 | 1758 | 10143 |
| T=2048 cap=16 | 5886 | 6017 | **843** | **888** | 3974 |
| T=2048 cap=∞ | 7308 | 4952 | **843** | **888** | 3636 |
| T=8192 cap=8 | 5888 | 5194 | 9270 | 1434 | 8323 |
| T=16384 cap=4 | 5888 | 5586 | 10955 | 1913 | 10412 |

- **`VLLM_DP_LB_MAX_PENDING_REQUESTS`（默认 4）是主要瓶颈**：`pending_load` 在
  `T*4 = 8192` token 处饱和，8K/16K/32K/128K 的 prompt 在打分里完全一样。
  `single_long` 里短请求 p99 从 12.6 s 只降到 11 s；把 cap 提到 16 后降到 0.84 s（15×）。
- **cap 的代价**只在“均匀长 prompt、到达分散”时出现：`staggered` 上 cap=∞ 让
  makespan +2%、短尾 +35%；cap=16 只 +9%。均匀长 prompt 时把 cap 调大没有收益，
  但也不会更差（T≥8192 时与 off 完全一致）。
- **最优 cap 随 DP 规模增长**：burst 上 DP4 在 cap≈16–32 到平台，DP8 在 cap≈8–16
  到平台；默认 4 对两种规模都偏小。
- **预算模式**（`VLLM_PREFILL_STEP_BUDGET_MS=300`）把 ITL p99 从 ~840 ms 压到
  ~460 ms，并让短请求在长 prefill 中间插队（online_hot 短尾 10.0 s → 1.5 s），
  代价是长请求 TTFT 和 makespan 变差（128K 长尾 13 s → 23 s）。
- **shared load** 只在到达偏斜时有用：4 server、75% 请求打到 server 0 时，
  online 短尾 1853 → 905 ms；8 server 时 4094 → 916 ms。无偏斜时中性。
  单 API server（Rust 前端恒为 1）是 no-op。

## 注意

- 表中的 `cap` 对应 `VLLM_DP_LB_MAX_PENDING_REQUESTS`（默认 4，0 表示不封顶）。
- 模拟是相对比较，不是绝对步长；`own_i`、ragged 罚项、KV 压力都未建模。
- 文档里“uncapped 让 32×16K 偏成 10/8/7/7、慢 25%”在本模型里只复现出
  makespan +2%（`staggered`），没有 25%；结论方向一致、量级不同。
- 预算模式的 `VLLM_PREFILL_COST_*` 必须按 **prefill 形态步**拟合
  （fixed≈130 ms、0.087 ms/token）。用 decode 步的拟合值（fixed≈10 ms、
  0.021 ms/token）会让 `max_prefill_tokens` 恒等于 8192，预算完全不生效。
