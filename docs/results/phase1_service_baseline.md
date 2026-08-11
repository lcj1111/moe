# Phase 1 — BF16/FP8 service baseline

> Consolidated from the dated reports listed below. Source content is retained; only trailing whitespace was normalized. SHA-256 values are computed from the UTF-8 Git blob (LF-normalized); machine-readable artifacts keep their original paths for reproducibility.

## Source integrity

| Original file | UTF-8 bytes | SHA-256 of Git blob |
|---|---:|---|
| `docs/Q-TopoMoE_Phase1_BF16_FP8_实测报告.md` | 3670 | `106AD62B9E3D22CF98B524888EF6B095090E27A5869F2F3B0CE524D9FB047E7A` |
| `docs/Q-TopoMoE_Phase1_BF16_实测报告.md` | 2096 | `05F39ACF3FB85508F57C68AA56FAC4584D8738455E535ACAF0CAED9D3F9A03D8` |
| `docs/Q-TopoMoE_Phase1_statistics_20260804.md` | 6144 | `A872EB632E2087D072D8E8F86F4CDAFD820AD8FEB90160DD3A952644FC38506B` |

---

## Source: `docs/Q-TopoMoE_Phase1_BF16_FP8_实测报告.md`

# Q-TopoMoE Phase 1：BF16/FP8 服务基线实测报告

执行日期：2026-08-04（gpu-111）

## 结论

- FP8：10 个 TP2/TP4 拓扑均完成真实服务验收；每个配置完成短 workload（256/128）和中 workload（2048/256），并发 1/8/32 共 60 个筛选单元，410 个 smoke 请求全部成功。
- BF16：已补齐 checkpoint 并完成真实服务矩阵。TP4-NUMA0、TP4-NUMA1、TP8-SYS 通过；全部 TP2 配置因 32 GB/卡容量不足而启动失败，保留 OOM 证据。
- TP8-SYS：SGLang 和 vLLM 均在加载阶段失败，原因相同：FP8 权重块大小为 128，而 TP8 分片后的 expert 维度为 64，违反 kernel 的 block 对齐约束。该项保留为明确的兼容性失败证据，不报告为 OOM。

## 5.1 拓扑配置

配置文件：`configs/experiments/phase1_service.yaml`。

执行了四组同 NUMA、跨 PIX 分支的 TP2-NODE，四组 PIX 负面对照，两个单 NUMA TP4，以及一个跨 NUMA TP8-SYS。所有服务统一设置 `NCCL_IB_DISABLE=1`，NUMA 绑定由 `numactl` 完成。

## 5.2 服务端与客户端

- `serving/start_server.sh`：只负责启动 SGLang/vLLM 服务，服务端参数来自环境变量；自动把锁定 venv 的 `bin/` 放入 `PATH`，确保 FlashInfer 能找到 `ninja`。
- `serving/acceptance.sh`：依次验收 health、模型发现、真实 completion、metrics，并保存 JSON/Prometheus 证据。
- `clients/smoke.py`：独立客户端，固定 seed，支持并发 1/8/32，记录 TTFT、TPOT、E2E、成功/失败数。
- `scripts/phase1_matrix.sh`：按拓扑逐组启动、验收、筛选、释放服务并写出 `matrix.tsv` 与 `phase1_summary.json`。

## 5.3–5.5 启动与验收

FP8 推荐组 `TP2-NODE (GPU 0,2)` 的独立预检通过：

- health、模型发现、真实 chat completion、metrics 全部通过；completion 返回 `42`。
- 32 请求 smoke：32/32 成功、失败 0；短 workload 下 TTFT p50 约 1.43 s、p95 约 2.27 s，TPOT p50 约 6.38 ms。

FP8 矩阵中 10 个 TP2/TP4 配置均重复通过四级验收和 6 个 workload/concurrency 单元。BF16 结果见 `Q-TopoMoE_Phase1_BF16_实测报告.md`。

## 5.6 筛选矩阵摘要

正式 FP8 结果目录：`artifacts/phase1/20260804T063000Z/fp8/`（原始日志按 `.gitignore` 排除）。

在并发 32 下，短 workload 的 TTFT p95（ms）示例：

| 配置 | TTFT p95 | TPOT p50 | E2E p95 |
|---|---:|---:|---:|
| TP2-NODE 0,2 | 1879 | 6.29 | 2541 |
| TP2-NODE 4,6 | 1998 | 6.43 | 2667 |
| TP2-PIX 0,1 | 2119 | 7.22 | 2812 |
| TP4-NUMA0 | 1081 | 10.08 | 2318 |
| TP4-NUMA1 | 1077 | 9.72 | 2291 |

中 workload 并发 32 下，TP4-NUMA0 的 TTFT p95 为约 1527 ms、E2E p95 为约 4025 ms；TP2-NODE 0,2 的 TTFT p95 为约 2961 ms、E2E p95 为约 4220 ms。PIX 对照在同 NUMA 下的 TTFT/TPOT 均劣于 NODE 组，支持拓扑感知筛选方向。

当前可保留的 FP8 Pareto 候选为 `tp2_node_0_2`（较低 TPOT、推荐通信拓扑）和 `tp4_numa0`（较低并发尾延迟）；正式候选仍需在 BF16 基线补齐后重新按同一数据和指标比较。

## 失败证据与复现入口

TP8-SYS 的核心错误为：

```text
ValueError: The output_size of gate's and up's weight = 64 is not divisible by weight quantization block_n = 128
```

vLLM 的同类错误为 `Weight input_size_per_partition = 64 is not divisible by weight quantization block_k = 128`。复现入口：

```bash
source env/project.env
FORMAT=fp8 RUN_ID=<RUN_ID> scripts/phase1_matrix.sh
```

补齐 BF16 checkpoint 后，用完全相同的 `RUN_ID` 规则、拓扑顺序、seed、输入/输出长度和并发重新运行 `FORMAT=bf16`；不要修改已完成的 FP8 原始结果。

---

## Source: `docs/Q-TopoMoE_Phase1_BF16_实测报告.md`

# Q-TopoMoE Phase 1：BF16 服务基线实测报告

执行日期：2026-08-04（gpu-111）

## 结果

BF16 checkpoint 使用 `/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`，服务环境为 SGLang 0.5.16、torch 2.11.0、CUDA 13.0、`NCCL_IB_DISABLE=1`。

- TP4-NUMA0：通过四级验收，短/中 workload × 并发 1/8/32 共 6 个单元全部成功。
- TP4-NUMA1：通过四级验收，6 个单元全部成功。
- TP8-SYS：通过四级验收，6 个单元全部成功。
- 4 个 TP2-NODE 和 4 个 TP2-PIX：均因单卡 32 GB 容量不足启动失败；不是服务协议或拓扑失败。
- BF16 正式服务单元：18 个，完成请求失败数为 0；启动失败的 TP2 配置没有进入 completion 测试。

## 并发 32 指标

| 配置 | workload | TTFT p95 (ms) | TPOT p50 (ms) | E2E p95 (ms) |
|---|---|---:|---:|---:|
| TP4-NUMA0 | 256/128 | 1072 | 10.00 | 2306 |
| TP4-NUMA1 | 256/128 | 1188 | 9.80 | 2406 |
| TP8-SYS | 256/128 | 1178 | 9.77 | 2419 |
| TP4-NUMA0 | 2048/256 | 1626 | 10.65 | 4255 |
| TP4-NUMA1 | 2048/256 | 1678 | 10.33 | 4202 |
| TP8-SYS | 2048/256 | 1562 | 11.16 | 4253 |

当前 BF16 候选：短请求尾延迟优先保留 `tp4_numa0`；跨 NUMA 压力项保留 `tp8_sys`。TP4-NUMA1 作为同配置 NUMA 对照保留。

## TP2 OOM 证据

首个 TP2-NODE 启动失败原因为：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
GPU 1 ... 31.30 GiB in use, 47.38 MiB free
```

BF16 总权重约 67 GB，TP2 后每卡承载约 35 GB 权重/运行时开销，超过 32 GB 显存；因此不再将 TP2 OOM 误判为通信或服务框架故障。

## 复现与原始结果

```bash
source env/project.env
RUN_ID=<RUN_ID> FORMAT=bf16 \
  TOPOLOGIES=tp4_numa0,tp4_numa1,tp8_sys,tp2_node_0_2,tp2_node_1_3,tp2_node_4_6,tp2_node_5_7,tp2_pix_0_1,tp2_pix_2_3,tp2_pix_4_5,tp2_pix_6_7 \
  scripts/phase1_matrix.sh
```

原始结果目录：`artifacts/phase1/20260804T071500Z/bf16/`；矩阵汇总为 `artifacts/reports/phase1_bf16_matrix_20260804.json`。原始日志和 JSONL 不提交 Git。

---

## Source: `docs/Q-TopoMoE_Phase1_statistics_20260804.md`

# Q-TopoMoE Phase 1 BF16/FP8 statistical analysis

Completed means `failed == 0` and `completed == requests`; the legacy `acceptance` flag is reported but not used to discard measurements.

- Matrix rows: 78 (completed: 78, failed/incomplete: 0)
- acceptance=true rows: 0

## Grouped medians

| Format | Topology | Workload | C | Runs | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | tp4_numa0 | medium | 1 | 1 | 487.96 | 4.23 | 1236.28 |
| bf16 | tp4_numa0 | medium | 32 | 1 | 1625.68 | 12.14 | 4255.19 |
| bf16 | tp4_numa0 | medium | 8 | 1 | 537.17 | 6.25 | 2040.00 |
| bf16 | tp4_numa0 | short | 1 | 1 | 125.95 | 3.85 | 615.03 |
| bf16 | tp4_numa0 | short | 32 | 1 | 1071.77 | 10.50 | 2306.31 |
| bf16 | tp4_numa0 | short | 8 | 1 | 359.80 | 5.62 | 1037.13 |
| bf16 | tp4_numa1 | medium | 1 | 1 | 487.28 | 3.99 | 1193.23 |
| bf16 | tp4_numa1 | medium | 32 | 1 | 1678.00 | 11.66 | 4202.28 |
| bf16 | tp4_numa1 | medium | 8 | 1 | 500.56 | 6.21 | 1991.00 |
| bf16 | tp4_numa1 | short | 1 | 1 | 112.54 | 3.88 | 605.28 |
| bf16 | tp4_numa1 | short | 32 | 1 | 1187.65 | 10.13 | 2406.24 |
| bf16 | tp4_numa1 | short | 8 | 1 | 399.85 | 5.34 | 1071.26 |
| bf16 | tp8_sys | medium | 1 | 1 | 478.16 | 4.84 | 1335.59 |
| bf16 | tp8_sys | medium | 32 | 1 | 1561.54 | 12.44 | 4252.47 |
| bf16 | tp8_sys | medium | 8 | 1 | 570.20 | 6.77 | 2235.23 |
| bf16 | tp8_sys | short | 1 | 1 | 104.64 | 4.81 | 715.33 |
| bf16 | tp8_sys | short | 32 | 1 | 1178.43 | 9.83 | 2418.59 |
| bf16 | tp8_sys | short | 8 | 1 | 699.17 | 6.29 | 1492.45 |
| fp8 | tp2_node_0_2 | medium | 1 | 1 | 353.29 | 4.58 | 1484.68 |
| fp8 | tp2_node_0_2 | medium | 32 | 1 | 2960.63 | 7.45 | 4220.07 |
| fp8 | tp2_node_0_2 | medium | 8 | 1 | 533.26 | 5.70 | 1768.11 |
| fp8 | tp2_node_0_2 | short | 1 | 1 | 110.79 | 4.57 | 691.16 |
| fp8 | tp2_node_0_2 | short | 32 | 1 | 1879.17 | 6.30 | 2540.89 |
| fp8 | tp2_node_0_2 | short | 8 | 1 | 443.07 | 5.30 | 1109.61 |
| fp8 | tp2_node_1_3 | medium | 1 | 1 | 361.72 | 4.65 | 1509.10 |
| fp8 | tp2_node_1_3 | medium | 32 | 1 | 3198.04 | 8.16 | 4445.79 |
| fp8 | tp2_node_1_3 | medium | 8 | 1 | 536.68 | 5.73 | 1784.82 |
| fp8 | tp2_node_1_3 | short | 1 | 1 | 89.82 | 4.52 | 663.48 |
| fp8 | tp2_node_1_3 | short | 32 | 1 | 1836.45 | 6.56 | 2505.88 |
| fp8 | tp2_node_1_3 | short | 8 | 1 | 456.27 | 5.36 | 1130.70 |
| fp8 | tp2_node_4_6 | medium | 1 | 1 | 434.04 | 4.60 | 1570.33 |
| fp8 | tp2_node_4_6 | medium | 32 | 1 | 2928.07 | 8.13 | 4143.28 |
| fp8 | tp2_node_4_6 | medium | 8 | 1 | 526.41 | 5.77 | 1774.34 |
| fp8 | tp2_node_4_6 | short | 1 | 1 | 95.79 | 4.65 | 685.93 |
| fp8 | tp2_node_4_6 | short | 32 | 1 | 1998.38 | 6.50 | 2666.57 |
| fp8 | tp2_node_4_6 | short | 8 | 1 | 381.63 | 5.48 | 1041.53 |
| fp8 | tp2_node_5_7 | medium | 1 | 1 | 402.14 | 4.58 | 1534.44 |
| fp8 | tp2_node_5_7 | medium | 32 | 1 | 2836.08 | 7.86 | 4203.52 |
| fp8 | tp2_node_5_7 | medium | 8 | 1 | 602.53 | 5.98 | 1888.71 |
| fp8 | tp2_node_5_7 | short | 1 | 1 | 113.31 | 4.38 | 669.45 |
| fp8 | tp2_node_5_7 | short | 32 | 1 | 2047.94 | 6.35 | 2716.05 |
| fp8 | tp2_node_5_7 | short | 8 | 1 | 322.97 | 5.34 | 993.56 |
| fp8 | tp2_pix_0_1 | medium | 1 | 1 | 456.88 | 4.58 | 1588.87 |
| fp8 | tp2_pix_0_1 | medium | 32 | 1 | 3245.37 | 8.25 | 4551.00 |
| fp8 | tp2_pix_0_1 | medium | 8 | 1 | 555.33 | 5.86 | 1844.22 |
| fp8 | tp2_pix_0_1 | short | 1 | 1 | 105.69 | 4.45 | 670.84 |
| fp8 | tp2_pix_0_1 | short | 32 | 1 | 2118.59 | 8.15 | 2811.88 |
| fp8 | tp2_pix_0_1 | short | 8 | 1 | 414.24 | 5.54 | 1112.53 |
| fp8 | tp2_pix_2_3 | medium | 1 | 1 | 444.38 | 4.63 | 1588.73 |
| fp8 | tp2_pix_2_3 | medium | 32 | 1 | 3118.73 | 8.91 | 4479.59 |
| fp8 | tp2_pix_2_3 | medium | 8 | 1 | 584.92 | 6.05 | 1881.25 |
| fp8 | tp2_pix_2_3 | short | 1 | 1 | 115.79 | 4.49 | 685.76 |
| fp8 | tp2_pix_2_3 | short | 32 | 1 | 2115.09 | 7.80 | 2811.90 |
| fp8 | tp2_pix_2_3 | short | 8 | 1 | 398.45 | 5.57 | 1101.22 |
| fp8 | tp2_pix_4_5 | medium | 1 | 1 | 454.11 | 4.58 | 1585.34 |
| fp8 | tp2_pix_4_5 | medium | 32 | 1 | 3142.14 | 9.05 | 4484.07 |
| fp8 | tp2_pix_4_5 | medium | 8 | 1 | 675.46 | 6.19 | 2138.08 |
| fp8 | tp2_pix_4_5 | short | 1 | 1 | 105.77 | 4.45 | 670.63 |
| fp8 | tp2_pix_4_5 | short | 32 | 1 | 2136.46 | 7.38 | 2829.03 |
| fp8 | tp2_pix_4_5 | short | 8 | 1 | 383.08 | 5.55 | 1080.83 |
| fp8 | tp2_pix_6_7 | medium | 1 | 1 | 454.86 | 4.60 | 1591.03 |
| fp8 | tp2_pix_6_7 | medium | 32 | 1 | 3141.16 | 8.12 | 4369.36 |
| fp8 | tp2_pix_6_7 | medium | 8 | 1 | 577.46 | 5.90 | 1855.16 |
| fp8 | tp2_pix_6_7 | short | 1 | 1 | 106.34 | 4.47 | 674.32 |
| fp8 | tp2_pix_6_7 | short | 32 | 1 | 2085.95 | 7.42 | 2779.86 |
| fp8 | tp2_pix_6_7 | short | 8 | 1 | 389.37 | 5.54 | 1087.03 |
| fp8 | tp4_numa0 | medium | 1 | 1 | 449.07 | 4.80 | 1672.92 |
| fp8 | tp4_numa0 | medium | 32 | 1 | 1527.30 | 11.69 | 4024.90 |
| fp8 | tp4_numa0 | medium | 8 | 1 | 550.52 | 6.14 | 2039.64 |
| fp8 | tp4_numa0 | short | 1 | 1 | 103.25 | 4.54 | 679.84 |
| fp8 | tp4_numa0 | short | 32 | 1 | 1081.11 | 10.44 | 2317.60 |
| fp8 | tp4_numa0 | short | 8 | 1 | 521.16 | 5.69 | 1237.60 |
| fp8 | tp4_numa1 | medium | 1 | 1 | 449.46 | 4.62 | 1628.82 |
| fp8 | tp4_numa1 | medium | 32 | 1 | 1665.61 | 11.63 | 4095.90 |
| fp8 | tp4_numa1 | medium | 8 | 1 | 519.39 | 6.00 | 1983.40 |
| fp8 | tp4_numa1 | short | 1 | 1 | 104.65 | 4.49 | 675.45 |
| fp8 | tp4_numa1 | short | 32 | 1 | 1077.22 | 9.90 | 2290.71 |
| fp8 | tp4_numa1 | short | 8 | 1 | 334.60 | 5.64 | 1044.74 |

## Overlapping BF16/FP8 configurations

| Topology | Workload | C | FP8 E2E p95 delta | FP8 TTFT p95 delta |
|---|---|---:|---:|---:|
| tp4_numa0 | medium | 1 | +35.32% | -7.97% |
| tp4_numa0 | medium | 8 | -0.02% | +2.49% |
| tp4_numa0 | medium | 32 | -5.41% | -6.05% |
| tp4_numa0 | short | 1 | +10.54% | -18.03% |
| tp4_numa0 | short | 8 | +19.33% | +44.85% |
| tp4_numa0 | short | 32 | +0.49% | +0.87% |
| tp4_numa1 | medium | 1 | +36.51% | -7.76% |
| tp4_numa1 | medium | 8 | -0.38% | +3.76% |
| tp4_numa1 | medium | 32 | -2.53% | -0.74% |
| tp4_numa1 | short | 1 | +11.59% | -7.01% |
| tp4_numa1 | short | 8 | -2.48% | -16.32% |
| tp4_numa1 | short | 32 | -4.80% | -9.30% |
