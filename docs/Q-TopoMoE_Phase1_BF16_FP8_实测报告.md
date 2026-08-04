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
