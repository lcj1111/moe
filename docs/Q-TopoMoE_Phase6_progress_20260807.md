# Q-TopoMoE Phase 6：TP/DP/EP 系统矩阵（4 卡部分）

> 生成日期：2026-08-07（Asia/Shanghai）
> 范围：4×RTX 5090（NUMA0：GPU0-3）；GPU4-7 被 llama-server 占用，
> 8 卡配置（TP8×DP1、TP2×DP4、TP1×DP8、EP8）待显存可用后补测。

## 1. 环境与前置

- vLLM cleanroom `33c50587d`，完整支持 `--tensor-parallel-size`、
  `--data-parallel-size`、`--enable-expert-parallel`、`--all2all-backend`
  （`allgather_reducescatter`）、`--enable-eplb`。
- BF16 模型：Qwen3.6-35B-A3B snapshot；W4A16 canonical（triton MoE）。
- 每格：启动 → health → 32 请求 smoke（并发 8，256 in / 64 out）→ summary。
- 执行脚本：`scripts/run_phase6_matrix.sh`。

## 2. 四卡矩阵结果

| 配置 | 模型 | TTFT p50 | TTFT mean | TPOT p50 | TPOT mean | e2e p50 | 通过 |
|---|---|---:|---:|---:|---:|---:|---:|
| TP4×DP1 | BF16 | 173.7 ms | 785.5 ms | 6.47 ms | 7.06 ms | 820.2 ms | 32/32 |
| TP2×DP2 | BF16 | 612.7 ms | 1,081.2 ms | 8.21 ms | 8.95 ms | 1,586.2 ms | 32/32 |
| TP1×DP4 | W4A16 | 292.3 ms | 618.7 ms | 12.04 ms | 12.06 ms | 1,072.3 ms | 32/32 |
| EP4 static | W4A16 | 187.9 ms | 496.0 ms | 8.67 ms | 9.09 ms | 735.4 ms | 32/32 |

## 3. 观察

1. **TPOT（稳态 decode）**：TP4×DP1 最低（6.47 ms），张量并行对 decode
   最有利；TP1×DP4 最高（12.04 ms），符合单卡推理预期。
2. **TTFT（prefill）**：TP2×DP2 明显偏高（612.7 ms p50），DP 协调 + 更细
   TP 分片对 prefill 不利；TP4×DP1 与 EP4 static 接近（174 vs 188 ms）。
3. **e2e p50**：EP4 static 最低（735 ms），W4+EP 组合在混合负载下最优；
   TP2×DP2 最差（1,586 ms）。
4. TP1×DP4 证明 W4A16 canonical 单卡容量通过（Runbook 前置条件），
   为 TP1×DP8 提供了依据。

## 4. 边界与下一步

- 本矩阵为 32 请求 smoke 级正确性 + 时序；正式性能需更大并发与多轮。
- 8 卡配置（TP8×DP1、TP2×DP4、TP1×DP8、EP8/static、EP8+EPLB、
  EP8+冗余专家 1）在 GPU4-7 可用后补测；TP8 曾因 llama-server 抢占
  显存导致 EngineCore 初始化失败（已确认根因，非配置问题）。
- EP 准入顺序第 1 步（EP4/单 NUMA static）已完成；后续 EP8/static、
  EP8+原生 EPLB、EP8+冗余专家 1、自研 topology-aware EPLB 待 8 卡。

原始数据：
[Q-TopoMoE_Phase6_matrix_4gpu_20260807.json](Q-TopoMoE_Phase6_matrix_4gpu_20260807.json)。
