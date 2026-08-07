# Q-TopoMoE Phase 6：TP/DP/EP 系统矩阵（P2P 后全量）

> 生成日期：2026-08-07（Asia/Shanghai）
> 范围：4×RTX 5090（NUMA0: GPU0-3）与 8×RTX 5090（双 NUMA）。
> P2P 启用后所有多卡性能数字已重测；旧 P2P 禁用版数据归档至
> `docs/archive/`（仅作历史对照）。

## 1. 环境与前置

- vLLM cleanroom `33c50587d`，支持 `--tensor-parallel-size`、
  `--data-parallel-size`、`--enable-expert-parallel`、`--all2all-backend`
  （`allgather_reducescatter`）、`--enable-eplb`。
- BF16：Qwen3.6-35B-A3B snapshot；W4A16 canonical（triton MoE）。
- 每格：启动 → health → 32 请求 smoke（并发 8，256 in / 64 out）→ summary。
- 执行脚本：`scripts/run_phase6_matrix.sh`。

## 2. 四卡矩阵（P2P 后）

| 配置 | 模型 | TTFT p50 | TPOT p50 | e2e p50 | 旧 e2e | 通过 |
|---|---|---:|---:|---:|---:|---:|
| TP4×DP1 | BF16 | 190.0 ms | 5.80 ms | 620.5 ms | 820.2 ms | 32/32 |
| TP2×DP2 | BF16 | 214.0 ms | 6.40 ms | 693.1 ms | 1,586.2 ms | 32/32 |
| TP1×DP4 | W4A16 | 238.3 ms | 11.32 ms | 1,010.2 ms | 1,072.3 ms | 32/32 |
| EP4 static | W4A16 | 194.0 ms | 8.17 ms | 711.6 ms | 735.4 ms | 32/32 |

原始数据：[Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json](Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json)。

## 3. 八卡矩阵（P2P 后）

| 配置 | 模型 | TTFT p50 | TPOT p50 | e2e p50 | 旧 e2e | 通过 |
|---|---|---:|---:|---:|---:|---:|
| TP8×DP1 | BF16 | 173.3 ms | 6.25 ms | 627.6 ms | 864.9 ms | 32/32 |
| TP4×DP2 | BF16 | 316.6 ms | 7.86 ms | 785.4 ms | 1,528.7 ms | 32/32 |
| TP2×DP4 | BF16 | 276.3 ms | 8.27 ms | 1,041.6 ms | 1,314.8 ms | 32/32 |
| EP8 TP1 | W4A16 | 227.4 ms | 12.54 ms | 1,024.6 ms | 2,055.4 ms | 32/32 |
| EP4×TP2+EPLB | W4A16 | 182.2 ms | 16.28 ms | 1,221.5 ms | 1,234.6 ms | 32/32 |

原始数据：[Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json](Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json)。

## 4. 观察

1. **P2P 全面提升多卡性能**：e2e 普遍改善 20-50%，其中 TP2×DP2（-56%）、
   TP4×DP2（-49%）、EP8 TP1（-50%）最显著；TP8×DP1 改善 27%。
2. **TPOT（稳态 decode）**：TP4×DP1 最低（5.80 ms），TP8×DP1 次之
   （6.25 ms）；TP1×DP4/EP8 最高（11-13 ms），符合单卡推理预期。
3. **TTFT（prefill）**：TP8×DP1 最优（173 ms）；TP4×DP2 明显偏高
   （317 ms），DP 协调 + 更细 TP 分片对 prefill 不利。
4. **e2e p50**：TP4×DP1 最低（620 ms）；8 卡矩阵中 TP8×DP1（628 ms）
   与 TP4×DP1 相当，TP2×DP4 与 EP 配置略高。
5. TP1×DP4 证明 W4A16 canonical 单卡容量通过（Runbook 前置条件），
   为 TP1×DP8 提供依据。

## 5. 边界与下一步

- 本矩阵为 32 请求 smoke 级正确性 + 时序；正式性能需更大并发与多轮。
- TP1×DP8（W4A16）仍受静态 group scale 对齐限制（group_size=128 与
  DP8 每分区 64 不整除），可用 block64 变体（`..._g64`）支持；TP8×FP8
  同理受 block 对齐限制，保留为格式不兼容证据。
- EP 准入顺序：EP4/单 NUMA static（完成）、EP8/static（完成，TP1×EP8）、
  EP8+原生 EPLB（EP4×TP2 验证 EPLB 可用；纯 EP8 无 TP/DP 维度被 vLLM
  拒绝）、EP8+冗余专家 1（待重测）、自研 topology-aware EPLB（待做）。
- 旧 P2P 禁用版四卡矩阵数据：`docs/archive/Q-TopoMoE_Phase6_matrix_4gpu_20260807.json`。
