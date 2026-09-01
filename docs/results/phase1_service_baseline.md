# 阶段 1：BF16/FP8 服务基线

## 1. 数据范围

本文只汇总 P2P 生效后的真实服务矩阵：

- [BF16 P2P 矩阵](../Q-TopoMoE_Phase1_BF16_matrix_p2p_20260807.json)；
- [FP8 P2P 矩阵](../Q-TopoMoE_Phase1_FP8_matrix_p2p_20260807.json)。

两份文件均来自 gpu-111，使用 short（256/128）和 medium（2048/256）
workload，并发为 1、8、32。BF16 共 18 个单元、246 个请求；FP8 共 60 个单元、
820 个请求；失败数均为 0。

## 2. 检查点与服务入口

BF16 checkpoint：

```bash
QTOPOMOE_BF16_MODEL=/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master
```

该快照包含 26 个 `model-*.safetensors` 分片，总大小约 67 GB。
`config.json` SHA-256 为
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`，
`model.safetensors.index.json` SHA-256 为
`41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`。

统一执行链：

- `serving/start_server.sh`：按环境变量启动 SGLang/vLLM；
- `serving/acceptance.sh`：检查 health、模型发现、真实 completion 和 metrics；
- `clients/smoke.py`：固定 seed，记录 TTFT、TPOT、E2E 与失败数；
- `scripts/phase1_matrix.sh`：按拓扑运行矩阵并释放服务。

## 3. BF16 结果

BF16 在 TP4-NUMA0、TP4-NUMA1 和 TP8-SYS 上均通过。并发 32 的 p50 如下：

| 拓扑 | workload | TTFT（ms） | TPOT（ms） | E2E（ms） |
|---|---|---:|---:|---:|
| TP4-NUMA0 | short | 979.08 | 8.05 | 2,003.06 |
| TP4-NUMA1 | short | 1,029.89 | 8.05 | 2,040.97 |
| TP8-SYS | short | 1,158.35 | 8.05 | 2,165.64 |
| TP4-NUMA0 | medium | 1,530.99 | 9.22 | 3,411.05 |
| TP4-NUMA1 | medium | 1,510.28 | 8.99 | 3,312.18 |
| TP8-SYS | medium | 1,481.45 | 8.67 | 3,055.71 |

短请求优先保留 TP4-NUMA0；TP8-SYS 在 medium/c32 的 E2E 最低，可作为长输入
压力项。BF16 TP2 不进入正式矩阵，因为约 67 GB 权重加运行时开销超过两张
32 GB GPU 的容量，这属于容量约束，不是通信故障。

## 4. FP8 结果

FP8 的四组 PIX TP2、四组 NODE TP2 和两个单 NUMA TP4 均通过。并发 32 的
short 结果如下：

| 拓扑 | TTFT（ms） | TPOT（ms） | E2E（ms） |
|---|---:|---:|---:|
| TP2-NODE 0,2 | 1,209.92 | 5.72 | 1,937.37 |
| TP2-NODE 1,3 | 1,264.40 | 5.08 | 1,920.30 |
| TP2-NODE 4,6 | 1,324.86 | 5.08 | 1,967.86 |
| TP2-NODE 5,7 | 1,175.56 | 5.04 | 1,818.24 |
| TP2-PIX 0,1 | 1,107.11 | 5.36 | 1,790.92 |
| TP2-PIX 2,3 | 1,085.24 | 5.75 | 1,819.42 |
| TP2-PIX 4,5 | 1,031.81 | 5.25 | 1,702.86 |
| TP2-PIX 6,7 | 1,046.73 | 5.29 | 1,720.95 |
| TP4-NUMA0 | 1,030.08 | 7.89 | 2,034.90 |
| TP4-NUMA1 | 1,008.07 | 7.82 | 2,006.57 |

short/c32 下 PIX TP2 整体更优，GPU4,5 的 E2E 最低。medium/c32 下不同映射
受 prefill、排队和服务调度共同影响，NODE TP2 的 E2E 为 2.69–2.79 秒，PIX TP2
为 2.90–3.02 秒；因此部署不能只按通信微基准选择，必须结合目标 workload 的
端到端数据。

## 5. 当前结论

1. P2P 环境下 BF16/FP8 的正式服务矩阵均为零请求失败。
2. BF16 以 TP4 单 NUMA 为常规候选，TP8 保留为长输入与跨 NUMA 压力项。
3. FP8 的低 TPOT 候选是 TP2；short workload 优先 PIX 对，medium workload
   需要按真实服务 E2E 复核。
4. 通信成本库用于筛选候选，端到端矩阵用于最终排序，两者不能互相替代。
5. 本阶段是服务 smoke 和基线，不替代 Phase 8 多轮、独立集、canary 与回滚验收。

## 6. 复现入口

```bash
source env/project.env

RUN_ID=<RUN_ID> FORMAT=bf16 +  TOPOLOGIES=tp4_numa0,tp4_numa1,tp8_sys +  scripts/phase1_matrix.sh

RUN_ID=<RUN_ID> FORMAT=fp8 +  TOPOLOGIES=tp2_node_0_2,tp2_node_1_3,tp2_node_4_6,tp2_node_5_7,tp2_pix_0_1,tp2_pix_2_3,tp2_pix_4_5,tp2_pix_6_7,tp4_numa0,tp4_numa1 +  scripts/phase1_matrix.sh
```

运行前必须先用[阶段 0 验证脚本](../q_topomoe_phase0_verify.sh)确认 P2P、NCCL
和 NUMA 状态与正式基线一致。
