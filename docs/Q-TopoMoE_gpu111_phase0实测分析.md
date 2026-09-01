# 阶段 0：gpu-111 八卡 RTX 5090 拓扑与 P2P 实测

## 1. 权威基线

本文只记录 2026-08-07 P2P 配置生效后的正式结果。当前通信成本、并行映射和
后续 placement 必须使用以下两项：

- [Phase 0 验证脚本](q_topomoe_phase0_verify.sh)；
- [当前 NCCL 成本库](../configs/communication/nccl_cost_db.json)。

成本库的原始统计来自服务器目录
`artifacts/raw/20260807T120000Z_nccl_formal_p2p/nccl/statistics.json`。若驱动、
BIOS、内核参数、NCCL 或 P2P 状态发生变化，必须重新测量并生成新的成本库。

## 2. 系统条件

| 项目 | 实测结果 | 状态 |
|---|---|---|
| 服务器 | 裸金属，双路 Intel Xeon Platinum 8575C | 通过 |
| NUMA | 2 节点，GPU0–3 属于 NUMA0，GPU4–7 属于 NUMA1 | 通过 |
| GPU | 8× RTX 5090，约 32 GB/卡，计算能力 12.0 | 通过 |
| Driver / CUDA | 580.126.09 / CUDA Driver 13.0 / NVCC 13.0.88 | 通过 |
| PyTorch / NCCL | PyTorch 2.11.0+cu130 / NCCL 2.28.9 | 通过 |
| BF16 | 8 卡逐卡 BF16 GEMM 正确 | 通过 |
| PCIe | 8 卡负载时均为 Gen5 ×16 | 通过 |
| NVLink | 无 | 已确认 |
| CUDA P2P | 8×8 非对角 peer read/write 全部可用 | 通过 |
| NCCL 数据路径 | 日志显示 `P2P/direct pointer` | 通过 |

默认 ERDMA/RoCE 插件路径曾在当前软件组合中触发初始化故障。服务使用
`NCCL_IB_DISABLE=1`，必要时用 `NCCL_NET=Socket` 兜底；升级驱动、NCCL
或网络插件后需要重新验证。

## 3. 物理拓扑

```text
NUMA 0                                      NUMA 1
  PIX: GPU0—GPU1                              PIX: GPU4—GPU5
       \     /                                     \     /
        NODE                                        NODE
       /     \                                     /     \
  PIX: GPU2—GPU3                              PIX: GPU6—GPU7

NUMA0 与 NUMA1 之间：SYS
所有非对角 GPU 对：CUDA peer access 可用
```

代表映射：

- PIX：`0,1`；
- NODE：`0,2`；
- SYS：`0,4`；
- 单 NUMA 四卡：`0,1,2,3` 或 `4,5,6,7`；
- 八卡：`0,1,2,3,4,5,6,7`。

## 4. 正式 NCCL 成本

成本库覆盖双卡、四卡和八卡的 all-reduce、send/recv、all-gather、
reduce-scatter 与 all-to-all；每项五次重复，错误计数均为 0。大消息点的
描述性中位成本如下：

| 映射 | 中位成本（μs/GB） | 用途 |
|---|---:|---|
| TP2 PIX | 23,458.4 | TP2 首选通信映射 |
| TP2 NODE | 43,015.5 | 同 NUMA 跨分支对照 |
| TP4 NUMA0 | 28,969.5 | 四卡单 NUMA |
| TP4 NUMA1 | 28,964.5 | 四卡单 NUMA |
| TP4 SYS 0,4 | 47,480.7 | 跨 NUMA 压力项 |
| TP8 SYS | 36,669.7 | 八卡压力项 |

`effective_us_per_gb` 是对已测点的描述，不是可任意外推的线性带宽模型。选择器
应按 collective、消息大小和映射查询原始点；端到端服务仍以真实服务矩阵为准。

## 5. 部署与实验约束

1. TP2 优先使用 PIX 对 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)`。
2. TP4 优先放在单一 NUMA 域，TP8 和跨 NUMA映射作为压力项。
3. W4A16/NVFP4 若能单卡容纳，优先用 TP1 降低模型并行通信。
4. EP placement 同时考虑真实 route load、显存余量、迁移字节与当前成本库；
   不能只根据 `nvidia-smi topo -m` 的标签排序。
5. 所有线上计划必须再通过请求零失败、计划哈希一致、恢复 p99 和 rollback Gate。

## 6. Gate 状态

| Gate | 状态 | 证据 |
|---|---|---|
| GPU/NUMA 拓扑 | accepted | 八卡与双 NUMA 关系已确认 |
| CUDA P2P | accepted | 所有非对角 peer read/write 可用 |
| PCIe 负载链路 | accepted | 八卡 Gen5 ×16 |
| NCCL correctness | accepted | 正式矩阵错误数为 0 |
| 当前通信成本库 | accepted | 2026-08-07 P2P 正式矩阵 |
| 服务并行策略 | 分阶段准入 | 以 Phase 1、Phase 6 和 Phase 8 在线 Gate 为准 |

阶段 1 服务数据见[BF16/FP8 服务基线](results/phase1_service_baseline.md)，
Phase 6–7 多卡和迁移数据见[阶段 4–7 工程结果](results/phase4_to_phase7_engineering.md)。
