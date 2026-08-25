# 阶段 0：gpu-111 八卡 RTX 5090 拓扑与 P2P 实测

> **当前结论以 2026-08-07 P2P 启用后的重测为准。** 本文前半部保留
> 2026-08-03 的 P2P 禁用基线，用于说明通信路径变化前后的因果关系；其中
> “P2P 不可用、PIX 最慢、TP2 优先 NODE”已被后续实测推翻，不可再作为
> 当前部署建议。

## 文档范围

本文已经吸收首轮拓扑评估、2026-08-03 实测、P2P 启用记录和 2026-08-07 重测。
这些拆分稿不再作为仓库入口；需要追溯编辑过程时查看 Git 历史，当前结论和复现路径以本文
及 [Phase 0 验证脚本](q_topomoe_phase0_verify.sh) 为准。

> 执行日期：2026-08-03
> 原始结果目录：`/home/k8s-ops/artifacts/q_topomoe_phase0`
> 结果性质：关键双卡 AllReduce 与 EP4/EP8 All-to-All 已完成 5 次独立重复；每次 20 次 warmup、100 次正式迭代，报告 95% t 置信区间。8 卡 rank 排序仍为单次机制筛选。

## 1. 2026-08-03 历史结论摘要（P2P 禁用基线）

这台服务器适合研究“无 NVLink、无 CUDA P2P 的 PCIe 多 GPU MoE 推理”，而且实测揭示了一个不能仅凭 `nvidia-smi topo` 推断的关键现象：

1. 所有 GPU 间直接 CUDA P2P 均不可用，GPU 间通信由 NCCL 通过共享主机内存等路径完成。
2. 标记为 `PIX` 的相邻双卡并不是最快组合。对于 64–256 MiB 的双卡 AllReduce，`NODE` 组合约为 `PIX` 的 1.9–2.0 倍；`SYS` 组合也明显快于 `PIX`。
3. 因此 TP2 的首选应由原先的 PIX 对改为同一 NUMA、跨两个 PIX 分支的 NODE 对：`(0,2)`、`(1,3)`、`(4,6)`、`(5,7)`。
4. EP4 应优先限制在单 NUMA 内；EP8 在 4–64 KiB 小消息下延迟约为 EP4 的 2.3–2.6 倍，在 64 MiB 时吞吐低约 20.0%。
5. 默认 NCCL 会选择 ERDMA/RoCE 并触发 SIGSEGV。单独设置 `NCCL_IB_DISABLE=1` 或单独设置 `NCCL_NET=Socket` 的最小 smoke 均通过。当前建议优先使用 `NCCL_IB_DISABLE=1`，并把 `NCCL_NET=Socket` 作为显式兜底。
6. 改变 `CUDA_VISIBLE_DEVICES` 的 8 卡顺序不能强制改变 NCCL 的物理 Ring；NCCL 会根据物理拓扑重新排序。逻辑排序仍可用于定义 TP/EP 子组成员，但不应被当作全局 Ring 优化手段。

课题由此应聚焦为：**量化降低模型并行度 + NUMA 域内专家放置/副本 + 面向 SM120 的本地计算与打包算子优化**。在本机上，“避免通信”比“假设存在 GPU 直连后优化通信”更有研究价值。

## 2. 已确认的系统条件

| 项目 | 实测结果 | 状态 |
|---|---|---|
| 服务器形态 | `systemd-detect-virt: none`，裸金属 | 通过 |
| CPU | 2× Intel Xeon Platinum 8575C；48 核/96 线程每路 | 通过 |
| NUMA | 2 节点；距离 10/21 | 通过 |
| GPU | 8× RTX 5090，32607 MiB/卡 | 通过 |
| GPU–NUMA | GPU0–3 属于 NUMA0；GPU4–7 属于 NUMA1 | 通过 |
| Driver / CUDA | 580.126.09 / CUDA Driver 13.0 / NVCC 13.0.88 | 通过 |
| PyTorch | 2.11.0+cu130；CUDA 13.0；NCCL 2.28.9 | 通过 |
| 架构 | 计算能力 12.0；PyTorch arch list 包含 `sm_120` | 通过 |
| BF16 | 8 卡逐卡 BF16 GEMM 正确 | 通过 |
| PCIe | 8 卡负载时均升至 Gen5 ×16 | 通过 |
| NVLink | 无 | 已确认 |
| CUDA P2P | 任意两张不同 GPU 均为 false | 已确认不可用 |
| NCCL 默认路径 | 初始化 ERDMA/RoCE 后 SIGSEGV | 阻塞，已有规避方案 |
| NCCL Socket/SHM | 8 卡 AllReduce 与 All-to-All 正确 | 条件通过 |

内核启动参数包含 `iommu=pt` 与 `pci=disable_acs_redir=pci:0:0`。因此，P2P 不可用不是因为尚未尝试关闭 IOMMU；不要为了实验继续修改 ACS/IOMMU 安全设置。

## 3. 实际拓扑

```text
NUMA 0                                      NUMA 1
  PIX: GPU0—GPU1                              PIX: GPU4—GPU5
       \     /                                     \     /
        NODE                                        NODE
       /     \                                     /     \
  PIX: GPU2—GPU3                              PIX: GPU6—GPU7

NUMA0 与 NUMA1 之间：SYS，经 CPU/UPI 与主机内存路径
所有非对角 GPU 对：无 CUDA peer access；无 NVLink
```

代表组合：

- PIX：`0,1`
- NODE：`0,2`
- SYS：`0,4`
- NUMA0：`0,1,2,3`
- NUMA1：`4,5,6,7`

## 4. 双卡通信实测

### 4.1 256 MiB，NCCL 五次独立重复

| 路径 | GPU | AllReduce 时间（均值±95% CI） | AllReduce algbw（均值±95% CI） | SendRecv 筛选值 | 相对判断 |
|---|---|---:|---:|---:|---|
| PIX | 0,1 | 17.333±0.073 ms | 15.488±0.069 GB/s | 17.79 GB/s | 最慢；共享分支疑似形成争用 |
| NODE | 0,2 | 8.655±0.066 ms | 31.016±0.233 GB/s | 24.01 GB/s | 最快；TP2 首选 |
| SYS | 0,4 | 9.131±0.031 ms | 29.396±0.106 GB/s | 21.48 GB/s | 大消息快于 PIX，但依赖跨 NUMA |

AllReduce 在 1 MiB、64 MiB、256 MiB 上的 algbw（均值±95% CI）：

| 消息大小 | PIX 0,1 | NODE 0,2 | SYS 0,4 |
|---:|---:|---:|---:|
| 1 MiB | 11.680±0.203 GB/s | 17.134±0.017 GB/s | 16.518±0.102 GB/s |
| 64 MiB | 15.414±0.064 GB/s | 29.864±0.177 GB/s | 28.622±0.101 GB/s |
| 256 MiB | 15.488±0.069 GB/s | 31.016±0.233 GB/s | 29.396±0.106 GB/s |

4 KiB 时三类双卡路径的 AllReduce 延迟都约 10–11 μs，差异很小；从 64 KiB 开始 NODE/SYS 的优势变得稳定。因此，控制器不能只使用静态顺序 `PIX < NODE < SYS`，至少应按消息区间使用实测代价表。

CUDA sample 的 fallback 拷贝结果与 NCCL 趋势一致：PIX 双向约 44 GB/s，NODE/SYS 多数约 55–57 GB/s。CUDA sample 自身声明不是正式性能基准，所以策略判断以 NCCL 数据为主。

## 5. EP4 与 EP8 的 All-to-All

所有测试均使用正确 NUMA 绑定；8 卡使用 `numactl --interleave=0,1`。

| 消息大小 | EP4 / NUMA0 延迟 | EP4 / NUMA1 延迟 | EP8 延迟 | EP4 algbw | EP8 algbw |
|---:|---:|---:|---:|---:|---:|
| 4 KiB | 21.484±0.088 μs | 21.684±0.174 μs | 55.452±0.349 μs | 0.190 GB/s | 0.070 GB/s |
| 64 KiB | 24.332±0.212 μs | 24.180±0.067 μs | 54.944±0.389 μs | 2.692/2.710 GB/s | 1.190±0.009 GB/s |
| 1 MiB | 72.430±0.077 μs | 72.398±0.100 μs | 84.504±0.254 μs | 14.478/14.484 GB/s | 12.408±0.033 GB/s |
| 64 MiB | 2.945±0.038 ms | 2.947±0.075 ms | 3.682±0.089 ms | 22.786/22.780 GB/s | 18.234±0.444 GB/s |

这说明 EP8 的主要风险并非大块带宽完全失效，而是 MoE dispatch 常见的小消息/稀疏 token 场景下固定延迟被显著放大。后续应按真实每专家 token 数构造 4 KiB–4 MiB 的 dispatch 分布，不能只用 64–256 MiB 集合通信推断 MoE 性能。

## 6. 8 卡逻辑排序实验

固定 `NCCL_ALGO=Ring`，测试三种可见顺序：

| 标签 | `CUDA_VISIBLE_DEVICES` | 256 MiB AllReduce algbw | AllGather algbw | ReduceScatter algbw | All-to-All algbw |
|---|---|---:|---:|---:|---:|
| default | `0,1,2,3,4,5,6,7` | 11.00 GB/s | 21.23 GB/s | 21.23 GB/s | 20.23 GB/s |
| node_ring | `0,2,1,3,4,6,5,7` | 10.96 GB/s | 21.22 GB/s | 21.30 GB/s | 19.89 GB/s |
| sys_ring | `0,4,1,5,2,6,3,7` | 11.01 GB/s | 21.19 GB/s | 21.25 GB/s | 19.98 GB/s |

差异处于单次运行噪声范围。NCCL 日志显示：

- 默认逻辑序列的物理 Ring 为 `0,1,2,3,4,5,6,7`；
- NODE 逻辑序列下，NCCL 通道 rank 为 `0,2,1,3,4,6,5,7`，映射回物理 GPU 后仍是 `0,1,2,3,4,5,6,7`；
- SYS 逻辑序列也被重构回同一物理序列。

因此：用 `CUDA_VISIBLE_DEVICES=0,2` 建立一个 TP2 子服务是有效的；但仅把 8 卡进程的可见顺序改成 `0,2,1,3,...`，不会强制 NCCL 使用该物理 Ring。

## 7. NCCL 故障与稳定运行配置

默认 8 卡 PyTorch collective 在 NCCL 初始化 ERDMA `erdma_0:uverbs0:1/RoCE` 后所有 rank 收到 SIGSEGV。以下两种最小配置分别 smoke 通过：

```bash
# 推荐：禁用当前有问题的 IB/RoCE 插件，NCCL 自动回退到 Socket
export NCCL_IB_DISABLE=1

# 或显式强制 Socket；可作为启动脚本兜底
export NCCL_NET=Socket
```

正式服务建议先采用：

```bash
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

若仍看到 ERDMA 或网络插件初始化，再追加 `NCCL_NET=Socket`。这不是性能调优选项，而是当前驱动/NCCL/ERDMA 组合的稳定性条件。后续若升级驱动、NCCL 或 ERDMA 驱动，必须重新测试默认路径，不能永久假设故障仍存在。

## 8. 对量化、并行与 CUDA 优化方案的修订

### 8.1 推荐实验配置

| 精度/目标 | 首选部署 | 对照 | 原因 |
|---|---|---|---|
| BF16 | 两个 TP4：`0–3`、`4–7` | TP8 | 单 NUMA 内运行，避免跨 NUMA 模型并行 |
| FP8 | 四个 TP2：`0,2`、`1,3`、`4,6`、`5,7` | PIX TP2 与 TP4 | NODE 对实测约为 PIX 的两倍大消息 AllReduce 带宽 |
| W4A16 | TP1×8 | NODE TP2×4 | 如果单卡可容纳，彻底消除模型并行通信 |
| NVFP4/W4A4 | TP1 优先，NODE TP2 兜底 | FP8 TP2 | 验证 SM120 原生低比特路径及量化收益 |
| MoE EP | EP4/NUMA 优先 | EP8、EP8+EPLB | 先建立局部域基线，再评估跨 NUMA 负载均衡收益 |

PIX TP2 `0,1`、`2,3`、`4,5`、`6,7` 应保留为负面对照，而非最优配置。

### 8.2 动态负载均衡设计

把机器视为两个 NUMA 域，每域四张 GPU：

1. 专家初始放置优先局限在本 NUMA 域。
2. 高频专家在两个 NUMA 域各保留副本，优先本地路由。
3. 迁移/复制收益函数至少包含：预计减少的 SYS token 字节、复制显存、迁移时间、热点持续窗口和副本一致性成本。
4. 使用滞回阈值和最短驻留时间，防止专家在两个 NUMA 域之间振荡。
5. 分别报告负载方差、跨 NUMA token 比例、每层 dispatch 字节、All-to-All 时间和端到端 TPOT；不能只报告吞吐。

### 8.3 CUDA 算子优化优先级

本机没有 CUDA P2P，因此不应把“自研 GPU 直连通信 kernel”列为第一目标。优先级应为：

1. SM120 FP8/NVFP4 grouped GEMM，以及按专家 token 数动态选择 tile/kernel。
2. routing + prefix-sum + token packing 融合，减少小 kernel 和全局内存往返。
3. 量化/反量化与 grouped GEMM epilogue 融合。
4. 本 NUMA pinned host buffer、双缓冲和计算/SHM 通信重叠。
5. 仅在 profiler 证明 dispatch/pack 占比足够高后，再开发专用融合 kernel。

## 9. 当前 Gate 状态

| Gate | 状态 | 证据/后续 |
|---|---|---|
| G0 拓扑与 NUMA | 通过 | PIX/NODE/SYS、双 NUMA、裸金属均已确认 |
| G0 CUDA P2P | 通过（结论为不可用） | PyTorch peer API 与 CUDA sample 一致 |
| G0 PCIe 负载链路 | 通过 | 8 卡负载时 Gen5 ×16 |
| G0 可利用通信差异 | 通过 | NODE/PIX 在 64–256 MiB 上差异约 93–102% |
| G1 PyTorch/SM120/BF16 | 通过 | torch 2.11.0+cu130；8 卡 BF16 GEMM |
| G1 NCCL correctness | 条件通过 | 禁用 IB 或强制 Socket 后 2/4/8 卡均无错误 |
| G1 正式通信统计 | 通过 | 关键配置完成 5 次独立重复；每次 20 warmup + 100 iterations；总错误数 0 |
| G1 模型服务 | 未执行 | 需确定模型 checkpoint、服务框架和 API alias |
| G2 FP8/W4A16/NVFP4 | 未执行 | 按 TP1/TP2/TP4 容量门逐级推进 |
| G3 自研 kernel | 未执行 | 先用真实模型 profiler 确定热点与理论上限 |

## 10. 下一阶段最短执行顺序

1. 选定一个可复现 MoE checkpoint，先做 BF16 TP4/TP8 的真实加载、completion、显存峰值与 profiler。
2. 依次验证 FP8 TP2-NODE、W4A16 TP1、NVFP4 TP1/TP2 的容量与正确性。
3. 对通过容量门的配置运行固定 workload，报告 TTFT、TPOT/ITL、吞吐、队列时间、HBM 峰值、功耗和 NCCL 时间占比。
4. 采集每层专家 token 直方图，离线回放 EP4、EP8、NUMA-aware placement、热点专家副本和 EPLB。
5. 用 Nsight Systems/Compute 确定 grouped GEMM、routing/packing、量化 epilogue 或 host staging 中的首要热点，再进入 CUDA 实现。

## 11. 原始证据位置

- PyTorch peer/BF16：`runtime/gpu-peer-bf16.json`
- 默认 NCCL 崩溃：`runtime/nccl-default.log`
- Socket/SHM 正确性：`runtime/nccl-no-ib-socket.log`
- CUDA P2P sample：`p2p/p2pBandwidthLatencyTest.log`
- NCCL 主矩阵：`nccl/matrix/`、`nccl/matrix-summary.tsv`
- NUMA/小消息矩阵：`nccl/numa-small/`、`nccl/numa-small-summary.tsv`
- Rank 排序矩阵：`nccl/rank-orders/`、`nccl/rank-orders-summary.tsv`
- 5 次正式重复：`nccl/formal5/`、`nccl/formal5/formal5-statistics.csv`、`nccl/formal5/formal5-statistics.json`
- 最小 NCCL 环境隔离：`nccl/ib_disable_only.log`、`nccl/socket_only.log`
- PCIe/功耗监控：`nccl/pcie-power-monitor.log`

以上路径均相对于服务器目录 `/home/k8s-ops/artifacts/q_topomoe_phase0`。

---

## 12. 2026-08-07 P2P 启用后的结论反转

P2P 配置生效后，8×8 peer access 全部可用，NCCL 日志显示
`via P2P/direct pointer`。双卡 AllReduce 五次重复结果如下：

| 组合 | 1 MiB 旧→新 | 64 MiB 旧→新 | 256 MiB 旧→新 |
|---|---:|---:|---:|
| PIX (0,1) | 11.7→27.2 GB/s | 15.4→43.3 GB/s | 15.5→45.0 GB/s |
| NODE (0,2) | 17.1→17.2 GB/s | 29.5→28.7 GB/s | 30.6→29.4 GB/s |
| SYS (0,4) | 16.5→16.7 GB/s | 28.6→28.1 GB/s | 29.4→28.8 GB/s |

PIX 在 256 MiB 上由最慢变为最快，约提升 2.9 倍。当前 TP2 首选恢复为
PIX 对 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)`；TP4 优先单 NUMA，跨 NUMA
继续作为压力项。通信代价应使用
[当前 NCCL 成本库](../configs/communication/nccl_cost_db.json)。服务器原始矩阵目录为
`artifacts/raw/20260807T120000Z_nccl_formal_p2p/`，它不随 Git 仓库分发；若驱动、BIOS
或 P2P 状态变化，必须重测，不能沿用本结论。

新旧 `effective_us_per_gb`：TP2 PIX 65,200.6→23,458.4（-64%），TP2 NODE
42,025.8→43,015.5（+2%），TP4 NUMA 约 46,824→28,969（-38%），TP4 SYS
46,983.8→47,480.7（+1%），TP8 SYS 54,982.3→36,669.7（-33%）。矩阵覆盖
双卡 all_reduce/sendrecv、四卡和八卡 all_gather、reduce_scatter、alltoall，
共 75 个日志，全部 `rc=0`，每项五次重复并报告 95% 置信区间。

## 13. P2P 生效后的全阶段重测摘要

- Phase 1 SGLang FP8 short/c32：TP2 PIX 的 TTFT/TPOT/e2e 为
  1107 ms/5.36 ms/1791 ms，优于 TP2 NODE 的 1210/5.72/1937；TP4 NUMA0
  e2e 由 2310 降至 2035 ms。BF16 TP2 仍因容量 OOM，TP4/TP8 通过。
- Phase 6 vLLM：BF16 TP8×DP1、TP4×DP2、TP2×DP4 的 e2e 分别为
  627.6/785.4/1041.6 ms；W4 EP8 为 1024.6 ms，EP4×TP2+EPLB 为
  1221.5 ms，重复 run2 为 1200.7 ms。`256+1=257` 不能被 EP2/4/8 整除，
  单冗余专家变体属于框架格式限制。
- Phase 3c BF16 TP4 pilot：official 协议 p50/mean 为 2066/2001 ms，约比
  旧值快 15 倍；thinking pilot 为 6839/7449 ms，约快 16 倍。
- Phase 7 单专家 4 MiB 迁移：同 NUMA 47.71 μs、跨 NUMA 73.45 μs、
  同 GPU 10.14 μs。
- Phase 8 修复 cost DB 未接入问题后，四条 observation 均从
  `fp8_tp2_node02` 反转为 `fp8_tp2_pix01_triton`。

因此，所有 P2P 启用前的多卡延迟、吞吐与耗时只保留作历史对照；质量分数、
route trace/漂移和单卡 kernel 延迟不受通信路径变化影响，仍可使用。
