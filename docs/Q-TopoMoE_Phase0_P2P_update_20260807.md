# Q-TopoMoE Phase 0 更新：P2P 启用后通信画像反转

> 生成日期：2026-08-07（Asia/Shanghai）
> 性质：**推翻** 2026-08-03 Phase 0 实测分析中的核心结论。
> 原因：该机已配置启用 GPU P2P；此前 P2P 被禁用导致所有跨卡链路退化为
> 共享主机内存路径，PIX 名义拓扑无法发挥，实测"PIX 最慢、NODE 最快"是
> 禁用 P2P 的假象。

## 1. 状态变化

- **此前**（2026-08-03）：`torch.cuda.can_device_access_peer` 任意两张
  不同 GPU 均 false；NCCL 走主机内存。
- **现在**（2026-08-07）：8×8 全部 peer 可用；NCCL 通道日志显示
  `via P2P/direct pointer`。

## 2. 双卡 AllReduce busbw（GB/s，5 次重复）

| 组合 | 1 MiB OLD→NEW | 64 MiB OLD→NEW | 256 MiB OLD→NEW |
|---|---:|---:|---:|
| PIX (0,1) | 11.7 → **27.2** | 15.4 → **43.3** | 15.5 → **45.0** |
| NODE (0,2) | 17.1 → 17.2 | 29.5 → 28.7 | 30.6 → 29.4 |
| SYS (0,4) | 16.5 → 16.7 | 28.6 → 28.1 | 29.4 → 28.8 |

**PIX 从最慢（15.5 GB/s）跃升为最快（45.0 GB/s，约 2.9x）**；NODE 与
SYS 基本不变。物理拓扑（PIX 同 switch 直连）现在与实测一致。

## 3. 通信代价库更新

新库：`configs/communication/nccl_cost_db_p2p.json`
（数据源 `artifacts/raw/20260807T120000Z_nccl_formal_p2p/nccl/statistics.json`）。

| mapping | OLD us/GB | NEW us/GB | 变化 |
|---|---:|---:|---|
| tp2_pix_0_1 | 65,200.6 | **23,458.4** | -64% |
| tp2_node_0_2 | 42,025.8 | 43,015.5 | +2% |
| tp4_numa0 | 46,824.4 | 28,969.5 | -38% |
| tp4_numa1 | 46,661.7 | 28,964.5 | -38% |
| tp4_sys_0_4 | 46,983.8 | 47,480.7 | +1% |
| tp8_sys | 54,982.3 | 36,669.7 | -33% |

旧库 `nccl_cost_db.json` 保留为禁用 P2P 时代的历史记录；新库启用后
Phase 8 cost model 应指向 `nccl_cost_db_p2p.json`。

## 4. 对既有结论的影响

1. **Phase 0（2026-08-03）**：结论 1-3 推翻——P2P 并非"不可用"，PIX 也
   不是最慢；TP2 首选应从 NODE 对改回 **PIX 对**（0,1 / 2,3 / 4,5 / 6,7）。
2. **Phase 1**：TP2 配置此前因单卡 32 GB 容量 OOM 未测，未产生基于
   PIX/NODE 的性能结论，无需推翻；但 Phase 1 矩阵中的拓扑优先级注解
   需要按新画像更新。
3. **Phase 6**：TP8×DP1 等跨卡配置的通信成本应使用新代价库重估；
   TP4-numa（28,969 us/GB）现在明显优于 tp4_sys（47,481 us/GB）。
4. **Phase 7**：离线 placement 的跨 NUMA 惩罚系数应基于新代价库复核；
   同 NUMA 内部 PIX 对现在是最优，EPLB 迁移收益计算需更新。

## 5. 边界

- 新矩阵覆盖双卡 all_reduce/sendrecv、四卡/八卡 all_gather/
  reduce_scatter/alltoall（75 个日志，全部 rc=0，5 次重复 + 95% CI）。
- 原始日志保留在 `artifacts/raw/20260807T120000Z_nccl_formal_p2p/`。
- 若后续 P2P 配置再次变化（如驱动/BIOS 恢复禁用），需重测并重新登记。
