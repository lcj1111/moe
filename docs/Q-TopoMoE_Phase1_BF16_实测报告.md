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
