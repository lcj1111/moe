# Q-TopoMoE

面向 8×RTX 5090 PCIe 多 GPU 系统的量化感知 MoE 推理并行、动态负载均衡与 SM120 算子研究工程。

## 快速开始

```bash
cd /home/k8s-ops/moe
source env/activate.sh
make check
qtopomoe_gpu_status
qtopomoe_new_run fp8_tp2_node
```

选择框架环境：

```bash
qtopomoe_use_sglang
# 或新开 shell 后：
qtopomoe_use_vllm
```

`env/activate.sh` 只设置项目、CUDA 和 NCCL 环境，不会自动占用 GPU 或启动服务。

## 目录约定

- `configs/`：模型、拓扑、workload 和实验配置。
- `env/`：项目变量、环境检查和依赖锁。
- `topology/`：硬件采集与通信成本模型。
- `quantization/`：LLM Compressor、ModelOpt 和 checkpoint 审计。
- `traces/`：路由捕获与重放。
- `kernels/`：CUTLASS、Triton、selector 和测试。
- `serving/`、`clients/`：服务端和客户端严格分离。
- `controller/`：placement、EPLB 和联合策略选择器。
- `experiments/`、`analysis/`：矩阵执行、聚合和绘图。
- `artifacts/`：原始结果只增不改；大文件不提交 Git。
- `third_party/`：外部仓库，必须锁定完整 commit SHA。

## 安全边界

- 开始实验前检查 GPU PID、命令和端口。
- 不执行 `pkill python`、`killall` 或清理不属于本项目的进程。
- 当前主机 NCCL 默认 ERDMA/RoCE 路径已知不稳定，项目默认设置 `NCCL_IB_DISABLE=1`。
- checkpoint 路径与 API served name 分开记录。
