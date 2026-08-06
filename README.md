# Q-TopoMoE

面向 8×RTX 5090 PCIe 多 GPU 系统的量化感知 MoE 推理并行、动态负载均衡与
SM120 算子研究工程。

## 当前进度（2026-08-06）

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 0 | 实机拓扑评估与实验矩阵 | 完成 |
| Phase 1 | BF16 / FP8 服务实测 | 完成 |
| Phase 2 | W4A16 量化、canonical checkpoint 与质量 gate | 完成（official-like v2） |
| Phase 3a | Route trace 全量采集（BF16 + W4A16-triton） | 完成（116 prompts × 全 token） |
| Phase 3b | 全量 trace 漂移分析（Jaccard / flip / 相关 / CV） | 完成 |
| Phase 3c | Full-set 官方协议评测（MMLU-Pro + C-Eval test） | 资产已冻结，计时 pilot 待跑 |
| Phase 4/8 | Kernel benchmark 与策略回放 | 框架就绪，待实测数据填充 |

结果与报告统一归档在 `docs/`，按阶段索引见 [docs/README.md](docs/README.md)。

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

## 目录结构

- `configs/`：模型、拓扑、workload、策略与评测配置（frozen 输入 + manifest）。
- `env/`：项目变量、环境检查和依赖锁。
- `topology/`：硬件采集与通信成本模型。
- `quantization/`：LLM Compressor、ModelOpt 和 checkpoint 审计。
- `traces/`：路由捕获（`capture_routes.py`）与采集 manifest。
- `analysis/`：route trace 漂移分析（`route_drift.py`）。
- `evaluation/`：评测输入冻结、官方资产抓取、full-set 冻结与对比。
- `clients/`：质量评测与 smoke 客户端。
- `selector/`、`phase4/`：SM120 内核/后端选择器与 M-bucket workload 生成。
- `serving/`：服务端启动与验收（与客户端严格分离）。
- `scripts/`：checkpoint gate、环境 bootstrap、矩阵执行与聚合。
- `tests/`：单元测试。
- `docs/`：阶段报告与结果（见 `docs/README.md` 索引）。

## 复现流程

1. **环境**：`env/activate.sh` + `env/requirements-lock/` + `scripts/bootstrap_qwen35_cleanroom.sh`。
2. **模型 gate**：`scripts/gate_qwen35_checkpoint.sh`（BF16 或 W4A16 canonical + vLLM/SGLang）。
3. **质量评测**：`clients/quality_eval.py --manifest <frozen jsonl> --summary <out.json>`，
   对比用 `evaluation/compare_quality.py`。
4. **Route trace 采集**：`traces/capture_routes.py`（需要 vLLM cleanroom
   `enable_return_routed_experts`），输出 npy 分片 + `traces.jsonl` +
   `capture_manifest.json` + `expert_token_histogram.json`。
5. **漂移分析**：`analysis/route_drift.py --reference-dir <bf16_dir> --candidate-dir <w4_dir>`。
6. **Full-set 冻结**：`evaluation/fetch_official_protocol_assets.py` →
   `evaluation/freeze_full_set.py`（大 JSONL 不入库，manifest 钉住 SHA-256）。

大体积产物（`*.jsonl`、`*.parquet`、`*.npz`、`*.log`、trace npy 分片）不提交
Git，均以 manifest（SHA-256 + 版本 + 采样参数）钉住，保证可复现。

## 安全边界

- 开始实验前检查 GPU PID、命令和端口。
- 不执行 `pkill python`、`killall` 或清理不属于本项目的进程。
- 当前主机 NCCL 默认 ERDMA/RoCE 路径已知不稳定，项目默认设置 `NCCL_IB_DISABLE=1`。
- checkpoint 路径与 API served name 分开记录。
