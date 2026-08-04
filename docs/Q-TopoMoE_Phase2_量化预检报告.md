# Q-TopoMoE Phase 2 量化预检报告

日期：2026-08-04  
阶段：6.1 量化环境冻结、6.2 W4A16 入口准备  
主机：gpu-111（8 × RTX 5090）

## 结论

6.1 已完成并可复现。独立环境位于 `/data/models/test/qtopomoe_quant_env`，其 CUDA 运行时复用了已验收的服务环境 torch，但量化工具包独立安装并锁定。正式 6.2 目前被校准数据门禁阻塞：项目只有 64 条 smoke seed，没有经数据集版本、样本索引、tokenizer revision、chat template 和 SHA-256 冻结的 256 条校准集，因此没有启动 35B PTQ。

这不是失败的量化结果，而是为了保持可复现性而保留的输入门禁。禁止把 64 条 smoke 数据复制填充到 256 条。

## 6.1 实测环境

执行：

```bash
cd /home/k8s-ops/moe
bash scripts/create_quant_env.sh
```

关键输出：

| 项目 | 实测值 |
|---|---|
| Python | 3.12.3 |
| torch | 2.11.0+cu130 |
| Transformers | 5.10.1 |
| accelerate | 1.13.0 |
| datasets | 5.0.0 |
| safetensors | 0.8.0 |
| llmcompressor | 0.12.0.1 |
| compressed-tensors | 0.17.1 |
| auto-round | 0.13.0 |
| CUDA 可用 / 设备数 | True / 8 |
| pip check | 量化栈通过；仅报告共享 SGLang 要求 Transformers 5.12.1 的边界冲突 |

锁文件：`env/requirements-lock/quant.txt`。共享服务环境中的 SGLang 不属于量化环境，不应为了量化而修改。

## 6.2 预检实测

模型快照：

`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`

预检命令：

```bash
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quant_preflight.py \
  --model /home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master \
  --output artifacts/reports/phase2_quant_preflight_formal_20260804.json
```

已验证：`model_type=qwen3_5_moe`、`Qwen3_5MoeForConditionalGeneration`、40 层、256 experts、8 卡可见、配置 SHA-256 与 Phase 1 manifest 一致，以及约 248 GB 模型盘可用空间。

正式预检返回 `BLOCKED`，唯一门禁是 `QTOPOMOE_CALIBRATION_JSONL is not set`。环境-only 预检返回 `PASS`，用于证明工具链本身可用。

另外使用现有 64 条 smoke seed 对正式入口做了守门测试，入口在加载 35B 权重前返回：

```text
refusing to quantize: calibration has 64 records, requires 256
```

因此当前没有产生任何 W4A16 checkpoint，也没有把 smoke 数据重复扩充为校准集。

## 已准备的正式入口

`quantization/quantize_w4a16.py` 已提交，使用 llmcompressor 0.12.0.1 的：

- `load_quantizable_moe(AutoModelForCausalLM)`，将 fused 3-D experts 线性化；
- `GPTQModifier(targets="Linear", scheme="W4A16")`；
- `moe_calibrate_all_experts=True`；
- 排除 `linear_attn`、router/shared-expert gate、embedding、`lm_head`；
- 256 样本、4096 token、`compressed-tensors` 输出和量化 manifest。

拿到冻结校准集后，先执行正式预检，再执行：

```bash
export QTOPOMOE_CALIBRATION_JSONL=/path/to/pinned_calibration.jsonl
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quantize_w4a16.py \
  --model /home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master \
  --calibration "$QTOPOMOE_CALIBRATION_JSONL" \
  --output /data/models/test/qtopomoe_w4a16 \
  --samples 256 --max-seq-length 4096
```

量化完成后仍需按 Runbook 6.4/6.5 做 safetensors 静态覆盖审计、TP1 容量门禁、SGLang/VLLM 实际加载、`/health`、`/v1/models`、真实 completion、`/metrics`，以及并发 1/32 的对照基线；当前均未宣称完成。
