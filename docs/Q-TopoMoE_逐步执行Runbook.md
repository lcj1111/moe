# Q-TopoMoE 逐步执行 Runbook（平台路径无关版）

> 课题：面向 PCIe 多 GPU 系统的量化感知 MoE 推理并行策略选择、动态负载均衡与 SM120 算子协同优化
> 目标平台：单机 8×RTX 5090，双 NUMA，PCIe 互连
> 原则：所有绝对路径、端口和模型位置由执行者填写；每一步都有输入、操作、产物、通过条件和失败处理。

## 1. 首先填写项目参数

不要直接修改后续命令。在一个独立环境文件中填写所有路径：

```bash
export PROJECT_ROOT=<PROJECT_ROOT>
export ARTIFACT_ROOT=<ARTIFACT_ROOT>
export MODEL_ROOT=<MODEL_ROOT>

export BF16_MODEL=<BF16_CHECKPOINT>
export FP8_MODEL=<FP8_CHECKPOINT>
export W4A16_MODEL=<W4A16_OUTPUT_CHECKPOINT>
export NVFP4_MODEL=<NVFP4_CHECKPOINT>

export SERVE_ENV=<SERVING_VENV>
export QUANT_ENV=<QUANTIZATION_VENV>
export KERNEL_ENV=<KERNEL_VENV>

export SERVED_NAME=<API_MODEL_ALIAS>
export PORT=<FREE_PORT>
```

变量含义：

- `*_MODEL` 是包含 `config.json` 的 checkpoint 目录，不是 Hugging Face cache 根目录。
- `SERVED_NAME` 是 API 请求中使用的模型别名，不是 checkpoint 路径。
- `ARTIFACT_ROOT` 只保存实验结果；源代码放在 `PROJECT_ROOT`。

检查变量：

```bash
for v in PROJECT_ROOT ARTIFACT_ROOT MODEL_ROOT SERVE_ENV; do
  eval "value=\${$v:-}"
  test -n "$value" || { echo "missing $v"; exit 1; }
done
```

## 2. 创建标准仓库结构

```bash
install -d \
  "$PROJECT_ROOT"/{configs/{models,workloads,experiments},env/{containers,requirements-lock},topology,quantization/{llm_compressor,modelopt},traces,kernels/{cutlass_sm120,triton_tuner,selector,tests},serving,clients,controller,experiments,analysis,third_party} \
  "$ARTIFACT_ROOT"/{hardware,manifests,checkpoints,raw,logs,profiles,reports,figures,failures}
```

初始化 Git：

```bash
cd "$PROJECT_ROOT"
git init
git add .
git commit --allow-empty -m "chore: initialize q-topomoe"
```

所有 raw 数据只增不改。每次运行生成唯一目录：

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_<CONFIG_ID>"
export RUN_DIR="$ARTIFACT_ROOT/raw/$RUN_ID"
install -d "$RUN_DIR"/{server,client,metrics,env,profiles}
```

## 3. 阶段 0：硬件与通信画像

如果已有同一硬件、驱动、内核和 NCCL 版本下的正式结果，可以直接登记并跳过；任一版本变化则重跑。

### 步骤 3.1：检查占用

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
  --format=csv
ss -ltnp
```

目标 GPU 有未知任务时停止。禁止 `pkill python`、`killall` 或终止不属于本项目的 PID。

### 步骤 3.2：采集硬件

```bash
export HW_DIR="$ARTIFACT_ROOT/hardware/$(date -u +%Y%m%dT%H%M%SZ)"
install -d "$HW_DIR"

nvidia-smi -L | tee "$HW_DIR/nvidia-smi-L.txt"
nvidia-smi -q | tee "$HW_DIR/nvidia-smi-q.txt"
nvidia-smi topo -m | tee "$HW_DIR/topo-m.txt"
nvidia-smi topo -p2p r | tee "$HW_DIR/topo-p2p-read.txt"
nvidia-smi topo -p2p w | tee "$HW_DIR/topo-p2p-write.txt"
lspci -tv | tee "$HW_DIR/lspci-tree.txt"
numactl --hardware | tee "$HW_DIR/numa.txt"
lscpu -e | tee "$HW_DIR/lscpu-e.txt"
uname -a | tee "$HW_DIR/uname.txt"
cat /proc/cmdline | tee "$HW_DIR/kernel-cmdline.txt"
nvcc --version | tee "$HW_DIR/nvcc.txt"
```

### 步骤 3.3：CUDA peer/BF16 检查

```bash
"$SERVE_ENV/bin/python" - <<'PY' | tee "$HW_DIR/torch-gpu.json"
import json, torch
r = {"torch": torch.__version__, "cuda": torch.version.cuda,
     "devices": [], "peer": []}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    x = torch.randn((1024,1024), device=i, dtype=torch.bfloat16)
    y = x @ x
    r["devices"].append({"id": i, "name": p.name,
        "capability": torch.cuda.get_device_capability(i),
        "memory": p.total_memory, "bf16_finite": bool(y.isfinite().all())})
for i in range(torch.cuda.device_count()):
    r["peer"].append([True if i == j else
        bool(torch.cuda.can_device_access_peer(i,j))
        for j in range(torch.cuda.device_count())])
print(json.dumps(r, indent=2))
PY
```

### 步骤 3.4：NCCL 正式矩阵

至少覆盖：PIX 代表对、同 NUMA 跨交换分支的 NODE 对、跨 NUMA SYS 对、单 NUMA 四卡、全八卡。

```bash
export NCCL_TESTS=<NCCL_TESTS_BUILD_DIR>
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 将 GPU_PAIR 分别替换为 PIX、NODE、SYS 代表对。
CUDA_VISIBLE_DEVICES=<GPU_PAIR> \
  "$NCCL_TESTS/all_reduce_perf" \
  -b 4K -e 256M -f 4 -g 2 -w 20 -n 100 -c 1

CUDA_VISIBLE_DEVICES=<NUMA_LOCAL_4_GPUS> \
  numactl --cpunodebind=<NODE> --membind=<NODE> \
  "$NCCL_TESTS/alltoall_perf" \
  -b 4K -e 64M -f 4 -g 4 -w 20 -n 100 -c 1

CUDA_VISIBLE_DEVICES=<ALL_8_GPUS> \
  numactl --interleave=0,1 \
  "$NCCL_TESTS/alltoall_perf" \
  -b 4K -e 64M -f 4 -g 8 -w 20 -n 100 -c 1
```

每项独立运行 5 次，计算均值、样本标准差和 95% t 置信区间。

Gate G0：8 卡 BF16 正确；peer 能力已明确；负载时 PCIe 达到预期代际/宽度；collective `wrong=0`；存在可复现的路径差异。若静态 PIX/NODE/SYS 顺序与实测不一致，以实测代价表为准。

## 4. 冻结软件、模型和数据

### 步骤 4.1：冻结服务环境

```bash
export ENV_SNAPSHOT="$ARTIFACT_ROOT/manifests/env_$(date -u +%Y%m%dT%H%M%SZ)"
install -d "$ENV_SNAPSHOT"

"$SERVE_ENV/bin/python" -VV > "$ENV_SNAPSHOT/python.txt"
"$SERVE_ENV/bin/python" -m pip freeze --all > "$ENV_SNAPSHOT/pip-freeze.txt"
"$SERVE_ENV/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())' \
  > "$ENV_SNAPSHOT/torch-cuda-nccl.txt"

<SERVING_CLI> --help > "$ENV_SNAPSHOT/serving-help.txt"
```

先用现有环境建立基线，不在基线环境中直接升级。量化和 kernel 使用独立 venv。

### 步骤 4.2：冻结模型

```bash
for model in "$BF16_MODEL" "$FP8_MODEL" "$NVFP4_MODEL"; do
  [[ -n "$model" && -f "$model/config.json" ]] || continue
  readlink -f "$model"
  sha256sum "$model/config.json"
  [[ -f "$model/model.safetensors.index.json" ]] && \
    sha256sum "$model/model.safetensors.index.json"
done | tee "$ENV_SNAPSHOT/model-identities.txt"
```

模型注册表必须记录：repo ID、revision/commit、本地路径、served alias、量化格式、config/index hash、验证状态。

### 步骤 4.3：冻结校准与评测数据

固定：数据集 revision、样本索引、顺序、seed、tokenizer revision、chat template。生成后保存 SHA256：

```bash
sha256sum <CALIBRATION_JSONL> <EVAL_MANIFEST_JSONL> \
  > "$ARTIFACT_ROOT/manifests/data.sha256"
```

量化建议使用同一组 256 条校准样本、最大长度 4096、seed 42。

## 5. 阶段 1：BF16/FP8 服务基线

### 步骤 5.1：定义拓扑配置

根据实测代价建立：

```text
TP2-NODE：四组同 NUMA、跨 PIX 分支的双卡组
TP2-PIX：四组 PIX 双卡，仅作负面对照
TP4-NUMA：两个单 NUMA 四卡组
TP8-SYS：八卡跨 NUMA 压力项
```

不要按 GPU 编号直觉判断距离。

### 步骤 5.2：创建独立服务端脚本

`serving/start_server.sh` 只负责服务端：

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?}"
: "${SERVED_NAME:?}"
: "${GPU_IDS:?}"
: "${TP_SIZE:?}"
: "${PORT:?}"
: "${SERVER_LOG:?}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1
export PATH="$SERVE_ENV/bin:$PATH"

exec <SERVING_CLI> serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  2>&1 | tee "$SERVER_LOG"
```

启动前必须用锁定版本的 `--help` 核对参数名。不要假设 vLLM 与 SGLang 参数相同。

### 步骤 5.3：按 NUMA 启动

单 NUMA：

```bash
numactl --cpunodebind=<NODE> --membind=<NODE> \
  env MODEL_PATH="$FP8_MODEL" SERVED_NAME=<FP8_ALIAS> \
      GPU_IDS=<NODE_TP2_GPUS> TP_SIZE=2 PORT="$PORT" \
      SERVER_LOG="$RUN_DIR/server/server.log" \
  "$PROJECT_ROOT/serving/start_server.sh"
```

八卡：

```bash
numactl --interleave=0,1 \
  env MODEL_PATH="$FP8_MODEL" SERVED_NAME=<FP8_ALIAS> \
      GPU_IDS=<ALL_GPUS> TP_SIZE=8 PORT="$PORT" \
      SERVER_LOG="$RUN_DIR/server/server.log" \
  "$PROJECT_ROOT/serving/start_server.sh"
```

### 步骤 5.4：四级服务验收

```bash
curl -fsS "http://127.0.0.1:$PORT/health"
curl -fsS "http://127.0.0.1:$PORT/v1/models" \
  | tee "$RUN_DIR/client/models.json"

curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"只回答数字：6乘以7等于多少？\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | tee "$RUN_DIR/client/completion.json"

curl -fsS "http://127.0.0.1:$PORT/metrics" \
  > "$RUN_DIR/metrics/prometheus.txt"
```

通过条件：health、模型发现、真实 completion、metrics、预定 GPU、无 OOM/rank crash 全部通过。只 import 成功或进程存在不算服务可用。

### 步骤 5.5：32 请求 smoke

服务端保持不变，客户端独立运行：

```bash
"$SERVE_ENV/bin/python" "$PROJECT_ROOT/clients/smoke.py" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$SERVED_NAME" \
  --input-tokens 256 \
  --output-tokens 128 \
  --concurrency 32 \
  --requests 32 \
  --seed 42 \
  --output "$RUN_DIR/client/smoke.jsonl"
```

### 步骤 5.6：筛选矩阵

对 BF16、FP8 分别运行：

```text
TP2-NODE（容量允许时）
TP2-PIX（负面对照）
TP4-NUMA
TP8-SYS
```

每个配置只做一次短/中筛选：256/128 和 2048/256；并发 1、8、32。记录 peak HBM、TTFT、TPOT/ITL、queue、requests/s、output tokens/s、功耗。

Gate：BF16 和 FP8 至少各一个配置通过真实服务；每种格式保留最多两个 Pareto 配置。

## 6. 阶段 2：量化检查点

### 步骤 6.1：创建独立量化环境

```bash
python3 -m venv "$QUANT_ENV"
"$QUANT_ENV/bin/pip" install --upgrade pip
"$QUANT_ENV/bin/pip" install \
  torch transformers datasets accelerate safetensors \
  llmcompressor compressed-tensors
"$QUANT_ENV/bin/pip" check
"$QUANT_ENV/bin/pip" freeze --all \
  > "$PROJECT_ROOT/env/requirements-lock/quant.txt"
```

确认 PyTorch/CUDA/SM120 后再量化；不得覆盖已验证服务环境。

### 步骤 6.2：W4A16 recipe

量化脚本必须：

1. 加载当前 Transformers 对应的 Qwen3.6 MoE 类；
2. 使用 `load_quantizable_moe(...)` 线性化 fused 3D experts；
3. `targets="Linear"`；
4. `moe_calibrate_all_experts=True`；
5. 显式排除 `re:.*linear_attn.*`，避免 32-wide 层触发 Marlin tile 错误；
6. router/gate、embedding、lm_head 保持高精度；
7. 保存为 `compressed-tensors`，同时保存 recipe、tokenizer、日志和 manifest。

执行：

```bash
CUDA_VISIBLE_DEVICES=<QUANT_GPUS> \
NCCL_IB_DISABLE=1 \
numactl --interleave=0,1 \
  "$QUANT_ENV/bin/python" \
  "$PROJECT_ROOT/quantization/llm_compressor/quantize_w4a16.py" \
  --model "$BF16_MODEL" \
  --calibration-jsonl <CALIBRATION_JSONL> \
  --num-samples 256 \
  --max-seq-len 4096 \
  --seed 42 \
  --exclude-regex '.*linear_attn.*' \
  --moe-calibrate-all-experts \
  --output "$W4A16_MODEL" \
  2>&1 | tee "$ARTIFACT_ROOT/logs/quantize_w4a16.log"
```

### 步骤 6.3：NVFP4

先验证已有/官方 NVFP4 checkpoint；通过后再用 LLM Compressor exact recipe 自生成：

- 256 条固定校准样本；
- 长度 4096；
- `targets="Linear"`；
- `moe_calibrate_all_experts=True`；
- 输出 `compressed-tensors`；
- 不覆盖外部 checkpoint。

ModelOpt mixed（NVFP4 experts + FP8 attention）只有在 FP8、W4A16 和纯 NVFP4 至少两条服务线稳定后再开始。

### 步骤 6.4：覆盖审计

```bash
"$QUANT_ENV/bin/python" "$PROJECT_ROOT/quantization/audit_checkpoint.py" \
  --checkpoint "$W4A16_MODEL" \
  --expected-layers 40 \
  --expected-experts 256 \
  --expected-expert-linears 3 \
  --require-format compressed-tensors \
  --require-excluded-regex '.*linear_attn.*' \
  --json "$ARTIFACT_ROOT/reports/w4a16-audit.json"
```

40×256×3=`30720` 个 expert linear 是覆盖基线。scale 数量随量化粒度变化，不能只看文件大小或 block 日志。

### 步骤 6.5：真实加载 Gate

每个 checkpoint 依次执行：静态审计 → TP1 容量 → NODE-TP2 → health/models/completion/metrics → concurrency 1 → concurrency 32。TP1 OOM 只说明容量不足；loader 或 completion 失败才是兼容性失败。

Gate G1：FP8/W4A16 必须稳定多卡；NVFP4 通过 TP1/2、concurrency 1/32 后才准入 EP/EPLB。

## 7. 阶段 3：质量和路由漂移

### 步骤 7.1：质量评测

固定 GSM8K、MMLU-Pro、C-Eval/CMMLU 二选一、一个代码子集。固定 revision、样本索引、prompt、sampling、seed。顺序：BF16 → FP8 → W4A16 → 通过 Gate 的 NVFP4。

预注册工程门槛：FP8 相对 BF16 ≤0.5、NVFP4 ≤1.5、W4A16 ≤2.0 个绝对百分点。失败配置退出正式系统主线，但保留证据。

### 步骤 7.2：捕获 route trace

在 router top-k 产生后、token permute 前捕获：

```text
trace_id,model_id,quant_format,prompt_id,token_position,
layer_id,topk_expert_ids,topk_weights,expert_service_time_us
```

```bash
"$SERVE_ENV/bin/python" "$PROJECT_ROOT/traces/capture_routes.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$SERVED_NAME" \
  --prompt-manifest <PROMPT_MANIFEST> \
  --output "$RUN_DIR/routes" \
  --seed 42
```

同一 prompt 必须使用相同 token IDs。trace 模式批量落盘，与正式性能模式分开。

### 步骤 7.3：路由分析

```bash
python3 "$PROJECT_ROOT/analysis/route_drift.py" \
  --reference <BF16_TRACE> \
  --candidate <QUANT_TRACE> \
  --output "$ARTIFACT_ROOT/reports/route-drift-<FORMAT>.json"
```

输出逐层 top-k Jaccard、router probability KL、expert load CV/Gini、最大/均值、跨 NUMA token 比例，并与 p99 TPOT 做相关分析。

## 8. 阶段 4：SM120 Level 1 kernel/backend 选择器

### 步骤 8.1：锁定第三方仓库

```bash
git clone https://github.com/NVIDIA/cutlass.git "$PROJECT_ROOT/third_party/cutlass"
git clone https://github.com/flashinfer-ai/flashinfer.git "$PROJECT_ROOT/third_party/flashinfer"
git clone https://github.com/sgl-project/sglang.git "$PROJECT_ROOT/third_party/sglang"
git clone https://github.com/vllm-project/vllm.git "$PROJECT_ROOT/third_party/vllm"

for repo in cutlass flashinfer sglang vllm; do
  git -C "$PROJECT_ROOT/third_party/$repo" checkout <FULL_COMMIT_SHA>
  printf '%s\t%s\n' "$repo" \
    "$(git -C "$PROJECT_ROOT/third_party/$repo" rev-parse HEAD)"
done | tee "$PROJECT_ROOT/env/repos.lock.tsv"
```

### 步骤 8.2：CUTLASS SM120 官方基线

```bash
cmake -S "$PROJECT_ROOT/third_party/cutlass" \
  -B "$PROJECT_ROOT/kernels/cutlass_sm120/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=120 \
  -DCUTLASS_ENABLE_TESTS=OFF

cmake --build "$PROJECT_ROOT/kernels/cutlass_sm120/build" \
  --target 79d_blackwell_geforce_nvfp4_grouped_gemm \
  -j"$(nproc)"
```

目标名变化时先运行 `cmake --build <BUILD> --target help | grep -E '79d|nvfp4|grouped'`。不能拿 SM100 二进制当 SM120 基线。

### 步骤 8.3：真实 M 桶

从 trace 统计每层每专家 token 数：`M=0、1–4、5–8、9–16、17–32、33–64、65–128、129–256、>256`。保存每种 workload/量化格式的桶权重。

### 步骤 8.4：backend/tile 搜索

```bash
python3 "$PROJECT_ROOT/kernels/triton_tuner/run_search.py" \
  --trace <ROUTE_TRACE> \
  --backends cutlass,flashinfer,triton \
  --warmup 50 \
  --iterations 200 \
  --repeats 5 \
  --verify-reference torch \
  --output "$RUN_DIR/kernel-search.jsonl"
```

每个 shape 测 M=0、M=1、小 M、非对齐和极端偏斜；记录误差、p50/p95 latency、TFLOPS、DRAM bytes、launch 数。至少一个重要 M 桶稳定改善 ≥10%，否则只交付 selector，不开发新 kernel。

## 9. 阶段 5：Level 2 CUDA 增强

只选一个目标，优先 `permute + activation quant/scale + pack` 融合。

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output "$RUN_DIR/profiles/moe" <FIXED_REPRO_COMMAND>

ncu --set full --target-processes all \
  --export "$RUN_DIR/profiles/kernel" <FIXED_MICRO_COMMAND>
```

先计算 permute、quant、FC1、activation、FC2、unpermute 占比。目标阶段不足 MoE 时间 10% 时停止。

实现顺序：Torch reference → CUDA/CUTLASS → 极端 shape 正确性 → 真实 trace → vLLM modular experts 接入 → 端到端 A/B。两周内关键 M 桶无正确且可重复的 ≥10% micro 收益，则停止 Level 2。

## 10. 阶段 6：TP/DP/EP 系统矩阵

### 步骤 10.1：普通 TP/DP

候选：

```text
TP8×DP1：八卡
TP4×DP2：每个 TP4 限制在一个 NUMA
TP2×DP4：连续逻辑 TP 组映射到四个实测最优 NODE 对
TP1×DP8：量化模型单卡容量通过后启用
```

TP2×DP4 的 `CUDA_VISIBLE_DEVICES` 按 NODE 对连续排列。它控制子组成员，但不能强制 NCCL 全局物理 Ring。

启动前核对 CLI：

```bash
grep -E -- '--tensor-parallel-size|--data-parallel-size|--enable-expert-parallel|--all2all-backend|--enable-eplb' \
  "$ENV_SNAPSHOT/serving-help.txt"
```

### 步骤 10.2：EP 准入顺序

1. EP4/单 NUMA static；
2. EP8/static；
3. EP8 + 原生 EPLB；
4. EP8 + 冗余专家 1；
5. 自研 topology-aware EPLB。

只对通过 G1 的 checkpoint 加 `--enable-expert-parallel` 和通用 PCIe All-to-All backend（如锁定版本支持的 `allgather_reducescatter`）。每一步先 32 请求 smoke 和并发正确性，再做性能。

## 11. 阶段 7：拓扑+量化感知 EPLB

### 步骤 11.1：测迁移成本

分别测同 GPU 重新映射、同 NUMA 复制、跨 NUMA 复制。记录专家大小、迁移字节、阻塞时间、恢复时间和受影响请求 p99。

### 步骤 11.2：离线 placement

输入：通信代价矩阵、量化后专家大小、每层 expert-token 直方图、HBM 余量、service time。目标同时惩罚计算不均衡、跨 NUMA dispatch 字节和迁移成本。

输出：`expert_to_gpu`、副本、预测跨 NUMA bytes、预测 p99、模型/trace hash。

### 步骤 11.3：在线初始参数

```text
窗口：500 ms 或 1000 requests，先达到者
EMA alpha：0.2
触发：load CV > 0.25，连续 3 窗口
收益门：预测 p99 改善 >= 5%
迁移收益/成本：>= 2
最短驻留：10 窗口
冷却：10 窗口
回滚：迁移后连续 3 窗口 p99 退化 > 5%
控制开销：<1%
```

严格比较：static EP、原生 EPLB、自研 load-only、load+topology、load+topology+quant-aware。每次只改变策略。

## 12. 阶段 8：联合策略选择器

候选字段：`quant_format,checkpoint,tp,dp,ep,gpu_mapping,eplb_policy,redundant_experts,kernel_backend,kernel_config`。

先删除 OOM、不支持和质量不合格候选，再计算：

```text
predicted_p99 = compute_cost(real_M_hist, kernel_db)
              + communication_bytes * measured_cost(mapping)
              + imbalance_penalty(route_hist, placement)
              + migration_penalty
```

对测试 workload 穷举已通过 Gate 的配置得到 oracle。报告 top-1、median regret、p95 regret、决策开销、不可行配置误选率。

Gate：median regret ≤5%、p95 ≤10%、控制开销 <1%。未达到时简化模型并报告误差，不直接引入 RL。

## 13. Workload 和实验漏斗

### 13.1 smoke

每个 checkpoint/backend/并行组合只跑 8–32 请求，筛加载、正确性和显存。

### 13.2 单次全量筛选

```text
W1：256/128，并发 1、8、32、128
W2：2048/256，并发 1、8、32
W3：8192/256，并发 1、8、16
W4：32768/128，并发 1、4
```

prefix cache 0%、50%、90%，closed-loop、Poisson、burst 分开报告。

### 13.3 消融

固定 checkpoint/workload，只改变一个因素：量化 only、kernel only、mapping only、EPLB only、联合方案。

### 13.4 正式结果

只保留约 6 个 Pareto 配置。每个配置：独立启动 → health → warmup → 测量 → 保存 metrics/profile → 正常停止 → 冷却 30 秒。随机化配置顺序，独立重复 5 次，bootstrap 95% CI。

## 14. 每次运行的 manifest

```yaml
run_id: <UTC>_<CONFIG_ID>
hostname: <HOST>
model_path: <CHECKPOINT>
served_model_name: <ALIAS>
checkpoint_hash: <SHA256>
quant_format: <BF16|FP8|W4A16|NVFP4|MIXED>
engine: <VLLM|SGLANG>
engine_version: <EXACT>
git_shas: {}
cuda_visible_devices: []
gpu_uuids: []
numa_policy: <NODE0|NODE1|INTERLEAVE>
tp: 1
dp: 1
ep: 1
eplb: false
nccl_ib_disable: 1
workload_config: <YAML>
prompt_manifest_hash: <SHA256>
seed: 42
server_command: <FULL_COMMAND>
client_command: <FULL_COMMAND>
status: planned
```

完成后补充 start/end UTC、退出码、peak HBM、日志、客户端结果、metrics、失败分类。raw 文件不改写。

## 15. Gate 总表

| Gate | 通过条件 | 失败动作 |
|---|---|---|
| G0 硬件 | 拓扑、P2P、PCIe、NCCL 正式统计完成 | 环境变化后重跑 |
| 服务 | health/models/completion/metrics/32 请求 | 不进性能矩阵 |
| 容量 | 无 OOM，记录 HBM/上下文 | 提高 TP，生成新配置 |
| 量化覆盖 | 40 层、256 experts、30720 linears、排除明确 | 修 recipe 重量化 |
| 质量 | FP8≤0.5、NVFP4≤1.5、W4A16≤2.0 点下降 | 退出主线或 mixed |
| NVFP4 EP | TP1/2、并发 1/32、多卡 completion | 仅保留 TP/kernel/失败证据 |
| Kernel | 全 shape 正确，关键 M 桶 ≥10% | 停 Level 2，保留 selector |
| 系统收益 | ≥5% tokens/s 或 ≥5% p99 TPOT | 报告瓶颈，不宣称优化成功 |
| EPLB | 开销 <1%，无抖动，可回滚 | 使用 static/native |
| Selector | median regret≤5%，p95≤10% | 简化模型和误差分析 |

## 16. 推荐执行顺序

1. 填参数并创建仓库。
2. 完成或登记 Phase 0。
3. 冻结环境、模型、校准/评测数据。
4. 建立 BF16/FP8 服务基线。
5. 完成 FP8 的 NODE/PIX/TP4/TP8 筛选。
6. 生成并审计 W4A16。
7. 验证 NVFP4 TP1/TP2 与并发 1/32。
8. 做固定质量评测。
9. 捕获 route trace 和真实 M 桶。
10. 完成 CUTLASS/FlashInfer/Triton Level 1 selector。
11. profiler 通过后才进入一个 Level 2 kernel。
12. 运行 TP/DP/EP 和 static/native EPLB。
13. 实现 topology+quant-aware EPLB。
14. 实现联合 selector，与 oracle 比较。
15. 选择 Pareto 6 组，随机顺序、5 次重复。
16. 发布代码、lockfile、manifest、raw、失败矩阵和一键 smoke。

## 17. 16 周落地安排

| 周 | 具体操作 | 产物 |
|---:|---|---|
| 1–2 | Phase 0、环境冻结、BF16/FP8 服务 | 拓扑/collective、baseline |
| 3 | FP8 NODE/PIX/TP4/TP8 筛选 | 并行策略表 |
| 4–5 | W4A16、NVFP4、mixed gate、质量 | checkpoint/audit/质量表 |
| 6 | route trace、Jaccard/KL/CV/Gini | 可回放 trace |
| 7–8 | SM120 backend/tile 搜索 | kernel config DB |
| 9 | 单一 Level 2 目标与停损判断 | patch 或停止证据 |
| 10–11 | TP/DP/EP、mapping、EPLB 消融 | Pareto 候选集 |
| 12–13 | topology+quant-aware EPLB、selector | 控制器与 regret |
| 14 | 完整 workload 单次筛选 | Pareto 6 组 |
| 15 | 随机顺序、每组 5 次、95% CI | 最终统计 |
| 16 | 复现包、论文、失败矩阵 | release tag |

最小成功闭环是：BF16/FP8/W4A16 可服务、实测成本矩阵、量化感知 TP/DP/EP、路由漂移、拓扑 EPLB、真实 trace 驱动的 SM120 selector和一条命令 smoke。自研融合 kernel 是增强项，不是成败单点。
