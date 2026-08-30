# 阶段 2：量化检查点与质量准入

本阶段验证两条量化路线：RedHatAI NVFP4 与 canonical W4A16。检查顺序固定为静态覆盖审计、
原生后端真实加载、服务检查、并发 smoke 和同协议质量比较。当前发布路线及文件哈希以
[Release manifest](../Q-TopoMoE_release_manifest_20260825.json) 为准。

## 1. RedHatAI NVFP4

当前 NVFP4 检查点来自 `RedHatAI/Qwen3.6-35B-A3B-NVFP4`，本地路径为
`/data/models/test/redhatai_qwen36_nvfp4`，格式是 compressed-tensors
`nvfp4-pack-quantized`。模型由 40 层、256 个 routed experts 组成。

[覆盖审计](../Q-TopoMoE_Phase2_NVFP4_audit_20260807.json)确认：

- packed weight、weight scale、weight global scale 和 input global scale 均覆盖 30,720 个 routed-expert projection；
- down、gate、up 三类 projection 各 10,240 个；
- linear attention 保持非量化；
- 模型加载没有回退到仿真路径，SM120 使用真实 NVFP4 MoE kernel。

在冻结 vLLM cleanroom 中，TP1 和 TP2 均通过 health、模型发现、completion、metrics 与
32 请求 smoke。TP1 加载后权重显存约 21.88 GiB，TP2 约为每卡 11.04 GiB。

116 条 official-like v2 的同协议结果如下：

| 数据集 | BF16 | NVFP4 | 差值 |
|---|---:|---:|---:|
| C-Eval（52） | 49/52 = 94.23% | 48/52 = 92.31% | -1.92 个百分点 |
| MMLU-Pro（64） | 59/64 = 92.19% | 59/64 = 92.19% | 0 |
| 合计（116） | 108/116 = 93.10% | 107/116 = 92.24% | -0.86 个百分点 |

该小样本只用于服务回归；正式质量结论来自 24,330 条共同可评分样本。在 full-set 上，
NVFP4 相对 BF16 下降 0.6946 个百分点，低于项目设定的 1.5 个百分点上限。详见
[三格式 full-set 质量收尾](phase3_fullset_quality_closeout_20260812.md)。

## 2. W4A16

W4A16 使用 compressed-tensors packed-int4 权重和 Triton WNA16 MoE 路径。冻结源检查点
包含 93,093 个张量，权重文件大小为 20,928,221,896 bytes；来源与 SHA-256 记录在
[W4A16 freeze manifest](../Q-TopoMoE_W4A16_freeze_20260805.json)。

固定 BF16 与 W4A16 Triton 在 official-like v2 上的结果为：

| 数据集 | BF16 | W4A16 Triton |
|---|---:|---:|
| C-Eval（52） | 49/52 = 94.23% | 50/52 = 96.15% |
| MMLU-Pro（64） | 59/64 = 92.19% | 58/64 = 90.63% |
| 合计（116） | 108/116 = 93.10% | 108/116 = 93.10% |

当前 W4A16 结论只使用 Triton 后端，避免把不同 kernel 的数值行为混入同一结果。

## 3. checkpoint 规范化

源检查点的权重位于 `model.language_model.*` 命名空间，而 text-only Qwen3.5 MoE loader
要求 `model.*`。[`canonicalize_qwen35_text_checkpoint.py`](../../scripts/canonicalize_qwen35_text_checkpoint.py)
执行确定性键映射，不改变配置、tokenizer、shape、dtype 或 tensor value。

规范化产物为 `/data/models/test/qtopomoe_w4a16_canonical_text_v1`：

- 93,093/93,093 个张量通过 shape、dtype 与逐值一致性检查；
- 张量总字节数为 20,915,187,456；
- 权重 SHA-256 为 `0eb2775989321d038ea9534041a6030f536a0d0d2293e7087837390e9738c1d2`；
- vLLM 与 SGLang 的 TP1 原生加载、服务检查均通过；TP2 的 32 请求 smoke 均为 32/32。

## 4. block128 到 block64 的可逆布局变换

TP8 分片后的宽度为 64。为满足分组对齐，
[`reblock_w4a16_128_to_64.py`](../../quantization/reblock_w4a16_128_to_64.py)
把每个 128 宽 scale group 拆成两个 64 宽子组并复制同一 scale；packed int4 权重保持不变。

六类代表张量验证结果为 packed 权重完全一致，反量化最大差值为 0。变换后的独立产物是
`/data/models/test/qtopomoe_w4a16_canonical_text_v1_g64`，关系记录在
[reblock manifest](../Q-TopoMoE_W4A16_reblock_manifest_20260807.json)，不会覆盖 block128 版本。
TP8 的 32 请求 smoke 全部完成。

## 5. 复现入口与边界

- official-like 输入：[`official_like_smoke_v2.jsonl`](../../configs/evaluation/official_like_smoke_v2.jsonl)
  与 [`official_like_smoke_v2.manifest.json`](../../configs/evaluation/official_like_smoke_v2.manifest.json)；
- NVFP4 审计：[`quantization/audit_nvfp4.py`](../../quantization/audit_nvfp4.py)；
- W4A16 审计：[`scripts/audit_w4a16.py`](../../scripts/audit_w4a16.py)；
- checkpoint 准入：[`scripts/gate_qwen35_checkpoint.sh`](../../scripts/gate_qwen35_checkpoint.sh)。

sampled 结果不能与模型卡的官方 benchmark 分数直接比较。模型格式、后端、prompt、
tokenizer/chat template、seed 和输出上限必须同时固定；full-set 结论只在共同可评分样本上计算。
