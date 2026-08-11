# Route trace 采集

`capture_routes.py` 通过 vLLM cleanroom 的原生
`enable_return_routed_experts` 选项，记录每个 token 的 MoE 专家路由 ID。

## 输出文件

每次采集生成：

- `traces/<prompt_id>.npy`：形状为 `[seq_len, num_layers, top_k]` 的 uint8 专家 ID。
- `traces.jsonl`：逐 token 展开记录（行数为 tokens × 40 层）。
- `capture_manifest.json`：模型、prompt 哈希、shape，以及每个 npy 的 SHA-256。
- `expert_token_histogram.json`：按 M-bucket 汇总的专家 token 分布。

## 已知边界

- 当前只导出 expert IDs；`topk_weights` 和 `expert_service_time_us` 为 null，因此不能计算 router-probability KL，也不能从 trace 伪造 kernel 延迟。
- 原始 npy/jsonl 体积较大，不进入仓库；`manifests/` 中的 capture manifest 固定文件哈希和协议参数，用于审计与复现。

## 服务器上的全量采集

BF16 与 W4A16（Triton）各 116 条 prompt、共 3,793,080 行，位于
`/data/models/test/qtopomoe_traces/v1/{bf16_full,w4a16_full}`。漂移分析结果见
`docs/Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json`。
