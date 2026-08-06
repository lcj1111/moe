# Route trace capture

`capture_routes.py` 通过 vLLM cleanroom 原生
`enable_return_routed_experts` 采集每个 token 的 MoE 路由专家 ID。

## 输出

每次采集生成：

- `traces/<prompt_id>.npy`：`[seq_len, num_layers, top_k]` uint8 专家 ID；
- `traces.jsonl`：逐 token 展开（行数 = tokens × 40 层）；
- `capture_manifest.json`：模型、prompt 哈希、shape、每个 npy 的 SHA-256；
- `expert_token_histogram.json`：M-bucket 分布。

## 已知边界

- 只导出 expert IDs；`topk_weights` 与 `expert_service_time_us` 为 null，
  因此无法计算 router-probability KL 或 kernel 延迟。
- 原始 npy/jsonl 体积大，不入库；`manifests/` 内的 capture manifest 钉住
  全部文件哈希与协议参数，供审计与复现。

## 服务器上的全量采集

BF16 与 W4A16（triton）各 116 prompts、4,793,080 行，位于
`/data/models/test/qtopomoe_traces/v1/{bf16_full,w4a16_full}`；
漂移分析结果见 `docs/Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json`。
