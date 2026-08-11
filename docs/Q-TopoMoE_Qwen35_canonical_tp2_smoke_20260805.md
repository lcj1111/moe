# Qwen3.5 MoE 规范化 W4A16 TP2 smoke（2026-08-05）

两个原生后端在 PIX GPU0-1 上均完成 32/32 请求且无失败。固定 smoke 使用 256 个输入 token、64 个输出 token、并发 8、seed=42。

| 后端 | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 | 总耗时 |
|---|---:|---:|---:|---:|
| vLLM | 265/779 ms | 71.4/81.1 ms | 4.76/5.89 s | 5.89 s |
| SGLang | 251/7341 ms | 63.7/66.4 ms | 4.27/11.33 s | 11.33 s |

这是功能 smoke，不是正式排序。SGLang 的 TTFT 长尾包含首次编译和调度影响，必须先 warmup 并重复运行后才能用于比较。

## 显存解读

匹配的 TP2 日志没有显示明显的权重显存差异：

- vLLM 每卡：10.15 GiB weight/non-Torch，另有 17.91 GiB hybrid cache；
- SGLang 每卡：10.08 GiB 权重、8.25 GiB Mamba state cache、9.18 GiB KV cache；
- SGLang 还在基准卡预留最多 1 GiB multimodal CUDA IPC，这是自动多模态传输造成的无关开销。

后续 text-only Gate 使用 `--language-only` 去除该预留。正式显存比较必须固定 cache 容量或最大并发，不能只把两个框架的 memory-fraction 参数设为相同值。
