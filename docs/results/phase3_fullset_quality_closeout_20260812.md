# 阶段 3：FP8/NVFP4 full-set 质量收尾

> 更新日期：2026-08-13（Asia/Shanghai）
> 当前状态：NVFP4 基础轮 24,374/24,374 完成，`failed=0`，其中 971 条
> 因输出达到长度上限而进入独立续跑；FP8 必须等待该续跑闭合后再启动。

## 1. 已冻结的评测输入

- 总样本数：24,374；A/B 分片各 12,187 条。
- seed：42；客户端并发：每个分片 4。
- tokenizer/chat template：使用候选 checkpoint 自带配置，`enable_thinking=false`。
- A 分片 SHA-256：`4c761274f20b85fb42f0c746cd2dbdaf04eb5e3e4987dd7e3c2b2b1b3396d10a`。
- B 分片 SHA-256：`d56082d0cd681f8fd5883fb810516a957e737fa8393e196f80f45b8c33f53da8`。

服务器上的冻结输入路径为：

- `/data/models/test/qtopomoe_quality/full_set_protocol_shard_a.jsonl`
- `/data/models/test/qtopomoe_quality/full_set_protocol_shard_b.jsonl`

## 2. NVFP4 正式任务

- 输出根目录：`/data/models/test/qtopomoe_quality_runs/full_official_nvfp4_ep4_pair_v1`
- 分片 A：GPU0–3、TP4/EP4、NUMA0、端口 31620。
- 分片 B：GPU4–7、TP4/EP4、NUMA1、端口 31621。
- 模型：`/data/models/test/redhatai_qwen36_nvfp4`。
- vLLM：`0.26.1rc1.dev343+g33c50587d` cleanroom。
- NVFP4 MoE 后端：两个服务均明确记录为 `MARLIN`。
- EP truth：每个服务均创建 4 个 EP rank，每 rank 64/256 experts。
- 四级服务验收：两个服务的 health、model discovery、completion、metrics 均通过；验收回答均为 `42`。
- 基础轮最终状态：`base_completed_rerun_required`；A/B 分片分别有
  499/472 条截断，合计 971 条（3.98%），请求失败为 0。基础轮服务与客户端
  已正常退出。

截断续跑使用 `scripts/build_fullset_truncation_manifest.py` 从基础轮结果按 ID
筛选，不改变 messages、协议、答案、seed 或采样参数。C-Eval 的 `max_tokens`
由 2048 提高到 8192，MMLU-Pro 由 4000 提高到 12000；服务
`MAX_MODEL_LEN=32768`，输出写入独立目录
`/data/models/test/qtopomoe_quality_runs/full_official_nvfp4_ep4_pair_trunc_v1`，
不覆盖基础轮。

续跑已于 2026-08-13 09:34（Asia/Shanghai）启动，状态为
`running_clients`，管理 PID 为 `1193994`，A/B 客户端 PID 为
`1212200/1212201`。A/B 续跑 manifest 分别为 499/472 条，SHA-256 为
`6610d7ebbdf4c5787e775c5d56439b821baef3bf46ff0c774f1ba746336fea5b`
和 `8d6bc412e88e1d3757d0d82ce7549d72cdddbe160a86a19df2c66ac7ef2abc6c`。
两个服务的 health、model discovery、completion 和 metrics 再次全部通过。

客户端每完成 100 条便原子更新一次结果文件，并使用 `--resume` 跳过已有成功样本。关闭 Codex 或 SSH 不会终止任务。

## 3. 历史 BF16 失败的重新判定

旧 BF16 A/B 服务并没有发生模型加载、CUDA、NCCL 或 vLLM 架构崩溃。两个独立服务在 2026-08-06 12:34:27 同一秒收到外部 `SIGTERM`，关停前持续正常生成；后续 HTTP 500 是服务关停后的连带错误。因此旧 BF16 full-set 只能判为“编排生命周期中断”，不能用于模型质量结论。

本轮使用独立 session、持久 PID、状态文件、周期性 checkpoint 和断点续跑，消除了同类风险。

## 4. 后续顺序与 Gate

1. NVFP4 基础轮完成后审计 `failed`、`truncated`、答案抽取与分科准确率。
2. 仅补跑截断样本，提高 `max_tokens` 后合并，并保证每个样本只有一个最终记录。
3. 生成 NVFP4 全量摘要、输入/输出哈希和 Gate。
4. 使用同一冻结输入与协议运行 FP8 TP2 双分片，并执行相同补跑/合并流程。
5. 只在两种格式均 `failed=0` 且截断闭合后比较质量；运行中的部分结果不得作为准确率结论。

执行入口为 [`scripts/run_fullset_quality_pair.sh`](../../scripts/run_fullset_quality_pair.sh)，客户端为 [`clients/quality_eval.py`](../../clients/quality_eval.py)。
