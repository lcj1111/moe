# 阶段 3：FP8/NVFP4 full-set 质量收尾

> 更新日期：2026-08-13（Asia/Shanghai）
> 当前状态：NVFP4 基础轮与 971 条续跑已完成严格合并，24,374 个 ID 完整且
> `failed=0`；18 条推理循环样本按“显式未完成”封板。FP8 基础轮 24,374 条
> 已完成且 `failed=0`；1,019 条截断有限续跑已通过四级服务 Gate，客户端运行中。

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

续跑于 2026-08-13 09:34（Asia/Shanghai）启动并于 10:27 左右完成。A/B
续跑 manifest 分别为 499/472 条，SHA-256 为
`6610d7ebbdf4c5787e775c5d56439b821baef3bf46ff0c774f1ba746336fea5b`
和 `8d6bc412e88e1d3757d0d82ce7549d72cdddbe160a86a19df2c66ac7ef2abc6c`。
两个服务的 health、model discovery、completion 和 metrics 再次全部通过。

续跑 A/B 均 `completed=requested`、`failed=0`，但分别仍有 11/7 条截断；合计
18 条，其中 C-Eval 15 条、MMLU-Pro 3 条。因此 manager 状态为
`base_completed_rerun_required`。

逐条审计确认，这 18 条全部满足：`finish_reason=length`、实际输出 token 数等于
新上限，响应尾部仍在重复枚举、重新推导或自我否定，没有接近稳定最终答案。
继续提高输出上限只会放大无效生成，因此不进行无界二次续跑。即便局部答案抽取
碰巧等于标准答案，`correct` 仍必须为 `null`，不得静默计对或计错。

使用 [`scripts/merge_fullset_quality_results.py`](../../scripts/merge_fullset_quality_results.py)
完成严格合并。工具要求基础轮 ID 与冻结 manifest 完全一致、续跑 ID 与基础轮
截断集合完全一致，并校验 `id/benchmark/expected/score_type` 身份字段；输出按原
manifest 顺序恢复。任何重复、缺失、额外替换或身份漂移都会直接失败。

合并结果：

| 指标 | 结果 |
|---|---:|
| manifest / 合并记录 / 唯一 ID | 24,374 / 24,374 / 24,374 |
| 使用基础轮 / 使用续跑 | 23,403 / 971 |
| 请求失败 / 显式未完成 | 0 / 18 |
| 可评分 / 正确 | 24,356 / 21,014 |
| MMLU-Pro（仅可评分分母） | 10,111 / 12,029 = 84.0552% |
| C-Eval（仅可评分分母） | 10,903 / 12,327 = 88.4481% |

- 合并目录：`/data/models/test/qtopomoe_quality_runs/full_official_nvfp4_ep4_pair_merged_v1`
- 合并 JSONL SHA-256：`d8e17e7106b2575ff1ac7b020ddc7d93f7d3d5698794310b163b959d75601920`
- 合并摘要 SHA-256：`24449c7e684f3e8ad1539d92009a6cd93991f800b5cc3a891d684d5e7596436f`
- Gate：完整性与请求 Gate 均接受；总体为 `closed_with_unfinished`，不是
  “24,374 条全部完成”。机器可读摘要见
  [`Q-TopoMoE_Phase3_NVFP4_fullset_merge_20260813.json`](../Q-TopoMoE_Phase3_NVFP4_fullset_merge_20260813.json)。

客户端每完成 100 条便原子更新一次结果文件，并使用 `--resume` 跳过已有成功样本。关闭 Codex 或 SSH 不会终止任务。

## 3. 历史 BF16 失败的重新判定

旧 BF16 A/B 服务并没有发生模型加载、CUDA、NCCL 或 vLLM 架构崩溃。两个独立服务在 2026-08-06 12:34:27 同一秒收到外部 `SIGTERM`，关停前持续正常生成；后续 HTTP 500 是服务关停后的连带错误。因此旧 BF16 full-set 只能判为“编排生命周期中断”，不能用于模型质量结论。

本轮使用独立 session、持久 PID、状态文件、周期性 checkpoint 和断点续跑，消除了同类风险。

## 4. FP8 正式任务与后续 Gate

FP8 于 2026-08-13 10:46（Asia/Shanghai）启动。启动前确认外部 RobustGEMQ
CUDA 测试已自然退出、端口空闲、目标目录不存在，未终止任何其他项目进程。

- 输出根目录：`/data/models/test/qtopomoe_quality_runs/full_official_fp8_tp2_pair_v1`
- 分片 A：GPU0–1、TP2、NUMA0、端口 31620。
- 分片 B：GPU4–5、TP2、NUMA1、端口 31621。
- 模型：`/data/models/test/models/Qwen--Qwen3.6-35B-A3B-FP8/snapshots/master`。
- vLLM：`0.26.1rc1.dev343+g33c50587d` cleanroom。
- MoE 后端：命令行固定 `--moe-backend triton`。
- 冻结输入、seed=42、每分片并发 4 与 NVFP4 完全相同。
- manager PID：`1447425`；启动脚本 SHA-256：
  `34dc03fac99bd28d959f2a30dcbb5dc336e465a71c9934dadcfd997e9655b1f7`。
- A/B 的 health、model discovery、completion、metrics 四级验收均通过，验收回答
  均为 `42`。2026-08-13 10:51:39 manager 进入 `running_clients`，客户端 PID
  分别为 `1466816` 和 `1466817`。

FP8 基础轮于 2026-08-13 13:50 左右结束，manager 状态为
`base_completed_rerun_required`。这里的 `client_rc_a=1/client_rc_b=1` 表示存在
截断项，并非请求、服务或 CUDA 失败。完整性审计结果如下：

| 指标 | 分片 A | 分片 B | 合计 |
|---|---:|---:|---:|
| manifest / 结果 / 唯一 ID | 12,187 | 12,187 | 24,374 |
| 请求失败 | 0 | 0 | 0 |
| 截断 | 519 | 500 | 1,019 |
| C-Eval 截断 | 232 | 195 | 427 |
| MMLU-Pro 截断 | 287 | 305 | 592 |

- A 片结果 SHA-256：`19eac38fdb045ade96d2d5dce7116fd51c8c79bc68957aef529ad5e9f2875a5d`
- B 片结果 SHA-256：`35ab64803ea3afe4eb7f48d81a526a95f5ac6b2c00713c5ad85c618f1269c7d9`
- 两片 ID 均与冻结 manifest 完全相等、顺序一致且无重复；所有截断项的
  `correct` 均为 `null`。

使用 `scripts/build_fullset_truncation_manifest.py` 只提取上述 1,019 个 ID。
C-Eval 的 `max_tokens` 从 2,048 提高到 8,192，MMLU-Pro 从 4,000 提高到
12,000，messages、协议、答案、seed 和采样参数保持不变：

- A 片续跑 manifest：519 条，SHA-256
  `be6de5c8c6c98093ae9fcc61cb54c0f8b9ba79e8ea78ca22ab6b175479771113`
- B 片续跑 manifest：500 条，SHA-256
  `5f3be6f53fd6e74dafabb065a9704f4bb9f02c3978520a95283e40c0d2324855`

续跑于 2026-08-13 13:56（Asia/Shanghai）启动，输出目录为
`/data/models/test/qtopomoe_quality_runs/full_official_fp8_tp2_pair_trunc_v1`。
服务使用相同 FP8 checkpoint、TP2 GPU0–1/4–5 和 Triton MoE backend，仅将
`MAX_MODEL_LEN` 提高到 32,768。A/B 四级验收再次全部通过，验收回答均为
`42`；manager 于 14:00:53 进入 `running_clients`：

- manager PID：`2159292`
- A/B 服务 PID：`2159307` / `2159309`
- A/B 客户端 PID：`2182950` / `2182951`

机器可读基础轮及续跑启动证据见
[`Q-TopoMoE_Phase3_FP8_fullset_base_20260813.json`](../Q-TopoMoE_Phase3_FP8_fullset_base_20260813.json)。

后续必须按以下顺序执行：

1. 持续检查两个续跑客户端、周期 checkpoint、请求失败和残余截断；不以进程存活代替结果完整。
2. 对残余截断逐条审计响应尾部；有合理收敛空间才允许有限二次补跑，否则显式标为未完成。
3. 使用同一严格合并工具生成 FP8 唯一结果、哈希与 Gate。
4. 只在 FP8 `failed=0` 且截断处理闭合后，与 NVFP4 做同分母、同协议比较；
   部分运行结果不得作为准确率结论。

执行入口为 [`scripts/run_fullset_quality_pair.sh`](../../scripts/run_fullset_quality_pair.sh)，
合并入口为 [`scripts/merge_fullset_quality_results.py`](../../scripts/merge_fullset_quality_results.py)，
客户端为 [`clients/quality_eval.py`](../../clients/quality_eval.py)。
