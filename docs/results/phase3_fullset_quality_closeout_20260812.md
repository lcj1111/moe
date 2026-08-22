# 阶段 3：BF16/FP8/NVFP4 full-set 质量收尾

> 更新日期：2026-08-16（Asia/Shanghai）
> 当前状态：BF16、FP8、NVFP4 三种格式的基础轮、有限截断续跑、残余审计和
> 严格合并均已完成；请求失败均为 0。三格式共同完成的 24,330 条样本上，
> BF16/FP8/NVFP4 的准确率分别为 86.9955%/86.9749%/86.3009%。BF16 与 FP8
> 总体近似持平，NVFP4 相对 BF16 下降 0.6946 个绝对百分点，仍在项目预注册的
> 1.5 点门槛内。19/26/18 条长输出分别按“显式未完成”封板，不静默计对或计错。

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

FP8 续跑 A/B 分别完成 519/519 与 500/500，`failed=0`，残余截断分别为
15/11 条。26 条均精确达到 8,192 或 12,000 token，`finish_reason=length`，
响应尾部仍在重复枚举、重新推导、纠结题面或反复验证，没有接近稳定答案。
因此不再进行无界二次续跑；即使答案抽取碰巧等于标准答案，`correct` 仍保持
`null`。

使用严格合并工具替换全部 1,019 条基础轮截断记录。FP8 合并结果如下：

| 指标 | 结果 |
|---|---:|
| manifest / 合并记录 / 唯一 ID | 24,374 / 24,374 / 24,374 |
| 使用基础轮 / 使用续跑 | 23,355 / 1,019 |
| 请求失败 / 显式未完成 | 0 / 26 |
| 可评分 / 正确 | 24,348 / 21,177 |
| MMLU-Pro（仅可评分分母） | 10,172 / 12,028 = 84.5693% |
| C-Eval（仅可评分分母） | 11,005 / 12,320 = 89.3263% |

- 合并目录：`/data/models/test/qtopomoe_quality_runs/full_official_fp8_tp2_pair_merged_v1`
- 合并 JSONL SHA-256：`6dc2df8a92f83eabd657d43cf1e77aa12a7c2088a40ed3061c50d45c3eb4851f`
- 合并摘要 SHA-256：`a857d838b5f4e1a45b41f03acbee2e6fd43a7917ade3aa42244b19673d7a4467`
- Gate：完整性与请求 Gate 接受；总体为 `closed_with_unfinished`。

## 5. FP8 与 NVFP4 的共同分母对比

FP8 有 26 条显式未完成，NVFP4 有 18 条，其中重叠 9 条、并集 35 条。直接比较
各自 scored-only 分母会引入选择性缺失偏差，因此主结论只使用两种格式均完成的
24,339 个 ID：

| 基准 | 共同 ID | NVFP4 | FP8 | FP8−NVFP4 |
|---|---:|---:|---:|---:|
| 总体 | 24,339 | 86.2977% | 86.9715% | +0.6738 pp |
| C-Eval | 12,313 | 88.4837% | 89.3202% | +0.8365 pp |
| MMLU-Pro | 12,026 | 84.0595% | 84.5668% | +0.5072 pp |

共同分母上，二者同时答对 20,372 条、同时答错 2,539 条；NVFP4 独有答对
632 条，FP8 独有答对 796 条。因此本冻结协议下 FP8 的质量高于该 NVFP4
checkpoint。这个结论不能替代 BF16 full-set，也不能直接与官方不同 harness、
prompt 或采样口径的分数混写。

机器可读的 FP8 合并及共同分母对比见
[`Q-TopoMoE_Phase3_FP8_merge_NVFP4_compare_20260813.json`](../Q-TopoMoE_Phase3_FP8_merge_NVFP4_compare_20260813.json)。

上述 FP8/NVFP4 基础轮、续跑、残余审计、严格合并和共同分母对比均已完成。
三格式正式结论要求重新执行同一冻结协议的 BF16 full-set；该要求已由下述
第 6、7 节完成，不使用历史被外部 SIGTERM 中断的 BF16 结果补齐。

执行入口为 [`scripts/run_fullset_quality_pair.sh`](../../scripts/run_fullset_quality_pair.sh)，
合并入口为 [`scripts/merge_fullset_quality_results.py`](../../scripts/merge_fullset_quality_results.py)，
客户端为 [`clients/quality_eval.py`](../../clients/quality_eval.py)。

## 6. BF16 同协议补测

BF16 full-set 于 2026-08-13 15:38（Asia/Shanghai）启动。它不是对历史中断目录
的续写，而是在独立输出目录中重新执行全部 24,374 条冻结样本：

- 输出根目录：`/data/models/test/qtopomoe_quality_runs/full_official_bf16_tp4_pair_v2`
- 模型：`/data/models/REAP/models/Qwen3.6-35B-A3B`
- 模型大小与分片：67 GiB、26 个 safetensors 权重分片
- `config.json` SHA-256：`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`
- 分片 A：GPU0–3、TP4、NUMA0、端口 31620
- 分片 B：GPU4–7、TP4、NUMA1、端口 31621
- vLLM：`0.26.1rc1.dev343+g33c50587d` cleanroom
- 服务参数：`max_model_len=8192`、`max_num_seqs=8`、显存利用率 0.90、
  prefix caching 开启、`--enforce-eager`
- 客户端参数：seed 42、每分片并发 4、超时 1,200 秒、每 100 条原子 checkpoint、
  `--resume`
- manager PID：`2580631`
- A/B 服务 PID：`2580635` / `2580637`
- A/B 客户端 PID：`2593122` / `2593123`
- 启动脚本 GitHub commit：`029847a`
- 启动脚本 SHA-256：`f51210735fcf831378ae1df55b40f692e1d54bbace346a77d3f8ae4bbc5ee894`

15:40 左右两个服务均完成 26/26 权重加载；每张 GPU 的 BF16 权重占用约
16.52 GiB，总显存占用约 29 GiB。日志明确记录 `quantization=None` 路径采用
Triton unquantized MoE backend，没有把 FP8/NVFP4 量化后端误用于 BF16。

A/B 两套服务的 health、model discovery、completion、metrics 四项验收均为
`true`，验收 completion 均返回 `42`。基础轮于 2026-08-15 03:59 左右自然
结束，服务与客户端正常退出并释放 8 张 GPU；manager 状态为
`base_completed_rerun_required`，其中 `client_rc_a=1/client_rc_b=1` 仅表示
存在截断项，不是请求、服务、CUDA 或 NCCL 失败。

基础轮严格审计结果：

| 指标 | 分片 A | 分片 B | 合计 |
|---|---:|---:|---:|
| manifest / 结果 / 唯一 ID | 12,187 | 12,187 | 24,374 |
| 请求失败 | 0 | 0 | 0 |
| 截断 | 524 | 484 | 1,008 |
| C-Eval 截断 | 226 | 196 | 422 |
| MMLU-Pro 截断 | 298 | 288 | 586 |

- A 片基础结果 SHA-256：`24315bef13678dcc0dd91e9bc3fc78cc5f202c27d487f8e97b284c0a98fdda85`
- B 片基础结果 SHA-256：`419748a64723b97292e00e2ca487755cae1c03724ffb0ab2dd3376d021933ca7`
- C-Eval 临时 scored-only：10,713 / 11,920 = 89.8742%
- MMLU-Pro 临时 scored-only：9,904 / 11,446 = 86.5280%

临时分数排除了截断项，不能作为 BF16 封板结果。生成续跑 manifest 前，工具会
强制校验 manifest/result 数量、ID 唯一性与顺序、benchmark/答案/评分类型/
`max_tokens` 身份字段、`failed=0`、截断项 `correct=null` 以及长度证据。
长度证据与客户端语义一致：`finish_reason=length`，或 completion token 数达到
请求上限。后者用于兼容 vLLM 偶尔返回 `finish_reason=stop` 但 token 数精确达到
上限的响应；这类记录仍按截断处理，没有修改原始结果。

严格审计和续跑清单生成工具对应 GitHub commit 为 `f458b88` 与 `1b567e4`。
BF16 续跑清单如下：

- A：524 条；SHA-256：`fc066c407713ed5c0a1a86f219fa101b1a5dbbb08c6e4973821a003269509eb5`
- B：484 条；SHA-256：`0cb590461ef5b1726ee957028a2998c74a4921e5e34b8e9c5e48486159fc1b93`
- 输入目录：`/data/models/test/qtopomoe_quality/bf16_trunc_v1`
- C-Eval `max_tokens`：2,048 → 8,192
- MMLU-Pro `max_tokens`：4,000 → 12,000
- messages、答案、协议、seed 和采样参数保持不变

续跑于 2026-08-15 11:00（Asia/Shanghai）启动，输出目录为
`/data/models/test/qtopomoe_quality_runs/full_official_bf16_tp4_pair_trunc_v1`。
服务仍使用相同 BF16 checkpoint、TP4 GPU0–3/4–7 和 `--enforce-eager`，仅将
`MAX_MODEL_LEN` 提高到 32,768。A/B 四项服务验收再次全部通过，验收回答均为
`42`；manager 已进入 `running_clients`：

- manager PID：`1541696`
- A/B 服务 PID：`1541701` / `1541703`
- A/B 客户端 PID：`1549993` / `1549994`

续跑于 2026-08-15 21:52 左右自然结束。A/B 分别完成 524/524 与 484/484，
`failed=0`；服务日志中没有 `ERROR` 或 `Traceback`。结果 SHA-256 分别为：

- A：`b610eeb029caf49cfb8e1189c0c312cff0db626873986735bab19922f3914326`
- B：`e0986bb24afb9342095c0e60c419059a6894a717b2040fcf211bfd068b394166`

续跑后仍有 19 条显式未完成：C-Eval 17 条，MMLU-Pro 2 条。逐条核对确认它们
全部 `finish_reason=length`，completion token 数精确等于各自上限（8,192 或
12,000），响应尾部仍在重复枚举、自我否定或未收敛推导。它们的 `correct` 均为
`null`。继续无界提高上限只会放大无效生成，因此不再进行第二轮续跑。

使用 [`scripts/merge_fullset_quality_results.py`](../../scripts/merge_fullset_quality_results.py)
严格替换全部 1,008 条基础轮截断记录。合并结果如下：

| 指标 | 结果 |
|---|---:|
| manifest / 合并记录 / 唯一 ID | 24,374 / 24,374 / 24,374 |
| 使用基础轮 / 使用续跑 | 23,366 / 1,008 |
| 请求失败 / 显式未完成 | 0 / 19 |
| 可评分 / 正确 | 24,355 / 21,183 |
| 总体（仅可评分分母） | 86.9760% |
| C-Eval（仅可评分分母） | 10,974 / 12,325 = 89.0385% |
| MMLU-Pro（仅可评分分母） | 10,209 / 12,030 = 84.8628% |

- 合并目录：`/data/models/test/qtopomoe_quality_runs/full_official_bf16_tp4_pair_merged_v1`
- 合并 JSONL SHA-256：`98b10220241353f28e8ab31a3d2a17178ef8a613a565a88edc49bdb14863b1f2`
- 合并摘要 SHA-256：`7c264d35f5bbcdbd71833a3f6c667b59326dd1705350177668c1a6a7caf8eb7a`
- Gate：完整性和请求 Gate 接受；总体为 `closed_with_unfinished`。

## 7. 三格式共同分母最终比较

三种格式各自的显式未完成数不同（BF16 19、FP8 26、NVFP4 18），因此最终
格式比较只使用三者均可评分的 24,330 个 ID。比较由
[`scripts/compare_fullset_quality_results.py`](../../scripts/compare_fullset_quality_results.py)
执行，并校验每个输入的 ID 唯一性、身份字段、完整性及共同分母。

| 基准 | 共同 ID | BF16 | FP8 | NVFP4 | FP8−BF16 | NVFP4−BF16 |
|---|---:|---:|---:|---:|---:|---:|
| 总体 | 24,330 | 86.9955% | 86.9749% | 86.3009% | -0.0206 pp | -0.6946 pp |
| C-Eval | 12,306 | 89.0704% | 89.3223% | 88.4853% | +0.2519 pp | -0.5851 pp |
| MMLU-Pro | 12,024 | 84.8719% | 84.5725% | 84.0652% | -0.2994 pp | -0.8067 pp |

总体共同分母上的正确数为 BF16 21,166、FP8 21,161、NVFP4 20,997。
BF16 与 FP8 只差 5 题：FP8 在 C-Eval 略高，BF16 在 MMLU-Pro 略高，两个方向
基本抵消。因此只能表述为“本冻结协议下总体近似持平”，不能把 0.0206 点差异
夸大为稳定优劣。NVFP4 比 BF16/FP8 分别低 0.6946/0.6741 点，但仍通过项目
预注册的“相对 BF16 不超过 1.5 个绝对百分点”质量门槛。

共同分母正确性组合如下：

| BF16/FP8/NVFP4 正确性 | 样本数 |
|---|---:|
| 三者都正确 | 20,048 |
| 仅 BF16、FP8 正确 | 534 |
| 仅 BF16、NVFP4 正确 | 311 |
| 仅 BF16 正确 | 273 |
| 仅 FP8、NVFP4 正确 | 317 |
| 仅 FP8 正确 | 262 |
| 仅 NVFP4 正确 | 321 |
| 三者都错误 | 2,264 |

- 比较结果目录：`/data/models/test/qtopomoe_quality_runs/full_official_three_format_compare_v1`
- 比较 JSON SHA-256：`2e34750125da1f420b07952e144b88cd93ab906919ab894422a993b6a2e753c9`
- 输入 JSONL SHA-256：BF16 `98b10220241353f28e8ab31a3d2a17178ef8a613a565a88edc49bdb14863b1f2`；
  FP8 `6dc2df8a92f83eabd657d43cf1e77aa12a7c2088a40ed3061c50d45c3eb4851f`；
  NVFP4 `d8e17e7106b2575ff1ac7b020ddc7d93f7d3d5698794310b163b959d75601920`。

机器可读的最终摘要见
[`Q-TopoMoE_Phase3_BF16_FP8_NVFP4_质量总结_20260816.json`](../Q-TopoMoE_Phase3_BF16_FP8_NVFP4_质量总结_20260816.json)。
以上结论只适用于冻结的 24,374 条协议、当前 checkpoint、tokenizer/chat template、
seed 与采样参数；不得与不同 harness 的官方模型卡分数直接混写。
