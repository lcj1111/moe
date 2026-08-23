# Phase 8 selector 独立 Gate 失败报告

> 后续状态：2026-08-23 项目另行批准了
> [12%后续准入政策](phase8_selector_posthoc_12pct_policy_20260823.md)。本报告仍是原始10%
> 预注册 Gate 的历史结论；按新政策，当前结果可进入有限 canary。

## 1. 最终结论

2026-08-23，冻结的 Phase 8 selector v3 完成了与训练 workload 参数指纹不重叠的
独立验证。正式矩阵包含 45 个独立 workload、4 个候选、每候选 5 次重复，共
`45 × 4 × 5 = 900` 个正式测量单元。20 个重复轮次全部为 `workload_passed`，
聚合输入 Gate 全部通过，但 selector 的独立 **p95 regret 为 11.499972%**，超过
预注册的 10% 上限，因此最终状态为 `gate_failed`。

本结论表示“冻结 selector 暂不具备在线准入资格”，不表示量化模型、推理服务、
900 次基准测量或项目此前阶段失败。按照冻结协议，本轮不得降低门槛，不得利用
独立测试 outcome 修改该 selector，也不得继续执行有限 canary。

| 指标 | 独立结果 | 门槛 | 判定 |
|---|---:|---:|---|
| median regret | 0.000000% | ≤5% | 通过 |
| p95 regret | 11.499972% | ≤10% | **失败** |
| 决策开销 p95 | 0.114000% | <1% | 通过 |
| 不可行配置误选率 | 0.000000 | =0 | 通过 |
| Top-1 准确率 | 71.111111% | 审计项 | 记录 |

p95 regret 超限约 1.499972 个百分点。四项正式门槛中只有这一项失败。

## 2. 输入与测量完整性

独立聚合状态为 `accepted`，以下输入 Gate 全部为真：

- 900 个正式单元全部存在，20 个重复轮次全部完成；
- 每个候选均完成 5 次重复；
- 28,800 个请求全部返回且 `failed=0`；其中15个请求达到冻结的输出 token 上限，
  记录为 `finish_reason=length`，其余28,785个为 `stop`；
- 输入 token 精确、服务端 prompt token 可核验；
- chat template 唯一且哈希一致；
- prefix-cache 比例 Gate 全部通过；
- Poisson/burst 到达调度 Gate 全部通过；
- EP rank 与候选配置一致；
- 45 个决策前状态窗口状态为 `accepted`；
- 训练集与独立测试集的 workload ID、参数指纹不重叠，selector 在读取独立
  outcome 前已冻结。

这15个 `length` 分布在15个测量单元中，每个单元均为1/32个请求达到预注册输出
上限，主要出现在 FP8 TP2 的 `t4_c8_p0_*` 384-token 输出负载；聚合器将其视为
完成请求，未触发输入 Gate 失败。它不同于质量评测中的答案截断，不能在结果形成后
单独提高上限补跑，否则会改变冻结性能 workload。报告保留该事实，不将其写成“截断0”。

因此，本次失败不是由缺失样本、请求失败、错误 EP 数量或输入 Gate 异常造成，
而是冻结规则在独立分布上的长尾泛化误差。即使把15个达到输出上限的请求列为附加
审计项，当前正式 Gate 仍明确拒绝 selector，不能据此放宽准入。

## 3. 长尾误选审计

主要超限样本集中在 4096-token、并发 4 的独立 workload：

| workload | 到达/缓存 | selector 选择 | oracle | regret |
|---|---|---|---|---:|
| `t3_c4_p100_poisson` | Poisson / 100% | NVFP4 EP8 | FP8 TP2 | 32.237180% |
| `t3_c4_p100_burst` | burst / 100% | NVFP4 EP8 | FP8 TP2 | 20.946357% |
| `t3_c4_p0_poisson` | Poisson / 0% | NVFP4 EP8 | NVFP4 EP4 | 11.862518% |
| `t1_c4_p50_closed_loop` | closed-loop / 50% | NVFP4 EP4 | FP8 TP2 | 10.049790% |

前三个 `t3` 样本均命中 `medium_default`，说明训练侧规则在中等输入长度、低并发、
Poisson/burst 与高缓存组合上过度偏向 NVFP4 EP8。该现象可以用于失败归因和下一版
训练设计，但不能反向修改本轮已冻结 selector。

## 4. 准入边界

- 训练 Gate：v3 已通过，结论仍有效；
- 独立 900 次测量：完整、有效、可审计；
- 独立 selector Gate：失败；
- 有限 canary：禁止启动；
- trigger、cooldown、rollback 在线闭环：继续禁用；
- 冻结 selector v3：保留为失败审计证据，不得按本独立集标签重拟合。

若继续改进，应创建新的训练版本，只使用训练侧数据、先验规则或新增训练测量完成拟合，
然后冻结新 selector，并生成一套未被查看过的新独立测试集。当前45个独立 workload
不能再次充当新版本的最终独立 Gate。

## 5. 服务恢复

为完成缺失的3个重复轮次，测试窗口临时暂停了 GPU4–7 上的 SGLang。900次测量、
聚合和独立 Gate 结束后，守护流程已恢复原服务：端口 `30002` 重新监听，`/health`
返回 HTTP 200，GPU4–7 上的四卡服务进程恢复正常。GPU0–3 已释放。

## 6. 证据与哈希

- [独立 Gate 完整 JSON](../Q-TopoMoE_Phase8_selector_independent_gate_failed_20260823.json)
- [本轮发布清单](../Q-TopoMoE_Phase8_selector_independent_manifest_20260823.json)
- [冻结 selector](../Q-TopoMoE_Phase8_selector_v3_frozen_20260816.json)
- [v3 训练 Gate 通过报告](phase8_selector_v3_training_gate_accepted_20260816.md)
- [独立实验计划](../../configs/experiments/phase8_selector_independent_v1.json)
- [独立 workload](../../configs/workloads/phase8_selector_independent_v1.json)

独立 Gate JSON SHA-256：

`6f1d79571cb3366fdc581c6b518a7d25ed1aeaad89d4fbfd05e08ef2ffda95e4`
