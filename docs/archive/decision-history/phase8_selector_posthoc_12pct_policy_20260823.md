# Phase 8 selector 12% 后续准入政策

## 1. 当前结论

2026-08-23，项目批准将 Phase 8 selector 后续运行政策中的 p95 regret 上限由
10% 调整为 12%。该变更发生在独立900次测量结果揭晓之后，因此属于后续政策修订，
不是最初预注册门槛的回溯修改。

在新政策下，冻结 selector v3 的独立 p95 regret 为 11.499972%，满足不超过12%的
要求；其他三项 Gate 也全部满足，因此状态为
`accepted_under_posthoc_12pct_policy`，允许进入有限 canary。

| 指标 | 独立结果 | 后续政策门槛 | 判定 |
|---|---:|---:|---|
| median regret | 0.000000% | ≤5% | 通过 |
| p95 regret | 11.499972% | ≤12% | 通过 |
| 决策开销 p95 | 0.114000% | <1% | 通过 |
| 不可行配置误选率 | 0.000000 | =0 | 通过 |
| Top-1 准确率 | 71.111111% | 审计项 | 记录 |

## 2. 与原始 Gate 的关系

- 原始预注册门槛仍为10%，原始独立 Gate 的 `gate_failed` 结论保持不变；
- 新政策不修改冻结 selector、独立 workload、900次原始测量或原始 Gate JSON；
- 同一批独立数据只用于评估经批准的新运行政策，不能再声称为未揭晓独立验证；
- “原始 Gate 失败”和“按12%后续政策准入”是两个不同时间点、不同政策语义的结论。

保留这一区分，可让后续复现者同时确认实验原貌和当前运行决策。

## 3. 当前准入边界

- 有限 canary：允许执行；
- 全量上线：尚未允许；
- trigger、cooldown、rollback 自动闭环：尚未启用；
- 只有 canary 达到请求零失败、计划哈希一致、恢复 p99 不超过基线105%且
  rollback 可用，才能继续验收自动闭环。

## 4. 证据

- [12%后续准入政策 JSON](../../../configs/strategies/phase8_selector_gate_policy_v2_20260823.json)
- [原始独立 Gate JSON](../../Q-TopoMoE_Phase8_selector_independent_gate_failed_20260823.json)
- [原始独立 Gate 报告](phase8_selector_independent_gate_failed_20260823.md)
- [冻结 selector](../../Q-TopoMoE_Phase8_selector_v3_frozen_20260816.json)
- [独立实验计划](../../../configs/experiments/phase8_selector_independent_v1.json)
