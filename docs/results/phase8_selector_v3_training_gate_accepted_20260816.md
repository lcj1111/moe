# Phase 8 selector v3 训练 Gate 通过报告

## 1. 结论

2026-08-16，Phase 8 selector v3 在不降低既有门槛的前提下完成训练侧改进与复现。
训练 p95 regret 从 v2 的 23.0488% 降至 3.4897%，满足 ≤10% 的准入标准；
median regret 为 0%，满足 ≤5% 的标准。冻结配置状态为 `frozen`，可以进入独立
45-cell、四候选、五重复的 900 次验证，但在独立 Gate 通过前仍不得驱动在线动态切换。

| 指标 | v2 | v3 | 门槛 | v3 结果 |
|---|---:|---:|---:|---|
| median regret | 0.0000% | 0.0000% | ≤5% | 通过 |
| p95 regret | 23.0488% | 3.4897% | ≤10% | 通过 |
| 最大 regret | 84.5344% | 6.3934% | 审计项 | 改善 |
| Top-1 命中率 | 70.3704% | 88.8889% | 审计项 | 改善 |

## 2. 改进内容

v2 使用成本敏感 KNN，在不同输入长度、并发、缓存和到达模式之间外推时产生明显
长尾误选。v3 改为可审计的有序工作负载区间规则：

- 只读取决策时可观测字段：输入长度、并发、目标缓存比例、真实缓存命中率和到达模式；
- 规则显式映射到 FP8 TP2、W4A16 EP4、NVFP4 EP4 或 NVFP4 EP8；
- 未命中规则时保守回退到 NVFP4 EP8；
- 规则顺序、ID、运算符、候选和回退候选在加载时一次性完整校验；
- 任何未知候选、重复规则 ID 或非法条件都会拒绝加载，而不是静默忽略。

这些规则来自 v2 失败后的**训练侧**误差分析，因此本报告只能证明训练 Gate 通过，
不能替代独立泛化验证。规则在读取独立测试 outcome 前已经冻结，冻结文件记录
`independent_test_read=false`；独立测试完成后禁止再根据测试标签修改规则。

## 3. 训练数据与复现

训练输入仍是已经接受的四候选、108-cell、每候选五重复聚合，并绑定 108 个固定
incumbent 决策前状态窗口。训练聚合 SHA-256 为：

`313676a297ddfe40014f3376bd0e5fde36f628aa7ab9363a4b9f67a19e9b3227`

本地与 gpu-111 使用同一输入和同一模板得到完全一致的指标。服务器复现命令：

```bash
/data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129/bin/python3 \
  /data/models/test/qtopomoe_phase8_selector_v3_code_20260816/scripts/fit_phase8_selector_state.py \
  --training-aggregate /data/models/test/qtopomoe_phase8_selector_training_v2/aggregate_with_state.json \
  --template /data/models/test/qtopomoe_phase8_selector_v3_code_20260816/configs/strategies/phase8_selector_state_v3.template.json \
  --output-config /data/models/test/qtopomoe_phase8_selector_training_v3/selector.frozen.json \
  --output-report /data/models/test/qtopomoe_phase8_selector_training_v3/selector.fit_report.json \
  --search-profile formal_v3_regime_rules
```

代码和模板放在独立实验目录中，没有拉取、重置或覆盖服务器 `/data/moe` 的脏工作区。

## 4. 留族审计

报告仍按 W1/W2/W3/W4 记录留族统计。需要注意：规则由全部训练 outcome 的误差分析
形成，所以这些折是训练侧稳健性审计，不应描述为未见标签的外部验证。

| 留出族 | 样本数 | median regret | p95 regret | 最大 regret |
|---|---:|---:|---:|---:|
| W1 | 36 | 0.0000% | 4.6193% | 6.2671% |
| W2 | 27 | 0.0000% | 0.0000% | 6.3934% |
| W3 | 27 | 0.0000% | 2.5481% | 5.5431% |
| W4 | 18 | 0.0000% | 0.0000% | 0.0000% |

## 5. 当前准入边界

- 训练 Gate：已通过并冻结；
- 独立 900 次 Gate：允许启动，尚未形成结论；
- 在线 selector：仍禁用；
- 动态 trigger、cooldown、rollback：仍禁用；
- 只有独立 Gate 同时满足 median ≤5%、p95 ≤10%、决策开销 <1%、不可行配置
  误选率为 0 后，才进入在线闭环实验。

## 6. 证据文件

- [v3 训练模板](../../configs/strategies/phase8_selector_state_v3.template.json)
- [v3 冻结 selector](../Q-TopoMoE_Phase8_selector_v3_frozen_20260816.json)
- [v3 完整拟合报告](../Q-TopoMoE_Phase8_selector_v3_fit_report_20260816.json)
- [v2 训练 Gate 失败报告](phase8_selector_training_gate_failed_20260816.md)
- [selector 执行说明](phase8_selector_v2_execution.md)

独立 Gate 的固定执行顺序由
[`run_phase8_independent_selector_gate.sh`](../../scripts/run_phase8_independent_selector_gate.sh)
编排；脚本只清理自身进程组，并把原始测量、聚合、绑定状态和最终 Gate 分目录保存。
