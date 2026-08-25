# Phase 8 selector v2 训练 Gate 失败报告

## 1. 结论

2026-08-16，Phase 8 selector v2 完成正式训练侧搜索与留族交叉验证。训练流程正常完成，
但最终模型没有达到预注册的 p95 regret 门槛，因此状态为 `failed`，不得冻结为生产
selector，也不得启动独立测试或在线闭环。

这不是项目整体失败。Phase 8 的 108 个 workload 测量矩阵、决策前状态采集和遥测覆盖
均已通过；失败范围仅限于“依靠当前状态特征自动选择四种候选策略”的上线准入 Gate。

| 指标 | 实测值 | 门槛 | 结果 |
|---|---:|---:|---|
| median regret | 0.0000% | ≤5% | 通过 |
| p95 regret | 23.0488% | ≤10% | **失败** |
| 最大 regret | 84.5344% | 审计项 | 未作为准入门槛 |
| Top-1 命中率 | 70.3704% | 审计项 | 未作为准入门槛 |

## 2. 数据与执行边界

训练输入是已接受的四候选、108 cells、每候选五次重复的正式测量结果，并绑定同一
workload 的独立决策前窗口。决策前审计结果如下：

- workload 状态：108/108；
- `state_phase=pre_decision`：108/108；
- `decision_eligible=true`：108/108；
- 服务端遥测覆盖率：100%；
- 服务端等待队列指标：108/108 可用；
- 缓存命中率范围：0 至 0.9990234375；
- 等待队列 p95 范围：0 至 71；
- KV-cache 使用率 p95 范围：0 至 0.5785310734463277；
- 决策前状态 Gate：`accepted`。

训练聚合文件 SHA-256：
`313676a297ddfe40014f3376bd0e5fde36f628aa7ab9363a4b9f67a19e9b3227`。

正式训练命令为：

```bash
python3 scripts/fit_phase8_selector_state.py \
  --training-aggregate /data/models/test/qtopomoe_phase8_selector_training_v2/aggregate_with_state.json \
  --template configs/strategies/phase8_selector_state_v2.template.json \
  --output-config /data/models/test/qtopomoe_phase8_selector_training_v2/selector.training_gate_failed.json \
  --output-report /data/models/test/qtopomoe_phase8_selector_training_v2/selector.fit_report.json \
  --search-profile formal_v2_extended
```

脚本在写出完整报告与失败配置后以退出码 1 结束。这是 Gate 失败的预期行为，不是运行
异常；本地与 gpu-111 服务器复现所得最佳配置和指标完全一致。

## 3. 搜索方法与最佳配置

正式搜索只读取训练数据，采用 W1/W2/W3/W4 留族交叉验证，共评估 2,500 组组合：

- selector：带成本函数的 KNN；
- 邻居数：3、5、7、9、15；
- 投票：均匀或距离倒数；
- cost：原始 regret 或按 oracle 真值归一化；
- 形状、到达、缓存、队列、KV 和短窗 p99 特征采用预注册权重网格；
- 缺失值不伪造为零；双方都不适用时跳过该维度，只有单侧缺失时使用显式惩罚。

最佳训练侧配置：

| 项目 | 值 |
|---|---|
| selector | `telemetry_aware_knn_cost` |
| 邻居数 | 7 |
| 投票方式 | 距离倒数 |
| cost 归一化 | 按 oracle 真值归一化 |
| 形状/到达/缓存/队列/KV 权重 | 1/1/1/1/1 |
| 短窗 p99 权重 | 16 |

各留族折的 p95 regret：

| 留出族 | p95 regret |
|---|---:|
| W1 | 23.6668% |
| W2 | 46.8761% |
| W3 | 14.2955% |
| W4 | 0.0000% |

W2 的尾部误差最明显，说明当前观测状态下，跨 workload 家族的最优候选边界仍不可稳定
分离。继续扩大同一训练矩阵上的搜索，或降低门槛，都会增加过拟合和错误上线风险。

## 4. 与前序 selector 的关系

本轮方法相对旧 selector 已有明显改善，但仍不足以上线：

| 版本 | p95 regret | 定位 |
|---|---:|---|
| 旧版最近邻 | 43.17% | 已拒绝历史基线 |
| 显式空值语义后的单邻居 | 31.9762% | 诊断迭代 |
| 标准 KNN 网格 | 27.4005% | 诊断迭代 |
| 正式扩展 KNN | 23.0488% | 本次正式 Gate，失败 |

Ridge 与浅层成本树只用于训练侧诊断，没有被选为正式模型，也没有接触独立测试标签。

## 5. 独立测试与在线控制状态

独立测试方案已经冻结并通过 dry-run：45 个新 cells、四个候选、每候选五次，共 900 次
测量。其 workload ID 和参数指纹与训练集不重叠。

由于训练 Gate 失败，本轮严格执行以下隔离规则：

- **没有启动独立 900 次测试**；
- `independent_test_read=false`；
- 没有利用独立集结果回调权重；
- 没有生成 `status=frozen` 的生产 selector；
- 动态 trigger、cooldown 和 rollback 继续禁用；
- 现有服务策略不受失败配置控制。

## 6. 后续建议

下一轮不应继续在同一 108-cell 标签上盲目扩网格。优先级如下：

1. 增加真正能在决策时获得的候选区分信号，例如候选轻量探针、通信压力和专家负载
   分布摘要；
2. 预注册新的训练协议与门槛，再采集新增训练窗口；
3. 仅在新的训练 Gate 同时满足 median ≤5%、p95 ≤10% 后，冻结 selector；
4. 冻结后才执行已经准备好的 900 次独立 Gate；
5. 独立 Gate 通过后，才启用动态 trigger、cooldown 和 rollback 实验。

## 7. 证据文件

- [决策前状态审计](../../Q-TopoMoE_Phase8_selector_v2_predecision_audit_20260816.json)
- [训练 Gate 失败配置](../../Q-TopoMoE_Phase8_selector_v2_training_gate_failed_20260816.json)
- [完整拟合报告](../../Q-TopoMoE_Phase8_selector_v2_fit_report_20260816.json)
- [selector v2 执行说明](phase8_selector_v2_execution.md)
