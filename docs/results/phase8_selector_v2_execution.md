# Phase 8 selector v2 执行说明

## 1. 为什么不能直接继续调旧的 108-cell 选择器

旧版最近邻 Gate 在四候选、108 个 cell、每候选五次重复的已接受测量矩阵上得到：

- median regret：0%；
- p95 regret：43.17%；
- 要求：median ≤5%、p95 ≤10%；
- 结论：测量矩阵有效，但 selector 不可上线。

旧聚合只保留 workload 形状、前缀目标比例、到达模式和冻结请求率。它没有完整保留
真实缓存命中、服务端队列、客户端背压、近期到达率和短窗口 p99。直接在同一个
108-cell 矩阵上反复调参，还会把测试标签间接泄漏到模型选择中。

## 2. v2 的数据流

```text
固定 incumbent 服务
  → 独立 pre_decision 请求窗口
  → /metrics 服务端队列与 KV-cache 遥测
  → 客户端缓存命中、背压、近期到达率、短窗 p99
  → selector_state.v1
  → 绑定已有训练 outcome（不改写原始结果）
  → 只用训练集拟合并冻结 selector 配置
  → 生成参数指纹不重叠的新 workload
  → 对新 workload 测全部候选 oracle
  → 独立 selector Gate
```

普通候选 benchmark 的状态被标记为 `post_workload` 和
`decision_eligible=false`，只能用于审计。只有单独运行的决策前窗口会标记为
`pre_decision` 和 `decision_eligible=true`。

## 3. 新增字段

`clients/smoke.py` 现在额外输出：

- `service_telemetry.num_requests_waiting`：vLLM 服务端等待队列；
- `service_telemetry.num_requests_running`：服务端运行请求数；
- `service_telemetry.kv_cache_usage_perc`：KV-cache 使用率；
- `selector_state.actual_cache_hit_ratio`：决策窗口真实缓存 token 比例；
- `selector_state.client_queue_delay_ms`：客户端 worker pool 背压，不能称为服务端队列；
- `selector_state.recent_arrival_rate_rps`：窗口内实际到达率；
- `selector_state.short_window_*_p99_ms`：服务时钟与到达时钟短窗 p99。

Prometheus 指标按稳定后缀匹配，并保留采样覆盖率。缺失指标保持 `null`，不会按 0
处理。正式决策前窗口要求服务端队列指标存在且采样成功率至少 90%。
closed-loop 没有 open-loop worker 调度队列，因此其客户端排队字段为“不适用”；
距离计算仅在训练与测试双方都不适用时跳过该维度，不会把它伪造为零。

## 4. 执行命令

在固定 incumbent 已启动且健康检查通过后：

```bash
python3 scripts/run_phase8_predecision_windows.py \
  --matrix configs/workloads/phase8_formal_controlled_v1.json \
  --base-url http://127.0.0.1:31540/v1 \
  --model qtopomoe-phase8-selector-incumbent-fp8 \
  --tokenizer /data/models/test/models/Qwen--Qwen3.6-35B-A3B-FP8/snapshots/master \
  --incumbent-candidate-id fp8_tp2_pix01_triton \
  --cache-block-tokens 1056 \
  --output-root /data/models/test/qtopomoe_phase8_selector_training_prewindows_v1_20260816 \
  --python-bin /data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129/bin/python3
```

把状态绑定到已有训练 outcome：

```bash
python3 scripts/attach_phase8_predecision_state.py \
  --aggregate /data/models/test/qtopomoe_phase8_formal_controlled_combined_v1/aggregate.json \
  --predecision-states /data/models/test/qtopomoe_phase8_selector_training_prewindows_v1_20260816/predecision_states.json \
  --output /data/models/test/qtopomoe_phase8_selector_training_v2/aggregate_with_state.json
```

只用绑定后的训练数据拟合并冻结配置：

```bash
python3 scripts/fit_phase8_selector_state.py \
  --training-aggregate /data/models/test/qtopomoe_phase8_selector_training_v2/aggregate_with_state.json \
  --template configs/strategies/phase8_selector_state_v2.template.json \
  --output-config /data/models/test/qtopomoe_phase8_selector_training_v2/selector.training_gate_failed.json \
  --output-report /data/models/test/qtopomoe_phase8_selector_training_v2/selector.fit_report.json \
  --search-profile formal_v2_extended
```

拟合使用 W1/W2/W3/W4 留族交叉验证和固定的组权重网格。只有训练集 median/p95
regret Gate 均通过才会输出 `status=frozen`；失败时禁止读取独立测试结果。

正式独立 Gate 必须使用冻结配置：

```bash
python3 scripts/evaluate_phase8_independent_selector.py \
  --training-aggregate /path/to/training_with_state.json \
  --test-aggregate /path/to/disjoint_test_with_state.json \
  --selector-config /path/to/frozen_selector.json \
  --output /path/to/independent_gate.json
```

模板 `configs/strategies/phase8_selector_state_v2.template.json` 的状态是
`draft_not_fitted`，评估器会拒绝它。必须先只用训练数据确定特征权重并改成
`frozen`；一旦读取独立测试结果，不得再回调权重。

独立矩阵由 `build_phase8_independent_workload.py` 根据
`phase8_selector_independent_design_v1.json` 生成。每个新 shape 取训练矩阵中两个
最近 shape 的较小冻结速率，再乘 0.8 并向下保留六位小数；该规则只读取训练资料。

## 5. 正式 Gate

- 训练与测试 `workload_id` 不重叠；
- 输入、输出、并发、前缀、到达模式、请求率组成的参数指纹不重叠；
- 所有 selector 状态均来自决策前窗口；
- 服务端队列遥测真实可用；
- median regret ≤5%；
- p95 regret ≤10%；
- 决策开销占服务 p99 <1%；
- 不可行配置误选率为 0。

全部通过前，动态 trigger、cooldown 和 rollback 继续保持禁用。

## 6. 2026-08-16 训练 Gate 结论

决策前窗口已完成 108/108，状态和服务端遥测 Gate 均为 `accepted`。正式扩展搜索
完成 2,500 组训练侧组合，最佳 selector 的 median regret 为 0%，但 p95 regret 为
23.0488%，未达到 ≤10% 的门槛，因此训练 Gate 状态为 `failed`。

脚本按设计写出失败配置与完整报告后返回非零退出码；这表示准入被正确拦截，并非脚本
运行异常。由于训练 Gate 未通过，冻结的 45-cell × 4 候选 × 5 重复（900 次）独立
测试没有启动，独立标签也没有被读取。

完整原因、折间误差、搜索空间和后续边界见
[训练 Gate 失败报告](phase8_selector_training_gate_failed_20260816.md)。
