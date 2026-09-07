# Q-TopoMoE Phase 8 暖态 placement 最终验收

日期：2026-08-25

## 最终结论

新暖态候选 `warm_swap_008_slots_per_layer_v1` 已完成当前 Runbook 要求的剩余验证，最终状态为
`accepted`：

- 当前路由稳定性上限为 1%；短臂 0.045%、长臂 0.718%，均通过。
- 有限 canary 完成 384/384 请求，零失败，三段请求流一致，8/8 rank 提交相同 map 哈希；
  recovery p99 为 stable 的 61.65%，通过 105% 上限。
- 自动闭环完成 20 个窗口、2560 个长生成请求，全部零失败并固定输出；三窗口 trigger、
  十窗口 cooldown、三窗口 rollback、8-rank apply 与 identity rollback 均按预注册时序通过。
- 决策开销 p95 为 0.000102%，低于 1%；回滚后 p99 为基线的 61.61%，低于 105%。

这表示 Phase 8 技术验证链已经完成，但不表示已执行生产部署。扩大流量、长期运行或生产切换
应作为独立运维变更处理。

本页采用后续 12% selector 和 1% 路由政策；它们与最初目标的差异见
[准入政策版本](phase8_benchmark_history.md#准入政策版本)。

## 指标解释

- recovery/stable p99 和回滚后 p99/基线用于检查恢复是否超过 105% 上限，
  不作为 placement 的加速比。各阶段按时序执行，预热、缓存和运行状态的影响未由
  此项验收单独排除；证明性能收益需要匹配条件下的多轮对照。
- 0.000102% 的分子仅为控制器 `observe_cv` 调用时间，分母为对应负载窗口的
  `wall_time_s`，对各窗口比例取 p95。不包含路由采集、候选求解、模型推理和迁移耗时，
  与独立 selector 测试的决策开销指标不能直接比较。
- 闭环计划固定了候选收益和迁移成本输入；本轮验证控制状态机与运行时提交。
  观测值故障注入不等同于真实服务故障或 rank 掉线测试。
- canary 的 `finish_reason=length` 来自固定输出长度和 `ignore_eos=true` 的负载协议；
  不等同于质量评测中要求补跑的意外截断。

## 验收关系

```text
质量 A/B/A accepted
    + 当前 1% 路由稳定性 Gate accepted
        └─ 有限 canary
            ├─ 384/384、failed=0
            ├─ 显式激活前无 placement 应用
            ├─ 8/8 rank 同哈希提交
            └─ recovery/stable p99 = 61.65% → accepted
                └─ 自动闭环
                    ├─ trigger: hold → hold → rebalance
                    ├─ generation=1 apply: 8/8 rank
                    ├─ cooldown: 连续 10 窗 hold
                    ├─ fault: hold → hold → rollback
                    ├─ generation=2 identity rollback: 8/8 rank
                    └─ 回滚后 p99/基线 = 61.61% → accepted
```

## 有限 canary

冻结计划为
[phase8_warm_placement_limited_canary_v1.json](../../configs/experiments/phase8_warm_placement_limited_canary_v1.json)，
SHA-256 为 `62e4e3eea760cd99b9e0a1a69c47fadef6bd86f062c67a3f559dcb4ce41c39fd`。
正式结果目录为
`/data/models/test/qtopomoe_phase8_warm_placement_limited_canary_v1_20260825`，
`canary_gate.json` SHA-256 为
`5e3eb834c3f7cf74f90cd0e1f7ae156795a648a8f26be5c770afc058495631a5`。

| 指标 | 结果 | Gate |
|---|---:|---|
| stable / migration / recovery 请求 | 128 / 128 / 128 | 三段均完成 |
| 失败数 | 0 / 0 / 0 | 通过 |
| finish reason | 三段均 128 条 `length` | 通过 |
| 请求流 | 六个身份字段逐条一致 | 通过 |
| placement apply | call=2，8/8 rank，同一 map 哈希 | 通过 |
| stable p99 | 5795.27 ms | 基线 |
| migration p99 | 4150.09 ms | 记录 |
| recovery p99 | 3572.80 ms | 记录 |
| recovery / stable | 61.65% | ≤105%，通过 |

## trigger、cooldown、rollback 自动闭环

冻结计划为
[phase8_warm_placement_closed_loop_acceptance_v1.json](../../configs/experiments/phase8_warm_placement_closed_loop_acceptance_v1.json)，
SHA-256 为 `bef8884df0f16bd363e1019fa6283dfb7ba7ee5b03f0ff9448bad07c0f9e70f1`。
正式结果目录为
`/data/models/test/qtopomoe_phase8_warm_placement_closed_loop_acceptance_v1_20260825`，
`closed_loop_gate.json` SHA-256 为
`97933e65892d27076e3be2e0d59b8829fb8591ace29cb8cb179b4ec44692255d`。

| Gate | 结果 |
|---|---|
| 输入有限 canary | accepted |
| 三窗口 trigger | `hold → hold → rebalance` |
| generation=1 apply | 8/8 rank |
| 十窗口 cooldown | 全部 `hold`，末窗 remaining=0 |
| 三窗口退化观测 | `hold → hold → rollback` |
| generation=2 identity rollback | 8/8 rank |
| 请求完整性 | 20 窗 × 128 = 2560，零失败、固定输出 |
| observe 调用耗时 / 窗口耗时 p95 | 0.000102%，低于本轮 1% 上限；非全链路开销 |
| 回滚后 p99 / 基线 | 3586.07 / 5820.90 = 61.61%，低于 105% |

故障注入只将控制器看到的 p99 放大到冻结参照的 106%，没有延迟或丢弃真实请求；因此它验证
的是自动 rollback 信号链路和提交行为，不把故障注入值当作真实服务延迟。

## SGLang 与最终边界

用户明确要求“不用管 SGLang”，因此正式 canary 和闭环均不启动、不暂停、不恢复 30002，
也不以其状态判定 Gate。实验 vLLM 在每轮结束后退出，最终 GPU 无残留计算进程。

机器摘要见
[Q-TopoMoE_Phase8_warm_placement_final_acceptance_20260825.json](../Q-TopoMoE_Phase8_warm_placement_final_acceptance_20260825.json)。
