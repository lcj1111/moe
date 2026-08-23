# Phase 8 暖态 placement 候选重新生成报告

## 结论

2026-08-23 已停止调整旧 Gate，并把旧 selector、旧 placement map 和旧失败结果全部保留为
历史只读证据。本轮从最终代表负载的真实暖态逐层专家计数重新生成 placement 候选。

新候选 `warm_swap_008_slots_per_layer_v1` 已成功生成，但同进程暖态 A/B/A/B Gate 最终为
**rejected**，不得进入自动闭环或上线。拒绝不是因为 p99：候选的 p99 中位数相对 identity
为 101.83%，满足 105% 上限；rank CV 也实际改善 36.55%。阻塞项是逻辑专家负载分布相对
identity 系统性变化 14.28%，超过冻结的 10% 稳定范围，同时精确输出哈希 Gate 未通过。

## 执行关系

```text
旧 Gate / 旧 map 冻结
  └─ identity 暖态采集（48 个窗口，accepted）
      └─ 限制迁移量候选生成（8/16/32 槽位每层 + full LPT）
          └─ 选择满足离线改善门槛且迁移量最小的 8 槽位候选
              └─ 同进程 identity → candidate → identity → candidate
                  ├─ rank CV、p99、请求与 8-rank 提交通过
                  └─ 逻辑专家分布稳定性、精确输出哈希失败 → rejected
```

## 暖态采集

正式有效采集位于：
`/data/models/test/qtopomoe_phase8_warm_placement_capture_v3_20260823`。

| 项目 | 结果 |
|---|---:|
| 导出窗口 | 48 |
| 请求 | 4 阶段 × 128 |
| failed | 0 |
| 固定输出 | 每请求 384 tokens |
| identity map 哈希 | `46a5de9abf04b3a6d4ae62897b2fe17dc159adf01069a7a99b8b117ec5795671` |
| 暖态窗口文件哈希 | `13b418189b2f917595a7639031e7d4f9fa3d53e984f6f41995c65a258acfde12` |
| 原 SGLang 恢复 | 端口 30002 返回 200，仅 GPU4–7 占用 |

前两个尝试没有进入候选数据：v1 因服务器旧版请求客户端不支持 `ignore_eos` 而停止；
v2 发现 identity/hold 模式没有推进旧 rebalance 调用计数器，无法触发定期导出。两次均使用
独立目录、没有可用窗口，并在退出后恢复原服务。v3 改为独立负载窗口计数器后通过。

## 候选生成

生成器比较每层最多移动 8、16、32 个物理槽位的候选及 full LPT。离线门槛要求逐层
rank CV 中位数至少改善 20%、p95 至少改善 15%；满足门槛后选择迁移槽位最少者。

最终选择每层 8 槽位候选：总计移动 320/10240 个槽位，即 3.125%，明显小于旧 map
接近全量重排的范围。

| 指标 | identity | 新候选 | 候选/identity |
|---|---:|---:|---:|
| 逐层 rank CV 中位数 | 0.3022 | 0.0467 | 15.44% |
| 逐层 rank CV p95 | 0.4160 | 0.0899 | 21.61% |
| 全局 rank CV | 0.0674 | 0.0099 | 14.73% |

候选 map 位于
[phase8_warm_swap_008_slots_per_layer_v1.json](../../configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json)，
文件哈希为 `3386d1457a62f4c2a9cb5cdc56b71225c8f07fae73063a0e51c142f4c0b1f29e`，
map 哈希为 `24f09c93ff8e6c410917a3fcb087307130f8914f45c028adf23088c39fc80ac6`。

## A/B/A/B 在线结果

测量严格使用同一服务进程和顺序 `identity A1 → candidate B1 → identity A2 → candidate B2`。
每次 apply/rollback 后先运行独立预热段，预热和迁移提交时延不计入测量段。

| 指标 | identity 两轮 | candidate 两轮 | 结果 |
|---|---:|---:|---|
| p99 ms | 3385.42、3519.11 | 3531.35、3499.59 | 中位数比 101.83%，通过 |
| rank CV | 0.07566、0.07676 | 0.04870、0.04801 | 比值 63.45%，通过 |
| 逻辑专家 CV | 2.1733、2.1850 | 1.8879、1.8480 | 比值 85.72%，超出 90%–110%，失败 |
| 请求与固定输出 | 8 阶段均 failed=0 | 每请求 384 tokens | 通过 |
| 控制提交 | generation 1 apply、2 rollback、3 apply | 每代均 8/8 rank | 通过 |
| 原服务恢复 | 30002=200，GPU4–7 | 正常 | 通过 |

精确输出哈希为失败项，但单独看它不能证明迁移造成语义错误：identity A1 与 identity A2
也是 0/128 条完全一致，说明这套长生成在当前分布式运行时不是逐 token 位级确定性的。
冻结 Gate 没有因此被事后修改，结果仍保持 rejected。逻辑专家 CV 在 identity 两轮内部很稳定，
候选两轮内部也稳定，但两组之间存在 14.28% 的系统性差异，因此仍有足够理由阻止准入。

## 当前边界与下一步

1. 旧 Gate、旧 selector、旧 map 均不再调整。
2. 新候选已生成并证明能改善 rank 负载，但尚未证明语义/路由等价，不得启用自动闭环。
3. 下一步应建立独立质量等价验证：使用有标准答案的冻结 official-like 子集，比较 identity 与
   candidate 的答案抽取、分科准确率、failed、truncated 和 finish_reason；不能再使用无语义的
   长 `x` 提示输出哈希作为唯一正确性依据。
4. 只有质量等价和逻辑路由稳定性通过后，才为这个新候选设计新的 cooldown/rollback Gate；
   不重开或改写旧 Gate。

机器可读摘要见
[Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json](../Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json)。
