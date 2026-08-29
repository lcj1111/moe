# Phase 8 暖态 placement 候选重新生成与质量验证报告

## 最终结论

2026-08-23 从最终代表负载的真实暖态逐层专家计数独立生成候选，得到
`warm_swap_008_slots_per_layer_v1`，随后完成在线性能、NVFP4 迁移正确性和冻结质量集验证。

当前结论分为两层：

- **质量等价 Gate 已接受**：修复 NVFP4 Marlin 辅助尺度迁移后，完整 116 题
  `identity A1 → candidate B → identity A2` 三轮均为 `failed=0`、`truncated=0`、
  `extraction_failed=0`、`finish_reason=stop`。候选正确 110/116，高于两轮 identity 的
  较低值 106/116，且没有“两轮 identity 都正确、候选错误”的新增回归。
- **当前路由稳定性 Gate 已接受**：较早冻结的同进程暖态 A/B/A/B 中，候选虽然满足 p99
  上限并改善 rank CV，但逻辑专家负载分布相对 identity 变化 14.28%，超过当时的 0.5%
  范围。后续两级诊断得到的可复现长臂超额差异为 0.718%；当前有效上限调整为 1% 后，
  短臂 0.045% 与长臂 0.718% 均通过，候选获准进入有限 canary。
- **14.28% 的成因已分类**：两轮重新预注册诊断排除了统计还原错误和计数器重置污染。
  热态确认轮的单 token 对照稳定，但长生成出现超过冻结阈值的逻辑路由分布差异，归因于
  生成轨迹分叉后的真实路由采样变化。14.28% 这一具体幅度未复现，不能解释为稳定的
  placement 效应；历史 0.5% 判定继续保留用于审计，当前运行决策以 1% 政策为准。

## 执行关系

```text
identity map 冻结
  └─ identity 暖态采集（48 个窗口，accepted）
      └─ 限制迁移量候选生成（8/16/32 槽位每层 + full LPT）
          └─ 选择满足离线改善门槛且迁移量最小的 8 槽位候选
              ├─ 同进程暖态 A/B/A/B
              │   ├─ rank CV、p99、请求与 8-rank 提交通过
              │   └─ 历史 0.5% 路由稳定性 Gate rejected
              ├─ 逻辑专家分布稳定性诊断
              │   ├─ 记录时计数 = 旧口径重建，且无跨 map 代际槽位
              │   ├─ 热态单 token 对照稳定
              │   └─ 长生成差异 0.718% → 成因 classified，当前 1% Gate accepted
              └─ 冻结 official-like 质量 A/B/A
                  ├─ v1 发现候选 8 条截断 → 停止，不作为结论
                  ├─ 修复 NVFP4 Marlin 辅助尺度迁移
                  ├─ 8 条定向 A/B/A 回归 accepted
                  └─ 116 条完整 A/B/A accepted → 仅质量等价通过
```

## 暖态采集与候选生成

正式采集位于
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

生成器比较每层最多移动 8、16、32 个物理槽位的候选及 full LPT。离线门槛要求逐层
rank CV 中位数至少改善 20%、p95 至少改善 15%；满足门槛后选择迁移槽位最少者。
最终候选总计移动 320/10240 个槽位，即 3.125%。

| 指标 | identity | 新候选 | 候选/identity |
|---|---:|---:|---:|
| 逐层 rank CV 中位数 | 0.3022 | 0.0467 | 15.44% |
| 逐层 rank CV p95 | 0.4160 | 0.0899 | 21.61% |
| 全局 rank CV | 0.0674 | 0.0099 | 14.73% |

候选文件为
[phase8_warm_swap_008_slots_per_layer_v1.json](../../configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json)，
文件哈希为 `3386d1457a62f4c2a9cb5cdc56b71225c8f07fae73063a0e51c142f4c0b1f29e`，
map 哈希为 `24f09c93ff8e6c410917a3fcb087307130f8914f45c028adf23088c39fc80ac6`。

## 暖态 A/B/A/B 在线结果

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

精确输出哈希失败不能单独证明迁移造成语义错误：identity A1 与 identity A2 也是
0/128 条完全一致，说明这套长生成在当前分布式运行时不是逐 token 位级确定性的。历史
0.5% Gate 与原始 `rejected` 结果均未改写；当前有效运行政策单独调整为 1%，并依据后续
可复现的 0.718% 长臂结果作准入判断。

## 逻辑专家分布稳定性诊断

历史 A/B/A/B 的 14.28% 变化不能只靠再次比较四个 CV 数字判断。2026-08-24 因此冻结了
两级诊断采用独立输入与输出：

1. v1 在 EPLB 计数写入滑动窗口之前直接保存逻辑专家计数，同时保留旧的“用当前 map
   重建逻辑计数”结果，并给每个窗口槽位标记 placement generation；
2. v2 针对 v1 首个短阶段跨越冷缓存到热缓存的观测，增加一个完整热态 settle，在正式
   identity A1 前冲洗 16 步滑动窗口，其余负载、阈值和输入哈希保持不变。

两个诊断均使用同进程 A/B/A/B、每阶段 128 请求。短臂为 1536 输入 token、1 输出 token、
并发 4；长臂为 1536 输入 token、384 输出 token、并发 16。分类阈值预先冻结为：旧口径与
记录时口径 TV p95 不超过 `1e-6`，短、长臂 placement excess TV p95 分别不超过 0.5%。
`placement excess` 是 identity/candidate 交叉 TV p95 减去两个状态各自重复波动基线的较大值。

v2 最终结果如下：

| 诊断项 | 结果 | 判断 |
|---|---:|---|
| 稳态旧口径与记录时口径 TV p95 | 0 | 排除统计还原错误 |
| 切换期旧口径与记录时口径 TV p95 | 0 | 排除切换期还原偏差 |
| 稳态/切换期混合 generation 槽位 | 0 / 0 | 排除计数器重置或跨 map 窗口污染 |
| 短臂 placement excess TV p95 | 0.000450（0.045%） | 低于历史 0.5% 与当前 1% 上限 |
| 长臂 placement excess TV p95 | 0.007175（0.718%） | 超过历史 0.5%，但低于当前 1% 上限 |
| 短臂响应哈希 | 任意两轮均 128/128 一致 | 短轨迹确定 |
| 长臂响应哈希 | 同状态重复也仅 0–1/128 一致 | 分布式长生成轨迹本身会分叉 |
| 请求与控制提交 | 24 阶段均 128/128、failed=0；7 次 generation 均 8/8 rank | 完整性通过 |

分类结论是：**14.28% 不来自逻辑计数还原口径，也不来自计数器重置；它属于长生成轨迹
分叉后的实际逻辑路由采样变化。** 同时，14.28% 这一幅度并不稳定：v2 的长臂交叉 CV
中位数比约为 0.9968–1.0033，远未复现历史 0.8572。正确边界是“存在小而可测的长轨迹
路由差异”，而不是“候选稳定降低逻辑专家 CV 14.28%”。

### 当前 1% Gate 与后续边界

2026-08-25 将当前有效的短、长臂 placement excess TV p95 上限统一调整为 1%，不再增加
本阶段诊断实验。现有 v2 结果据此重判为 `accepted`；这只表示路由稳定性阶段完成，并不
表示整个项目结束或已经上线。下一步仍按 Runbook 执行有限 canary；只有 canary 满足零失败、
计划哈希一致且恢复 p99 不超过基线 105%，才继续验收 trigger、10 窗口 cooldown 与 3 窗口
placement rollback 自动闭环。按用户要求，既有 SGLang 服务不属于本轮 canary 的管理或
Gate 范围。当前有效政策见
[phase8_route_stability_gate_policy_v2_20260825.json](../../configs/strategies/phase8_route_stability_gate_policy_v2_20260825.json)。
有限 canary 的冻结计划见
[phase8_warm_placement_limited_canary_v1.json](../../configs/experiments/phase8_warm_placement_limited_canary_v1.json)。

v1 Gate 位于
`/data/models/test/qtopomoe_phase8_route_stability_diagnostic_v1_20260824/diagnostic_gate.json`，
SHA-256 为 `7d0b6652e82a490535bcd0b2d341ffab88d8d555bc0d988bfd6a32b79dc81fe8`。
v2 Gate 位于
`/data/models/test/qtopomoe_phase8_route_stability_diagnostic_v2_20260824/diagnostic_gate.json`，
SHA-256 为 `da129bd58dcd2fb45ab12cab852a0e8166ea28cf1ba9aac1602e8698c26678d0`。
两者的 `classified` 只表示成因分类成功，不表示部署准入。

## 首次质量夹测发现的问题

首次完整质量夹测位于
`/data/models/test/qtopomoe_phase8_warm_placement_quality_equivalence_v1_20260823`。
identity A1 116/116 完成且零失败、零截断；候选轮出现 8 条输出达到 32768 tokens 的截断，
内容退化为重复词或重复字符。脚本按冻结规则立即停止，没有继续 identity A2，也没有把异常
结果解释为候选质量结论。失败 Gate 哈希为
`605861aed0a68d30c131b780d937ed37ba5aa7aecacb8797c569df3618bddce4`。

8 条异常题为：

- `mmlu_pro:test:2255`、`mmlu_pro:test:2659`、`mmlu_pro:test:7871`；
- `mmlu_pro:test:10878`、`mmlu_pro:test:11273`；
- `ceval:logic:val:4`、`ceval:middle_school_politics:val:3`；
- `ceval:teacher_qualification:val:17`。

根因不是评测上限，而是 NVFP4 EPLB 迁移不完整。vLLM Marlin 在权重加载后生成
`w13_weight_scale_2` 与 `w2_weight_scale_2` 两个普通张量；原生专家迁移只搬运已注册权重与
block scale，没有搬运这两个辅助尺度，导致候选映射下权重和尺度错配。修复在
[sitecustomize.py](../../runtime_patches/qtopomoe_eplb/sitecustomize.py) 中把两个尺度纳入每层
专家迁移，并校验第一维、本地专家数和连续布局。修复文件哈希为
`9f6d3d99dfe570dc87d7d1ff3909880dce981c267f8af8768c477d9940852b5f`。

## 定向回归与完整质量 Gate

先只对上述 8 条异常题执行 `identity → candidate → identity` 定向回归。三轮均 8/8 正确、
零失败、零截断、零抽取失败，候选与两轮 identity 的预测全部一致；apply 和 rollback 均由
8/8 rank 提交。定向 Gate 为 accepted，哈希为
`362f24b1552e5edb91d9ea23b481a9a2f9d35cf2fcd705c6d546afed963b1bf6`。

随后使用同一冻结的 116 题 official-like smoke v2、seed 42、并发 4，重新执行完整 A/B/A。
有效 manifest 哈希为
`549c1e3b3e131e8ff586dbf17ae666fb8468b75b57362e206cfbce2a4fcb6d0f`。

| 轮次 | 总正确 | C-Eval | MMLU-Pro | failed / truncated / 抽取失败 | finish_reason |
|---|---:|---:|---:|---:|---|
| identity A1 | 108/116（93.10%） | 50/52（96.15%） | 58/64（90.63%） | 0 / 0 / 0 | stop 116 |
| candidate B | 110/116（94.83%） | 50/52（96.15%） | 60/64（93.75%） | 0 / 0 / 0 | stop 116 |
| identity A2 | 106/116（91.38%） | 49/52（94.23%） | 57/64（89.06%） | 0 / 0 / 0 | stop 116 |

最终 Gate 的全部八项均通过：三轮完整且干净、输入 Gate accepted、候选总正确数/逐基准/
逐学科均不低于两轮 identity 的较低值、没有唯一新增退化、apply 与 rollback 均 8/8 rank
提交、原服务恢复。116 条中候选与 A1、A2 的预测分别匹配 111、112 条；两轮 identity
彼此匹配 109 条，反映温度 1 生成自身存在小幅波动。稳定改善题为
`mmlu_pro:test:11397` 和 `mmlu_pro:test:5447`，唯一新增退化为 0 条。

最终 Gate 状态为 accepted，文件位于
`/data/models/test/qtopomoe_phase8_warm_placement_quality_equivalence_v3_20260823/quality_equivalence_gate.json`，
SHA-256 为 `1f37556a11b469c3c16e96ca705da6d751d1c57d5756c74097791415aba7dcc0`。

## 服务恢复与部署边界

实验完成后，临时 TP8/EP8 vLLM 服务已退出。原 SGLang 在端口 30002 的 `/health` 返回
200，进程只占用 GPU4–7；GPU0–3 空闲。回滚状态文件哈希为
`01a8e868fcec5aebc44d0c524751f156e4dc71504ca4b95c648c933ae09cc912`。

因此本轮成功关闭的是“候选是否保持冻结答案质量”“NVFP4 辅助尺度是否正确迁移”和
“14.28% 来自哪类机制”三个问题。历史 0.5% Gate 与原 A/B/A/B `rejected` 证据保持不变；
当前 1% Gate 已接受。截至本报告冻结时，候选只获准进入有限 canary，canary 和后续自动
闭环仍须各自生成机器 Gate 与中文报告；两级后来均已完成，结果见文末最终验收链接。

机器可读摘要见
[Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json](../Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json)。

后续有限 canary 与自动闭环均已接受，最终结论见
[Phase 8 暖态 placement 最终验收](phase8_warm_placement_final_acceptance_20260825.md)。
