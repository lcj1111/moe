# Phase 8 暖态 placement 候选重新生成与质量验证报告

## 最终结论

2026-08-23 已停止调整旧 Gate，并把旧 selector、旧 placement map 和旧失败结果保留为
历史只读证据。本轮从最终代表负载的真实暖态逐层专家计数重新生成候选
`warm_swap_008_slots_per_layer_v1`，随后完成在线性能、NVFP4 迁移正确性和冻结质量集验证。

当前结论分为两层：

- **质量等价 Gate 已接受**：修复 NVFP4 Marlin 辅助尺度迁移后，完整 116 题
  `identity A1 → candidate B → identity A2` 三轮均为 `failed=0`、`truncated=0`、
  `extraction_failed=0`、`finish_reason=stop`。候选正确 110/116，高于两轮 identity 的
  较低值 106/116，且没有“两轮 identity 都正确、候选错误”的新增回归。
- **部署 Gate 仍拒绝**：较早冻结的同进程暖态 A/B/A/B 中，候选虽然满足 p99 上限并改善
  rank CV，但逻辑专家负载分布相对 identity 系统性变化 14.28%，超过冻结稳定范围。
  质量通过不能覆盖这个独立失败项，因此候选仍不得进入 canary、自动闭环或上线。

## 执行关系

```text
旧 Gate / 旧 map 冻结
  └─ identity 暖态采集（48 个窗口，accepted）
      └─ 限制迁移量候选生成（8/16/32 槽位每层 + full LPT）
          └─ 选择满足离线改善门槛且迁移量最小的 8 槽位候选
              ├─ 同进程暖态 A/B/A/B
              │   ├─ rank CV、p99、请求与 8-rank 提交通过
              │   └─ 逻辑专家分布稳定性失败 → 部署 Gate rejected
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
0/128 条完全一致，说明这套长生成在当前分布式运行时不是逐 token 位级确定性的。冻结 Gate
没有事后修改，结果保持 rejected。逻辑专家 CV 在两组内部稳定、组间变化 14.28%，仍足以
阻止部署准入。

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

因此本轮成功关闭的是“候选是否保持冻结答案质量”和“NVFP4 辅助尺度是否正确迁移”两个
问题。尚未关闭的是在线逻辑专家分布稳定性。后续不得降低或重写旧 Gate，也不得直接启动
canary；若继续，应为这个新候选设计独立的路由稳定性诊断，形成新的预注册实验与新 Gate。

机器可读摘要见
[Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json](../Q-TopoMoE_Phase8_warm_placement_regeneration_summary_20260823.json)。
