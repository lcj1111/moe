# Phase 8 selector 自动闭环最终拒绝报告

## 最终结论

2026-08-23 在 gpu-111 完成 one-shot 修复、短验证、五轮配对 canary 和四次闭环诊断后，
Phase 8 selector 自动闭环最终判定为 **rejected**。不得进入扩大流量、长期 canary 或生产启用。

最终代表负载使用 8 张 GPU、TP8/EP8、每窗口 128 请求、并发 16、输入 1536 tokens、
固定输出 384 tokens。动态触发和 8-rank placement apply 均正确执行，但迁移前 p99 EMA
为 3488.36 ms，迁移后连续三个窗口为 3766.33、3691.37、3718.51 ms，均超过回滚
参照的 105%。控制器在第三个窗口正确发出 rollback，正式 Gate 随即停止。

这不是整个 Q-TopoMoE 项目失败。量化质量、服务兼容性、拓扑测量、静态候选、selector
离线评测和安全控制器均形成了可复现成果；失败的是当前冻结 placement 与自动闭环候选的
上线准入。

## 最终 Gate

| 项目 | 结果 | 说明 |
|---|---:|---|
| 五轮 one-shot 输入 Gate | 通过 | `repeated_gate.json` 状态为 accepted，哈希匹配 |
| 连续 3 窗口 expert-load CV > 0.25 | 通过 | 三窗约 2.146、2.164、2.131 |
| 第 3 窗口触发 rebalance | 通过 | 动作为 hold、hold、rebalance |
| placement 哈希与 8 rank 提交 | 通过 | 8/8 rank 各提交一次 generation 1 |
| 10 窗口 cooldown 保持 hold | **失败** | 第 3 窗口已满足真实 p99 回滚条件 |
| 连续 3 窗口 p99 退化时发出 rollback | 通过 | 控制器第三窗动作是 rollback |
| 原生 generation 2 rollback 提交 | 未验收 | Gate 在非预期提前回滚处停止，未写入 generation 2 控制文件 |
| 请求正确性 | 通过 | 已完成窗口均 failed=0，固定输出长度一致 |
| 原 SGLang 恢复 | 通过 | 端口 30002 返回 200，仅占用 GPU4–7；31682 已关闭 |

## 为什么五轮曾接受、最终却拒绝

五轮 one-shot canary 每轮固定执行 `stable → migration → recovery`，并在每轮重新启动服务。
五轮 stable p99 约 5.50–5.78 秒，migration 约 4.41–4.48 秒，recovery 约 3.47–3.61 秒，
呈现几乎单调的自然预热趋势。由于没有交叉阶段顺序或暖态 A/B/A/B，对 recovery/stable 的
低比值不能排除冷启动、编译缓存和 GPU 热状态影响。

最终 v4 在同一服务进程内先完成 baseline 和三个暖态触发窗口，迁移前稳定 p99 已收敛到
3.47–3.50 秒。应用 placement 后，apply-confirm 为 4.62 秒，随后三窗仍为 3.69–3.77 秒。
这组数据消除了“首段冷、末段热”的主要顺序偏差，因此其拒绝结论优先级高于五轮 accepted。

## 四次闭环诊断

| 版本 | 目的与发现 | 结论 |
|---|---|---|
| v1 | 初始实现把所有层汇总为 rank-load CV，实测仅 0.08–0.10，未达到0.25 | 负控成功，未误触发；指标口径不符 Runbook |
| v2 | 改为各 MoE 层逻辑 expert-load CV 的中位数；第3窗触发，但单点 p99 作为回滚参照导致过敏 | 拒绝；需要稳定回滚参照 |
| v3 | 用迁移前 p99 EMA 冻结值作参照；短负载在 cooldown 第9窗真实回滚 | 拒绝；短负载与收益准入负载不一致 |
| v4 | 负载严格对齐五轮代表负载，保留原0.25/5%/10窗/3窗门槛 | 最终拒绝；placement 在暖态下真实退化 |

所有诊断均使用新的输出目录和预注册配置，未覆盖历史结果，未修改冻结 selector 或
placement map。v2–v4 失败后均自动恢复原 SGLang，健康码为 200。

## 已修复的工程问题

1. 将在线 placement 从每周期重复迁移改为显式 one-shot apply，后续同 map 为 no-op。
2. `cooldown` 从实现默认 20 窗口修正为 Runbook 的 10 窗口，并修正少算一个窗口的问题。
3. 控制窗口按“500 ms 或 1000 请求先到者”判断，不再只接受 1000 请求。
4. 将 trigger 指标修正为逐层逻辑 expert-load CV 的中位数，rank CV仅作为旁证。
5. rollback 参照由单个触发窗口 p99 改为迁移前 p99 EMA，并在迁移后冻结。
6. 失败路径现在也写出机器 Gate、恢复状态和中文可解释结论。

这些修复证明控制器能够拒绝不满足条件的触发、只执行一次 placement、识别连续 p99 退化
并保护原服务；但它们不能把性能退化的 placement 变成可上线候选。

## 后续建议

1. 暂停当前 placement 候选，不再围绕同一 map 调整 Gate。
2. 重新生成 placement 时，把暖态窗口的 rank-load CV、跨 NUMA 通信和实测 p99 共同纳入
   目标，而不是只依赖离线预测收益。
3. 新候选必须使用交叉顺序或暖态 A/B/A/B：至少 `identity → candidate → identity → candidate`，
   各段先预热再计数，禁止固定冷到热顺序。
4. 新候选通过多轮暖态配对后，再重新验收 10 窗口 cooldown 和 generation 2 原生 rollback。
5. 在新的正式 Gate 通过前，保持自动闭环禁用。

机器可读最终 Gate：`docs/Q-TopoMoE_Phase8_selector_closed_loop_final_gate_rejected_20260823.json`。
服务器原始目录：`/data/models/test/qtopomoe_phase8_selector_closed_loop_acceptance_v4_20260823`。
