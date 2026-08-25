# Q-TopoMoE 实验决策记录

本文件只回答两个问题：哪些实验构成当前主线，哪些失败或被替代的分支不再参与结论。
机器结果和冻结配置保留原路径以维持哈希、脚本和复现关系；详细历史报告统一放在
[`archive/decision-history/`](archive/decision-history/README.md)。

## 当前主线

```text
硬件拓扑与 P2P/NCCL
  → BF16/FP8 服务基线
  → W4A16/官方 NVFP4 覆盖、加载和质量 Gate
  → 三格式 full-set 与 route trace
  → kernel/backend、通信与迁移成本
  → 暖态窗口生成最小迁移量 placement
  → 质量 A/B/A + 当前 1% 路由稳定性
  → 有限 canary
  → trigger/cooldown/apply/rollback 自动闭环
  → 技术验收完成
```

当前 Phase 8 唯一正式候选是 `warm_swap_008_slots_per_layer_v1`。权威结论见
[暖态 placement 最终验收](results/phase8_warm_placement_final_acceptance_20260825.md)，
部署边界见[项目发布与生产部署清单](Q-TopoMoE_项目发布与生产部署清单_20260825.md)。

## 已关闭的实验分支

| 分支 | 判定 | 关闭原因 | 当前处理 |
|---|---|---|---|
| P2P 生效前的通信和系统矩阵 | superseded | 主机 P2P 状态改变，旧性能不能代表最终环境 | 只保留环境变化和对照证据 |
| 自生成 NVFP4 v1 | rejected | 服务与覆盖通过，但 sampled 质量相对 BF16 下降 3.45 个百分点 | 退出正式候选；官方 RedHatAI NVFP4 进入主线 |
| W4A16 Marlin | rejected backend | 同一 checkpoint 在 Marlin 上出现严重质量回退 | W4A16 正式结果固定为 Triton |
| 早期错误 M 语义 kernel 数据 | invalid measurement | 把 batch 总 token 错当成每专家 M，部分 shape 被放大约 32 倍 | 旧数据仅供故障审计；正式矩阵已重测 |
| 早期 Phase 8 replay | superseded | 缺少完整 oracle，且被正式重复矩阵覆盖 | 机器结果保存在 `docs/archive/` |
| 2026-08-12 Phase 7–8 联合收尾 | superseded report | Phase 7 在线迁移仍有效，但其中的旧 selector 结论已被后续链路覆盖 | 联合报告归档；Phase 7 当前结论并入 Phase 4–7 聚合报告 |
| selector v2 | rejected | 训练侧 p95 regret 23.05%，超过 10% Gate | 不冻结、不进入独立测试 |
| selector v3 原始独立 Gate | rejected | 900 次独立测量完整，但 p95 regret 11.50% 超过原始 10% Gate | 原始结论保持失败，不用独立标签回调模型 |
| selector 12% 后续政策 | superseded policy | 只允许旧 selector 进入有限诊断，不等于原始 Gate 通过 | 不作为当前暖态 placement 的准入依据 |
| 旧 selector 有限 canary | rejected | recovery/stable p99 为 136.36%，超过 105% | 停止旧候选上线链路 |
| 固定 `stable → migration → recovery` 五轮 one-shot | diagnostic only | 阶段顺序包含明显冷启动、编译缓存和预热偏差 | 不作为性能收益证据 |
| 旧 placement 闭环 v1–v4 | rejected | 依次暴露 CV 口径、回滚参照、负载不一致和暖态 p99 退化 | 冻结旧 map，重新生成暖态候选 |
| 暖态质量 A/B/A v1 | rejected diagnostic | 候选出现 8 条长输出截断，定位为 NVFP4 Marlin 辅助尺度未迁移 | 修复后先做 8 条定向回归，再完成 116 题 v3 |
| 路由稳定性历史 0.5% Gate | superseded policy | 长生成可复现 excess TV p95 为 0.718% | 历史判定保留；当前使用后续冻结的 1% 政策 |
| 无效 canary 编排尝试 | invalid run | 服务前检或 control/activation 模式不一致，未形成完整可比较结果 | 只保留 manifest，不并入正式结果 |

## 证据语义

- `accepted`：可以进入当前阶段结论。
- `rejected`：实验本身有效，但候选未通过 Gate。
- `superseded`：曾用于判断，后被更严格或环境一致的实验覆盖。
- `diagnostic only`：用于定位机制或实现问题，不能支持性能收益。
- `invalid run`：输入、环境或编排不满足实验合同，不属于性能样本。

失败证据不会删除，也不会改写为 accepted。主 README 和 `docs/results/` 只展示当前主线；
历史细节从本文件进入归档，机器 JSON、manifest 和冻结配置继续由
[数据与结果清单](DATA_CATALOG.md)索引。

## 后续修改规则

1. 新实验若替代旧结论，必须在本表增加一行并注明 `supersedes` 关系。
2. 阈值变化必须创建新的 policy 文件；不得覆盖原始 Gate。
3. 无效运行只进入 manifest 和归档，不新增主线结果报告。
4. `docs/results/` 每个阶段只保留当前聚合报告；详细失败叙述进入归档。
5. 生产部署是独立变更，不修改本项目已经冻结的技术验收结论。
