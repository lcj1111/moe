# Q-TopoMoE 代码导读

这份导读只回答三个问题：一次请求怎样进入系统，实验结果怎样形成，出现异常时从哪里查。
函数名、配置键和命令保持代码原样，解释使用中文。

## 一次请求经过哪些模块

```text
冻结配置
  → 启动服务
  → 客户端发送请求
  → runner 保存逐请求结果和运行元数据
  → 聚合器检查完整性并计算指标
  → Gate 给出 accepted / rejected
  → 报告引用 Gate 和输入哈希
```

服务入口是 [start_server.sh](../serving/start_server.sh)，基础验收由
[acceptance.sh](../serving/acceptance.sh)完成。质量请求走
[quality_eval.py](../clients/quality_eval.py)，普通并发请求走
[smoke.py](../clients/smoke.py)。runner 不应直接把“进程退出码为 0”写成实验通过，必须再由
聚合器核对请求数、失败数、身份字段和哈希。

## 质量评测

| 环节 | 代码 | 作用 |
|---|---|---|
| 冻结正式输入 | [freeze_full_set_official.py](../evaluation/freeze_full_set_official.py) | 固定数据 revision、样本顺序、prompt 和 manifest |
| 发送并评分 | [quality_eval.py](../clients/quality_eval.py) | 请求服务、抽取答案、记录 finish reason 和逐题结果 |
| 处理续跑 | [build_fullset_truncation_manifest.py](../scripts/build_fullset_truncation_manifest.py) | 只生成允许续跑的样本清单 |
| 严格合并 | [merge_fullset_quality_results.py](../scripts/merge_fullset_quality_results.py) | 拒绝重复、缺失、身份漂移和越权替换 |
| 同分母比较 | [compare_fullset_quality_results.py](../scripts/compare_fullset_quality_results.py) | 在各格式都完成的题目上比较准确率 |

质量目录中的 `quality.results.jsonl`、`quality.summary.json` 等文件是运行后生成的大体积
产物，不在 Git 中。仓库只保存生成规则、manifest、摘要和哈希。

## 路由采集与漂移

[capture_routes.py](../traces/capture_routes.py)调用支持返回 routed experts 的 vLLM，按 token
保存专家 ID；[route_drift.py](../analysis/route_drift.py)比较 BF16 与量化模型的 Jaccard、
flip、相关性和负载 CV。`traces/manifests/` 只保存采集身份与哈希，npy 分片留在服务器。

这条链路只能说明“选择了哪些专家”，不能恢复 router 的完整概率分布。报告中出现 KL 或
概率差异时，必须确认是否来自另一条真实记录路径，不能从 expert ID 反推。

## kernel、通信和候选选择

Phase 4 先由 [generate_m_buckets.py](../phase4/workload/generate_m_buckets.py)构造 M 桶，
[bench_moe_kernel.py](../scripts/bench_moe_kernel.py)测量 kernel，结果由
[kernel_db.py](../selector/kernel_db.py)读取。只有 `measured=true` 且 `valid=true` 的记录能进入
默认选择；缺失测量不能用零填充。

通信成本由 [build_nccl_cost_db.py](../scripts/build_nccl_cost_db.py)写入
[nccl_cost_db.json](../configs/communication/nccl_cost_db.json)。
[backend_selector.py](../selector/backend_selector.py)选择本地 kernel，
[strategy_selector.py](../selector/strategy_selector.py)综合计算、通信、负载不均和迁移成本选择
TP/DP/EP 候选。

## placement 与在线闭环

[placement.py](../phase7/placement.py)处理逻辑专家到物理槽位的放置，
[migration_cost.py](../phase7/migration_cost.py)估算迁移成本。
[build_runtime_placement_plan.py](../scripts/build_runtime_placement_plan.py)把离线结果转换成运行时
计划；[sitecustomize.py](../runtime_patches/qtopomoe_eplb/sitecustomize.py)负责 NVFP4 专家权重和
辅助尺度的实际迁移。

Phase 8 的线上状态机在 [eplb_policy.py](../selector/eplb_policy.py)：

```text
连续三个合格高负载窗口
  → rebalance
  → 提交 generation=1 placement
  → 十个 cooldown 窗口只观察
  → 连续三个退化窗口
  → rollback
  → 提交 generation=2 identity map
```

[run_phase8_selector_limited_canary.py](../scripts/run_phase8_selector_limited_canary.py)验证单次显式
激活和恢复；[run_phase8_selector_closed_loop_acceptance.py](../scripts/run_phase8_selector_closed_loop_acceptance.py)
验证 trigger、cooldown、8-rank 提交和 rollback。最终可用组合不要从某个单独配置猜测，直接看
[release manifest](Q-TopoMoE_release_manifest_20260825.json)。

## 文件状态怎么判断

- `configs/` 是运行前冻结输入；文件里的状态只代表冻结时点。
- `docs/*.json` 是机器结果或 Gate，先读 `status`，再核对输入哈希。
- `docs/results/*.md` 是给人看的结论，不替代机器 Gate。
- `docs/archive/` 是失败或被替代的历史证据，不进入当前选择。
- `/data/...`、`artifacts/...` 和仅写 basename 的 `status.json`、`gate_status.json` 通常是运行时
  输出，不是仓库文件；文档必须同时说明它由哪个命令生成、位于哪个输出目录。

## 修改代码后先做什么

```bash
python scripts/validate_configs.py
python scripts/check_document_references.py
python -m unittest discover -s tests
```

涉及 CUDA、服务加载或多卡通信的改动还要在目标运行环境执行对应 smoke。单元测试通过不等于
模型可加载，服务可启动也不等于质量或性能 Gate 已通过。
