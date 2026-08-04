# 量化期间可执行的 Phase 4/8 离线任务

这些产物只依赖 M-bucket 清单、已有 NCCL 正式矩阵和 BF16/FP8 结果，不读取或启动 W4A16 模型。

## Phase 4 kernel benchmark plan

```bash
python3 scripts/plan_phase4_kernel_benchmark.py \
  --m-buckets configs/workloads/m_buckets.json \
  --output configs/kernels/phase4_benchmark_plan.json
```

当前计划包含 8 个 M bucket × 3 个 backend × 2 个 precision，共 48 个运行单元，状态全部为 `planned`。脚本没有执行 command；量化结束后由实际 CUDA wrapper 消费该计划，并把 `p50_us/p95_us/measured/source` 写入 kernel DB。

## NCCL mapping cost DB

```bash
python3 scripts/build_nccl_cost_db.py \
  --input artifacts/raw/20260804T040000Z_nccl_formal/nccl/statistics.json \
  --output configs/communication/nccl_cost_db.json
```

已归一化 132 个 `time_us` 实测点，覆盖 6 个 mapping，所有点 `wrong_total=0`。`effective_us_per_gb` 是描述性指标；不同 collective 和消息大小不能直接压成一个线性带宽常数，Phase 8 应优先按 collective/size 查点或拟合后再注入 mapping cost。

## Phase 8 offline replay

```bash
python3 scripts/replay_phase8.py \
  --candidates configs/strategies/phase8_candidates.json \
  --kernel-db configs/kernels/phase4_kernel_db.json \
  --observations configs/strategies/phase8_observations.json \
  --output docs/Q-TopoMoE_Phase8_replay_20260804.json
```

当前回放结果为 `blocked_missing_kernel_measurements`，这是预期状态：kernel DB 仍为空模板，不能伪造 kernel latency。候选注册表和观察样本已准备好；Phase 4 CUDA 实测写入后可直接重跑得到 regret/gate 结果。
