# 阶段 8：基准测试与校准历史

> 本文件将 Phase 8 的 7 份历史叙述报告合并为中文说明。机器可读 JSON、配置路径、哈希、命令和服务端目录保持原样；原始文件名及 SHA-256 见下表。

## 源文件完整性

| 原始文件 | UTF-8 字节数 | SHA-256 |
|---|---:|---|
| `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.md` | 4882 | `BEFC17B978267D5F0AC7C9BBED648FD6B380C17FBA9E31AFB2FE16C0EE00D686` |
| `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.md` | 4183 | `64EF6FE5C55120C8AEEB1ED9FD990C2BD1AA3CF360C4112DF8C8438810CA54A9` |
| `docs/Q-TopoMoE_Phase8_cache_arrival_pilot_v3_20260810.md` | 2526 | `1F13A9F2A2695BD4922988BC451B657E50F9250F2899F3633ACDE3EC88399780` |
| `docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.md` | 4570 | `2B54CDC8591ED12F13459A703F30CB66BB6F5A47323D3349E73F824B4E15A65E` |
| `docs/Q-TopoMoE_Phase8_capacity_rate_freeze_20260810.md` | 2462 | `30DA2FB452E868B293B1F21D75674677037C1F2C1FD0E32CB2EB0D27D799A790` |
| `docs/Q-TopoMoE_Phase8_repeated_run_incidents_20260809.md` | 1604 | `8D168CD632A928EDA2B2848EFE25BBC43AB24A984380D5ADD4DE59532E82135B` |
| `docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.md` | 3187 | `3D198C82056B37B0479B267386429B2EEDEAB280EBC91723BCD6312A3488B504` |

## 1. 五次重复 bootstrap Gate

跨格式单次筛选保留的 4 个候选按 seed=42 随机顺序启动 20 次服务，每个候选重复 5 次，并对每个服务执行冻结的 12-cell Runbook matrix。240 个 candidate/repetition/workload summary 全部通过：所有请求完成、`failed=0`、渲染输入 token 数精确、chat-template 哈希一致、实际 EP rank 为预期的 0/4/4/8。10,000 次重采样聚合被接受；对应机器数据是 `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json`，原始请求、GPU 样本、元数据和日志仍在 gpu-111 的 `/data/models/test/qtopomoe_phase8_repeated_v4/`。

证据哈希：workload matrix `de91dd4a05fa610d73d2e8554ea4fabb7d4959f0423defc057c1190089a33e74`；随机调度 `9746e1873c9b504d99b10cf4e300cd250a8df1132cd331df923e6aed6c1e8740`；chat-template `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`；冻结 vLLM `0.26.1rc1.dev343+g33c50587d`。

| 候选 | GPU | EP rank | median e2e p99（ms） | median 输出 tok/s | oracle 胜出 | 峰值显存和（MiB） |
|---|---:|---:|---:|---:|---:|---:|
| FP8 TP2 PIX 0,1 Triton | 2 | 0 | 1135.29 | 828.65 | 7 | 58878 |
| W4A16 static EP4 Triton | 4 | 4 | 1065.03 | 874.54 | 1 | 119652 |
| RedHat NVFP4 static EP4 NUMA0 | 4 | 4 | 1481.23 | 918.60 | 0 | 117540 |
| RedHat NVFP4 static EP8 SYS | 8 | 8 | 1249.63 | 1031.66 | 4 | 235896 |

四个候选都位于资源感知 Pareto 集合，但适用场景不同：FP8 TP2 是默认的资源效率选择，NVFP4 EP8 适合长 prefill/高并发 cell。重复运行的 cell 变异系数为 FP8 TP2 0.86%、W4A16 EP4 1.04%、NVFP4 EP4 5.55%、NVFP4 EP8 18.02%；不能只依据单次最快重复结果训练 selector。

## 2. 单次跨格式筛选与故障

冻结的 12-cell 矩阵对 6 个 BF16、FP8、W4A16、NVFP4 配置各运行一次。所有候选 12/12 cell 通过，`failed=0`，输入 token 数精确，chat-template SHA-256 为 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。资源感知 Pareto 集合推进 FP8 TP2、W4A16 EP4、NVFP4 EP4 和 NVFP4 EP8；BF16 TP4 与 W4A16 DP4 保留为基线。

| 候选 | GPU | median e2e p99（ms） | median 输出 tok/s | oracle 胜出 | 峰值显存和（MiB） |
|---|---:|---:|---:|---:|---:|
| BF16 TP4 NUMA0 | 4 | 1471.14 | 831.80 | 1 | 117814 |
| FP8 TP2 PIX 0,1 Triton | 2 | 1220.20 | 776.34 | 7 | 58826 |
| W4A16 TP1×DP4 Triton | 4 | 1236.89 | 750.84 | 0 | 119528 |
| W4A16 static EP4 Triton | 4 | 1157.84 | 850.65 | 1 | 119570 |
| RedHat NVFP4 static EP4 | 4 | 1356.06 | 928.58 | 0 | 117546 |
| RedHat NVFP4 static EP8 | 8 | 1334.32 | 881.51 | 3 | 235898 |

首次 FP8 `auto` backend 因 DeepGEMM 报 `Unknown SF transformation` 在服务前拒绝；强制 `VLLM_MOE_BACKEND=triton` 后通过全部 12 个 cell，未修改框架或 checkpoint。W4A16 DP4 的 `EngineDeadError` 发生在 runner 有意关闭后，所有请求均已 HTTP 200，因此属于关闭产物而非推理失败。

## 3. cache / arrival pilot v3

该 pilot 只验证 workload 机制，不能用于部署排序或服务模型校准：单个 FP8 TP2 候选、每个 cell 12 请求、输入 4224 token、输出 16 token。9/9 cell、108/108 请求完成，`failed=0`；服务端 prompt 长度全部为 4224，`prompt_tokens_details.cached_tokens` 每次都有值，缓存比例和到达调度 Gate 全部通过。证据目录为 `/data/models/test/qtopomoe_phase8_cache_arrival_pilot_v3_20260810`。

| 语义共享前缀 | 引擎可实现缓存比例 | 实测比例 | 结果 |
|---:|---:|---:|---|
| 0% | 0% | 0% | 通过 |
| 50% | 50% | 50% | 通过 |
| 100% | 75% | 75% | 通过 |

语义 100% 不等于缓存命中 100%。冻结 vLLM 至少要计算最后一个 prompt token，Qwen3.5 MoE 混合 attention/Mamba cache page 按 1056 token 对齐；4224 token pilot 的最大命中为 `floor((4224 - 1) / 1056) * 1056 = 3168`，即 75%。Poisson/burst 六个 cell 的 p95 请求启动延迟为 0.289–1.644 ms，Gate 上限为 125 ms。

## 4. 校准准备与容量冻结

接受的五次重复聚合已转换为 12 个 Phase 8 observation，每个包含 4 个候选的 median e2e-p99 与 n=5、10,000 次 bootstrap 95% 区间；W4A16 缺失 bucket 通过真实 packed-int4 vLLM Triton WNA16 路径测量，并经过反量化数值 Gate，最大绝对误差为 `7.6195e-06`。要求的 M bucket 为 `1,4,8,16,32,128,256,2048,8192,16384`，四个候选均 ready。

容量预跑覆盖 4 个候选、每个 3 次重复和 12 个基础 cell；144 个 summary 全部通过。正式 open-loop 速率为四候选三次重复中的最小观测 req/s × 0.70，并向下取 6 位小数，避免按候选分别使用更容易的 offered load。正式矩阵为 12 个基础 cell × 3 个前缀比例 × 3 种到达模式，共 108 个 cell，所有 stream seed 唯一。

关键机器数据：`docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.json`、`docs/Q-TopoMoE_Phase8_capacity_audit_20260810.json`、`configs/experiments/phase8_formal_controlled_v1.json`。原始容量预跑目录为 `/data/models/test/qtopomoe_phase8_capacity_prepass_v1_20260810`。

## 5. 透明服务校准结论

透明校准按 Runbook regret Gate 正确失败：未校准 top-1 33.33%、median regret 33.89%、p95 regret 101.80%；分组 held-out 校准 top-1 41.67%、median 13.64%、p95 54.77%；只有 overhead p95 0.04–0.05% 通过。原因是当前请求没有构造受控 0/50/100% prefix-hit，也没有 Poisson/burst 到达，队列、批处理和隐式 prefix reuse 无法从现有特征识别。不得记忆 12 个 oracle 选择，也不得引入 RL。

当前重复筛选启用了 `enable_prefix_caching=True`，但未构造可验证的 prefix-hit population；因此完整数据的回归系数只用于审计和诊断，不能作为部署 manifest。正式下一步是增加 prefix identity hash、记录真实 inter-arrival，再在保留候选上执行正式交叉矩阵并重新拟合透明 selector。

## 6. 重复运行事故

- v1：runner 默认使用旧版 vLLM 0.26.0，启动前拒绝；修复为固定 cleanroom 版本和入口 SHA。
- v2：venv 入口未把 `bin` 加入 `PATH`，FlashInfer 找不到已安装的 `ninja`；修复 PATH、ninja 路径和 SHA Gate。
- v3：32 请求 warmup 成功后因日志 backend 字符串 Gate 过严而拒绝；改为稳定子串并增加 `Unknown SF transformation` 禁止项。

正式下一步按顺序为：先执行候选无关的 closed-loop capacity，再按最慢候选 × 安全利用率冻结 open-loop 速率，扩展前缀与到达模式正式矩阵；只用重复正式结果重新拟合透明 selector。
