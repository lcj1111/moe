# Q-TopoMoE P2P 启用后全量性能重测总结

> 生成日期：2026-08-07（Asia/Shanghai）
> 触发：该机配置启用 GPU P2P，旧 Phase 0 画像（P2P 不可用、PIX 最慢）
> 被推翻；所有多卡性能数字在 P2P 禁用假象下测得，一律作废并重测。

## 判断标准

- **作废重测**：任何 TP>1 / DP>1 / EP>1 多卡服务上测得的延迟/吞吐/耗时
  （通信路径从"共享主机内存"变为"P2P/direct pointer"）。
- **保留不重测**：质量分数（同输入同输出）、route trace/漂移（expert ID
  分布与通信无关）、单卡 kernel 延迟（无跨卡通信）。

## 各阶段新旧对比

### Phase 1（SGLang，short c32）

FP8：
| 拓扑 | TTFT OLD→NEW | TPOT OLD→NEW | e2e OLD→NEW |
|---|---:|---:|---:|
| tp2_node_0_2 | 1044→1210 | 6.29→5.72 | 1845→1937 |
| tp2_pix_0_1 | 1177→**1107** | 7.22→**5.36** | 2089→**1791** |
| tp4_numa0 | 1049→1030 | 10.08→7.89 | 2310→2035 |

PIX 从"最差"反转为"最优"（e2e 1791 vs NODE 1937）。TP4 通信改善，
TPOT -22%。

BF16：TP2 全部 OOM（单卡 32GB，预期，与 P2P 无关）；TP4-NUMA0/1、
TP8-SYS accepted。TP4-NUMA0 e2e 2003ms、TP8-SYS 2166ms，TPOT 均 8.05ms
（旧 10.00/9.77，-20%）。

### Phase 6（vLLM，32 请求，并发 8）

| 配置 | TTFT | TPOT | e2e | 旧 e2e |
|---|---:|---:|---:|---:|
| TP8×DP1（BF16） | 173.3 | 6.25 | 627.6 | 864.9 |
| TP4×DP2（BF16） | 316.6 | 7.86 | 785.4 | 1528.7 |
| TP2×DP4（BF16） | 276.3 | 8.27 | 1041.6 | 1314.8 |
| EP8 TP1（W4） | 227.4 | 12.54 | 1024.6 | 2055.4 |
| EP4×TP2+EPLB（W4） | 182.2 | 16.28 | 1221.5 | 1234.6 |

TP8/DP/EP 配置的 e2e 普遍改善 20-50%；EP8 all-to-all 在 P2P 下 TTFT
-63%、TPOT -43%。

EPLB 行另跑一次重复验证（run2）：e2e 1,221.5→1,200.7 ms（-1.7%），
结果稳定。冗余专家 1 变体不可测：vLLM 要求专家数可被 EP rank 整除，
256+1=257 为质数（EP2/4/8 均不整除），`--eplb-config
'{"num_redundant_experts": 1}'` 报 `even distribution of experts across
ranks`——框架限制，与 P2P 无关，保留为格式不兼容证据。

### Phase 3c pilot 计时（BF16 TP4，8 条 official 协议）

单题 request_ms：**p50 2066ms、mean 2001ms**（旧 23667/30242ms，**约 15x
加速**）。旧的全量外推基数完全失效，已用新值替代。

thinking 模式 pilot（32 条，同协议采样）：**p50 6839ms、mean 7449ms**
（旧 p50 112711ms / mean 116484ms，**约 16x 加速**）。

### Phase 7 迁移成本（单专家 4MB）

同 NUMA 78.77→**47.71μs**（P2P 直连，-39%）；跨 NUMA 75.32→73.45μs；
同 GPU 10.14μs 不变。

### Phase 8 replay（新代价库 + PIX 候选）

修复了 replay 未接入 `nccl_cost_db` 的缺口（此前用默认 1000us/GB 兜底，
旧/新代价库都没生效）。重跑后 4 条 observation 全部选择
**fp8_tp2_pix01_triton**（旧库时为 fp8_tp2_node02）——P2P 直接改变联合
策略选择，验证了 RQ1/RQ4 的假设。

## 结论

1. P2P 启用后通信画像反转（PIX 最快），并让多卡性能普遍提升
   （TP8/EP/DP e2e -20~50%，pilot 计时 -93%）。
2. 联合策略选择因通信代价变化发生反转（node→pix），说明通信测量必须
   基于当前 P2P 状态。
3. 质量分数与 trace/漂移结论不受影响，保留。

## 归档

- 各阶段 P2P 重测 JSON：`Q-TopoMoE_Phase{0,1,3c,6,7,8}_*_p2p_*`
- 通信代价库：`configs/communication/nccl_cost_db.json`（P2P 版）
- 原始 NCCL 矩阵：`artifacts/raw/20260807T120000Z_nccl_formal_p2p/`
