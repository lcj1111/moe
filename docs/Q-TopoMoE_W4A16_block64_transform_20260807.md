# Q-TopoMoE W4A16 block64 无损变换验证

> 生成日期：2026-08-07（Asia/Shanghai）
> 目的：在不重新量化的前提下，让 W4A16 canonical（group_size=128）适配
> TP8（中间维每分区 64，要求 group 可整除 64）。

## 1. 方案

把每个 128 宽的 scale group 拆成两个 64 宽子组，各复制同一 scale：

```
scale [R, G]  (G = 中间维/128)
  -> repeat_interleave(2, dim=1) -> [R, 2G]  (中间维/64)
```

packed int4 权重不变，config `weights.group_size` 128 → 64。
W4A16 为对称量化（symmetric=true, zp_dtype=null），反量化 `int4 × scale`
在拆组后逐位一致。

脚本：`quantization/reblock_w4a16_128_to_64.py`；
产物：`/data/models/test/qtopomoe_w4a16_canonical_text_v1_g64`
（manifest：`reblock_128_to_64.manifest.json`，标注 source 与变换关系，
不覆盖 block128 canonical）。

## 2. 数值验证

对 6 个代表性张量（专家 down/gate/up、shared down、attention q/o）：

- packed 权重完全一致；
- scale 形状 `(2048,4)→(2048,8)`、`(512,16)→(512,32)`、
  `(2048,32)→(2048,64)`、`(8192,16)→(8192,32)`；
- **反量化逐位一致（max diff = 0.0）**。

## 3. 功能验证

| 检查 | 结果 |
|---|---|
| TP8×W4A16（block64，此前 group128 被拒） | **通过**，32/32 smoke |
| TP1 block128 vs block64 同一固定 prompt | 输出一致（"42"） |
| 4 个多样 prompt 对比 | 3/4 完全一致；1 题贪心边界个别 token 不同 |
| official-like v2 quality（116 条） | C-Eval 49/52、MMLU-Pro 59/64 |

与 block128 官方质量对比（C-Eval 50/52、MMLU-Pro 58/64）：差 1 题，
属于贪心解码边界差异（源自 group 变化导致的 kernel 浮点累加顺序不同），
整体持平，非系统性退化。

## 4. 结论与边界

1. block64 变换是**无损的**（反量化逐位一致），且让 TP8×W4A16 从"格式
   不兼容"变为"可用"，无需 9.7 小时重新量化。
2. 质量评测显示与 block128 基本一致（±1 题边界差异）。
3. 边界：若需完全消除边界差异，需重新量化（方案 A）或接受此差异并在
   论文标注；TP8 本身对 35B-A3B 无显著吞吐收益（Phase 1：TPOT 与 TP4
   相当），block64 主要价值是解锁 TP8×W4A16 覆盖。
