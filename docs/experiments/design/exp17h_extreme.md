# exp17h：极端条件补强实验

> **状态**: 待运行（2026-09-11 启动）
> **上游依据**: `paper/planning/03-exp17h-extreme-conditions-design.md`（完整设计提案）、
> `paper/discussion/2026-09-11_narrative-and-experiment-decisions.md` §4.2/§4.5/§10

## 目的

为「走廊的贡献是条件性的、集中在高扰动与紧余量区域」这一论文叙事补充两个缺失维度的证据：

1. **Block A（s 扫描）**：空间扰动 s ∈ {0.3, 0.4} m——走廊独立增益随扰动幅度是否单调增长，
   还是 0.2 m 之后饱和/回落？（已有 s=0.2 高功效角点：tube_only vs none +3.57pp, p=0.0022）
2. **Block B（紧余量角点）**：12 m/s × s=0.2——时间余量显著更短时，走廊独立价值是否比
   9 m/s 同角点的 +3.57pp 更大？（标称速度维度已有 −2.0/+11.2/+22.4pp 的放大趋势）

## 假设

- H_A：tube_only vs none 的增益在 s ∈ {0.2, 0.3, 0.4} 上单调不减（走廊价值随误差增大）。
- H_B：12 m/s 角点的 tube_only vs none 增益 > 9 m/s 角点的 +3.57pp（余量越紧越值钱）。

## 参数

| 参数 | Block A | Block B |
|------|---------|---------|
| 球速 | 9 m/s | 12 m/s |
| 时间扰动 t | 0 ms | 0 ms |
| 空间扰动 s | 0.3、0.4 m（uniform[0,s]，符号随机） | 0.2 m |
| 消融档 | full / tube_only / softmin_only / none | 同左 |
| Seeds/格 | 1500 | 3000 |
| 规模 | 2 × 4 × 1500 = 12,000 | 4 × 3000 = 12,000 |

合计 **24,000 runs**；同 seed 四档对齐（同一次发球 + 同扰动交给四种档位）。
执行：`python scripts/exp/batch_runner.py exp17h_extreme --workers 16`（预计 ~65 min）。

## 统计功效

- n=1500/格：95% CI ≈ ±2.3pp；n=3000/格：≈ ±1.6pp（可分辨 ~3-4pp 差异）。
- 与 exp17g 的 s=0.2 角点（n≈2970/档）拼接成「走廊增益 vs s」三曲线，直接供应 fig6。

## 数据目录

`experiment_data/exp17h_extreme/`（batch_runner 自动生成 config.yaml + results.csv；
完成后写 `_.COMPLETE`）。提取：`python scripts/extract/extract_exp17h_results.py`。

## 结果走向（两种都能进论文）

| 结果 | 论文写法 |
|------|---------|
| 增益单调放大 / Block B 显著更大 | 条件性结论升级为「走廊价值随扰动幅度与时间余量紧张度单调放大」，附曲线图 |
| 饱和 / 回落 / 不显著 | 如实写「走廊在 s≤0.2 显著，更大偏差下不再增长」——同样有信息量的诚实结论 |

## 关联

- 数据回填：`paper/discussion/2026-09-11_narrative-and-experiment-decisions.md` §4.2 / §10，定 fig6 数据源；
- 报告：`docs/experiments/reports/2026-09-11_exp17h_extreme.md`（含人工分析决策栏）。
