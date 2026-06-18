# Exp13 — V12 架构对比验证

**日期**: 2026-06-18
**状态**: 设计完成，已执行

## 目的

验证 V12 新管线架构（EpisodeRunner + MPCController 策略化 + 可组合组件）相对于 V11 旧架构（inline main() 循环）的可靠性和性能差异。

## 假设

1. V12 的 `do_replan` 每次构造全新 HittingCost（无状态泄漏），应比 V11 的 inline replan（复用 base_cost_fn）更稳定
2. V12 统一走 do_replan 无重复代码路径，规划耗时应与 V11 相当或更优
3. 位置模式下两者表现接近（架构差异主要体现在规划层，执行层差异由 PD 增益主导）

## 参数

| 参数 | 值 |
|------|-----|
| 对比版本 | V11 (`rm65_mpc_v11.py`) vs V12 (`rm65_mpc_v12.py`) |
| 模式 | 力矩（默认）+ 位置（`--position-mode`） |
| Seeds | 1-85（每组 82-85 有效样本） |
| 球速 | 7 m/s |
| 发球模式 | `--serve-box` |
| 命中阈值 | pos_error < 0.153m |

## 对比指标

| 指标 | 来源 | 单位 |
|------|------|------|
| 命中率 | hit_type + pos_error | % |
| 位置误差 | pos_error | m |
| 速度误差 | vel_error | m/s |
| 首次规划耗时 | REPLAN log t=XXXms | ms |
| 总耗时 | wall_time | s |

## 实验脚本

`scripts/exp/compare_v11_v12.py` — 单文件，运行 N seeds × 4 配置。

## 结果

见 [报告](../reports/2026-06-18_exp13_v12_architecture_validation.md)
