# exp5: 实时性能分析

## 目的
收集 MPC 重规划时间开销，验证实时可行性。

## 参数
- ball_speed: 9, 12 m/s
- seeds: 0-19
- 实时模式: `--realtime`

## 关键指标
| 指标 | 说明 |
|------|------|
| first_replan_ms | 首次规划（冷启动）耗时 |
| avg_steady_replan_ms | 稳态重规划平均耗时 |
| max_steady_replan_ms | 稳态重规划最大耗时 |
| avg_step_ms | MPC 每步平均耗时 |
| mpc_realtime_ratio | MPC 实时比率 |
| buffer_exhaust_count | buffer 耗尽次数 |

## 状态
✅ 已完成 → 参见 `docs/experiments/reports/2026-05-31_exp5_realtime.md`
