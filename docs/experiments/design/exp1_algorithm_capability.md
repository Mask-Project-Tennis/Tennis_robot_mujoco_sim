# exp1: 算法能力上限（速度豁免模式）

## 目的
验证 iLQR+Tube 在理想高速执行器条件下的击打能力上限，不受真实关节速度限制。

## 假设
- 速度豁免（forward_pass_margin=3.0）下，算法能在高球速（≥15 m/s）保持命中
- Tube 在高球速下提供比无 Tube 更好的时间容错

## 参数矩阵
| 变量 | 取值 |
|------|------|
| ball_speed | 9, 12, 15, 18, 20, 25, 30 m/s |
| use_tube | true / false |
| seeds | 0-19 (20 个) |

固定参数 + 约束配置见 `experiment_data/exp1_algorithm_capability/config.yaml`。

## 脚本
- 主脚本: `scripts/rm65_mpc_tube_constraint.py`
- 批量运行 + 提取: 见 `experiment_data/exp1_algorithm_capability/`

## 对照分析
| 对比 | 回答的问题 |
|------|-----------|
| tube_on vs tube_off (各速度) | Tube 在无约束时是否有附加价值 |

## 状态
✅ 已完成 → `experiment_data/exp1_algorithm_capability/results.csv`
