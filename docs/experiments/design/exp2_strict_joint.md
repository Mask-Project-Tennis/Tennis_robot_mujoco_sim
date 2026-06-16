# exp2: 真实关节约束可行性

## 目的
在 RM-65B 真实关节约束下（qdot ≤ 1.0×，forward_pass_margin=1.0），测试系统可行球速范围。

## 假设
- 严格约束下可行球速上限远低于 exp1（预计 7-10 m/s）
- Tube 在严格约束下帮助有限（约束是主要瓶颈，非预测误差）

## 参数矩阵
| 变量 | 取值 |
|------|------|
| ball_speed | 7, 8, 9, 10, 11, 12 m/s |
| use_tube | true / false |
| seeds | 0-19 (20 个) |

约束配置（严格）: `forward_pass_margin=1.0, qdot_scale=1.0, forward_pass_q_tol_deg=0.0`
详见 `experiment_data/exp2_strict_joint_v3/config.yaml`。

## 脚本
- 包装（monkey-patch 约束）: `scripts/exp/_run_exp2_v3_strict.py`
- 批量: `scripts/exp/run_exp2_v3_batch.py`
- 提取: `scripts/extract/extract_exp2_v3_results.py`

## 状态
✅ 已完成 → `experiment_data/exp2_strict_joint_v3/results.csv`
