# exp4: Tube 鲁棒性（时间/空间扰动）

## 目的
测试 Tube 机制在球轨迹预测存在时间误差和空间偏移时的鲁棒性。

## 参数矩阵
| 子实验 | 变量 | 取值 |
|--------|------|------|
| 4a 时间扰动 | time_perturb_ms | -100, -50, -20, 0, 20, 50, 100 |
| 4b 空间偏移 | space_perturb_m | -0.10, -0.06, -0.03, 0, 0.03, 0.06, 0.10 |

固定: ball_speed=9, 严格关节约束, seeds 0-19。详见 `experiment_data/exp4_tube_robustness/config.yaml`。

## 对照组
同时运行 `--use_tube false` 的同等扰动，对比 Tube on/off 鲁棒性差异。

## 状态
✅ 已完成
