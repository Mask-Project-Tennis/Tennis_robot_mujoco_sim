# exp7: 噪声×Tube 消融

## 目的
验证观测噪声（模拟深度相机跟踪误差）对 MPC 命中率的影响，并测试 Tube 空间走廊结构能否补偿噪声造成的性能下降。

## 假设
- 观测噪声 → 预测错误击球点 → iLQR 规划偏错方向 → 命中率下降
- Tube 空间走廊可容忍"规划错了"的偏差 → 恢复命中率

## 参数矩阵
| 变量 | 取值 |
|------|------|
| noise | off (pos_std=0, vel_std=0) / on (pos_std=0.03, vel_std=0.3) |
| use_tube | true / false |
| ball_speed | 8, 10, 12, 14, 16 m/s |
| seeds | 0-4 |

详见 `experiment_data/exp7_noise_tube_ablation/config.yaml`。

## 对照分析
| 对比 | 回答的问题 |
|------|-----------|
| noise_off×tube_off vs noise_on×tube_off | 噪声对命中率的绝对影响 |
| noise_on×tube_off vs noise_on×tube_on | **Tube 能否补偿噪声**（核心结论） |

## 噪声注入方式
通过 `RM65Env(estimator_config=...)` 或 `preprocessor` 回调注入，而非旧版 monkey-patch。
详见 `src/utils/noise.py` 的 `add_observation_noise`。

## 状态
✅ 已完成 → `experiment_data/exp7_noise_tube_ablation/results.csv`
