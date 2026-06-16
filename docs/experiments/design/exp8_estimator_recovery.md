# exp8: KF 滤波恢复实验

## 目的
建立噪声等级到 KF 恢复率的完整映射，验证 6D 卡尔曼滤波器能否恢复因观测噪声损失的 MPC 命中率。

## 参数矩阵
5 级噪声 × 2 KF 开关 × 5 球速 × 20 seeds = 1000 runs。

详见 `experiment_data/exp8_estimator_recovery/config.yaml`。

## 核心结论
- off+kf: 52.2-52.6%（性能税 -20~29pp，根因 7.5mm 残留偏差）
- lo+kf: 13.4-15.2%（绝对恢复 +12-14pp）
- Tube 无交互效应（与 exp7 一致）

## 状态
✅ 已完成 → `experiment_data/exp8_estimator_recovery/results.csv`
