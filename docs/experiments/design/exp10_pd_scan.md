# exp10: PD 增益扫描

## 目的
位置模式下系统搜索最优 Kp / Kd_ratio / dq_max_fraction 组合。

详见 `experiment_data/exp10_pd_scan/config.yaml`。

## 核心结论
- 选定 Kp=25, Kd_r=0.08（位置模式最优）
- pos_ff 55.2% > pos_noff 44.8% > torque 36.4%

## 状态
✅ 已完成 → `experiment_data/exp10_pd_scan/results.csv`
