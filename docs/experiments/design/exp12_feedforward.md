# exp12: 前馈补偿综合评估

## 目的
验证位置模式下前馈补偿（重力+科氏力 mj_rne）对击打性能的影响。

## 参数矩阵
核心消融（FF on/off）+ PD 重扫描 + 噪声鲁棒性，1900 runs。

详见 `experiment_data/exp12_feedforward/config.yaml`。

## 核心结论
- pos_ff 55.2% > pos_noff 44.8% > torque 36.4%
- 前馈补偿提升位置模式命中率 +10.4pp

## 状态
✅ 已完成 → `experiment_data/exp12_feedforward/results.csv`
