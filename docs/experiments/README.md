# 实验设计索引

> 所有实验的可执行参数存放在 `experiment_data/expN/config.yaml`，本文档仅记录设计意图。
> 实验结果存放在 `experiment_data/expN/results.csv`。

## 目录结构

```
docs/experiments/
├── README.md                ← 本文件（索引）
├── design/                  ← 实验设计文档（"做什么"：目的/假设/参数）
│   ├── _template.md
│   ├── exp1_algorithm_capability.md
│   └── ...
└── reports/                 ← 实验报告（"结果如何"：分析/结论/统计）
    ├── 2026-06-02_exp2_strict_joint.md
    └── ...
```

## 实验矩阵

| 编号 | 名称 | 设计 | 报告 | 状态 |
|------|------|------|------|------|
| exp1 | 算法能力上限 | [design](design/exp1_algorithm_capability.md) | [2026-06-03](reports/2026-06-03_exp1_algorithm_capability.md) | ✅ |
| exp2 | 严格关节约束 | [design](design/exp2_strict_joint.md) | [2026-06-04](reports/2026-06-04_exp2_v3_strict_joint.md) | ✅ |
| exp3 | TCP+关节双约束 | [design](design/exp3_tcp_joint_dual.md) | — | ✅ |
| exp4 | Tube 鲁棒性 | [design](design/exp4_tube_robustness.md) | — | ✅ |
| exp5 | 实时性能 | [design](design/exp5_realtime_performance.md) | — | ✅ |
| exp6 | 消融实验 | [design](design/exp6_ablation.md) | — | ✅ |
| exp7 | 噪声×Tube 消融 | [design](design/exp7_noise_tube_ablation.md) | [2026-06-09](reports/2026-06-09_exp7_noise_tube_ablation_v4.md) | ✅ |
| exp8 | KF 恢复实验 | [design](design/exp8_estimator_recovery.md) | [2026-06-10](reports/2026-06-10_exp8_estimator_recovery_v2.md) | ✅ |
| exp9 | 观测频率鲁棒性 | [design](design/exp9_obs_freq_robustness.md) | [2026-06-10](reports/2026-06-10_exp9_obs_freq_robustness.md) | ✅ |
| exp10 | PD 增益扫描 | [design](design/exp10_pd_scan.md) | [2026-06-13](reports/2026-06-13_exp10_pd_finetune.md) | ✅ |
| exp11 | 回归测试 | [design](design/exp11_regression.md) | [2026-06-13](reports/2026-06-13_exp11_regression.md) | ✅ |
| exp12 | 前馈补偿评估 | [design](design/exp12_feedforward.md) | [2026-06-14](reports/2026-06-14_exp12_feedforward.md) | ✅ |
| exp13 | V12 架构对比验证 | [design](design/exp13_v12_architecture_validation.md) | [2026-06-18](reports/2026-06-18_exp13_v12_architecture_validation.md) | ✅ |
| exp14 | V12 位置模式 PD 扫描 | [design](design/exp14_v12_pd_scan.md) | [2026-06-22](reports/2026-06-22_exp14_v12_pd_scan.md) | ✅ |
| exp15 | V12 多球速鲁棒性 | [design](design/exp15_v12_speed_sweep.md) | [2026-06-22](reports/2026-06-22_exp15_v12_speed_sweep.md) | ✅ |
| exp16 | 真机限位对比 | [design](design/exp16_limits_comparison.md) | — | ✅ |

## 文档类型说明

- **设计文档**（`design/expN_*.md`）：实验前编写，包含目的、假设、参数矩阵、脚本路径、对照分析
- **实验报告**（`reports/YYYY-MM-DD_expN_*.md`）：实验后生成，包含结果统计、Agent 数据观察、人工分析决策
- **可执行配置**（`experiment_data/expN/config.yaml`）：实际运行参数
- **结果数据**（`experiment_data/expN/results.csv`）：CSV 汇总表

## 与 Skill 的关系

- **Skill**（`skills/experiment_design.md`）：描述"如何做实验"（流程、模板、规范）
- **设计文档**（`docs/experiments/design/`）：描述"做什么实验"（设计、参数、假设）
- **实验报告**（`docs/experiments/reports/`）：描述"实验结果如何"（分析、结论）
- **数据目录**（`experiment_data/`）：存放"实验结果"（CSV、日志、NPZ）
