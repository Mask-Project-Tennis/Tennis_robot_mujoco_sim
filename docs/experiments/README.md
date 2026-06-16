# 实验设计索引

> 所有实验的可执行参数存放在 `experiment_data/expN/config.yaml`，本文档仅记录设计意图。
> 实验结果存放在 `experiment_data/expN/results.csv`。

## 实验矩阵

| 编号 | 名称 | 目的 | 状态 |
|------|------|------|------|
| exp1 | [算法能力上限](exp1_algorithm_capability.md) | 速度豁免模式下测试 iLQR+Tube 击打上限 | ✅ 已完成 |
| exp2 | [严格关节约束](exp2_strict_joint.md) | RM-65B 真实关节约束下可行球速范围 | ✅ 已完成 |
| exp3 | [TCP+关节双约束](exp3_tcp_joint_dual.md) | TCP 线速度硬限制对性能影响 | ✅ 已完成 |
| exp4 | [Tube 鲁棒性](exp4_tube_robustness.md) | 时间/空间扰动下 Tube 鲁棒性 | ✅ 已完成 |
| exp5 | [实时性能](exp5_realtime_performance.md) | MPC 重规划实时可行性 | ✅ 已完成 |
| exp6 | [消融实验](exp6_ablation.md) | 各组件独立贡献 | ✅ 已完成 |
| exp7 | [噪声×Tube 消融](exp7_noise_tube_ablation.md) | 观测噪声下 Tube 补偿作用 | ✅ 已完成 |
| exp8 | [KF 恢复实验](exp8_estimator_recovery.md) | 卡尔曼滤波噪声恢复率 | ✅ 已完成 |
| exp9 | [观测频率鲁棒性](exp9_obs_freq_robustness.md) | 低频摄像机观测退化与恢复 | ✅ 已完成 |
| exp10 | [PD 增益扫描](exp10_pd_scan.md) | 位置模式 Kp/Kd 最优搜索 | ✅ 已完成 |
| exp11 | [回归测试](exp11_regression.md) | 力矩 vs 位置模式全面对比 | ✅ 已完成 |
| exp12 | [前馈补偿评估](exp12_feedforward.md) | 前馈补偿（重力+科氏力）消融 | ✅ 已完成 |

## 设计文档格式

每组实验的设计文档（`docs/experiments/expN_<name>.md`）包含：
- **目的**：一句话说明实验回答什么问题
- **假设**：核心因果关系假设
- **参数矩阵**：变量 × 取值，固定参数引用 `config.yaml`
- **脚本**：包装/批量/提取脚本路径
- **对照分析**：对比组设计
- **状态**：已执行/待执行 + 结果路径

## 与 Skill 的关系

- **Skill**（`skills/experiment_design.md`）：描述"如何做实验"（流程、模板、规范）
- **本文档**：描述"做什么实验"（设计、参数、假设）
- **数据目录**（`experiment_data/`）：存放"实验结果"（CSV、日志、NPZ）
