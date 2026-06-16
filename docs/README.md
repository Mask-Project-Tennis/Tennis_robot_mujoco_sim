# 项目文档索引

## 目录结构

```
docs/
├── README.md                          ← 本文件
├── experiments/                       ← 实验设计与报告（独立索引）
│   ├── README.md
│   ├── design/                        ← 实验设计文档（"做什么"）
│   └── reports/                       ← 实验报告（"结果如何"）
├── architecture_and_algorithm.md      ← 系统架构与算法详解
├── cpp_acceleration.md                ← C++ 加速模块构建指南
├── rm65_evaluate_usage.md             ← 评估脚本使用说明
├── rm65_mpc_ilqr_core.md              ← MPC+iLQR 核心机制
├── rm65_mpc_ilqr_pipeline.md          ← MPC+iLQR Pipeline
├── rm65_tennis_report.md              ← 项目技术总览
└── rm65_mpc_fast_group_report.md      ← 组会汇报材料
```

## 参考文档

| 文档 | 内容 | 适合谁读 |
|------|------|---------|
| `rm65_tennis_report.md` | 项目总览：MPC+iLQR+Tube 框架、硬件、方法 | 新人入门 |
| `architecture_and_algorithm.md` | 实时架构：far/near 分阶段、异步重规划、安全滤波 | 理解系统设计 |
| `rm65_mpc_ilqr_pipeline.md` | Pipeline：从球感知到挥拍的完整数据流 | 理解代码流程 |
| `rm65_mpc_ilqr_core.md` | 核心机制：iLQR 后向/前向、代价函数、R 退火 | 深入算法细节 |
| `cpp_acceleration.md` | C++ 扩展：编译、pybind11 绑定、线性化加速 | 性能优化 |
| `rm65_evaluate_usage.md` | 评估脚本：反弹球模式、图表生成 | 运行评估 |
| `rm65_mpc_fast_group_report.md` | 快速版方法框架，面向组会汇报 | 组会准备 |

## 实验文档

实验设计与报告存放在 `experiments/` 子目录，详见 [experiments/README.md](experiments/README.md)。

- **设计文档**（`experiments/design/`）：实验前编写，含目的、假设、参数矩阵
- **实验报告**（`experiments/reports/`）：实验后生成，含结果统计与分析结论
