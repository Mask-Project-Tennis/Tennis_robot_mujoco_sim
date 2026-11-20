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
├── rm65_mpc_fast_group_report.md      ← 组会汇报材料
├── r1d_v12_usage.md                   ← R1D×V12 使用/复现/限制（T9）
└── r1d_v12_parallel_repair_plan.md    ← 复审修复与球场白线的 subagent 并行计划（已实施，见第 15 节）
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
| `r1d_v12_usage.md` | R1D×V12：两模式固定案例、配置分层、v2 产物/回放、论文 smoke、留出集限制 | 复现 / 验收 / 维护 |
| [r1d_v12_parallel_repair_plan.md](r1d_v12_parallel_repair_plan.md) | 2026-11-20 复审问题、A–E 文件独占分工、接口与依赖、球场白线验收；已实施并提交（第 15 节） | 协调者 / subagent 实施与复审 |
| [r1d_v12_parallel_tuning.md](experiments/design/r1d_v12_parallel_tuning.md) | 多 worker 隔离、续跑与单写入者、9,940 回合预算、完整安全恢复指标；设计存档，未运行 | 调参实验 / 数据与算力规划 |
| `experiments/reports/2026-11-16-unitree-stage3.md` | Unitree R1-A7-D 单臂适配（阶段 3）工程验收：右臂固定案例、双侧渲染、与论文 RM-65B 的边界 | 工程验收 / 后续维护 |
| `experiments/reports/2026-11-18-r1d-v12-parity.md` | R1D×V12 parity（T7）：两模式真实 active、20 回合留出集、估计阻尼/非实时等限制 | 验收 / 论文边界 |

## 实验文档

实验设计与报告存放在 `experiments/` 子目录，详见 [experiments/README.md](experiments/README.md)。

- **设计文档**（`experiments/design/`）：实验前编写，含目的、假设、参数矩阵
- **实验报告**（`experiments/reports/`）：实验后生成，含结果统计与分析结论
