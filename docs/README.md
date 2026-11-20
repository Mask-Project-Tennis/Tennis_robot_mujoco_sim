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

### Unitree R1-A7-D 阶段报告

| 报告 | 内容 |
|------|------|
| `experiments/reports/2026-11-14-unitree-stage1.md` | 阶段 1 验收报告：多机器人描述加载层（RobotProfile/ModelBinding/入口隔离；含评审勘误后状态） |
| `experiments/reports/2026-11-16-unitree-stage2.md` | 阶段 2 验收报告：R1-A7-D 模型与运动学链（参数层/安全加载器/双侧场景/运动学/预览 CLI） |
| `experiments/reports/2026-11-16-unitree-stage3.md` | 阶段 3 验收报告：R1D 七维仿真管线与多机器人入口（含 baseline 数据报告与固定可达案例） |
| `experiments/reports/2026-11-16-r1d-sdk-control-contract.md` | R1D SDK 控制契约调研报告（G1 决策表：话题/IDL 布局/CRC/控制权生命周期证据链） |
| `experiments/reports/2026-11-17-unitree-stage4.md` | 阶段 4 报告：R1-A7-D 单臂驱动软件层（G0-G5 门槛状态、十二切片明细、回归/ruff 实测、阶段 5 交接；评审后状态=实现完成、验收评审已裁决并修复、待实机验收） |
| `experiments/reports/2026-11-17-stage4-review.md` | 阶段 4 团队评审裁决（5 只读 reviewer + Lead 复验：0 Critical / 10 Important / 36 Minor 与修复处置） |
| `experiments/reports/2026-11-18-stage4-acceptance-review.md` | 阶段 4 复审裁决（R1–R6：2 P1 + 4 P2 与修复轮处置、红线验证；§6 第二轮复验 2 Blocking 与处置） |
| `experiments/reports/2026-11-18-stage4-p1-fixes.md` | 阶段 4 三项 P1 修复实施报告（传输会话隔离 / 停止收尾合并异步故障 / 跟踪验收运动证据；红线、发布基准、1675+3 回归与终审） |
| `experiments/reports/2026-11-17-r1d-field-standby-protocol.md` | R1D 现场待命规程：4.9 只读核对 / 4.10 低速跟踪的 G3/G4 前置与现场步骤（不构成实机授权） |
