# scripts/ 目录结构

## 组织原则

- **根目录保留活跃核心脚本**（V12 + 工具入口）
- **按功能分类入子目录**（sim/exp/extract/plot/tools/test）
- **旧脚本归档至 `archive/`**（V6-V10 + tube + 旧实验，详见 `archive/README.md`）

## 目录说明

| 目录 | 数量 | 内容 | 运行方式 |
|------|------|------|---------|
| `scripts/` (根) | 7 | 活跃核心脚本 | `python scripts/xxx.py --args` |
| `sim/` | 11 | 独立仿真（MPC/iLQR/Training/工具） | `python scripts/sim/xxx.py --args` |
| `exp/` | ~20 | 活跃实验基础设施（exp9-15） | `python scripts/exp/xxx.py --args` |
| `extract/` | 9 | 结果提取：日志 → CSV | `python scripts/extract/xxx.py` |
| `plot/` | 14 | 论文图表生成 | `python scripts/plot/xxx.py` |
| `tools/` | 10 | 独立工具（查看器·扫描·诊断·可视化） | `python scripts/tools/xxx.py` |
| `test/` | 10 | 快速验证脚本 | `python scripts/test/xxx.py` |
| `archive/` | 53 | 已归档（V6-V10 + tube + 旧实验） | 详见 `archive/README.md` |

## 根目录（7 个）

| 文件 | 用途 |
|------|------|
| `rm65_mpc_v12.py` | ★ V12：EpisodeRunner 管线架构 + MPCController 策略化（命中率 85.7%） |
| `rm65_mpc_v11.py` | V11 薄壳（29 行，委托到 V12 main） |
| `rm65_mpc_ilqr_5_5.py` | Tube/iLQR 基线（工具脚本依赖） |
| `rm65_evaluate.py` | 评估脚本 |
| `run_20hits_video.py` | 连续 20 次击打视频生成 |
| `run_real_robot.py` | 真机入口 |
| `replay_trajectory.py` | 轨迹回放（含 --speed 安全检查） |

## sim/ — 独立仿真（11 个）

| 文件 | 用途 |
|------|------|
| `rm65_mpc_ilqt.py` | 简化 MPC+iLQR（无 Tube） |
| `rm65_mpc_ilqr_5_7_python.py` | 纯 Python iLQR benchmark |
| `rm65_mpc_fast.py` | 快速模式 |
| `rm65_mpc_fast_workspace.py` | 快速模式 + workspace 约束 |
| `rm65_constrained_fast.py` | 约束快速模式 |
| `rm65_joint_limit.py` | 关节限速版本 |
| `rm65_batch_viz.py` | 批量击球 + 回放 + 视频 |
| `rm65_realtime_batch.py` | 批量评估（20 球汇总统计） |
| `rm65_realtime_play.py` | 实时连续击球 |
| `train_mpc.py` | MPC Rolling Planner 训练 |
| `train_ilqt.py` | 单次 iLQR 优化 + 可视化 |

## exp/ — 活跃实验基础设施

活跃实验脚本（引用 V11/V12 或独立的批量运行器/提取器）。
完整清单见 `scripts/exp/` 目录。

## archive/ — 已归档（53 个）

V6-V10 + tube 变体 + 旧实验脚本，保留供 `git log --follow` 历史追溯。
详见 `archive/README.md`。
