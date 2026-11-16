# scripts/ 目录结构

## 组织原则

- **根目录保留活跃核心脚本**（V12 + 工具入口）
- **按功能分类入子目录**（sim/exp/extract/plot/tools/test）
- **旧脚本归档至 `archive/`**（V6-V10 + tube + 旧实验，详见 `archive/README.md`）

## 目录说明

| 目录 | 数量 | 内容 | 运行方式 |
|------|------|------|---------|
 | `scripts/` (根) | 7 | 活跃核心脚本 | `python scripts/xxx.py --args` |
| `sim/` | 5 | 独立仿真（MPC/iLQR/Training/工具） | `python scripts/sim/xxx.py --args` |
| `exp/` | ~20 | 活跃实验基础设施（exp9-15） | `python scripts/exp/xxx.py --args` |
| `extract/` | 9 | 结果提取：日志 → CSV | `python scripts/extract/xxx.py` |
| `plot/` | 14 | 论文图表生成 | `python scripts/plot/xxx.py` |
| `tools/` | 20 | 独立工具（查看器·扫描·诊断·可视化·轨迹残影渲染·R1D 模型构建与预览） | `python scripts/tools/xxx.py` |
| `test/` | 10 | 快速验证脚本 | `python scripts/test/xxx.py` |
| `archive/` | 59 | 已归档（V6-V10 + tube + 旧实验 + 旧 sim 脚本） | 详见 `archive/README.md` |

## 根目录（7 个）

| 文件 | 用途 |
|------|------|
| `rm65_mpc_v12.py` | ★ V12：EpisodeRunner 管线架构 + MPCController 策略化（命中率 85.7%）；阶段 3 起支持 R1D 单臂仿真入口（`--robot-config`/`--runtime-config`/`--sim-task-config`，见下方 R1D 小节） |
| `rm65_mpc_v11.py` | V11 薄壳（29 行，委托到 V12 main） |
| `rm65_mpc_ilqr_5_5.py` | Tube/iLQR 基线（工具脚本依赖） |
| `rm65_evaluate.py` | 评估脚本 |
| `run_20hits_video.py` | 连续 20 次击打视频生成 |
| `run_real_robot.py` | 真机入口（--robot-config 可选机器人描述，默认 configs/robots/rm65.yaml；描述加载/校验失败以日志+退出码 2 呈现） |
| `replay_trajectory.py` | 轨迹回放（含 --speed 安全检查） |

## R1D 单臂仿真入口（阶段 3）

`scripts/rm65_mpc_v12.py` 同时是 R1-A7-D 单臂闭环仿真入口（实现委托 `src/sim/r1d_runner.py::run_r1d_episode`）。
R1D 路径只接受 `--robot-config`/`--runtime-config`/`--sim-task-config`/`--output`/`--seed`/`--no-plot`/`--log-level`（其中 `--seed`/`--no-plot`/`--output`/`--log-level` 为共享基础参数，`--robot-config`/`--runtime-config`/`--sim-task-config` 为 R1D 专属配置项），其余 RM65 旧参数（`--serve-box`/`--position-mode`/`--async-replan`/`--fix-joint5`/`--dump-trajectory` 等）被拒绝（退出码 2）：

```bash
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_left.yaml \
  --runtime-config configs/robots/r1d_sim_runtime_left.yaml \
  --sim-task-config configs/r1d_sim_left.yaml \
  --output results/r1d_episode --seed 42 --no-plot
```

- 三份 YAML 必须显式给出，侧别（left/right）与路径一致，否则退出码 2。
- 产物：`--output` 目录下 `r1d_episode.json`（schema `r1d_sim_episode_v1`：输入 SHA256、固定场景参数、hit/碰撞/耗时/指标）+ `r1d_episode.npz`（`X`(N+1,14)/`U`(N,7)/`ball_pos`/`ball_vel`(N+1,3)/`t`）。
- 固定可达案例（阶段 3 验收）：`configs/scenarios/r1d_stage3_reachable.yaml`（`seed 42`，`expected_hit_type: active`；路径相对 scenario 文件解析）。
- 离线仿真口径：单回合首 plan 约 6–14 s、整回合 8–14 s（Python 有限差分后端），非实时。

## sim/ — 独立仿真（5 个）

| 文件 | 用途 |
|------|------|
| `rm65_mpc_ilqt.py` | 简化 MPC+iLQR（无 Tube） |
| `rm65_mpc_ilqr_5_7_python.py` | 纯 Python iLQR benchmark |
| `rm65_mpc_fast.py` | 快速模式 |
| `train_mpc.py` | MPC Rolling Planner 训练 |
| `train_ilqt.py` | 单次 iLQR 优化 + 可视化 |

> 已归档至 `archive/sim/`：`rm65_constrained_fast.py`、`rm65_joint_limit.py`、
> `rm65_mpc_fast_workspace.py`（HittingCost 子类，随组合化重构归档）、
> `rm65_batch_viz.py`、`rm65_realtime_batch.py`、`rm65_realtime_play.py`（旧批量/实时脚本）。

## exp/ — 活跃实验基础设施

活跃实验脚本（引用 V11/V12 或独立的批量运行器/提取器）。
完整清单见 `scripts/exp/` 目录。

## tools/ — 独立工具（阶段 2 新增 3 个）

| 文件 | 用途 |
|------|------|
| `inspect_robot_model.py` | R1D 模型预览与可追溯参数导出（`--robot-config` r1d_left/right.yaml + `--runtime-config` 必传，输出 JSON 报告 + front/side/oblique 三视图，`--render` 可选；异常配置退出 2） |
| `build_r1d_model.py` | R1D 官方整机 URDF → 双侧裸臂/环境/资产片段导出（字节确定，勿手改产物） |
| `build_d2_model.py` | D2 历史资产构建（型号更正后保留，不作 R1-A7-D 验收） |

其余 17 个既有工具（查看器·扫描·诊断·可视化·轨迹残影渲染）完整清单见 `scripts/tools/` 目录。

## archive/ — 已归档（59 个）

V6-V10 + tube 变体 + 旧实验脚本 + 旧 sim 脚本（HittingCost 子类 + 旧批量/实时），
保留供 `git log --follow` 历史追溯。详见 `archive/README.md`。
