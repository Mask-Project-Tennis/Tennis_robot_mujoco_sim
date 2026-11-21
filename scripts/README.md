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
| `exp/` | ~21 + 子包 | 活跃实验基础设施（exp9-15 + R1D 论文 smoke `paper_smoke/`） | `python scripts/exp/xxx.py --args` |
| `extract/` | 10 + 子包 | 结果提取：日志/产物 → CSV/JSON（含 `paired_analysis/`） | `python scripts/extract/xxx.py` |
| `plot/` | 15 + 子包 | 论文图表生成 + R1D per-case smoke 图（含 `paper/`） | `python scripts/plot/xxx.py` |
| `tools/` | 22 | 独立工具（查看器·扫描·诊断·可视化·轨迹残影渲染·R1D 模型构建与预览·R1D 只读/低速诊断·R1D 传输发布基准） | `python scripts/tools/xxx.py` |
| `test/` | 10 | 快速验证脚本 | `python scripts/test/xxx.py` |
| `archive/` | 59 | 已归档（V6-V10 + tube + 旧实验 + 旧 sim 脚本） | 详见 `archive/README.md` |

## 可维护性子包与兼容入口

| 关注点 | 新入口 | 兼容入口 |
|---|---|---|
| 论文 smoke manifest/任务/续跑 | `scripts.exp.paper_smoke` | `scripts/exp/run_r1d_paper_smoke.py` |
| 配对统计 | `scripts.extract.paired_analysis` | `scripts/extract/paired_stats.py` |
| 论文图表 | `scripts.plot.paper` | `scripts/plot/paper_figs.py` |

- 新代码优先 import 新包；兼容入口必须转发同一对象，禁止第二份实现。
- `scripts/` 不得被 `src/` 反向 import；`batch_runner.py`/`run_catalog.py` 属主代理独占。
- 详细说明见各子包 `README.md`。

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

## R1D 单臂控制 + 双侧渲染入口（阶段 3 旧路径，保留兼容）

> 2026-11-19 起新增 R1D×V12 共享管线入口（见下一节）。本节旧三件套路径仍可运行，
> 产物为旧 `r1d_sim_episode_v1`；新工作请优先使用 R1D×V12 路径。

`scripts/rm65_mpc_v12.py` 同时是 R1-A7-D 单臂闭环仿真入口（实现委托 `src/sim/r1d_runner.py::run_r1d_episode`）。
R1D 路径只接受 `--robot-config`/`--runtime-config`/`--sim-task-config`/`--output`/`--seed`/`--no-plot`/`--viewer`/`--log-level`（其中 `--seed`/`--no-plot`/`--viewer`/`--output`/`--log-level` 为共享基础参数，`--robot-config`/`--runtime-config`/`--sim-task-config` 为 R1D 专属配置项），其余 RM65 旧参数（`--serve-box`/`--position-mode`/`--async-replan`/`--fix-joint5`/`--dump-trajectory` 等）被拒绝（退出码 2）：

```bash
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml \
  --runtime-config configs/robots/r1d_sim_runtime_right.yaml \
  --sim-task-config configs/r1d_sim_right.yaml \
  --output results/r1d_episode_right --seed 42 --viewer
```

> 批处理/无头环境把 `--viewer` 换成 `--no-plot`（R1D 下仅表示不回放；`--viewer` 会阻塞到窗口关闭）。

- 三份 YAML 必须显式给出，侧别（left/right）与路径一致，否则退出码 2。
- 模型为“单侧主动臂 + 对侧静态渲染臂”：右臂模板中左臂无关节/无执行器，只作为视觉挂件；`nq/nv/nu=7` 不变。两侧模板都带 RM65 checker 球场/白线与落地支撑柱。
- 产物：`--output` 目录下 `r1d_episode.json`（schema `r1d_sim_episode_v1`：输入 SHA256、固定场景参数、hit/碰撞/耗时/指标）+ `r1d_episode.npz`（`X`(N+1,14)/`U`(N,7)/`ball_pos`/`ball_vel`(N+1,3)/`t`）。
- 可视化：加 `--viewer` 会在产物写完后用 MuJoCo 被动查看器循环回放（0.2× 慢放；窗口关闭后进程退出）。无图形环境时只记警告，不影响产物。
- 固定可达案例（阶段 3 验收）：`configs/scenarios/r1d_stage3_reachable.yaml`（右臂三路径、`seed 42`、`expected_hit_type: active`；路径相对 scenario 文件解析）。
- 离线仿真口径：单回合首 plan 约 6–14 s、整回合 8–14 s（Python 有限差分后端），非实时。
- 左侧命令仍可运行（`--robot-config ... r1d_left.yaml` 等三份 left 配置），但左侧现在是渲染/短回合验证，不再承担固定案例验收。

## R1D × V12 Parity（阶段 4，2026-11-19）

同一 `scripts/rm65_mpc_v12.py` 在 `--robot-config configs/robots/r1d_right.yaml` 时走
`src/sim/configuration` → `v12_robot_adapter.py` → `src/sim/episode` → `src/sim/artifacts`
共享管线；无需再传旧三件套（注册表自动指向 `r1d_v12_runtime_right.yaml`/`r1d_v12.yaml`/
`blade100_v9_reference.yaml`）。`--serve-box/--ball-speed/--position-mode/--seed/--no-plot/
--async-replan/--racket-config/--task-overrides` 均可用于 R1D；旧 `--runtime-config/--sim-task-config`
三件套仍兼容。

```bash
# 固定案例（两模式真实 active）完整命令：见 docs/r1d_v12_usage.md §4
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml --seed 42 --no-plot \
  --serve-box --ball-speed 7 --output results/r1d_active_position --position-mode \
  --task-overrides mpc.total_horizon=270 --task-overrides mpc.fixed_horizon=260 \
  --task-overrides mpc.racket_speed=1.5 --task-overrides mpc.target_speed=1.5 \
  --task-overrides mpc.normal_weight=0

# R1D 论文 smoke（19 组+3 额外依赖，默认每 cell 2 seeds 100/101）
python scripts/exp/run_r1d_paper_smoke.py \
  --manifest configs/experiments/r1d_paper_smoke.yaml \
  --output experiment_data/r1d_v12_parity/paper_smoke
```

- 产物：`v12_episode.json`（schema `v12_episode_v2`，含 ContactEvent/身份/哈希）+ `v12_episode.npz`
  + `snapshot/`；R1D 另写旧 `r1d_episode.json/.npz` 兼容投影。
- 统计/绘图：`scripts/extract/extract_r1d_smoke_stats.py`、`scripts/plot/plot_r1d_smoke_stats.py`、
  `scripts/plot/paper_figs.py --data-root/--out/--fig table3`、`scripts/tools/render_swing_hero.py`。
- 验收测试：`tests/test_r1d_v12_acceptance.py`；完整使用/限制/证据见 `docs/r1d_v12_usage.md`。

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

## tools/ — 独立工具（阶段 2 新增 3 个，阶段 4 新增 2 个）

| 文件 | 用途 |
|------|------|
| `inspect_robot_model.py` | R1D 模型预览与可追溯参数导出（`--robot-config` r1d_left/right.yaml + `--runtime-config` 必传，输出 JSON 报告 + front/side/oblique 三视图，`--render` 可选；异常配置退出 2） |
| `build_r1d_model.py` | R1D 官方整机 URDF → 双侧裸臂/环境/资产片段导出（字节确定，勿手改产物） |
| `build_d2_model.py` | D2 历史资产构建（型号更正后保留，不作 R1-A7-D 验收） |
| `test_unitree_r1d.py` | R1D 只读/低速诊断工具（阶段 4；四基础强制参数 `--robot-config`/`--hardware-config`/`--output-dir`/`--acquire-timeout-s`，`--mode read_only|tracking` 缺省 read_only；tracking 另强制 `--tracking-error-limit-rad` 与 `--stop-latency-limit-s` 验收阈值，且需 G4 冻结的 motion 档配置否则如实失败退出非 0；产物 JSON verdict + provenance 可追踪指标。**实机诊断入口：G3 通过前不得连接硬件，见 `docs/experiments/reports/2026-11-17-r1d-field-standby-protocol.md`**） |
| `bench_r1d_transport_publish.py` | R1D 传输层发布路径基准（离线 fake SDK，不加载 unitree_sdk2py/cyclonedds）：对照冻结基线 `9a5b3f7` 旧无锁 publish/close 与当前 `SdkR1DTransport`，测发布 p50/p99、组包+CRC、RLock 开销与 close 在途 Write 等待；强制自检 `write_in_flight_at_close` 基线与当前实现分别为 True/False，失败非零退出。`--pairs/--rounds/--build-iterations/--lock-iterations/--close-block-ms/--json`，非性能门禁、不外推实机 DDS 时序） |

其余 17 个既有工具（查看器·扫描·诊断·可视化·轨迹残影渲染）完整清单见 `scripts/tools/` 目录。

## archive/ — 已归档（59 个）

V6-V10 + tube 变体 + 旧实验脚本 + 旧 sim 脚本（HittingCost 子类 + 旧批量/实时），
保留供 `git log --follow` 历史追溯。详见 `archive/README.md`。
