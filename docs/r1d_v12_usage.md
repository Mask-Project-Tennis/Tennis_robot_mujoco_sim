# R1D × V12 Parity 使用与复现指南

> 版本：2026-11-19（T9 交付）
> 适用范围：本仓库 `feat/r1d-v12-parity` 分支上的 R1D 单臂 V12 共享管线、
> 实物规格球拍、两模式固定案例与论文实验 smoke 链路。
> 本文只描述**已经实现并被测试/产物验证**的命令；超出本文范围的参数组合
> 视为未验收。

## 1. 这是什么

原 V12 任务（`scripts/rm65_mpc_v12.py`）已从 RM65 专用流程改造成
“机器人无关的共享管线”：

```
scripts/rm65_mpc_v12.py
  └── src/sim/v12_config.py        resolve_v12_config(args, explicit) -> ResolvedV12Config
  └── src/sim/v12_robot_adapter.py build_v12_robot(config)           -> V12RobotBundle
  └── src/sim/v12_episode.py       run_v12_episode(config)           -> V12EpisodeResult
  └── src/sim/episode_artifacts.py write_episode_artifacts(...)      -> v12_episode.json/.npz
```

同一入口按 `--robot-config` 分派：

- RM65：`configs/robots/rm65.yaml`（默认），力矩/位置模式、原有输出与
  `__RESULT__` 行保持；
- R1D：`configs/robots/r1d_right.yaml`（右臂主动，左臂静态渲染），
  7 主动关节 + 自由球，执行环境 `nq=14/nv=13`、规划环境 `nq=nv=7`、`nu=7`；
- R1D 的配置注册表显式映射到 V12 专用 runtime/task/racket：
  - `configs/robots/r1d_v12_runtime_right.yaml`
  - `configs/r1d_v12.yaml`
  - `configs/rackets/blade100_v9_reference.yaml`

## 2. 环境

```powershell
conda activate mujoco_tennis
# 本机验证版本：Python 3.11.15 / mujoco 3.9.0 / numpy 2.4.6 / scipy 1.17.1
python setup.py build_ext --inplace      # 若 C++ 扩展未编译；R1D 走 Python FD 后端
```

> Windows 控制台中文乱码时设置 `$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'`。
> R1D 当前不强制/不使用 C++ 后端（`--solver-backend cpp` 会被明确拒绝），
> 产物 `backend.effective` 记录实际内核（通常 `python-fd`）。

## 3. 用户原命令（RM65，必须保持可用）

```bash
# 力矩模式（默认）
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --viewer

# 位置模式
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --viewer --position-mode

# 无渲染/离线，9 m/s
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 9 --no-plot
```

T0 基线证据（seed 42，2026-11-19）：

- torque：`hit_type=active, pos_error=0.118449, min_dist=0.108581, v_racket=1.486`；
- position：`hit_type=active, pos_error=0.149508, min_dist=0.149508, v_racket=1.659`；
- 两模式重复运行 npz 逐键相同（`all_equal=true`，`max_abs_diff=0.0`）；
- 证据：`results/r1d_v12_parity/baseline/` 与
  `experiment_data/r1d_v12_parity/baseline/manifest.json`。

## 4. R1D 固定案例（硬验收）

冻结场景：`configs/scenarios/r1d_v12_active.yaml`（同一成人拍/安装/目标区/
发球 7 m/s + bounce 0.75；两模式只在 MPC/执行器覆盖上不同）。

```bash
# torque：固定 seed42 真实拍面 active
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml --seed 42 --no-plot \
  --serve-box --ball-speed 7 --output results/r1d_active_torque \
  --task-overrides mpc.total_horizon=270 \
  --task-overrides mpc.fixed_horizon=260 \
  --task-overrides mpc.far_threshold=300 \
  --task-overrides mpc.racket_speed=3.0 \
  --task-overrides mpc.target_speed=3.0 \
  --task-overrides mpc.normal_weight=0 \
  --task-overrides mpc.no_follow_through=true

# position：固定 seed42 真实拍面 active
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml --seed 42 --no-plot \
  --serve-box --ball-speed 7 --position-mode --output results/r1d_active_position \
  --task-overrides mpc.total_horizon=270 \
  --task-overrides mpc.fixed_horizon=260 \
  --task-overrides mpc.racket_speed=1.5 \
  --task-overrides mpc.target_speed=1.5 \
  --task-overrides mpc.normal_weight=0
```

冻结结果（v2 产物，`experiment_data/r1d_v12_parity/baseline/` 不含此项，
逐回合产物在 `results/r1d_v12_parity/t7/` 与 `validation/`）：

| 模式 | seed | physical_hit_type | step | face_speed [m/s] | distance [m] | failure_reason | wall [s] |
|---|---|---|---|---|---|---|---|
| torque | 42 | active | 259 | 1.7893 | -0.007512 | null | ≈94.8 |
| position | 42 | active | 255 | 0.8412 | -0.005437 | null | ≈59.0 |

判据：`hit.physical_hit=true` 且 `physical_hit_type=active`，`contact_events` 中
`surface_kind=face`、`physical_contact=true`、`distance_m<=0`，同一物理步
`face_speed_m_s>0.3`。不得用 `pos_error`/回放合成反弹替代。

验收测试：

```bash
python -m pytest tests/test_r1d_v12_acceptance.py -q
# 3 passed；会真实跑两模式固定案例（约 2.5–4 分钟）
```

## 5. R1D 其他常用入口

```bash
# 通用短回合（默认 runtime + task overlay；不保证 active）
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml \
  --seed 7 --no-plot --serve-box --ball-speed 7 \
  --output results/r1d_short

# 显式旧球拍（RM65 parity / 回放对照）
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml \
  --racket-config configs/rackets/rm65_legacy.yaml \
  --seed 42 --no-plot --serve-box --ball-speed 7 \
  --output results/r1d_legacy_racket

# 旧 stage3 固定任务兼容入口（schema v1 仍写）
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml \
  --runtime-config configs/robots/r1d_sim_runtime_right.yaml \
  --sim-task-config configs/r1d_sim_right.yaml \
  --output results/r1d_stage3_legacy --seed 42 --no-plot
```

- `--racket-config` 缺省：R1D 注册表用 `blade100_v9_reference`；RM65 仍用 XML
  内建 legacy 拍，不需要该参数。
- `--task-overrides path=value` 只能覆盖 T3 白名单点路径；未知/非法值在加载
  模型前失败。
- R1D 命令不需要也不应再传 `--runtime-config`/`--sim-task-config`（注册表自动
  指向 V12 专用文件）；传旧三件套仍走旧兼容路径。
- 左臂配置（`r1d_left.yaml`）只作渲染/短回合验证，不承担验收。

## 6. 配置分层

| 层 | 文件 | 职责 |
|---|---|---|
| 机器人描述 | `configs/robots/rm65.yaml`、`configs/robots/r1d_right.yaml` | robot_id / model_path / 主动关节名；无任务数值 |
| 运行参数 | `configs/robots/rm65.yaml` 对应 runtime、`r1d_v12_runtime_right.yaml` | init_q、PD、限位、底座/升降/躯干、球拍安装；R1D 值多为 provisional |
| V12 任务 | `configs/default.yaml` → `v5_active_hit.yaml` → `mpc.yaml` → CLI | 所有机器人的共同解析基线 |
| R1D 覆盖 | `configs/r1d_v12.yaml` | 只放确有差异字段（target 等），不复制基线 |
| 球拍 | `configs/rackets/blade100_v9_reference.yaml`、`rm65_legacy.yaml` | 几何/质量/COM/惯量/安装/来源；TCP=拍面中心 |
| 场景冻结 | `configs/scenarios/r1d_v12_active.yaml` | 两模式覆盖、共同物理、seed、哈希、复现命令 |

优先级：`原 V12 解析 → R1D 稀疏覆盖 → 显式 CLI/--task-overrides`；
serve-box 自动推导值只作用于“未被显式指定”的字段。

### 6.1 球场白线 profile（V01）

R1D 模板磁盘字节固定以旧 RM65 白线生成；运行时由 `scene.court_profile`
选择渲染口径（只影响视觉 plane，不参与碰撞、不改物理与轨迹）：

| profile | 说明 |
|---|---|
| `legacy_v1`（默认） | 旧 RM65 的 10 条白线（0.10 m 线宽；整场十字中线/底线±6/发球线±3/边线），保持既有产物与对拍口径 |
| `itf_2026_v1` | ITF 2026 规则第 1 条标准 12 条线（双打 23.77×10.97、单打 8.23、发球线距网 6.40、中心标记 0.10；0.05 m 线宽，无横贯球网线） |

显式使用标准球场（不复制既有 runtime，仅叠加一个字段的变体文件）：

```bash
python scripts/rm65_mpc_v12.py \
  --robot-config configs/robots/r1d_right.yaml \
  --runtime-config configs/robots/r1d_v12_runtime_right_itf.yaml \
  --seed 42 --no-plot --serve-box --ball-speed 7 --output results/r1d_itf_demo
```

`configs/robots/r1d_v12_runtime_right_itf.yaml` 逐字段复制
`r1d_v12_runtime_right.yaml`，唯一差异是 `scene.court_profile: itf_2026_v1`
（`tests/test_court_scene_integration.py` 断言两者仅此字段不同，且同 seed
下两 profile 的 `X/U/ball_pos/ball_vel/hit` 逐位一致）。旧 runtime/YAML
缺该字段时按 `legacy_v1` 解释，磁盘模板不改写。为保持 T7 冻结产物的
`input_sha256["scene_snapshot"]` 不变，`legacy_v1`（既有默认）不向
`metrics.r1d_artifact.fixed_scene_params` 追加 court 字段；只有显式非 legacy
（如 `itf_2026_v1`）才记录 `court_profile/court_geometry_version/
court_line_count`，便于回放追溯视觉口径。

注意：ITF 产物的 `snapshot/files/...` 中 `model` 角色仍是磁盘上的 legacy 模板字节，
实际编译模型由 runtime 在内存中按 `court_profile` 重建白线。回放会依据
`runtime_path` 重建，因此球线布局一致；但若只审计快照里的 model 文件，会看到
legacy 布局——白线口径应以 `fixed_scene_params` 的 court 字段为准。

## 7. 产物与回放

每个回合输出（`--output DIR`）：

```
DIR/
  v12_episode.json        # schema v12_episode_v2（新）
  v12_episode.npz         # X(N+1,nq), U(N,nu), ball_pos/ball_vel/t, racket_pos
  snapshot/files/...      # robot/runtime/task/racket YAML + 模型 XML 快照
  r1d_episode.json/.npz   # 旧 R1D schema v1 兼容投影（仅 R1D）
```

v2 JSON 关键字段：`robot_id/mode/racket_id/joint_names/dt/config_sha256/
input_sha256/environment/seed/backend/serve/target/hit/metrics`。
`hit.contact_events[]` 为真实物理接触事件：

```
step, time_s, ball_geom_name, other_geom_name,
surface_kind(face/frame/handle/other), distance_m,
face_speed_m_s, physical_contact, hit_type
```

只有 `distance_m<=0` 才记为 `physical_contact`；正距离 margin 只进
`proximity`，不产生 hit。失败/安全停止回合也会落盘 `failure_reason`。

回放：

```bash
python scripts/replay_trajectory.py --help
# 或使用 T6 的 episode_artifacts 回放 API：
#   load_episode_artifacts / build_replay_bundle / replay_episode_artifacts
```

回放优先用 `snapshot/` 快照；若模型文件被替换且 sha 与 `input_sha256.model`
不符，会抛 `ValueError` 拒绝回放。调试慢放必须显式传速度参数；默认按正常
速度、按关节名映射状态（不假设前六个槽）。

回放还会消费产物中保存的 `runtime_overrides`（`max_tcp`、
`terminal_exempt_steps`、`dq_max`、limits 文件快照等），不会重新落回
默认限值。R1D 新产物必须携带装配依赖清单（`dependency_manifest_source=
result`）；缺清单会拒绝写入，旧产物按 legacy 分支逐项校验 SHA 后回放。

多机器人关节查看器：

```bash
python scripts/tools/rm65_joint_viewer.py                     # 默认 RM65，12 滑条
python scripts/tools/rm65_joint_viewer.py \
  --robot-config configs/robots/r1d_right.yaml --racket-config configs/rackets/blade100_v9_reference.yaml
# R1D 7 滑条，初值来自 runtime，限位来自模型
```

## 8. 论文实验 smoke 链路

实现位置：

- 运行器：`scripts/exp/run_r1d_paper_smoke.py`（manifest 驱动）；
- manifest：`configs/experiments/r1d_paper_smoke.yaml`
  （19 组正文 + 3 个额外依赖；每比较 cell ≥2 共同 seed 100/101；
  torque 全遍历、position 代表链）；
- 批量：`scripts/exp/batch_runner.py`（身份列/resume/timeout/data-root）；
- 统计：`scripts/extract/extract_r1d_smoke_stats.py`（physical/legacy 双口径）
  与 `scripts/extract/paired_stats.py` 等；
- 绘图：`scripts/plot/paper_figs.py --data-root/--out/--fig table3`、
  `scripts/tools/render_swing_hero.py`；
- per-case smoke 图：`scripts/plot/plot_r1d_smoke_stats.py`。

最小可复现命令：

```bash
# 1) 运行 2 seed × 1 case（调试规模；正式 smoke 不加 --limit）
python scripts/exp/run_r1d_paper_smoke.py \
  --data-root experiment_data/r1d_v12_parity/paper_smoke/batch \
  --output experiment_data/r1d_v12_parity/paper_smoke \
  --limit 2 --seeds 100 101

# 2) 单 case 统计
python scripts/extract/extract_r1d_smoke_stats.py \
  --experiment exp15_speed_v2 --case speed7 \
  --data-root experiment_data/r1d_v12_parity/paper_smoke/batch \
  --output results/r1d_v12_parity/t9/smoke_case_stats

# 3) 单 case 图
python scripts/plot/plot_r1d_smoke_stats.py \
  --stats-root results/r1d_v12_parity/t9/smoke_case_stats \
  --experiment exp15_speed_v2 --case speed7 \
  --output results/r1d_v12_parity/t9/smoke_case_figures

# 4) 统计 JSON → 论文图/表（小样本预览，不是论文结论）
python scripts/extract/paired_stats.py \
  --data-root experiment_data/r1d_v12_parity/paper_smoke/batch \
  --output results/r1d_v12_parity/paper_smoke/stats
python scripts/plot/paper_figs.py \
  --data-root experiment_data/r1d_v12_parity/paper_smoke \
  --stats results/r1d_v12_parity/paper_smoke/stats/06-stats-active.json \
  --out results/r1d_v12_parity/paper_smoke/figures \
  --fig 5 6 table3
```

实测（2026-11-19）：`--limit 2 --seeds 100 101` 只跑 exp15_speed_v2/speed7
两个 torque seed，均为 miss（`safety_filter_stop`）；因此 Fig.5/6 因数据不足
**如实跳过**，table3 五格全部 `--`（不可估计）。这证明链路可运行，不构成
论文统计结论；不得把不可估计格补成 0，也不得用 RM65 旧数据顶替。

全量重跑（论文矩阵 19 组）请使用不带 `--limit` 的同一命令；本任务只交付
smoke 规模，未跑完整批量。

## 9. 留出集与限制（必须如实报告）

20 回合留出集（seeds 0–9 × 两模式；冻结后未再调参）：

- `torque`：10/10 `physical_hit_type=active`（face 1.728–1.805 m/s），但
  接触后 7–11 步出现 `safety_filter_stop`（腕 pitch 越限）；
- `position`：3/10 active（seed 0/8/9），其余在接触前安全停止；
- 全部 20 行为 `status=ok`，0 error/timeout，无删行/重试覆盖；
- 数据：`experiment_data/r1d_v12_parity/validation/{results.csv,results.jsonl,runs/,logs/,manifest.json}`。

限制：

1. **不是实时**：R1D 固定案例单回合 wall 约 59–95 s，首 plan 36–61 s（Python FD）；
   论文计时章节不得引用本机 R1D 数值作为实时性能。
2. **R1D 关节阻尼/armature 为估计值**：R1D 官方 URDF 未声明，V12 路径在
   `src/sim/v12_robot_adapter.py` 构造时向内存模型注入
   `damping=(2,2,1,0.5,0.3,0.2,0.1)`、`armature=(0.2,0.2,0.1,0.05,0.03,0.02,0.01)`
   （estimated/provisional；参考 RM65 同口径，不改 XML/URDF）。替换官方值需
   重跑 T7 验收与留出集。该组常量仅存在于代码层，**不写入 v2 产物/snapshot
   哈希**；因此旧产物仍可通过哈希回放，但动力学不会随常量变化复现（已知限制，
   详见 T7 报告 §3.2）。
3. **左臂只渲染**：左臂固定姿态，不参与控制；左侧固定案例不承担验收。
4. **R1D X 墙未启用**（`enabled=false`），中线保护由 T2 的命名保护接触对承担；
   不要把 RM65 的 `world_x/-0.1` 直接套到 R1D。
5. **旧测试遗留失败**：全量 `pytest tests/` 中 12 failed + 3 errors 为 T0 基线
   既有（RM65 时代/阶段 3 旧测试与新 R1D 配置的预期不一致），逐 ID 见
   `results/r1d_v12_parity/t9/full_final.log` 与 T7 报告 §7；本次迁移未新增失败。
   R1D 验收以 `tests/test_r1d_v12_acceptance.py` 为准。
6. 论文 `paper/` 与旧 `experiment_data/exp*` 未被修改；本任务全部输出在
   `experiment_data/r1d_v12_parity/` 与 `results/r1d_v12_parity/`。
7. 本次不安装 `unitree_sdk2`、不连真机、不验证真机驱动/2 ms 控制周期。

## 10. 证据索引

| 内容 | 路径 |
|---|---|
| 设计 | `docs/superpowers/specs/2026-11-18-r1d-v12-parity-design.md` |
| 实施计划 | `docs/superpowers/plans/2026-11-18-r1d-v12-parity-implementation.md` |
| 施工方案 | `docs/superpowers/plans/2026-11-19-r1d-v12-parity-construction.md` |
| 论文矩阵 | `docs/experiments/design/r1d_v12_parity_matrix.md` |
| T0 基线 | `results/r1d_v12_parity/baseline/`、`experiment_data/r1d_v12_parity/baseline/` |
| T7 报告 | `docs/experiments/reports/2026-11-18-r1d-v12-parity.md` |
| T7 留出集 | `experiment_data/r1d_v12_parity/validation/` |
| T8 smoke | `experiment_data/r1d_v12_parity/paper_smoke/`、`results/r1d_v12_parity/paper_smoke/` |
| T9 预检 | `results/r1d_v12_parity/t9/pretest_full.log` |
| 施工 ledger | `.superpowers/sdd/2026-11-19-r1d-v12-parity-construction/progress.md` |
