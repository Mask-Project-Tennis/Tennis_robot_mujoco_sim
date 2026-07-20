# Trajectory Studio — 交互式轨迹工作台设计文档

日期: 2026-07-17
分支: `feat/real-robot-replay-testing`

## 目标

将现有的三步分离流程（`inspect_trajectory` 检查 → 手动看图 → `replay_trajectory` 执行）
整合为一个交互式终端工作台，支持：浏览轨迹 → 安全检查 → 可视化确认 → 真机重演。

## 设计原则

- **组合优于继承**：Studio 是编排层，调用已有函数，不创建新类/Protocol/基类
- **整合而非重建**：把已有工具连成一个 `input()` 菜单驱动器
- **零新依赖**：只用已有的 numpy、matplotlib、mujoco、pathlib

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `src/real/replay_pipeline.py` | 从 `replay_trajectory.py` 抽取 `pre_motion()` + `run_replay()` + `ReplayConfig` |
| 新建 | `scripts/trajectory_studio.py` | 交互菜单：浏览→检查→可视化→确认→重演 |
| 新建 | `tests/test_replay_pipeline.py` | `run_replay()` + FakeRobot 端到端 |
| 新建 | `tests/test_trajectory_studio.py` | 扫描/元数据提取/安全摘要（纯函数） |
| 改瘦 | `scripts/replay_trajectory.py` | 382→~50 行薄壳：argparse → `ReplayConfig` → `run_replay()` |
| 小改 | `scripts/tools/inspect_trajectory.py` | 抽取 `plot_trajectory()` 供 Studio 复用 |

## 复用映射

```
Studio 功能            调用的已有函数
────────────────────────────────────────────────────────
加载 .npz          →   TrajectoryRecorder.load()
关节限位检查        →   inspect_trajectory.check_joint_limits()
平滑性检查          →   inspect_trajectory.check_smoothness()
TCP 速度检查        →   inspect_trajectory.check_tcp_speed()
推荐 speed 因子      →   inspect_trajectory.check_tcp_speed()[1]
限位/TCP 加载       →   inspect_trajectory._load_limits()
2D 关节+TCP 图表   →   inspect_trajectory.plot_trajectory()  ← 待抽取
3D MuJoCo 预览     →   内联 ~25 行（RM65Env + launch_passive，运动学回放）
真机重演            →   replay_pipeline.run_replay(ReplayConfig)
真机录制对比        →   plot_trajectory(overlay=) 集成
```

## 架构

### 重演模块抽取（`src/real/replay_pipeline.py`）

从 `scripts/replay_trajectory.py` 提取核心逻辑为可 import 的模块：

```python
@dataclass
class ReplayConfig:
    trajectory_path: Path
    speed: float = 0.1
    use_actual: bool = True
    mock: bool = False
    record: Path | None = None
    pre_motion_duration: float = 10.0
    max_tcp_speed: float = 0.0
    target_dt: float | None = None
    force_mode: bool = False
    config_path: Path = _DEFAULT_CONFIG_PATH

def pre_motion(...) -> bool:   # 整体搬入，签名不变
def run_replay(cfg: ReplayConfig) -> int:  # 从 main() 提取，返回步数
```

`replay_trajectory.py` 变为：`_parse_args()` → 构造 `ReplayConfig` → `run_replay(cfg)`

### inspect_trajectory 抽取

从 `main()` 第 300-334 行的内联绘图代码提取为：

```python
def plot_trajectory(
    traj: ReplayTrajectory,
    q_check: np.ndarray,
    tcp_speeds: np.ndarray,
    title: str = "",
    overlay: tuple[ReplayTrajectory, np.ndarray, np.ndarray] | None = None,
) -> None:
```

`overlay` 参数为 `(traj2, q2, speeds2)` 三元组，用于真机录制对比模式。

### 3D MuJoCo 预览（内联函数）

运动学回放（不跑物理，仅设 qpos → mj_forward → viewer.sync）：

```python
def _preview_3d(traj: ReplayTrajectory, use_actual: bool) -> None:
    env = RM65Env(MODEL_PATH)
    q_data = traj.q_actual if use_actual else traj.q_desired
    env.reset(traj.init_q)
    env.init_q_left = traj.init_q_left
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        for i in range(len(q_data)):
            if not v.is_running(): return
            env.data.qpos[:6] = q_data[i]
            if i < len(traj.ball_pos):
                env.data.qpos[12:15] = traj.ball_pos[i]
            mujoco.mj_forward(env.model, env.data)
            v.sync()
            time.sleep(traj.dt)
```

## Studio 交互流程

```
╔══════════════════════════════════════════╗
║     轨迹工作台 Trajectory Studio          ║
╠══════════════════════════════════════════╣

规划轨迹:
  [1] stage0_traj_s1.npz            131步  0.35m  hit@65  ✓安全
  [2] stage0_traj_swing_s21.npz     153步  0.62m  hit@77  ✓安全
  ...

真机录制:
  [R1] stage3_real_swing_s21_speed050.npz   speed=0.5

选择编号 (r=刷新 q=退出): 2

┌─ stage0_traj_swing_s21.npz ───────────────────┐
│ 步数: 153    路径: 0.62m    击球步: 77        │
│ 模式: position    命中: clean_hit             │
│                                              │
│ 关节裕度: J2 距下限 12.3°               ✓    │
│ 平滑性:   max跳变 2.1°                   ✓   │
│ TCP峰值:  1.24 m/s → 建议 speed ≤ 0.81  ✓    │
└──────────────────────────────────────────────┘

  [1] 2D 图表 (关节角度 + TCP 位置/速度)
  [2] 3D 仿真预览 (MuJoCo 挥拍动画 + 球轨迹)
  [3] 对比真机录制
  [4] 真机重演
  [5] 返回列表
```

真机重演子流程：选速度 → 安全确认 → `run_replay(ReplayConfig(...))`

## 测试策略

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_replay_pipeline.py` | `run_replay()` + FakeRobot mock 端到端；`ReplayConfig` 字段映射 |
| `test_trajectory_studio.py` | `_scan_trajectories()`、`_format_summary()`、`_safety_card()` 纯函数 |
| `test_replay_trajectory.py` | import 路径从 `replay_trajectory` 迁移到 `replay_pipeline` |
