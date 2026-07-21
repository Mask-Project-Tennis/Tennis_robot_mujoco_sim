# 单关节跟踪实验脚本设计文档

**日期**: 2026-07-20
**状态**: 已批准（待实现）
**分支**: feat/real-robot-replay-testing

## 1. 概述

### 1.1 目标
为 RM-65B 机械臂实现单关节跟踪性能测试工具，通过发送周期性波形指令（正弦/三角/方波/Chirp/阶跃）并记录实际关节响应，量化评估关节的闭环跟踪带宽。

### 1.2 动机
- 项目使用位置模式控制 RM-65B（MPC 规划 q_desired → 固件 PD 跟踪）
- `FakeRobot` 是理想跟踪 Mock（q=q_desired 瞬时），但真实硬件必然存在滞后
- 需量化真实关节能跟踪多快的信号，判断网球挥拍轨迹（含高频加减速）的可行性

### 1.3 范围
- **包含**: 5 种波形生成、单/多关节测试、Bode 图分析、安全防护、速度控制
- **不包含**: 多关节 MIMO 辨识、自动传递函数拟合、三角/方波谐波分析（留 v2）

## 2. 整体架构

采用**组合模式**（组合优于继承），5 个独立组件 + 1 个编排器，通过 dataclass 传递数据。

```
┌─────────────────────────────────────────────────────────────┐
│         scripts/tools/joint_tracking_test.py (CLI)          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         src/joint_test/experiment.py                         │
│         TrackingExperiment (编排器)                          │
│   run_single(config) → TrackingResult + Metrics             │
│   run_sweep(configs)  → SweepResult                          │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   ▼          ▼          ▼          ▼          ▼
┌──────┐  ┌────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Wave- │  │Robot   │  │Tracking │  │Metrics  │  │Result   │
│form  │  │Adapter │  │Recorder │  │Analyzer │  │Plotter  │
│Gen   │  │        │  │         │  │         │  │         │
└──────┘  └────────┘  └─────────┘  └─────────┘  └─────────┘
              │
              ▼
      RobotArmInterface Protocol
      (RM65Env / FakeRobot / RobotInterface)
```

**速度控制**: 复用现有 `src.real.adaptive_timer.AdaptiveTimer`（不创建 TimePacer）

## 3. 模块设计

### 3.1 文件布局
```
src/joint_test/
├── __init__.py
├── types.py              # 数据结构（WaveformConfig/TrackingResult/Metrics/SweepResult/TestConfig）
├── waveform.py           # WaveformGenerator（5 种波形）
├── safety.py             # JointSafetyGuard（真机限幅/预检）
├── robot_adapter.py      # RobotAdapter（抹平 RM65Env/FakeRobot/RobotInterface）
├── recorder.py           # TrackingRecorder（采集+NPZ/CSV）
├── analyzer.py           # MetricsAnalyzer（RMSE/幅值比/相位/带宽）
├── plotter.py            # ResultPlotter（PNG+Bode+弹窗）
└── experiment.py         # TrackingExperiment（编排器，直接用 AdaptiveTimer）

scripts/tools/joint_tracking_test.py    # CLI 入口
tests/test_joint_tracking.py            # 单元测试（~72 tests）
```

### 3.2 核心数据结构（types.py）

```python
class WaveformType(Enum):
    SINE / TRIANGLE / SQUARE / CHIRP / STEP

class BackendType(Enum):
    SIM / FAKE / REAL

@dataclass(frozen=True)
class WaveformConfig:
    waveform: WaveformType
    joint_idx: int               # 0-5
    frequency_hz: float
    amplitude_rad: float
    offset_rad: float            # 默认 INIT_Q[joint_idx]
    duration_s: float
    end_frequency_hz: float | None = None  # Chirp
    step_target_rad: float | None = None   # Step

@dataclass
class TrackingResult:
    config: WaveformConfig
    backend: BackendType
    dt: float
    time: np.ndarray             # (N,)
    q_desired: np.ndarray        # (N, 6)
    q_actual: np.ndarray         # (N, 6)
    qdot_actual: np.ndarray      # (N, 6)
    tracking_error: np.ndarray   # (N,) 自动计算

@dataclass
class Metrics:
    waveform: WaveformType
    target_joint: int
    rmse_rad: float
    max_error_rad: float
    mean_error_rad: float
    amplitude_ratio: float | None = None      # 正弦/Chirp
    phase_lag_deg: float | None = None        # 正弦/Chirp
    rise_time_s: float | None = None          # Step
    settling_time_s: float | None = None      # Step
    overshoot_pct: float | None = None        # Step
    steady_state_error_rad: float | None = None  # Step

@dataclass
class SweepResult:
    joint_idx: int
    waveform: WaveformType
    amplitude_rad: float
    frequencies_hz: np.ndarray
    amplitude_ratios: np.ndarray
    phase_lags_deg: np.ndarray
    rmses_rad: np.ndarray
    individual_metrics: list[Metrics]

@dataclass
class TestConfig:
    waveform_cfg: WaveformConfig
    backend: BackendType
    speed_ratio: float = 1.0          # (0, 1.0]
    realtime_plot: bool = False
    save_npz: bool = True
    save_csv: bool = False
    save_png: bool = True
    print_metrics: bool = True
    compare_fake: bool = False
```

### 3.3 组件接口

#### WaveformGenerator（waveform.py）
```python
class WaveformGenerator:
    def __init__(self, config: WaveformConfig, base_q: np.ndarray, dt: float): ...
    def generate(self) -> np.ndarray:  # (N, 6)
```
- 所有波形预生成为 (N, 6) 数组，向量化
- Sine: `np.sin(2π·f·t)`
- Triangle: `scipy.signal.sawtooth(2π·f·t, width=0.5)`
- Square: `scipy.signal.square(2π·f·t)`
- Chirp: `scipy.signal.chirp(t, f0=f, f1=end_f, t1=T, method="linear")`
- Step: 前 10% 保持 offset，之后阶跃到 step_target（默认 offset+A）

#### JointSafetyGuard（safety.py）
```python
class JointSafetyGuard:
    def __init__(
        self, q_lower, q_upper, qdot_max,
        max_amplitude_rad=0.05,   # 保守默认（≈2.86°）
        max_frequency_hz=1.0,     # 保守默认
    ): ...
    def check_preconditions(self, cfg: WaveformConfig) -> list[str]: ...
    def clip_command(self, q_des: np.ndarray, q_current: np.ndarray) -> np.ndarray: ...
```

#### RobotAdapter（robot_adapter.py）
```python
class RobotAdapter:
    def __init__(self, robot, dt: float, backend: BackendType,
                 safety_guard: JointSafetyGuard | None = None): ...
    def reset(self, q0: np.ndarray) -> None: ...
    def step(self, q_des: np.ndarray) -> None: ...
    def get_q_qdot(self) -> tuple[np.ndarray, np.ndarray]: ...
    def emergency_stop(self) -> None: ...
```
- 通过 `hasattr` 检测后端类型（RM65Env vs FakeRobot/RobotInterface）
- RM65Env: `env.reset(q0) → env.step(q_des) → env.get_arm_state()`
- FakeRobot/RobotInterface: `send_joint_command(q_des) → get_arm_state()`

#### TrackingRecorder（recorder.py）
```python
class TrackingRecorder:
    def __init__(self, config: WaveformConfig, dt: float, backend: BackendType): ...
    def record(self, t, q_des, q_actual, qdot_actual) -> None: ...
    def finalize(self) -> TrackingResult: ...
    @staticmethod
    def save_npz(result: TrackingResult, path: Path) -> None: ...
    @staticmethod
    def save_csv(result: TrackingResult, path: Path) -> None: ...
```

#### MetricsAnalyzer（analyzer.py）
```python
class MetricsAnalyzer:
    def analyze(self, result: TrackingResult) -> Metrics: ...
```
- 正弦/Chirp: FFT 提取目标频率的幅值和相位
- Step: 10%-90% 上升时间、±2% settling time、超调
- 退化处理: A=0/常量信号/短信号/相位 wrapping

#### ResultPlotter（plotter.py）
```python
class ResultPlotter:
    def __init__(self, output_dir: Path, backend: str = "TkAgg"): ...
    def plot_single(self, result, metrics, fake_result=None) -> Path: ...
    def plot_bode(self, sweep: SweepResult) -> Path: ...
    def show_realtime(self, result, metrics) -> None: ...
    @staticmethod
    def print_metrics(metrics: Metrics) -> None: ...
```
- 单次图: 3 子图（指令vs实际、误差、Lissajous 相图）
- Bode 图: 幅频 + 相频（对数横轴）

#### TrackingExperiment（experiment.py）
```python
class TrackingExperiment:
    def __init__(
        self, robot_adapter, analyzer, plotter, dt, base_q,
        speed_ratio=1.0, backend=BackendType.SIM,
    ): ...
    def run_single(self, config: TestConfig) -> tuple[TrackingResult, Metrics]: ...
    def run_sweep(self, joint_idx, frequencies_hz, amplitude_rad,
                  waveform=WaveformType.SINE, duration_s=3.0,
                  compare_fake=False) -> SweepResult: ...
```

## 4. 速度控制设计

**复用 `AdaptiveTimer`**（src/real/adaptive_timer.py），不创建 TimePacer。

### 4.1 语义
`speed_ratio ∈ (0, 1.0]`:
- **1.0** = 最大速度（默认）
  - Sim: 不 pace，CPU 跑多快跑多快（`self._timer = None`）
  - Real: 实时（AdaptiveTimer at native rate）
- **0.5** = 半速（每步目标 2·dt 墙钟）
- **0.25** = 1/4 速

### 4.2 Experiment 内部配置
```python
if backend == BackendType.REAL:
    target_hz = (1.0 / dt) * speed_ratio
    self._timer = AdaptiveTimer(target_hz=target_hz)
elif speed_ratio < 1.0:
    target_hz = (1.0 / dt) * speed_ratio
    self._timer = AdaptiveTimer(target_hz=target_hz)
else:
    self._timer = None  # Sim max speed
```

### 4.3 主循环
```python
for k in range(len(q_traj)):
    if self._timer is not None:
        self._timer.tick_start()
    self._adapter.step(q_traj[k])
    q_act, qdot_act = self._adapter.get_q_qdot()
    recorder.record(k * dt, q_traj[k], q_act, qdot_act)
    if self._timer is not None:
        sleep_dt = self._timer.tick_end()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
```

## 5. 安全设计

### 5.1 真机默认参数（保守）
- `max_amplitude_rad = 0.05`（≈2.86°）
- `max_frequency_hz = 1.0`
- 需要时通过 CLI `--max-real-amplitude` / `--max-real-freq` 覆盖

### 5.2 三层防护
1. **预检查** (`check_preconditions`): 运行前返回警告列表
2. **指令裁剪** (`clip_command`): 每步裁剪 q_des 到关节限位
3. **强制 flag**: 真机模式必须 `--i-understand-real-risk`

### 5.3 安全测试矩阵（10 tests）
| 场景 | 预期 |
|------|------|
| 缺 flag | SystemExit |
| 幅值 > 0.05 | 拒绝 |
| 频率 > 1.0 | 拒绝 |
| 波形超限位 | 拒绝 |
| 指令被裁剪 | 下发值在范围内 |
| 关节位置越限 | emergency_stop |
| 关节速度越限 | emergency_stop |
| Ctrl+C | slow_stop + 退出 |
| 通信中断 | RuntimeError |
| 默认值保守 | A≤0.05, f≤1.0 |

## 6. CLI 设计

```bash
# 基本格式
python scripts/tools/joint_tracking_test.py \
    --backend {sim|fake|real} \
    --waveform {sine|triangle|square|chirp|step} \
    --joint 0-5 \
    --freq Hz \
    --amplitude rad \
    --duration s \
    [--speed (0,1.0]] \
    [--sweep --sweep-freqs f1,f2,...] \
    [--compare-fake] \
    [--no-plot] [--realtime] \
    [--save-npz] [--save-csv] [--save-png] \
    [--output-dir PATH] \
    [--i-understand-real-risk]  # 真机强制
```

`--speed` 验证: 必须在 (0, 1.0]，否则 argparse 报错。

## 7. 测试策略

### 7.1 TDD 流程
每个组件严格遵循 **RED → verify RED → GREEN → verify GREEN → REFACTOR**。

### 7.2 测试覆盖（~72 tests）
| 模块 | 测试数 | 重点 |
|------|--------|------|
| types | 4 | 数据结构正确性 |
| waveform | 14 | 5 种波形+9 边界 |
| safety | 13 | 预检+裁剪+真机保护 |
| robot_adapter | 10 | 后端适配+安全集成 |
| recorder | 7 | 采集+保存格式 |
| analyzer | 11 | FFT 精度+退化处理 |
| plotter | 5 | 文件生成 |
| experiment | 11 | 端到端+speed_ratio |
| CLI integration | 5 | 命令行运行 |
| safety matrix | 10 | 真机安全路径 |
| **adaptive_timer** | (已有 3) | 复用，不新增 |

### 7.3 覆盖率目标
≥ 90%（`pytest --cov=src/joint_test`）

## 8. 验证清单

```bash
# 全部测试
pytest tests/test_joint_tracking.py -v

# 覆盖率
pytest tests/test_joint_tracking.py --cov=src/joint_test --cov-report=term-missing

# 类型检查
mypy src/joint_test/

# Lint
ruff check src/joint_test/ tests/test_joint_tracking.py scripts/tools/joint_tracking_test.py

# 烟雾测试
python scripts/tools/joint_tracking_test.py --backend sim --waveform sine --joint 2 --freq 1.0 --amplitude 0.2 --duration 3.0
python scripts/tools/joint_tracking_test.py --backend sim --slow-mo --waveform sine --joint 2 --freq 1.0 --amplitude 0.2 --duration 3.0
python scripts/tools/joint_tracking_test.py --backend sim --sweep --joint 2 --amplitude 0.1
```

## 9. 全局约束（Global Constraints）

以下约束对所有任务生效，subagent 必须遵守：

### 9.1 代码规范
- **所有代码注释、docstring 使用中文**（遵循 AGENTS.md）
- 变量名、函数名、类名使用英文
- 类型提示（type hints）必须标注在所有函数签名上
- 公有函数必须有中文 docstring（Google 风格）
- 使用 `numpy` 进行数组运算，禁止对数组使用原生 Python 循环
- 文件路径使用 `pathlib.Path`，不拼接字符串
- 日志使用 Python `logging` 模块（CLI 报告例外可用 print）

### 9.2 复用现有代码
- `RobotArmInterface` Protocol: `src/real/robot_arm_protocol.py`
- `RobotLimits`: `src/ilqt/robot_limits.py`
- `INIT_Q` / `KP` / `KD` / `DT`: `src/robot/constants.py`
- `AdaptiveTimer`: `src/real/adaptive_timer.py`（**直接复用，不创建 TimePacer**）
- `FakeRobot`: `src/real/fake_robot.py`
- `RM65Env`: `src/sim/rm65_env.py`
- `load_mujoco_model`: `src/utils/mujoco_loader.py`

### 9.3 关键参数（必须使用精确值）
- `DT = 0.005`（来自 `src/robot/constants.py`）
- `INIT_Q = [-1.5, 1.57, -0.236, 0.404, 0.446, 2.45]`
- 真机保守默认: `max_amplitude_rad=0.05`, `max_frequency_hz=1.0`
- `speed_ratio ∈ (0, 1.0]`, 默认 1.0

### 9.4 TDD 要求
- 每个生产函数必须有测试
- 测试必须先失败（RED），再实现（GREEN）
- 测试用例必须覆盖边界条件
- 真机安全相关代码必须有专门的安全测试

### 9.5 组合优于继承
- 不创建基类让子类继承
- 通过构造函数注入依赖
- 每个组件单一职责、独立可测

## 附录 A：可选重构（Out of Scope，记录待未来执行）

### A.1 AdaptiveTimer 模块迁移

**当前状态**: `AdaptiveTimer` 位于 `src/real/adaptive_timer.py`，仅依赖标准库 `time`，与真机无关。

**问题**: 命名空间暗示"真机专用"，但实际是通用工具，sim 模块需跨 `src/real/` 导入。

**建议迁移**: `src/real/adaptive_timer.py` → `src/utils/adaptive_timer.py`

**影响范围**（需同步修改 import）:
- `src/real/real_runner.py:39`
- `src/real/trajectory_sink.py:21`
- `src/real/replay_pipeline.py:28`
- `scripts/run_real_robot.py:30`
- `tests/test_adaptive_timer.py:7`
- `tests/test_real_runner.py:19`
- `tests/test_replay_trajectory.py:17`
- `tests/test_create_runner.py`（间接）

**迁移步骤**（独立 PR）:
1. `git mv src/real/adaptive_timer.py src/utils/adaptive_timer.py`
2. 全局替换 `from src.real.adaptive_timer` → `from src.utils.adaptive_timer`
3. `pytest tests/ -k "adaptive or real_runner or trajectory_sink or replay"`
4. ruff/mypy 验证

**风险**: 低。纯文件移动 + import 路径替换。
**优先级**: 低。

### A.2 AdaptiveTimer 扩展 speed_ratio 参数（候选）

**未来可选**: 将 speed_ratio 参数移入 AdaptiveTimer 自身：
```python
class AdaptiveTimer:
    def __init__(self, target_hz=100.0, utilization=0.8, speed_ratio=1.0): ...
```

**决策**: 暂不实施（YAGNI）。现有调用方无 speed_ratio 需求，joint_tracking 的 experiment-side 逻辑已满足。等 2+ 个场景需要时再提取。

### A.3 RobotAdapter 进一步抽象（候选）

**当前**: `RobotAdapter` 用 `hasattr` 检测后端类型（RM65Env 不直接实现 RobotArmInterface）。

**未来可选**: 让 RM65Env 实现 RobotArmInterface Protocol（添加 send_joint_command 等）。

**决策**: 暂不实施。会修改核心仿真模块，影响面太大。属于独立重构议题。

## 附录 B：与现有项目代码的契合点

- **遵循** `scan_joint_safety.py` 的 CLI/输出风格（argparse + 格式化 print 报告）
- **遵循** `rm65_joint_viewer.py` 的 sys.path 注入模式
- **遵循** AGENTS.md 全部规范
- **复用** MuJoCo 模型加载通过 `load_mujoco_model()`（非 `from_xml_path`）
