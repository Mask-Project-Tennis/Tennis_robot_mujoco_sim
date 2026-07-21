# Task 1 Brief: types.py + __init__.py（数据结构基础）

## 任务背景

这是单关节跟踪实验脚本项目的第 1 个任务，建立数据结构基础。所有后续组件（waveform/safety/recorder/analyzer/plotter/experiment）都依赖此模块的 dataclass 和枚举。

完整 spec 文档: `docs/superpowers/specs/2026-07-20-joint-tracking-test-design.md`

## 你要做的事

### 1. 创建 `src/joint_test/__init__.py`

空文件或仅含 docstring：
```python
"""单关节跟踪实验模块。"""
```

### 2. 创建 `src/joint_test/types.py`

实现以下数据结构（**精确使用这些字段名和类型**）：

#### WaveformType 枚举
```python
class WaveformType(Enum):
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CHIRP = "chirp"
    STEP = "step"
```

#### BackendType 枚举
```python
class BackendType(Enum):
    SIM = "sim"      # RM65Env (MuJoCo)
    FAKE = "fake"    # FakeRobot (完美跟踪 Mock)
    REAL = "real"    # RobotInterface (真机 SDK)
```

#### WaveformConfig（frozen dataclass）
```python
@dataclass(frozen=True)
class WaveformConfig:
    """波形生成参数（不可变）。"""
    waveform: WaveformType
    joint_idx: int                    # 0-5
    frequency_hz: float
    amplitude_rad: float
    offset_rad: float                 # 直流偏置（默认 INIT_Q[joint_idx]）
    duration_s: float
    end_frequency_hz: float | None = None  # Chirp 终止频率
    step_target_rad: float | None = None   # Step 目标角度
```

#### TrackingResult（普通 dataclass，带 __post_init__）
```python
@dataclass
class TrackingResult:
    """单次跟踪实验的完整时序数据。"""
    config: WaveformConfig
    backend: BackendType
    dt: float
    time: np.ndarray                  # (N,)
    q_desired: np.ndarray             # (N, 6)
    q_actual: np.ndarray              # (N, 6)
    qdot_actual: np.ndarray           # (N, 6)
    tracking_error: np.ndarray = field(init=False)  # (N,) 自动计算

    def __post_init__(self) -> None:
        """自动计算 tracking_error = q_desired[:, joint_idx] - q_actual[:, joint_idx]。"""
        j = self.config.joint_idx
        object.__setattr__(self, 'tracking_error',
                           self.q_desired[:, j] - self.q_actual[:, j])
```

**注意**: 因为 `tracking_error` 是 `field(init=False)`，不能用普通赋值（dataclass 限制），需用 `object.__setattr__`。

#### Metrics（普通 dataclass，可选字段默认 None）
```python
@dataclass
class Metrics:
    """跟踪性能指标。None 表示该指标对此波形不适用。"""
    waveform: WaveformType
    target_joint: int
    rmse_rad: float
    max_error_rad: float
    mean_error_rad: float
    amplitude_ratio: float | None = None
    phase_lag_deg: float | None = None
    rise_time_s: float | None = None
    settling_time_s: float | None = None
    overshoot_pct: float | None = None
    steady_state_error_rad: float | None = None
```

#### SweepResult
```python
@dataclass
class SweepResult:
    """批量扫频聚合结果（用于 Bode 图）。"""
    joint_idx: int
    waveform: WaveformType
    amplitude_rad: float
    frequencies_hz: np.ndarray        # (M,)
    amplitude_ratios: np.ndarray      # (M,)
    phase_lags_deg: np.ndarray        # (M,)
    rmses_rad: np.ndarray             # (M,)
    individual_metrics: list[Metrics]
```

#### TestConfig
```python
@dataclass
class TestConfig:
    """单次测试的完整配置（传给编排器）。"""
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

### 3. 创建测试文件 `tests/test_joint_tracking.py`

只写 Task 1 相关的测试（4 个）：

```python
"""单关节跟踪实验组件单元测试。"""
import numpy as np
import pytest
from src.joint_test.types import (
    WaveformType, BackendType, WaveformConfig,
    TrackingResult, Metrics, SweepResult, TestConfig,
)


class TestTypes:
    """数据结构测试。"""

    def test_waveform_type_enum_values(self):
        """WaveformType 包含 5 个正确枚举值。"""
        assert WaveformType.SINE.value == "sine"
        assert WaveformType.TRIANGLE.value == "triangle"
        assert WaveformType.SQUARE.value == "square"
        assert WaveformType.CHIRP.value == "chirp"
        assert WaveformType.STEP.value == "step"

    def test_tracking_result_computes_error(self):
        """TrackingResult 自动计算 tracking_error。"""
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=2,
            frequency_hz=1.0, amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
        )
        n = 10
        q_des = np.ones((n, 6))
        q_des[:, 2] = 0.5  # joint 2 的指令
        q_act = np.ones((n, 6))
        q_act[:, 2] = 0.4  # joint 2 的实际（有 0.1 误差）
        result = TrackingResult(
            config=cfg, backend=BackendType.SIM, dt=0.1,
            time=np.arange(n) * 0.1,
            q_desired=q_des, q_actual=q_act,
            qdot_actual=np.zeros((n, 6)),
        )
        assert result.tracking_error.shape == (n,)
        assert np.allclose(result.tracking_error, 0.1)

    def test_waveform_config_is_frozen(self):
        """WaveformConfig 不可变。"""
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=1.0, amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
        )
        with pytest.raises((AttributeError, Exception)):
            cfg.frequency_hz = 2.0  # type: ignore[misc]

    def test_metrics_optional_fields_default_none(self):
        """Metrics 可选字段默认 None。"""
        m = Metrics(
            waveform=WaveformType.SINE, target_joint=0,
            rmse_rad=0.01, max_error_rad=0.02, mean_error_rad=0.005,
        )
        assert m.amplitude_ratio is None
        assert m.phase_lag_deg is None
        assert m.rise_time_s is None
        assert m.settling_time_s is None
        assert m.overshoot_pct is None
        assert m.steady_state_error_rad is None
```

## 全局约束（必须遵守）

- **所有 docstring 使用中文**（Google 风格）
- 类型提示必须标注
- 使用 `from __future__ import annotations` 启用 PEP 604 语法（`X | None`）
- 文件路径用 `pathlib.Path`
- 测试用 `pytest`，遵循现有 `tests/test_adaptive_timer.py` 的风格

## 验证

```bash
# 跑 Task 1 测试
pytest tests/test_joint_tracking.py::TestTypes -v

# Lint
ruff check src/joint_test/types.py tests/test_joint_tracking.py

# 类型检查
mypy src/joint_test/types.py
```

预期：4 tests 全绿，ruff/mypy 无错误。

## 不要做的事

- 不要实现 waveform.py / safety.py 等其他模块（后续任务）
- 不要在 `__init__.py` 中导出符号（保持简洁）
- 不要添加 spec 中未列出的字段或方法（YAGNI）
