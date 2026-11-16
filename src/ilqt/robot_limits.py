"""机器人硬约束参数与 feasibility check（含滑动窗口 qddot）— n 维通用。

历史版本面向 REALMAN RM65-B（n=6）；阶段 3 Task 3.7 起维度通用化：
- 全部检查函数按 ``RobotLimits`` 限值维数 n 取状态切片（q = x[:n]，
  qdot = x[n:]），不再硬编码 6；状态宽度与限值维数不符时显式报错；
- ``from_config`` 保留 RM65 六维缺省（n=6 时行为不变），其余维数要求
  配置显式提供数值；
- 新增 ``from_runtime_parameters``：RobotRuntimeParameters（阶段 2/3
  参数层）→ RobotLimits 适配 —— 完整参数才能构造，逐值直传（无 margin/
  scale 隐式缩放），模型 q 限位（jnt_range 回填权威）与人为运行约束互
  不覆盖、不互相顶替。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.robot.runtime_parameters import RobotRuntimeParameters

logger = logging.getLogger(__name__)


@dataclass
class RobotLimits:
    """机器人硬约束参数（全部 SI 单位，n 维通用；RM65 n=6 / R1D n=7）。"""

    q_lower: np.ndarray              # (n,) [rad]
    q_upper: np.ndarray              # (n,) [rad]
    qdot_max: np.ndarray             # (n,) [rad/s]
    qddot_max: np.ndarray            # (n,) [rad/s²]
    u_min: np.ndarray                # (n,) [Nm]
    u_max: np.ndarray                # (n,) [Nm]
    alpha_fallback: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2, 0.1, 0.05])
    forward_pass_margin: float = 1.5
    forward_pass_q_tol_rad: float = 0.0   # 前向传递 q 额外容忍 (rad)
    qdot_window_size: int = 3
    qddot_window_size: int = 5
    qddot_hard_reject: bool = False
    max_tcp_speed: float = float('inf')
    terminal_exempt_steps: int = 20
    dq_max: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        config: dict,
        dt: float,
        ctrlrange: np.ndarray,
    ) -> RobotLimits:
        """从旧 RM65 风格配置构建限位（n=6 缺省行为不变，n 维显式要求）。

        Args:
            config: 旧配置字典（度单位的 q/qdot/qddot 声明 + 过程参数）。
            dt: 时间步长 (s)。
            ctrlrange: (n,2) 执行器控制范围（u 界缺省来源）。

        Returns:
            RobotLimits 实例。

        Raises:
            ValueError: ctrlrange 形状不符，或 n≠6 时缺省项未显式提供。
        """
        deg_to_rad = np.pi / 180.0
        n = _validate_control_range(ctrlrange)

        # RM65 六维缺省仅在 n=6 时可用；其余维数必须显式配置
        def _deg_array(key: str, rm65_default: list[float]) -> np.ndarray:
            if key in config:
                arr = np.array(config[key], dtype=np.float64)
            elif n == 6:
                arr = np.array(rm65_default, dtype=np.float64)
            else:
                raise ValueError(
                    f"配置缺少 {key}，且非 RM65 六维（n={n}）无缺省值，"
                    "必须显式提供（或改用 from_runtime_parameters）"
                )
            if arr.shape != (n,):
                raise ValueError(f"配置 {key} 形状 {arr.shape} 应为 ({n},)")
            return arr

        q_min_raw = _deg_array("q_min_deg", [-178, -130, -135, -178, -128, -360])
        q_max_raw = _deg_array("q_max_deg", [178, 130, 135, 178, 128, 360])
        q_margin_cfg = config.get("q_margin_deg", 8.0)
        # 支持标量（统一裕度）和数组（per-joint 裕度）
        if np.ndim(q_margin_cfg) == 0:
            q_margin_rad = np.full(n, float(q_margin_cfg) * deg_to_rad, dtype=np.float64)
        else:
            q_margin_rad = np.array(q_margin_cfg, dtype=np.float64) * deg_to_rad
            if q_margin_rad.shape != (n,):
                raise ValueError(
                    f"配置 q_margin_deg 形状 {q_margin_rad.shape} 应为 ({n},)"
                )

        q_lower = (q_min_raw * deg_to_rad) + q_margin_rad
        q_upper = (q_max_raw * deg_to_rad) - q_margin_rad

        qdot_max_raw = _deg_array("qdot_max_deg_s", [180, 180, 225, 225, 225, 225])
        qdot_scale = float(config.get("qdot_scale", 0.9))
        qdot_max = qdot_max_raw * deg_to_rad * qdot_scale

        qddot_max_raw = _deg_array("qddot_max_deg_s2", [400, 400, 500, 500, 500, 500])
        qddot_scale = float(config.get("qddot_scale", 0.7))
        if qddot_scale > 0.0:
            qddot_max = qddot_max_raw * deg_to_rad * qddot_scale
        else:
            qddot_max = np.full(n, np.inf, dtype=np.float64)

        u_cfg_min = config.get("u_min", None)
        u_cfg_max = config.get("u_max", None)
        if u_cfg_min is not None and u_cfg_max is not None:
            u_min_arr = np.array(u_cfg_min, dtype=np.float64)
            u_max_arr = np.array(u_cfg_max, dtype=np.float64)
        else:
            u_min_arr = ctrlrange[:, 0].copy()
            u_max_arr = ctrlrange[:, 1].copy()

        return cls(
            q_lower=q_lower, q_upper=q_upper,
            qdot_max=qdot_max, qddot_max=qddot_max,
            u_min=u_min_arr, u_max=u_max_arr,
            alpha_fallback=[float(a) for a in config.get("alpha_fallback", [0.5, 0.3, 0.2, 0.1, 0.05])],
            forward_pass_margin=float(config.get("forward_pass_margin", 1.5)),
            forward_pass_q_tol_rad=float(config.get("forward_pass_q_tol_deg", 1.5)) * deg_to_rad,
            qdot_window_size=int(config.get("qdot_window", 3)),
            qddot_window_size=int(config.get("qddot_window", 5)),
            qddot_hard_reject=bool(config.get("qddot_hard_reject", False)),
            max_tcp_speed=float(config.get("max_tcp_speed", float('inf'))),
            terminal_exempt_steps=int(config.get("terminal_exempt_steps", 20)),
            dq_max=qdot_max * dt * float(config.get("dq_max_fraction", 0.5)),
        )

    @classmethod
    def from_runtime_parameters(
        cls,
        parameters: RobotRuntimeParameters,
        control_range: np.ndarray,
        config: dict | None = None,
    ) -> RobotLimits:
        """从 RobotRuntimeParameters 构建限位（阶段 3 · Task 3.7 适配）。

        语义约定（施工计划 Task 3.7）：
        - **完整参数才能构造**：q_lower/q_upper/qdot_max/qddot_max/
          torque_max/max_tcp_speed 任一为 None 即拒绝（未知限值不得进入
          仿真/执行；与 ``validate_runtime_parameters(purpose="simulation")``
          同一完整性口径，但此处不依赖 profile，独立逐项校验）；
        - **逐值直传**：不加 q margin、不乘 qdot/qddot scale —— runtime
          声明值即最终限值（与 ``from_config`` 的 RM65 裕度/缩放路径不同，
          两条路径互不影响）；
        - **不互相覆盖**：模型 q 限位（阶段 2 jnt_range 回填权威）与人为
          运行约束各自保留；``control_range`` 仅作维数锚定与有限性校验，
          不顶替 torque_max（u 界）也不顶替 q 界 —— 位置模式下
          control_range 是关节范围，力矩界限仍以 torque_max 为准；
        - u 界取 ±torque_max（对称幅值，来自 runtime）。

        Args:
            parameters: RobotRuntimeParameters 实例（通常为模型回填后的
                ``bundle.parameters``，q_lower/q_upper 已从 jnt_range 解析）。
            control_range: (n,2) 执行器控制范围（``env.control_range`` 副本；
                仅锚定维数，不参与限值合成）。
            config: 可选过程参数（键名同 ``from_config``：alpha_fallback /
                forward_pass_margin / forward_pass_q_tol_deg / qdot_window /
                qddot_window / qddot_hard_reject / terminal_exempt_steps /
                dq_max_fraction；缺省与 from_config 一致）。

        Returns:
            RobotLimits 实例（n = control_range 行数）。

        Raises:
            ValueError: 限值不完整（None）、control_range/数组形状不符、
                数组含非有限值或符号约束失败。
        """
        cfg = config or {}
        n = _validate_control_range(control_range)
        context = f"robot_id: {getattr(parameters, 'robot_id', '?')}"

        # 完整性：六个动态限值字段全部必须已知
        required = (
            "q_lower", "q_upper", "qdot_max", "qddot_max",
            "torque_max", "max_tcp_speed",
        )
        missing = [name for name in required if getattr(parameters, name, None) is None]
        if missing:
            raise ValueError(
                f"限值不完整：缺少 {missing}（完整参数才能构造 RobotLimits，"
                f"未知限值不得进入仿真/执行；{context}）"
            )

        def _array(name: str, *, positive: bool) -> np.ndarray:
            arr = np.array(getattr(parameters, name), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"字段 {name} 形状 {arr.shape} 与 control_range 维数 "
                    f"({n},) 不符（{context}）"
                )
            if not bool(np.all(np.isfinite(arr))):
                raise ValueError(f"字段 {name} 含非有限值（{context}）")
            if positive and bool(np.any(arr <= 0.0)):
                raise ValueError(f"字段 {name} 必须逐元素为正（{context}）")
            return arr

        q_lower = _array("q_lower", positive=False)
        q_upper = _array("q_upper", positive=False)
        if bool(np.any(q_lower >= q_upper)):
            raise ValueError(f"字段 q_lower 存在 >= q_upper 的关节（{context}）")
        qdot_max = _array("qdot_max", positive=True)
        qddot_max = _array("qddot_max", positive=True)
        torque_max = _array("torque_max", positive=True)
        tcp_speed_raw = parameters.max_tcp_speed
        if tcp_speed_raw is None:
            # 不可达（missing 检查已含 max_tcp_speed）；显式收窄供类型检查
            raise ValueError(
                f"限值不完整：缺少 ['max_tcp_speed']（完整参数才能构造，{context}）"
            )
        max_tcp_speed = float(tcp_speed_raw)
        if not np.isfinite(max_tcp_speed) or max_tcp_speed <= 0.0:
            raise ValueError(
                f"字段 max_tcp_speed={max_tcp_speed} 必须为正有限值（{context}）"
            )

        deg_to_rad = np.pi / 180.0
        dt = float(parameters.dt)
        return cls(
            q_lower=q_lower.copy(),
            q_upper=q_upper.copy(),
            qdot_max=qdot_max.copy(),
            qddot_max=qddot_max.copy(),
            u_min=-torque_max.copy(),
            u_max=torque_max.copy(),
            alpha_fallback=[float(a) for a in cfg.get("alpha_fallback", [0.5, 0.3, 0.2, 0.1, 0.05])],
            forward_pass_margin=float(cfg.get("forward_pass_margin", 1.5)),
            forward_pass_q_tol_rad=float(cfg.get("forward_pass_q_tol_deg", 1.5)) * deg_to_rad,
            qdot_window_size=int(cfg.get("qdot_window", 3)),
            qddot_window_size=int(cfg.get("qddot_window", 5)),
            qddot_hard_reject=bool(cfg.get("qddot_hard_reject", False)),
            max_tcp_speed=max_tcp_speed,
            terminal_exempt_steps=int(cfg.get("terminal_exempt_steps", 20)),
            dq_max=qdot_max * dt * float(cfg.get("dq_max_fraction", 0.5)),
        )


# ==============================================================================
#  维数解析与状态切片辅助（n 维通用）
# ==============================================================================


def _validate_control_range(ctrlrange: np.ndarray) -> int:
    """校验控制范围形状并返回维数 n。

    Raises:
        ValueError: ctrlrange 不是 (n,2) 有限数组。
    """
    arr = np.asarray(ctrlrange, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"ctrlrange 形状 {arr.shape} 应为 (n,2)，n ≥ 1")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("ctrlrange 含非有限值")
    return int(arr.shape[0])


def _nq_from_limits(limits: RobotLimits) -> int:
    """从限值数组解析维数 n 并校验各限值数组维数一致。

    Raises:
        ValueError: 任一限值数组形状与 q_lower 不一致。
    """
    n = int(np.asarray(limits.q_lower).shape[0])
    for name in ("q_upper", "qdot_max", "qddot_max", "u_min", "u_max"):
        if int(np.asarray(getattr(limits, name)).shape[0]) != n:
            raise ValueError(
                f"RobotLimits 限值数组维数不一致：q_lower 为 {n}，"
                f"{name} 为 {np.asarray(getattr(limits, name)).shape[0]}"
            )
    return n


def _validate_state_width(x: np.ndarray, nq: int, label: str) -> np.ndarray:
    """校验状态/控制向量宽度（q/qdot 分界 = nq），失败显式报错。

    Raises:
        ValueError: 状态向量宽度不等于 2*nq。
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (2 * nq,):
        raise ValueError(
            f"状态向量 {label} 形状 {arr.shape} 应为 ({2 * nq},)"
            f"（q/qdot 分界 = 限值维数 nq={nq}）"
        )
    return arr


# ==============================================================================
#  滑动窗口加速度估算
# ==============================================================================

def compute_qddot_filtered(
    qdot_history: list[np.ndarray],
    dt: float,
    window_size: int,
) -> np.ndarray:
    """滑动窗口加速度估算（n 维：返回与 history 元素同维的数组）。"""
    effective_len = min(len(qdot_history), window_size + 1)
    if effective_len < 2:
        # 窗口不足：返回与最近一次记录同形状的零加速度（无记录时为零长）
        if len(qdot_history) == 0:
            return np.zeros(0)
        return np.zeros_like(np.asarray(qdot_history[-1]))
    return (qdot_history[-1] - qdot_history[-effective_len]) / ((effective_len - 1) * dt)


def build_qdot_history(
    history: deque,
    qdot_new: np.ndarray,
    window_size: int,
) -> None:
    history.append(qdot_new.copy())
    while len(history) > window_size + 1:
        history.popleft()


# ==============================================================================
#  轨迹指标收集
# ==============================================================================

@dataclass
class TrajectoryMetrics:
    """单条 iLQR 前向传递轨迹的约束违反指标（规划层）。"""

    max_qdot_ratio: float = 0.0          # max_j(|qdot_j| / qdot_max_j)
    max_qdot_joint: int = -1
    max_qddot_ratio: float = 0.0         # max_j(|qddot_j| / qddot_max_j) 滑动窗口
    max_qddot_joint: int = -1
    max_joint_speed_rad_s: float = 0.0   # max ||qdot|| (rad/s)
    tcp_speed_max: float = 0.0           # FK 末端线速度 [m/s]，需 env 传入
    racket_face_speed_max: float = 0.0   # 球拍面线速度 [m/s]，需 env 传入
    mean_qddot_ratio_100ms: float = 0.0  # 最后 20 步平均 qddot ratio
    qddot_peak_before_hit: float = 0.0   # 最后 20 步内 qddot ratio 峰值
    n_rejected_alphas: int = 0           # 被拒的 alpha 数（线搜索用）


@dataclass
class ExecutionMetrics:
    """执行层累计指标（整个 MPC 生命周期）。"""

    max_qdot_ratio: float = 0.0
    max_qddot_ratio_filtered: float = 0.0
    max_u_ratio: float = 0.0
    max_du_ratio: float = 0.0
    max_tcp_speed: float = 0.0
    max_racket_face_speed: float = 0.0
    overspeed_duration_ms: float = 0.0   # qdot 超限持续时间
    overacc_duration_ms: float = 0.0     # qddot 超限持续时间
    fallback_count: int = 0
    emergency_stop_count: int = 0
    total_mpc_steps: int = 0


def compute_trajectory_metrics(
    X: np.ndarray,
    U: np.ndarray,
    limits: RobotLimits,
    dt: float,
    env=None,
) -> TrajectoryMetrics:
    """对已 rollout 的轨迹采集约束指标（n 维：q/qdot 分界取自 limits）。

    若提供 env，则用 FK 计算 TCP 和球拍面速度（仅后 10 步节省开销）。

    Raises:
        ValueError: 轨迹状态宽度与限值维数不一致。
    """
    nq = _nq_from_limits(limits)
    if len(X) > 0 and np.asarray(X[0]).shape != (2 * nq,):
        _validate_state_width(np.asarray(X[0]), nq, "X[0]")
    metrics = TrajectoryMetrics()
    qdot_hist: deque = deque(maxlen=limits.qddot_window_size + 1)
    n = len(U)
    hit_window_start = max(0, n - 20)
    qddot_ratios_hit: list[float] = []

    for k in range(n):
        qdot_k = X[k][nq:]
        qdot_next = X[k + 1][nq:]

        ratios_qdot = np.abs(qdot_next) / np.maximum(limits.qdot_max, 1e-8)
        max_j = int(np.argmax(ratios_qdot))
        if ratios_qdot[max_j] > metrics.max_qdot_ratio:
            metrics.max_qdot_ratio = float(ratios_qdot[max_j])
            metrics.max_qdot_joint = max_j

        build_qdot_history(qdot_hist, qdot_k, limits.qddot_window_size)
        if len(qdot_hist) >= 2:
            qddot_est = compute_qddot_filtered(
                list(qdot_hist), dt, limits.qddot_window_size,
            )
            ratios_qddot = np.abs(qddot_est) / np.maximum(limits.qddot_max, 1e-8)
            max_j = int(np.argmax(ratios_qddot))
            r_val = float(ratios_qddot[max_j])
            if r_val > metrics.max_qddot_ratio:
                metrics.max_qddot_ratio = r_val
                metrics.max_qddot_joint = max_j
            if k >= hit_window_start:
                qddot_ratios_hit.append(float(np.max(ratios_qddot)))

        metrics.max_joint_speed_rad_s = max(
            metrics.max_joint_speed_rad_s,
            float(np.linalg.norm(qdot_next)),
        )
        build_qdot_history(qdot_hist, qdot_next, limits.qddot_window_size)

    if qddot_ratios_hit:
        metrics.mean_qddot_ratio_100ms = float(np.mean(qddot_ratios_hit))
        metrics.qddot_peak_before_hit = float(np.max(qddot_ratios_hit))

    if env is not None and n > 0:
        check_indices = list(range(max(0, n - 10), n + 1))
        for k in check_indices:
            if k < len(X):
                env.set_arm_state(X[k])
                v_ee = env.get_ee_vel()
                metrics.tcp_speed_max = max(
                    metrics.tcp_speed_max,
                    float(np.linalg.norm(v_ee)),
                )
                if hasattr(env, 'get_racket_face_speed'):
                    metrics.racket_face_speed_max = max(
                        metrics.racket_face_speed_max,
                        env.get_racket_face_speed(),
                    )

    return metrics


def compute_tcp_speed_from_env(
    env, X: np.ndarray,
) -> float:
    """用环境 FK 计算末端 TCP 线速度（仅对后几步调用以节省开销）。"""
    if len(X) < 2:
        return 0.0
    max_speed = 0.0
    check_indices = list(range(max(0, len(X) - 10), len(X)))
    for k in check_indices:
        env.set_arm_state(X[k])
        v_ee = env.get_ee_vel()
        speed = float(np.linalg.norm(v_ee))
        if speed > max_speed:
            max_speed = speed
    return max_speed


# ==============================================================================
#  硬约束检查
# ==============================================================================

def check_step_feasibility(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    u_try: np.ndarray,
    limits: RobotLimits,
    dt: float,
    step: int = -1,
    margin: float = 1.0,
    skip_qdot: bool | str = False,
    skip_qddot: bool = False,
    qdot_history: deque | None = None,
    fp_q_tol: float = 0.0,
    actuator_mode: int = 0,
) -> tuple[bool, str]:
    """检查单步是否满足硬约束。

    Args:
        fp_q_tol: 前向传递中 q 约束的额外容忍 (rad)。
        skip_qdot:
            True:     跳过速度检查
            False/'hard': 硬拒绝
            'braking': 制动感知
        actuator_mode: 0=力矩模式(检查 u_min/u_max), 1=位置模式(检查 dq_max)。

    Returns:
        (pass, reason)

    Raises:
        ValueError: 状态/控制宽度与限值维数不一致。
    """
    nq = _nq_from_limits(limits)
    x_next_arr = _validate_state_width(np.asarray(x_next, dtype=np.float64), nq, "x_next")
    _validate_state_width(np.asarray(x_prev, dtype=np.float64), nq, "x_prev")
    u_arr = np.asarray(u_try, dtype=np.float64)
    if u_arr.shape != (nq,):
        raise ValueError(f"控制向量 u_try 形状 {u_arr.shape} 应为 ({nq},)")
    q_next = x_next_arr[:nq]
    qdot_next = x_next_arr[nq:]
    qdot_prev = np.asarray(x_prev, dtype=np.float64)[nq:]

    # 1. 关节角度（前向传递可以用 fp_q_tol 额外放宽）
    for j in range(nq):
        if q_next[j] < limits.q_lower[j] - fp_q_tol:
            reason = f"q lower bound violated, joint={j}"
            _log_rejection(step, "q", j, reason)
            return False, reason
        if q_next[j] > limits.q_upper[j] + fp_q_tol:
            reason = f"q upper bound violated, joint={j}"
            _log_rejection(step, "q", j, reason)
            return False, reason

    # 2. 关节速度
    if skip_qdot is True:
        pass
    elif skip_qdot is False or skip_qdot == "hard":
        # 硬拒绝模式
        for j in range(nq):
            if abs(qdot_next[j]) > limits.qdot_max[j] * margin:
                reason = f"qdot limit exceeded, joint={j}"
                _log_rejection(step, "qdot", j, reason)
                return False, reason
    elif skip_qdot == "braking":
        # 制动感知模式：只在超速+加速时拒绝
        for j in range(nq):
            abs_qdot = abs(qdot_next[j])
            limit_eff = limits.qdot_max[j] * margin
            if abs_qdot > limit_eff:
                qddot_est = (qdot_next[j] - qdot_prev[j]) / dt
                if qddot_est * np.sign(qdot_next[j]) > 0.0:
                    reason = (
                        f"qdot overspeeding+accelerating, joint={j}, "
                        f"|qdot|={abs_qdot:.2f} > {limit_eff:.2f}"
                    )
                    _log_rejection(step, "qdot-braking", j, reason)
                    return False, reason

    # 3. 控制量检查（分模式）
    if actuator_mode == 0:
        for j in range(nq):
            if u_try[j] < limits.u_min[j] * margin:
                reason = f"u lower bound violated, joint={j}"
                _log_rejection(step, "u", j, reason)
                return False, reason
            if u_try[j] > limits.u_max[j] * margin:
                reason = f"u upper bound violated, joint={j}"
                _log_rejection(step, "u", j, reason)
                return False, reason
    else:
        # 位置模式：不检查 dq_max（|u-q| 是位置误差，是产生力矩的必要条件）。
        # 实际运动安全由 qdot 检查（步骤2）和 forcerange（env 层）覆盖。
        pass

    # 4. 关节加速度（Phase1: 审计日志 + 滑窗估算，不硬拒）
    if not skip_qddot and np.isfinite(limits.qddot_max[0]):
        if qdot_history is not None and len(qdot_history) >= 2:
            qddot_est = compute_qddot_filtered(
                list(qdot_history), dt, limits.qddot_window_size,
            )
            # 一次性审计日志：验证 qddot_est * effective_dt ≈ qdot_delta
            _qddot_audited = getattr(limits, '_qddot_audit_done', False)
            if not _qddot_audited and step >= 0:
                qdot_delta = qdot_history[-1] - qdot_history[-len(qdot_history)]
                eff_dt = (len(qdot_history) - 1) * dt
                check_val = qddot_est * eff_dt
                err = np.max(np.abs(check_val - qdot_delta))
                logger.info(
                    "[QDDOT_AUDIT] effective_dt=%.4fs window=%d "
                    "max_qdot_delta=%.2f max_qddot=%.1f check_err=%.2e",
                    eff_dt, len(qdot_history) - 1,
                    float(np.max(np.abs(qdot_delta))),
                    float(np.max(np.abs(qddot_est))), err,
                )
                limits._qddot_audit_done = True  # type: ignore[attr-defined]

            for j in range(nq):
                if abs(qddot_est[j]) > limits.qddot_max[j] * margin:
                    if limits.qddot_hard_reject:
                        reason = f"qddot limit exceeded, joint={j}"
                        _log_rejection(step, "qddot", j, reason)
                        return False, reason
                    elif step >= 0 and step % 20 == 0:
                        logger.debug(
                            "[QDDOT_WARN] k=%d: joint=%d, |%.1f| > %.1f",
                            step, j, abs(qddot_est[j]), limits.qddot_max[j] * margin,
                        )

    return True, ""


def _log_rejection(step: int, constraint_type: str, joint: int, reason: str) -> None:
    if step < 0:
        return
    logger.debug("[HARD_CONSTRAINT] k=%d [%s] j=%d: %s", step, constraint_type, joint, reason)


# ==============================================================================
#  制动感知安全滤波器（执行层）
# ==============================================================================

def strict_braking_check(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    u_try: np.ndarray,
    limits: RobotLimits,
    dt: float,
    k_hit_remaining: int = 99,
    env=None,
    actuator_mode: int = 0,
) -> tuple[bool, str]:
    """执行层严格制动：q/u 硬拒绝，qdot 严格制动感知，TCP 速度限制。

    规则：
      - 未超速 → 下一步不允许新进入超速 (hard reject)
      - 已超速 → 下一步必须减速 (hard reject 加速方向控制)
      - TCP 速度超限 → hard reject
    终段豁免 (k_hit ≤ terminal_exempt_steps):
      - 只检查 q/u，不检查 qdot/TCP（击球需要速度）
      - terminal_exempt_steps=0 时全程无豁免
    actuator_mode: 0=力矩模式(检查 u_min/u_max), 1=位置模式(检查 dq_max)。

    Raises:
        ValueError: 状态/控制宽度与限值维数不一致。
    """
    nq = _nq_from_limits(limits)
    x_next_arr = _validate_state_width(np.asarray(x_next, dtype=np.float64), nq, "x_next")
    x_prev_arr = _validate_state_width(np.asarray(x_prev, dtype=np.float64), nq, "x_prev")
    u_arr = np.asarray(u_try, dtype=np.float64)
    if u_arr.shape != (nq,):
        raise ValueError(f"控制向量 u_try 形状 {u_arr.shape} 应为 ({nq},)")
    q_next = x_next_arr[:nq]
    qdot_prev = x_prev_arr[nq:]
    qdot_next = x_next_arr[nq:]

    for j in range(nq):
        if q_next[j] < limits.q_lower[j]:
            return False, f"q lower bound violated, joint={j}"
        if q_next[j] > limits.q_upper[j]:
            return False, f"q upper bound violated, joint={j}"

    if actuator_mode == 0:
        for j in range(nq):
            if u_try[j] < limits.u_min[j]:
                return False, f"u lower bound violated, joint={j}"
            if u_try[j] > limits.u_max[j]:
                return False, f"u upper bound violated, joint={j}"
    else:
        # 位置模式：不检查 dq_max，由 qdot 检查和 forcerange 覆盖
        pass

    if k_hit_remaining <= limits.terminal_exempt_steps:
        return True, ""

    for j in range(nq):
        abs_prev = abs(qdot_prev[j])
        abs_next = abs(qdot_next[j])
        limit_j = limits.qdot_max[j]

        if abs_prev <= limit_j:
            if abs_next > limit_j:
                return False, (
                    f"qdot entering overspeed, joint={j}, "
                    f"|qdot|={abs_next:.2f} > {limit_j:.2f}"
                )
        else:
            if abs_next >= abs_prev:
                return False, (
                    f"qdot overspeeding+not-decelerating, joint={j}, "
                    f"|cur|={abs_prev:.2f} → |next|={abs_next:.2f} > {limit_j:.2f}"
                )

    # TCP 速度硬限制
    if env is not None and np.isfinite(limits.max_tcp_speed):
        env.set_arm_state(x_next)
        env.update_kinematics()
        tcp_vel = env.get_ee_vel()
        tcp_speed = float(np.linalg.norm(tcp_vel))
        if tcp_speed > limits.max_tcp_speed:
            return False, f"tcp speed {tcp_speed:.2f} > {limits.max_tcp_speed:.2f} m/s"

    return True, ""


def check_one_step_feasibility(
    x_current: np.ndarray,
    u_try: np.ndarray,
    limits: RobotLimits,
    dt: float,
    step_predictor,
    k_hit_remaining: int = 99,
    env=None,
) -> tuple[bool, str]:
    x_next = step_predictor(x_current, u_try)
    actuator_mode = getattr(env, 'actuator_mode', 0) if env is not None else 0
    return strict_braking_check(
        x_current, x_next, u_try, limits, dt, k_hit_remaining, env=env,
        actuator_mode=actuator_mode,
    )


def check_limits_on_trajectory(
    X: np.ndarray,
    U: np.ndarray,
    limits: RobotLimits | None,
    dt: float,
    actuator_mode: int = 0,
) -> tuple[bool, int, str]:
    """整条轨迹逐步限位检查（n 维：q/qdot 分界取自 limits 维数）。

    Returns:
        (ok, 首个违规步索引 k 或 -1, reason)。
    """
    if limits is None:
        return True, -1, ""
    nq = _nq_from_limits(limits)
    qdot_hist: deque = deque(maxlen=limits.qddot_window_size + 1)
    for k in range(len(U)):
        build_qdot_history(
            qdot_hist, np.asarray(X[k], dtype=np.float64)[nq:],
            limits.qddot_window_size,
        )
        ok, reason = check_step_feasibility(
            X[k], X[k + 1], U[k], limits, dt, step=k,
            skip_qdot=False, skip_qddot=False,
            qdot_history=qdot_hist,
            actuator_mode=actuator_mode,
        )
        if not ok:
            return False, k, reason
        build_qdot_history(
            qdot_hist, np.asarray(X[k + 1], dtype=np.float64)[nq:],
            limits.qddot_window_size,
        )
    return True, -1, ""
