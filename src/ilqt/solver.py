"""iLQT（迭代线性二次跟踪器）求解器。"""

import logging
import numpy as np
from typing import Any, Literal
from src.ilqt.robot_env_protocol import RobotEnv
from src.dynamics.linearize import linearize_trajectory, linearize_analytical_trajectory
from src.ilqt.components.protocols import RunningCost, SmoothnessMixin
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.utils import (
    compute_total_cost,
    forward_pass_with_linesearch,
    forward_pass_single,
)

logger = logging.getLogger(__name__)


class ILQTSolver:
    """iLQT 求解器：后向-前向迭代主循环。"""

    def __init__(
        self,
        config: dict,
        use_analytical: bool = True,
        horizon_override: int | None = None,
    ) -> None:
        """初始化求解器。

        Args:
            config: iLQT 超参数字典，包含：
                max_iter, tol, horizon, mu_min, mu_max, mu_init,
                delta_0, alpha_list, lin_eps。
            use_analytical: 是否使用解析线性化（True=快速，False=有限差分）。
            horizon_override: 覆盖 config 中的 horizon 值（用于 MPC 短地平线）。
        """
        self.max_iter = int(config["max_iter"])
        self.tol = float(config["tol"])
        self.horizon = horizon_override if horizon_override is not None else int(config["horizon"])
        self.mu_min = float(config["mu_min"])
        self.mu_max = float(config["mu_max"])
        self.mu_init = float(config["mu_init"])
        self.delta_0 = float(config["delta_0"])
        self.alpha_list = [float(a) for a in config["alpha_list"]]
        self.lin_eps = float(config["lin_eps"])
        self.use_analytical = use_analytical

    def _linearize(self, env: RobotEnv, X: np.ndarray, U: np.ndarray,
                   ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """根据配置选择线性化方法。"""
        if self.use_analytical:
            actuator_mode = getattr(env, 'actuator_mode', 0)
            kp = getattr(env, 'kp', None)
            kd = getattr(env, 'kd', None)
            use_ff = getattr(env, 'use_feedforward', False)
            return linearize_analytical_trajectory(
                env, X, U, self.lin_eps,
                actuator_mode=actuator_mode, kp=kp, kd=kd,
                use_feedforward=use_ff,
            )
        return linearize_trajectory(env, X, U, self.lin_eps)

    def solve(
        self,
        env: RobotEnv,
        cost_fn: RunningCost,
        x0: np.ndarray,
        U_init: np.ndarray | None = None,
        limits: RobotLimits | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """运行 iLQT 优化。

        Args:
            env: MuJoCo 环境实例。
            cost_fn: 代价函数实例。
            x0: 初始臂状态，形状 (NX,)。
            U_init: 初始控制序列，形状 (N, NU)。若为 None 则用零初始化。
            limits: 真实机器人硬约束参数。None 表示不启用。

        Returns:
            (X_opt, U_opt, cost_history): 最优轨迹、最优控制、代价历史。
        """
        N = self.horizon
        n_u = env.NU

        if U_init is None:
            U = np.zeros((N, n_u))
        else:
            U = U_init.copy()

        X = self._rollout(env, x0, U)
        cost_old = compute_total_cost(env, cost_fn, X, U)
        cost_history: list[float] = [cost_old]

        mu = self.mu_init

        for iteration in range(self.max_iter):
            As, Bs, fs = self._linearize(env, X, U)

            l_xs, l_us, l_xxs, l_uxs, l_uus = self._running_cost_derivatives(
                cost_fn, X, U
            )
            l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])

            result = self._backward_pass(
                As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu
            )

            if result is None:
                mu = min(mu * self.delta_0, self.mu_max)
                if mu >= self.mu_max:
                    logger.warning("正则化参数达到上限，停止迭代")
                    break
                continue

            Ks, ks = result

            X_new, U_new, cost_new, accepted = forward_pass_with_linesearch(
                env, cost_fn, X, U, Ks, ks, self.alpha_list, cost_old,
                limits=limits,
            )

            if accepted:
                X = X_new
                U = U_new
                mu = max(mu / self.delta_0, self.mu_min)

                rel_improve = abs(cost_old - cost_new) / max(abs(cost_old), 1.0)
                cost_old = cost_new
                cost_history.append(cost_old)

                logger.info(
                    f"迭代 {iteration}: 代价 = {cost_old:.6f}, "
                    f"相对改进 = {rel_improve:.2e}, mu = {mu:.2e}"
                )

                if rel_improve < self.tol:
                    logger.info(f"收敛于迭代 {iteration}")
                    break
            else:
                mu = min(mu * self.delta_0, self.mu_max)
                cost_history.append(cost_old)
                logger.info(
                    f"迭代 {iteration}: 未改进, mu = {mu:.2e}"
                )
                if mu >= self.mu_max:
                    logger.warning("正则化参数达到上限，停止迭代")
                    break

        return X, U, cost_history

    def solve_few_iters(
        self,
        env: RobotEnv,
        cost_fn: RunningCost,
        x0: np.ndarray,
        U_init: np.ndarray,
        max_iter: int = 3,
        skip_linesearch: bool = True,
        limits: RobotLimits | None = None,
        use_fast_lin: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[float], bool]:
        """运行指定次数的 iLQT 迭代（用于 MPC 实时规划）。

        Args:
            env: MuJoCo 环境实例。
            cost_fn: 代价函数实例。
            x0: 初始臂状态，形状 (NX,)。
            U_init: 初始控制序列，形状 (N, NU)。
            max_iter: 最大迭代次数。
            skip_linesearch: 是否跳过线搜索（MPC 模式默认 True）。
            limits: 真实机器人硬约束参数。None 表示不启用。
            use_fast_lin: 是否使用快速线性化（跳过 ∂h/∂q, ∂h/∂qdot）。
                ``use_analytical=False``（FD 后端）时该快捷路径自动回退
                完整有限差分，不绕过 backend 选择（Task 3.5）。

        Returns:
            (X_opt, U_opt, cost_history, success):
              success=False 表示硬约束饱和，所有迭代 forward pass 均被拒绝。
        """
        U = U_init.copy()
        X = self._rollout(env, x0, U)
        cost_old = compute_total_cost(env, cost_fn, X, U)
        cost_history: list[float] = [cost_old]

        mu = self.mu_init
        rejection_streak: int = 0
        max_rejection_streak: int = 3
        solver_success: bool = True

        for iteration in range(max_iter):
            if use_fast_lin:
                As, Bs, fs = self._linearize_fast(env, X, U)
            else:
                As, Bs, fs = self._linearize(env, X, U)

            l_xs, l_us, l_xxs, l_uxs, l_uus = self._running_cost_derivatives(
                cost_fn, X, U
            )
            l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])

            result = self._backward_pass(
                As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu
            )

            if result is None:
                mu = min(mu * self.delta_0, self.mu_max)
                continue

            Ks, ks = result

            if skip_linesearch:
                X_new, U_new, cost_new, reject_reason = forward_pass_single(
                    env, cost_fn, X, U, Ks, ks, alpha=0.5,
                    limits=limits,
                )
                if X_new is None:
                    # 所有 alpha 被硬约束拒绝
                    rejection_streak += 1
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
                    logger.warning(
                        "迭代 %d: forward pass 硬约束拒绝 (%s), "
                        "rejection_streak=%d, mu=%.2e",
                        iteration, reject_reason, rejection_streak, mu,
                    )
                    if rejection_streak >= max_rejection_streak:
                        solver_success = False
                        logger.warning(
                            "迭代 %d: 连续 %d 次硬约束拒绝，求解失败",
                            iteration, rejection_streak,
                        )
                        break
                    continue
                else:
                    rejection_streak = 0

                # MPC 模式：始终接受更新，依赖重规划纠错
                if np.isfinite(X_new[-1]).all():
                    X = X_new
                    U = U_new
                    mu = max(mu / self.delta_0, self.mu_min)
                    cost_history.append(cost_new)
                else:
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
            else:
                X_new, U_new, cost_new, accepted = forward_pass_with_linesearch(
                    env, cost_fn, X, U, Ks, ks, self.alpha_list, cost_old,
                    limits=limits,
                )

                if accepted:
                    X = X_new
                    U = U_new
                    mu = max(mu / self.delta_0, self.mu_min)
                    cost_old = cost_new
                    cost_history.append(cost_old)
                    rejection_streak = 0
                else:
                    rejection_streak += 1
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
                    if rejection_streak >= max_rejection_streak:
                        solver_success = False
                        break
                    if mu >= self.mu_max:
                        solver_success = False
                        break

        return X, U, cost_history, solver_success

    def _rollout(
        self, env: RobotEnv, x0: np.ndarray, U: np.ndarray
    ) -> np.ndarray:
        """前向仿真获取轨迹（规划期间禁用球拍碰撞，避免干扰球轨迹）。"""
        has_collision_ctrl = hasattr(env, "set_arm_collision")
        if has_collision_ctrl:
            env.set_arm_collision(False)
        N = len(U)
        X = np.zeros((N + 1, env.NX))
        env.set_arm_state(x0)
        X[0] = x0.copy()
        for k in range(N):
            X[k + 1] = env.step_from_state(X[k], U[k])
        if has_collision_ctrl:
            env.set_arm_collision(True)
        return X

    def _running_cost_derivatives(
        self, cost_fn: RunningCost, X: np.ndarray, U: np.ndarray
    ) -> tuple[list, list, list, list, list]:
        """计算所有时间步的运行代价导数。"""
        N = len(U)
        l_xs, l_us, l_xxs, l_uxs, l_uus = [], [], [], [], []
        for k in range(N):
            if k > 0 and isinstance(cost_fn, SmoothnessMixin):
                cost_fn.set_u_prev(U[k - 1])
            lx, lu, lxx, lux, luu = cost_fn.running_derivatives(X[k], U[k], k)
            # 必须复制：CompositeCost（Flyweight 模式）返回持久引用，
            # 不复制会导致列表中所有元素指向同一数组（aliasing）
            l_xs.append(lx.copy())
            l_us.append(lu.copy())
            l_xxs.append(lxx.copy())
            l_uxs.append(lux.copy())
            l_uus.append(luu.copy())
        return l_xs, l_us, l_xxs, l_uxs, l_uus

    def _linearize_fast(
        self, env: RobotEnv, X: np.ndarray, U: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """快速线性化（仅 M^{-1}，跳过 ∂h/∂q, ∂h/∂qdot 有限差分）。

        后端选择保护（Task 3.5）：``use_analytical=False``（FD 后端，
        R1D 等非前缀布局环境的既定选择）时**回退完整有限差分**，
        确保任何 solve 路径（含 ``solve_few_iters(use_fast_lin=True)``
        快捷路径）都不会绕过 backend 选择去计算基于前缀布局假设的
        解析近似 A/B。
        """
        if not self.use_analytical:
            return self._linearize(env, X, U)
        from src.dynamics.linearize import linearize_fast_trajectory
        actuator_mode = getattr(env, 'actuator_mode', 0)
        kp = getattr(env, 'kp', None)
        kd = getattr(env, 'kd', None)
        return linearize_fast_trajectory(
            env, X, U, actuator_mode=actuator_mode, kp=kp, kd=kd,
        )

    def _backward_pass(
        self,
        As: list[np.ndarray],
        Bs: list[np.ndarray],
        l_xs: list[np.ndarray],
        l_us: list[np.ndarray],
        l_xxs: list[np.ndarray],
        l_uxs: list[np.ndarray],
        l_uus: list[np.ndarray],
        l_x_N: np.ndarray,
        l_xx_N: np.ndarray,
        mu: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """后向传递：计算增益矩阵 K_k, k_k。

        使用 np.linalg.solve 替代 Cholesky 分解求逆，减少开销。

        Args:
            As, Bs: 动力学线性化矩阵列表。
            l_xs, l_us, ...: 运行代价导数列表。
            l_x_N, l_xx_N: 终端代价导数。
            mu: 正则化参数。

        Returns:
            (Ks, ks) 增益列表，或 None（若正则化失败）。
        """
        N = len(As)
        n_u = Bs[0].shape[1]
        reg = mu * np.eye(n_u)

        Ks: list[np.ndarray] = []
        ks: list[np.ndarray] = []

        V_x = l_x_N.copy()
        V_xx = l_xx_N.copy()

        for k in range(N - 1, -1, -1):
            A_T = As[k].T
            B_T = Bs[k].T

            Q_x = l_xs[k] + A_T @ V_x
            Q_u = l_us[k] + B_T @ V_x
            Q_xx = l_xxs[k] + A_T @ V_xx @ As[k]
            Q_ux = l_uxs[k] + B_T @ V_xx @ As[k]
            Q_uu = l_uus[k] + B_T @ V_xx @ Bs[k]

            Q_uu_reg = Q_uu + reg

            try:
                # solve(Q_uu_reg, I) 等价于 Q_uu_reg^{-1}，但数值更稳定
                Q_uu_inv = np.linalg.solve(Q_uu_reg, np.eye(n_u))
            except np.linalg.LinAlgError:
                return None

            K_k = -Q_uu_inv @ Q_ux
            k_k = -Q_uu_inv @ Q_u

            Ks.insert(0, K_k)
            ks.insert(0, k_k)

            Q_ux_T = Q_ux.T
            V_x = Q_x - Q_ux_T @ Q_uu_inv @ Q_u
            V_xx = Q_xx - Q_ux_T @ Q_uu_inv @ Q_ux
            V_xx = 0.5 * (V_xx + V_xx.T)

        return Ks, ks


# 旧 C++ 内核硬编码维数（src/cpp/types.h kNQ/kNX/kNU），兼容性判定的依据
_CPP_KERNEL_NQ = 6
_CPP_KERNEL_NX = 12
_CPP_KERNEL_NU = 6

# 缺省 iLQT 超参数（build_solver 无 config 时使用，与旧版本逐值一致）
_DEFAULT_SOLVER_CONFIG: dict[str, Any] = {
    "max_iter": 10,
    "tol": 1e-4,
    "horizon": 60,
    "mu_min": 1e-6,
    "mu_max": 1e10,
    "mu_init": 0.01,
    "delta_0": 1.6,
    "alpha_list": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
    "lin_eps": 1e-6,
}


def is_cpp_kernel_compatible(env: RobotEnv) -> bool:
    """判定环境是否满足旧六维 C++ 内核的全部布局假设。

    C++ 内核（``src/cpp/*``，kNQ=6/kNX=12/kNU=6 硬编码）假设：
    1. 受控臂维数恰为 n=6（NX=12/NU=6）——**维数是必要条件而非充分条件**；
    2. Python 入口直接消费 RM65 专属属性 ``init_q_left`` 与
       ``_torque_ctrlrange``（缺失即运行期崩溃，见
       ``src/cpp/solver_cpp.py::_linearize_cpp/_forward_pass_cpp``）；
    3. 臂的 qpos/qvel/执行器地址为**前缀布局**（qpos[0:6]/qvel[0:6]/
       ctrl[0:6]）——六维但地址重排（如 free joint 前置或执行器乱序的
       绑定环境）同样会落入错误的内存布局，必须拒绝。

    带 ``binding`` 属性的环境（``BoundRobotEnv`` 族）按绑定地址显式核对
    前缀；旧 RM65 环境（RM65Env/PlanningEnv，无 binding 属性）按属性
    组合判定。本谓词是 build_solver 的 backend 判定规则，也被
    ``src/cpp/solver_cpp.py`` 的 Python 入口能力拒绝复用（Task 3.5）。

    Args:
        env: 待判定的机器人环境。

    Returns:
        True 表示环境可安全进入旧六维 C++ 内核。
    """
    if env.NQ != _CPP_KERNEL_NQ or env.NX != _CPP_KERNEL_NX or env.NU != _CPP_KERNEL_NU:
        return False
    # 条件 2：C++ Python 入口直接消费的 RM65 专属属性
    if not hasattr(env, "init_q_left") or not hasattr(env, "_torque_ctrlrange"):
        return False
    # 条件 3：绑定环境必须为前缀地址布局
    binding = getattr(env, "binding", None)
    if binding is not None:
        prefix = np.arange(_CPP_KERNEL_NQ)
        if not (
            np.array_equal(np.asarray(binding.qpos_indices), prefix)
            and np.array_equal(np.asarray(binding.qvel_indices), prefix)
            and np.array_equal(np.asarray(binding.actuator_ids), prefix)
        ):
            return False
    return True


def build_solver(
    *,
    env: RobotEnv | None = None,
    backend: Literal["auto", "python", "cpp"] = "auto",
    config: dict[str, Any] | None = None,
) -> Any:
    """构建 ILQTSolver（带后端选择，Task 3.5 扩展签名）。

    纯 ILQTSolver 工厂函数，零真机依赖，自然属于 ``src.ilqt``。
    ``src/real/runner_factory.py`` 重导出本函数以保持真机入口与测试零改动。

    Backend 判定规则（3.8 消费方以此为准）：
    - ``backend='python'``：恒返回纯 Python ``ILQTSolver``。兼容环境
      （六维前缀布局 RM65 族）沿用旧 Python 缺省 ``use_analytical=True``；
      非兼容环境（R1D n=7、六维地址重排等）强制 ``use_analytical=False``
      （Python FD）——解析线性化同样基于前缀布局假设，非前缀环境不可用。
    - ``backend='cpp'``：显式要求 C++ 后端。环境不兼容旧六维内核时
      ``ValueError``（判定先于导入尝试，与机器上是否装有扩展无关）；
      入口不可导入或扩展不可用（未编译/ABI 不匹配）时同样 ``ValueError``，
      不静默回退。
    - ``backend='auto'``（缺省）：
      - ``env=None``：与旧版零参行为逐字一致（入口可导入 → C++ 类，
        否则 Python 类）；
      - 兼容环境：同旧偏好（入口可导入 → C++ 类，否则 Python 解析）；
      - 非兼容环境：**强制 Python FD**（``use_analytical=False``），
        即使机器上装有可用扩展也不让七维/重排输入落进六维内核。

    Args:
        env: 目标规划环境；None 表示沿用旧零参缺省行为。
        backend: 'auto' | 'python' | 'cpp'。
        config: iLQT 超参数字典；在缺省值上做按键覆盖（浅合并）。

    Returns:
        ILQTSolver 实例（C++ 或 Python 实现，二者接口一致）。

    Raises:
        ValueError: backend 非法；显式 cpp 但环境不兼容或扩展不可用。
    """
    if backend not in ("auto", "python", "cpp"):
        raise ValueError(
            f"未知 backend={backend!r}（支持 'auto' / 'python' / 'cpp'）"
        )
    cfg: dict[str, Any] = {**_DEFAULT_SOLVER_CONFIG, **(config if config is not None else {})}
    compatible = True if env is None else is_cpp_kernel_compatible(env)

    if backend == "python":
        # 兼容环境沿用旧 Python 缺省（解析线性化）；非兼容环境强制 FD
        return ILQTSolver(cfg, use_analytical=compatible)

    if backend == "cpp":
        if env is not None and not compatible:
            raise ValueError(
                "显式要求 C++ 后端，但环境与旧六维内核不兼容"
                f"（NQ={env.NQ}/NX={env.NX}/NU={env.NU} 或地址布局非前缀；"
                "C++ 内核为 RM65 前缀布局专用，R1D 等请使用 "
                "backend='python'——auto 会自动选择 Python FD）"
            )
        try:
            from src.cpp.solver_cpp import ILQTSolver as CppILQTSolver
            from src.cpp import solver_cpp as _solver_cpp
        except ImportError as exc:
            raise ValueError(
                "显式要求 C++ 后端，但 src.cpp.solver_cpp 入口不可导入: "
                f"{exc}"
            ) from exc
        if not bool(getattr(_solver_cpp, "_CPP_AVAILABLE", False)):
            raise ValueError(
                "显式要求 C++ 后端，但 C++ 扩展模块不可用"
                "（未编译或 ABI 不匹配，如 cp313 扩展在 py3.11 下）；"
                "不静默回退 Python"
            )
        return CppILQTSolver(cfg)

    # backend == 'auto'
    if env is not None and not compatible:
        # R1D（n=7）及六维地址重排等非兼容环境：强制 Python FD，
        # 不做导入尝试——即使扩展可导入也不进旧内核
        return ILQTSolver(cfg, use_analytical=False)
    try:
        from src.cpp.solver_cpp import ILQTSolver as CppILQTSolver
    except ImportError:
        CppILQTSolver = ILQTSolver  # 旧回退语义：入口导入失败 → Python 类
    return CppILQTSolver(cfg)
