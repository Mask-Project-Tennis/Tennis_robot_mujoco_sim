"""iLQR 求解器（C++ 加速版）。

在 solve_few_iters 内部，当 C++ 模块可用时，使用 C++ 加速：
  - 解析动力学线性化（~10× 加速）
  - 前向传递（~6× 加速）

C++ 模块不可用时自动回退到 Python 实现。
用法与原始 ILQTSolver 完全一致。
"""

import logging
import numpy as np
from src.ilqt.robot_env_protocol import RobotEnv
from src.ilqt.components.protocols import RunningCost, SmoothnessMixin

logger = logging.getLogger(__name__)


def _get_model_ptr(model) -> int:
    """获取 MuJoCo MjModel / MjData C 结构体指针。"""
    try:
        return model._address  # MuJoCo 3.x Python bindings
    except AttributeError:
        pass
    try:
        return model.ptr
    except AttributeError:
        pass
    try:
        import ctypes
        return ctypes.addressof(model._model)
    except (AttributeError, TypeError):
        pass
    raise RuntimeError("无法获取 MuJoCo 结构体指针")


# 尝试加载 C++ 加速模块
try:
    from src.cpp.iLQR_Core import (
        linearize_analytical_batch,
        forward_pass_single as cpp_forward_pass_single,
        backward_pass as cpp_backward_pass,
    )
    _CPP_AVAILABLE = True
    logger.info("iLQR C++ 加速模块已加载")
except (ImportError, ModuleNotFoundError):
    _CPP_AVAILABLE = False
    logger.info("C++ 加速模块未找到，使用纯 Python iLQR")


def _backward_pass_numpy(
    As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu
):
    """纯 numpy Riccati 后向传递（C++ 回退参考实现）。

    Args:
        As: (N,) list of (12,12) 动力学雅可比。
        Bs: (N,) list of (12,6) 控制雅可比。
        l_xs/l_us/l_xxs/l_uxs/l_uus: 运行代价导数列表。
        l_x_N: (12,) 终端代价一阶导数。
        l_xx_N: (12,12) 终端代价二阶导数。
        mu: 正则化参数。

    Returns:
        (Ks, ks) 增益列表，或 None（Q_uu_reg 奇异）。
    """
    N = len(As)
    n_u = Bs[0].shape[1]
    reg = mu * np.eye(n_u)
    Ks = []
    ks = []
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

if not _CPP_AVAILABLE:
    # 如果没有 C++ 模块，直接导入原始求解器
    from src.ilqt.solver import ILQTSolver  # noqa: F401
else:
    from src.dynamics.linearize import linearize_trajectory, linearize_analytical_trajectory
    from src.ilqt.robot_limits import RobotLimits
    from src.ilqt.utils import (
        compute_total_cost,
        forward_pass_with_linesearch,
    )

    class ILQTSolver:
        """iLQR 求解器（自动使用 C++ 加速）。

        与 src.ilqt.solver.ILQTSolver 接口完全兼容。
        """

        def __init__(
            self,
            config: dict,
            use_analytical: bool = True,
            horizon_override: int | None = None,
        ) -> None:
            """初始化求解器。

            Args:
                config: iLQR 超参数字典。
                use_analytical: 是否使用解析线性化（True=快速，False=有限差分）。
                horizon_override: 覆盖 config 中的 horizon 值。
            """
            self.max_iter = int(config["max_iter"])
            self.tol = float(config["tol"])
            self.horizon = (
                horizon_override if horizon_override is not None
                else int(config["horizon"])
            )
            self.mu_min = float(config["mu_min"])
            self.mu_max = float(config["mu_max"])
            self.mu_init = float(config["mu_init"])
            self.delta_0 = float(config["delta_0"])
            self.alpha_list = [float(a) for a in config["alpha_list"]]
            self.lin_eps = float(config["lin_eps"])
            self.use_analytical = use_analytical
            self._ball_geom_start: int | None = None

        def _linearize(self, env, X, U):
            """线性化：C++ 解析 > Python 解析 > Python 有限差分。"""
            use_ff = getattr(env, 'use_feedforward', False)
            if self.use_analytical and _CPP_AVAILABLE:
                return self._linearize_cpp(env, X, U)
            if self.use_analytical:
                actuator_mode = getattr(env, 'actuator_mode', 0)
                kp = getattr(env, 'kp', None)
                kd = getattr(env, 'kd', None)
                return linearize_analytical_trajectory(
                    env, X, U, self.lin_eps,
                    actuator_mode=actuator_mode, kp=kp, kd=kd,
                    use_feedforward=use_ff,
                )
            return linearize_trajectory(env, X, U, self.lin_eps)

        def _linearize_fast(self, env, X, U):
            """快速线性化（仅 M^{-1}，跳过 ∂h/∂q, ∂h/∂qdot 有限差分）。"""
            from src.dynamics.linearize import linearize_fast_trajectory
            actuator_mode = getattr(env, 'actuator_mode', 0)
            kp = getattr(env, 'kp', None)
            kd = getattr(env, 'kd', None)
            return linearize_fast_trajectory(
                env, X, U, actuator_mode=actuator_mode, kp=kp, kd=kd,
            )

        def _linearize_cpp(self, env, X, U):
            """C++ 加速的解析线性化。"""
            N = len(U)
            A_all = np.zeros((N, 12, 12))
            B_all = np.zeros((N, 12, 6))
            x_next_all = np.zeros((N, 12))
            actuator_mode = getattr(env, 'actuator_mode', 0)
            kp = getattr(env, 'kp', None)
            kd = getattr(env, 'kd', None)
            use_ff = getattr(env, 'use_feedforward', False)
            linearize_analytical_batch(
                A_all, B_all, x_next_all, X, U,
                _get_model_ptr(env.model), _get_model_ptr(env.data),
                env.init_q_left,
                self.lin_eps, env.dt,
                actuator_mode, kp, kd,
                use_ff,
            )
            As = [A_all[k] for k in range(N)]
            Bs = [B_all[k] for k in range(N)]
            fs = [x_next_all[k] for k in range(N)]
            return As, Bs, fs

        def _rollout(self, env, x0, U):
            """前向仿真获取轨迹。"""
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

        def _running_cost_derivatives(self, cost_fn, X, U):
            l_xs, l_us, l_xxs, l_uxs, l_uus = [], [], [], [], []
            for k in range(len(U)):
                if k > 0 and isinstance(cost_fn, SmoothnessMixin):
                    cost_fn.set_u_prev(U[k - 1])
                lx, lu, lxx, lux, luu = cost_fn.running_derivatives(
                    X[k], U[k], k
                )
                # 必须复制：CompositeCost（Flyweight 模式）返回持久引用，
                # 不复制会导致列表中所有元素指向同一数组（aliasing）
                l_xs.append(lx.copy())
                l_us.append(lu.copy())
                l_xxs.append(lxx.copy())
                l_uxs.append(lux.copy())
                l_uus.append(luu.copy())
            return l_xs, l_us, l_xxs, l_uxs, l_uus

        def _backward_pass(self, As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus,
                           l_x_N, l_xx_N, mu):
            """后向传递：C++ 加速。

            C++ 路径报告奇异（ok=False）时返回 None，触发外层正则化。
            numpy 回退路径：模块级 ``_backward_pass_numpy``（测试 monkeypatch 使用）。
            """
            N = len(As)
            ok, Ks_arr, ks_arr = cpp_backward_pass(
                np.stack(As), np.stack(Bs),
                np.stack(l_xs), np.stack(l_us),
                np.stack(l_xxs), np.stack(l_uxs), np.stack(l_uus),
                l_x_N, l_xx_N, mu,
            )
            if ok:
                return [Ks_arr[k] for k in range(N)], \
                       [ks_arr[k] for k in range(N)]
            return None

        def _get_ball_geom_start(self, env) -> int:
            """获取球 body 的 geom 起始索引（惰性缓存）。"""
            if self._ball_geom_start is None:
                self._ball_geom_start = env.model.body("ball").geomadr[0]
            return self._ball_geom_start

        def _build_check_params(self, limits, dt, actuator_mode) -> dict:
            """从 RobotLimits 构造 C++ check_step 参数字典。"""
            return {
                "q_lo": limits.q_lower, "q_hi": limits.q_upper,
                "qd_max": limits.qdot_max,
                "u_lo": limits.u_min, "u_hi": limits.u_max,
                "qdd_max": limits.qddot_max,
                "margin": limits.forward_pass_margin,
                "fp_q_tol": limits.forward_pass_q_tol_rad,
                "actuator_mode": actuator_mode,
                "qdd_window": limits.qddot_window_size,
                "dt": dt,
                "qdd_hard_reject": limits.qddot_hard_reject,
            }

        def _forward_pass_single(self, env, cost_fn, X, U, Ks, ks, alpha=0.5,
                                 limits=None):
            """前向传递：优先 C++（含碰撞禁用），limits 用 alpha 回退。"""
            if limits is None:
                return self._forward_pass_cpp(env, X, U, Ks, ks, alpha)
            # limits 模式：C++ + alpha 回退
            actuator_mode = getattr(env, 'actuator_mode', 0)
            check_params = self._build_check_params(limits, env.dt, actuator_mode)
            alphas_to_try = [alpha]
            for a in limits.alpha_fallback:
                if a not in alphas_to_try:
                    alphas_to_try.append(a)
            for alpha_try in alphas_to_try:
                result = self._forward_pass_cpp(env, X, U, Ks, ks, alpha_try,
                                                check_params)
                if result[0] is not None:
                    return result
            return None, None, float("inf"), "all alphas rejected by limits"

        def _forward_pass_cpp(self, env, X, U, Ks, ks, alpha, check_params=None):
            """C++ 前向传递（含 collision disable + actuator mode + optional limits）。"""
            N = len(U)
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)

            actuator_mode = getattr(env, 'actuator_mode', 0)
            kp = getattr(env, 'kp', None)
            kd = getattr(env, 'kd', None)
            use_ff = getattr(env, 'use_feedforward', False)
            torque_max = env._torque_ctrlrange[:, 1]
            ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
            ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
            ball_geom_start = self._get_ball_geom_start(env)

            Ks_flat = np.asarray(Ks).reshape(N, 72) if not isinstance(Ks, np.ndarray) else Ks
            ks_arr = np.asarray(ks).reshape(N, 6) if not isinstance(ks, np.ndarray) else ks

            ok, reason = cpp_forward_pass_single(
                X_new, U_new, X, U, Ks_flat, ks_arr,
                _get_model_ptr(env.model), _get_model_ptr(env.data),
                env.init_q_left,
                ctrl_lo, ctrl_hi, alpha,
                actuator_mode, kp, kd, use_ff, torque_max,
                ball_geom_start, True,
                check_params if check_params is not None else None,
            )
            if ok:
                return X_new, U_new, 0.0, ""
            return None, None, float("inf"), reason

        def _forward_pass_linesearch(
            self, env, cost_fn, X, U, Ks, ks, cost_old,
            limits=None,
        ):
            """线搜索前向传递：Python（C++版缺 collision disable，结果不一致）。"""
            return forward_pass_with_linesearch(
                env, cost_fn, X, U, Ks, ks, self.alpha_list, cost_old,
                limits=limits,
            )

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
            """运行指定次数的 iLQR 迭代（MPC 模式）。

            C++ 模块可用时自动使用 C++ 加速线性化和前向传递。

            Args:
                env: MuJoCo 环境实例。
                cost_fn: 代价函数实例。
                x0: 初始臂状态，形状 (12,)。
                U_init: 初始控制序列，形状 (N, 6)。
                max_iter: 最大迭代次数。
                skip_linesearch: 是否跳过线搜索。
                limits: 真实机器人硬约束参数。None 表示不启用。
                use_fast_lin: 是否使用快速线性化（跳过 ∂h/∂q, ∂h/∂qdot）。

            Returns:
                (X_opt, U_opt, cost_history, success):
                  success=False 表示硬约束饱和。
            """
            U = U_init.copy()
            X = self._rollout(env, x0, U)
            cost_old = compute_total_cost(env, cost_fn, X, U)
            cost_history = [cost_old]

            mu = self.mu_init
            rejection_streak: int = 0
            max_rejection_streak: int = 3
            solver_success: bool = True

            for iteration in range(max_iter):
                if use_fast_lin:
                    As, Bs, fs = self._linearize_fast(env, X, U)
                else:
                    As, Bs, fs = self._linearize(env, X, U)

                l_xs, l_us, l_xxs, l_uxs, l_uus = \
                    self._running_cost_derivatives(cost_fn, X, U)
                l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])

                result = self._backward_pass(
                    As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus,
                    l_x_N, l_xx_N, mu,
                )

                if result is None:
                    mu = min(mu * self.delta_0, self.mu_max)
                    continue

                Ks, ks = result

                if skip_linesearch:
                    X_new, U_new, cost_new, reject_reason = \
                        self._forward_pass_single(
                            env, cost_fn, X, U, Ks, ks,
                            limits=limits,
                        )
                    if X_new is None:
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

                    if np.isfinite(X_new[-1]).all():
                        X = X_new
                        U = U_new
                        mu = max(mu / self.delta_0, self.mu_min)
                        cost_history.append(cost_new)
                    else:
                        mu = min(mu * self.delta_0, self.mu_max)
                        cost_history.append(cost_old)
                else:
                    X_new, U_new, cost_new, accepted = \
                        self._forward_pass_linesearch(
                            env, cost_fn, X, U, Ks, ks, cost_old,
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
