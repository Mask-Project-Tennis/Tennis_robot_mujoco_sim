"""异步重规划器：后台线程执行 iLQR，主线程永不阻塞。

架构：
  主线程（执行层，5ms/步）        后台线程（规划层）
  ──────────────────────        ──────────────
  读取球状态                      
  检查新规划就绪？                  
  ├─ 是: 切换 U_active            
  └─ 否: 继续旧 buffer            
  need_replan?                    
  ├─ 是: 提交请求 ──────→        拷贝球状态到 env_plan
  │   (非阻塞)                   执行完整规划流程
  │   继续执行 U_active          完成后写 U_pending
  │   ←─────── 通知就绪 ────     设 new_plan_ready
  u_cmd = U_active[idx++]
  safety_filter + env.step

关键设计：
  - 双缓冲：U_active（主线程消费）/ U_pending（后台写入）
  - 独立 MjData：后台线程使用独立 env_plan，避免竞态
  - 非阻塞提交：submit() 立即返回，后台线程异步执行
  - Buffer 耗尽保护：U_active 用完后 fallback 到 JT 控制
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from src.sim.rm65_env import RM65Env

if TYPE_CHECKING:
    from src.ilqt.mpc_controller import MPCConfig
    from src.ilqt.robot_env_protocol import RobotEnv
    from src.ilqt.robot_limits import RobotLimits
    from src.ilqt.solver import ILQTSolver

logger = logging.getLogger(__name__)


@dataclass
class PlanRequest:
    """规划请求：主线程 → 后台线程。"""

    x_current: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    step: int
    k_hit_current: int
    U_prev: np.ndarray
    p_hit_current: np.ndarray
    v_hit_desired: np.ndarray
    n_des_current: np.ndarray
    d_hat: np.ndarray                           # 期望击球方向（来球反方向）
    is_first_plan: bool = False
    d_follow: np.ndarray | None = None          # 随挥方向；None 时回退为 d_hat


@dataclass
class PlanResult:
    """规划结果：后台线程 → 主线程。"""
    U_buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    U_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    U_mpc_full: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    k_hit_new: int = 0
    p_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_des_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solver_ok: bool = True
    iters_plan: int = 0
    horizon_plan: int = 0
    fast_lin: bool = False
    fp_limits_was_none: bool = False
    plan_duration_ms: float = 0.0
    ball_unreachable: bool = False
    request_step: int = -1
    hitting_tube_data: object = None
    # 控制维数（R1D=7 与 RM65=6 共用本结构；空控制序列形状按 NU 归一化）
    nu: int = 6

    def __post_init__(self) -> None:
        """按 ``nu`` 归一化三条控制序列的形状。

        后台线程在“球不可达/规划异常”等路径直接构造 ``PlanResult``，
        此时尚未填充控制序列，必须让空数组的列数等于实际 NU：
        - 空序列统一改写为 ``(0, nu)``（R1D 为 ``(0, 7)``，RM65 为 ``(0, 6)``）；
        - 非空序列列数必须等于 ``nu``，否则主线程按列消费会静默错位。

        Raises:
            ValueError: nu 非正、序列不是二维、或非空序列列数不等于 nu。
        """
        nu = int(self.nu)
        if nu <= 0:
            raise ValueError(f"PlanResult.nu 必须为正整数，实际 {self.nu!r}")
        for name in ("U_buffer", "U_prev", "U_mpc_full"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2:
                raise ValueError(
                    f"PlanResult.{name} 必须是二维数组，实际 shape={value.shape}"
                )
            if value.size == 0:
                value = np.zeros((0, nu), dtype=np.float64)
            elif value.shape[1] != nu:
                raise ValueError(
                    f"PlanResult.{name} 列数应为 nu={nu}，实际 {value.shape[1]}"
                )
            setattr(self, name, value)


class AsyncReplanner:
    """异步重规划器：后台线程执行 iLQR 规划，主线程永不阻塞。

    使用双缓冲机制：
    - U_active：主线程正在消费的控制序列
    - U_pending：后台线程规划完成后写入的新序列
    主线程在合适的时机原子切换到新序列。

    Args:
        env: 主线程的 MuJoCo 执行环境（RM65 路径仅读取 model_path/dt；
            R1D 绑定环境路径不读取构造签名，改由 ``planning_env_factory``
            交付独立规划环境）。
        replan_fn: 规划函数，签名为
            (request, env_plan, state, config, robot_limits, solver) -> PlanResult。
        config: MPCConfig（规划参数集，类型安全直接消费）。
        robot_limits: 关节约束 + 安全滤波器。
        solver: ILQTSolver 实例（C++ 或 Python；后台线程独占实例）。
        state: 初始重规划状态（可选）。
        model_path: 模型 XML 路径（可选，自动推断；factory 路径忽略）。
        planning_env_factory: 规划环境工厂（可选）。提供时每个重规划器
            通过它构建**独立**的规划环境实例（独立 MjData），不依赖
            ``type(env)(model_path, dt=...)`` 构造签名、不调用
            ``clone_actuator_config``；R1D 异步路径必须提供本参数。
    """

    def __init__(
        self,
        env: "RobotEnv",
        replan_fn: "Callable[..., PlanResult]",
        config: "MPCConfig",
        robot_limits: "RobotLimits",
        solver: "ILQTSolver",
        state: object | None = None,
        model_path: "Path | None" = None,
        planning_env_factory: "Callable[[], RobotEnv] | None" = None,
    ) -> None:
        """初始化异步重规划器（RM65 延迟构造 / R1D 工厂注入两条路径）。

        Args:
            env: 主线程执行环境（只读引用；后台线程不得触碰其 MjData）。
            replan_fn: 规划函数（do_replan）。
            config: MPC 配置。
            robot_limits: 关节约束 + 安全滤波器。
            solver: 后台线程独占的求解器实例。
            state: 初始 ReplanState（MPCController 在 start() 中回填）。
            model_path: RM65 路径的模型 XML；factory 路径可为 None。
            planning_env_factory: 规划环境工厂；None 时保持 T0 RM65 行为
                （``type(env)(model_path, dt=...)`` 延迟构造）。

        Raises:
            ValueError: 传入绑定环境（BoundRobotEnv 族，R1D）但未提供
                ``planning_env_factory``。绑定环境无法按 RM65 构造签名
                延迟构造 env_plan，必须由调用方显式提供独立工厂，
                否则明确拒绝而非静默改同步或延迟崩溃。
        """
        if hasattr(env, "binding") and planning_env_factory is None:
            raise ValueError(
                "AsyncReplanner 不支持未提供 planning_env_factory 的绑定环境"
                "（R1D）：绑定环境无法按 RM65 构造签名（model_path, dt）"
                "延迟构造 env_plan。请注入 planning_env_factory（返回独立"
                "规划环境实例），或改用同步重规划。"
            )
        self._replan_fn = replan_fn
        # 静态配置：MPCConfig + RobotLimits + ILQTSolver 直接注入，
        # 后台线程调用 do_replan(request, env_plan, state, config, robot_limits, solver)
        self._config: "MPCConfig" = config
        self._robot_limits: "RobotLimits" = robot_limits
        self._solver: "ILQTSolver" = solver
        self._state = state
        self._env = env
        self._planning_env_factory = planning_env_factory

        # 创建独立 MjData 的规划环境（RM65：共享 MjModel 只读/独立 MjData；
        # R1D：工厂交付完全独立实例，后台线程独占）
        if planning_env_factory is not None:
            # 工厂路径不从执行环境推断模型路径（R1D 模型路径由工厂持有）
            self._model_path: Path | None = (
                Path(model_path) if model_path is not None else None
            )
        else:
            self._model_path = model_path or self._find_model_path(env)
        self._dt = env.dt
        self._NU: int = int(env.NU)
        self.env_plan: "RobotEnv | None" = None  # 延迟初始化
        self._env_plan_lock = threading.Lock()

        # 线程同步
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # 请求/结果
        self._request: PlanRequest | None = None
        self._result: PlanResult | None = None
        self._has_new_plan = False
        self._is_planning = False

        # 统计
        self.submit_count = 0
        self.complete_count = 0
        self.buffer_exhaustion_count = 0
        self.replan_durations_ms: list[float] = []
        self.replan_k_hit_history: list[int] = []

    @staticmethod
    def _find_model_path(env: RM65Env) -> Path:
        """从 env 实例推导 model 文件路径。"""
        # MuJoCo 3.x: model.filedir 给出 XML 目录
        filedir = env.model.filedir
        # 查找目录下的 XML 文件
        import glob
        xml_files = glob.glob(str(Path(filedir) / "*.xml"))
        if xml_files:
            return Path(xml_files[0])
        # 回退：使用已知路径
        return Path("src/robot/rm65_model.xml")

    def start(self) -> None:
        """启动后台规划线程（重复调用幂等）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        # 清理上一回合残留的请求/结果标志，保证重复回合互不污染
        with self._lock:
            self._request = None
            self._result = None
            self._has_new_plan = False
            self._is_planning = False
        self._thread = threading.Thread(target=self._planner_loop, daemon=True)
        self._thread.start()
        logger.info("AsyncReplanner: 后台规划线程已启动")

    def stop(self) -> None:
        """停止后台规划线程（幂等：重复调用/未启动均安全）。"""
        self._stop_event.set()
        self._request_event.set()  # 唤醒等待中的线程
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "AsyncReplanner: 后台线程在 5s 内未退出（可能仍在规划）"
                )
            self._thread = None
        with self._lock:
            self._request = None
            self._is_planning = False
        logger.info("AsyncReplanner: 后台规划线程已停止")

    def close(self) -> None:
        """释放后台线程资源（等价 ``stop()``，便于上下文管理器/收尾调用）。"""
        self.stop()

    def submit(self, request: PlanRequest) -> bool:
        """非阻塞提交规划请求。

        如果后台线程正在规划，则跳过（避免堆积请求）。

        Returns:
            True 表示请求已提交，False 表示后台忙（已跳过）。
        """
        with self._lock:
            if self._is_planning:
                logger.debug(f"步 {request.step}: 后台正在规划，跳过本次请求")
                return False
            # 深拷贝请求：主线程提交后可能立即复用/改写数组（如 ball_pos
            # 缓冲），后台线程必须持有提交时刻的不可变快照（T4 §5 要求）。
            self._request = copy.deepcopy(request)
            self._is_planning = True
            self.submit_count += 1
        self._request_event.set()
        return True

    def has_new_plan(self) -> bool:
        """检查后台规划是否完成且有新结果。"""
        return self._has_new_plan

    def apply_new_plan(self) -> PlanResult | None:
        """取出规划结果并重置标志。

        Returns:
            PlanResult 或 None（如果没有新结果）。
        """
        if not self._has_new_plan:
            return None
        with self._lock:
            result = self._result
            self._result = None
            self._has_new_plan = False
        if result is not None:
            self.complete_count += 1
            self.replan_durations_ms.append(result.plan_duration_ms)
            self.replan_k_hit_history.append(result.k_hit_new)
        return result

    def is_planning(self) -> bool:
        """后台线程是否正在规划。"""
        with self._lock:
            return self._is_planning

    def _ensure_env_plan(self) -> "RobotEnv":
        """确保独立规划环境就绪（线程安全，主线程/后台线程均可调用）。

        两条路径：
        - ``planning_env_factory`` 提供时：调用工厂构建独立规划环境
          （R1D：全新 MjModel/MjData + 已配置执行器模式），不触碰执行环境；
        - 否则保持 T0 RM65 行为：``type(env)(model_path, dt=...)`` 延迟构造，
          复制 ``init_q_left`` 并 ``clone_actuator_config``。

        Returns:
            独立规划环境实例（与执行环境不共享 MjData）。

        Raises:
            ValueError: 工厂返回 None 或返回执行环境本身（会破坏线程隔离）。
        """
        with self._env_plan_lock:
            if self.env_plan is not None:
                return self.env_plan
            if self._planning_env_factory is not None:
                env_plan = self._planning_env_factory()
                if env_plan is None:
                    raise ValueError("planning_env_factory 返回 None")
                if env_plan is self._env:
                    raise ValueError(
                        "planning_env_factory 返回了执行环境本身："
                        "后台规划必须使用独立实例（独立 MjData）"
                    )
                self.env_plan = env_plan
                logger.info(
                    "AsyncReplanner: planning_env_factory 创建规划环境"
                    "（独立实例，NQ=%d NU=%d）",
                    int(env_plan.NQ),
                    int(env_plan.NU),
                )
                return env_plan
            # T0 RM65 延迟构造路径（执行环境类型即规划环境类型）
            env_class = type(self._env)
            env_plan = env_class(self._model_path, dt=self._dt)
            if hasattr(self._env, "init_q_left"):
                env_plan.init_q_left = self._env.init_q_left.copy()
            self._env.clone_actuator_config(env_plan)
            # 初始化 env_plan 的 qpos 到默认值并调用 mj_forward
            import mujoco
            mujoco.mj_forward(env_plan.model, env_plan.data)
            self.env_plan = env_plan
            logger.info("AsyncReplanner: 规划环境已创建（独立 MjData）")
            return env_plan

    def _planner_loop(self) -> None:
        """后台线程主循环：等待请求 → 执行规划 → 通知主线程。

        env_plan 只在本线程/mpc 主线程经 ``_ensure_env_plan()`` 获取一次，
        循环内复用局部引用，避免每次都做锁竞争与 None 判定。
        """
        env_plan = self._ensure_env_plan()

        while not self._stop_event.is_set():
            # 阻塞等待规划请求
            self._request_event.wait(timeout=1.0)
            self._request_event.clear()

            if self._stop_event.is_set():
                break

            # 取出请求
            with self._lock:
                request = self._request
                self._request = None

            if request is None:
                continue

            t_start = time.perf_counter()
            result: PlanResult | None = None

            try:
                result = self._replan_fn(
                    request, env_plan, self._state,
                    self._config, self._robot_limits, self._solver,
                )
            except Exception as e:
                logger.error(f"AsyncReplanner: 规划异常: {e}", exc_info=True)
                # 异常结果按实际 NU 构造（R1D=7），主线程可安全判废
                result = PlanResult(
                    nu=self._NU, solver_ok=False, ball_unreachable=False,
                )
            finally:
                t_dur = (time.perf_counter() - t_start) * 1000
                if result is not None:
                    result.plan_duration_ms = t_dur

            assert result is not None  # 正常/异常路径均已赋值

            # 写入结果，通知主线程
            with self._lock:
                self._result = result
                self._has_new_plan = True
                self._is_planning = False

            logger.info(
                f"ASYNC_PLAN done: step={request.step} k_hit={result.k_hit_new} "
                f"iters={result.iters_plan} horizon={result.horizon_plan} "
                f"t={t_dur:.0f}ms fast_lin={result.fast_lin} ok={result.solver_ok}"
            )
