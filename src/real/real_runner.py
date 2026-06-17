"""真机部署主循环编排器 — 分步 start/step/stop 接口。

编排流程：
    读球感知 → MPC 规划 → 安全滤波 → 发送角度指令

使用方式：
    runner = RealRunner(env, robot, perceiver, safety, replanner,
                        replan_cfg, timer, replan_state)
    runner.start()
    while not runner.done:
        info = runner.step()
    metrics = runner.stop()

设计要点：
    - 首次规划同步完成（start 阶段），后续按 replan_interval 异步重规划
    - 位置模式下 U_buffer 存 q_desired（角度），直接发给机器人
    - 安全检查失败立即缓停并标记 done
    - 球不可达时优雅退出（done=True）
    - buffer 耗尽时 hold q（保持当前角度），等待新规划
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.replan_core import do_replan
from src.ilqt.tube_types import ReplanState
from src.real.adaptive_timer import AdaptiveTimer
from src.real.ball_perceiver import BallPerceiver
from src.real.robot_arm_protocol import RobotArmInterface
from src.real.safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)


class RealRunner:
    """真机部署主循环编排器 — 分步接口。

    所有依赖组件通过构造函数注入，支持 Mock 测试与真机部署复用。

    Args:
        env: 规划计算环境（PlanningEnv，MuJoCo 纯计算）。
        robot: 机器人接口（FakeRobot 或 RobotInterface）。
        ball_perceiver: 球感知器（sensor → KF 滤波 → pos/vel）。
        safety: 安全监控（关节位置/速度/TCP 速度检查）。
        replanner: 异步重规划器（后台线程 iLQR）。
        replan_cfg: do_replan 所需配置字典。
        timer: 自适应频率控制器。
        replan_state: 重规划可变状态快照。
    """

    def __init__(
        self,
        env: Any,
        robot: RobotArmInterface,
        ball_perceiver: BallPerceiver,
        safety: SafetyMonitor,
        replanner: AsyncReplanner,
        replan_cfg: dict,
        timer: AdaptiveTimer,
        replan_state: ReplanState,
    ) -> None:
        self._env = env
        self._robot = robot
        self._perceiver = ball_perceiver
        self._safety = safety
        self._replanner = replanner
        self._replan_cfg = replan_cfg
        self._timer = timer
        self._replan_state = replan_state

        # 运行状态
        self._step_count: int = 0
        self._max_steps: int = int(replan_cfg.get("total_horizon", 200))
        self._replan_interval: int = int(replan_cfg.get("replan_interval", 20))
        self._U_buffer: NDArray[np.floating] = np.zeros((0, env.NU))
        self._buffer_idx: int = 0
        self._x_current: NDArray[np.floating] = np.zeros(env.NX)
        self._u_cmd: NDArray[np.floating] = np.zeros(env.NU)
        self._last_ball_pos: NDArray[np.floating] | None = None
        self._last_ball_vel: NDArray[np.floating] | None = None

        # 结束标志
        self._done: bool = False
        self._ball_unreachable: bool = False
        self._safety_failed: bool = False

        # 指标
        self._safe_step_count: int = 0

        # 测试 hook：强制安全检查失败
        self._force_unsafe: bool = False

    # ── 启动 ──

    def start(self) -> None:
        """启动 episode：

        1. 连接机器人
        2. 启动球传感器
        3. 启动异步重规划线程
        4. 读取初始球状态 + 臂状态
        5. 计算 d_hat / v_hit_desired，更新 replan_cfg
        6. 同步执行首次规划
        7. 初始化 U_buffer / buffer_idx / step_count
        """
        # 1. 连接机器人
        self._robot.connect()
        logger.info("RealRunner: 机器人已连接")

        # 2. 启动球传感器
        self._perceiver.start_sensor()
        logger.info("RealRunner: 球传感器已启动")

        # 3. 启动异步重规划线程 + 确保规划环境就绪
        self._replanner.start()
        env_plan = self._replanner._ensure_env_plan()
        if env_plan is None:
            raise RuntimeError("RealRunner: 规划环境创建失败")
        logger.info("RealRunner: 异步重规划线程已启动")

        # 4. 读取初始球状态 + 臂状态
        self._perceiver.update()
        filtered = self._perceiver.get_latest_filtered()
        if filtered is None:
            logger.error("RealRunner: 初始球状态不可用（传感器无数据）")
            self._ball_unreachable = True
            self._done = True
            return
        ball_pos, ball_vel = filtered
        ball_pos = np.asarray(ball_pos, dtype=np.float64).copy()
        ball_vel = np.asarray(ball_vel, dtype=np.float64).copy()
        self._last_ball_pos = ball_pos
        self._last_ball_vel = ball_vel

        self._x_current = np.asarray(self._robot.get_arm_state(), dtype=np.float64).copy()

        # 5. 计算击球方向 d_hat（来球反方向）和期望击球速度
        ball_vel_norm = float(np.linalg.norm(ball_vel))
        if ball_vel_norm > 1e-6:
            d_hat = -ball_vel / ball_vel_norm
        else:
            d_hat = np.array([0.0, 1.0, 0.0])
        target_speed = float(self._replan_cfg.get("racket_speed", 1.8))
        if target_speed > 2.0:
            target_speed = 1.8  # 真机首版保守限速
        v_hit_desired = target_speed * d_hat
        self._replan_cfg["d_hat"] = d_hat
        self._replan_cfg["v_hit_desired"] = v_hit_desired
        self._replan_cfg["v_hit_at_contact"] = v_hit_desired
        logger.info(
            "RealRunner: d_hat=%s v_hit_desired=%s",
            np.round(d_hat, 3), np.round(v_hit_desired, 3),
        )

        # 6. 同步首次规划
        first_request = PlanRequest(
            x_current=self._x_current.copy(),
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            step=0,
            k_hit_current=int(self._replan_cfg.get("k_hit_total", self._max_steps)),
            U_prev=np.zeros((0, self._env.NU)),
            p_hit_current=ball_pos.copy(),
            v_hit_desired=v_hit_desired,
            n_des_current=d_hat.copy(),
            is_first_plan=True,
        )
        first_result = do_replan(
            first_request,
            self._replanner.env_plan,  # type: ignore[arg-type]
            self._replan_state,
            self._replan_cfg,
        )

        # 球不可达 → 优雅退出
        if first_result.ball_unreachable:
            logger.warning("RealRunner: 首次规划判定球不可达，提前结束 episode")
            self._ball_unreachable = True
            self._done = True
            return

        # 更新重规划状态
        self._replan_state.is_first_plan = False
        self._replan_state.k_hit_new = first_result.k_hit_new
        self._replan_state.p_hit_new = first_result.p_hit_new.copy()
        self._replan_state.v_ball_hit_new = first_result.v_ball_hit_new.copy()
        self._replan_state.current_n_des = first_result.n_des_new.copy()
        self._replan_state.U_prev = first_result.U_prev.copy()

        # 7. 初始化控制缓冲
        self._U_buffer = first_result.U_buffer.copy()
        self._buffer_idx = 0
        logger.info(
            "RealRunner: 首次规划完成 k_hit=%d horizon=%d buffer_len=%d solver_ok=%s",
            first_result.k_hit_new,
            first_result.horizon_plan,
            len(self._U_buffer),
            first_result.solver_ok,
        )

    # ── 单步 ──

    def step(self) -> dict[str, Any]:
        """执行一个控制 tick：

        1. 读球 (perceiver.update + get_latest_filtered)
        2. 读臂 (robot.get_arm_state)
        3. 重规划检查 (周期性 / buffer 耗尽)
        4. 提取控制 (U_buffer[idx] 或 hold q)
        5. 安全检查 (safety.is_safe)
        6. 发送指令 (robot.send_joint_command)
        7. 更新 step_count

        Returns:
            info dict，含 u_cmd / safe / step / done 等键。
        """
        if self._done:
            return {
                "u_cmd": self._u_cmd,
                "safe": False,
                "step": self._step_count,
                "done": True,
                "reason": "already_done",
            }

        # 1. 读球
        self._perceiver.update()
        filtered = self._perceiver.get_latest_filtered()
        if filtered is not None:
            self._last_ball_pos = np.asarray(filtered[0], dtype=np.float64)
            self._last_ball_vel = np.asarray(filtered[1], dtype=np.float64)

        # 2. 读臂
        self._x_current = np.asarray(
            self._robot.get_arm_state(), dtype=np.float64
        ).copy()

        # 3. 异步重规划：应用就绪的新规划
        if self._replanner.has_new_plan():
            result = self._replanner.apply_new_plan()
            if result is not None:
                if result.ball_unreachable:
                    logger.warning("RealRunner: 异步规划判定球不可达，结束 episode")
                    self._ball_unreachable = True
                    self._done = True
                    self._robot.slow_stop()
                    return {
                        "u_cmd": self._u_cmd,
                        "safe": False,
                        "step": self._step_count,
                        "done": True,
                        "reason": "ball_unreachable",
                    }
                self._U_buffer = result.U_buffer.copy()
                self._buffer_idx = 0
                self._replan_state.k_hit_new = result.k_hit_new
                self._replan_state.p_hit_new = result.p_hit_new.copy()
                self._replan_state.U_prev = result.U_prev.copy()

        # 周期性提交异步重规划请求
        need_submit = (
            self._step_count > 0
            and (
                self._step_count % self._replan_interval == 0
                or self._buffer_idx >= len(self._U_buffer)
            )
        )
        if need_submit and not self._replanner.is_planning():
            self._submit_replan()

        # 4. 提取控制
        if self._buffer_idx < len(self._U_buffer):
            u_cmd = np.asarray(
                self._U_buffer[self._buffer_idx], dtype=np.float64
            ).copy()
            self._buffer_idx += 1
        else:
            # buffer 耗尽 → hold q（保持当前角度）
            u_cmd = self._x_current[: self._env.NQ].copy()
            logger.debug("RealRunner 步 %d: buffer 耗尽，hold q", self._step_count)

        self._u_cmd = u_cmd

        # 5a. 位置模式 dq_max 限幅：逐步角度变化不超过 dq_max（平滑追踪规划轨迹）
        #     防止首规划（fp_limits=None）产生激进跳变直接发给真机
        robot_limits = self._replan_cfg.get("robot_limits")
        if robot_limits is not None and getattr(robot_limits, "dq_max", None) is not None:
            q_current = self._x_current[: self._env.NQ]
            dq = u_cmd - q_current
            dq_max = np.asarray(robot_limits.dq_max, dtype=np.float64)
            dq_clamped = np.clip(dq, -dq_max, dq_max)
            u_cmd = q_current + dq_clamped
            self._u_cmd = u_cmd

        # 5b. 安全检查（用雅可比精确计算 TCP 线速度）
        qdot = self._x_current[self._env.NQ:]
        try:
            self._env.set_arm_state(self._x_current)
            jacp = self._env.get_ee_jacp()
            tcp_speed = float(np.linalg.norm(jacp @ qdot))
        except Exception:
            tcp_speed = float(np.linalg.norm(qdot)) * 0.3  # 回退粗估
        safe = (not self._force_unsafe) and self._safety.is_safe(
            self._x_current, u_cmd, tcp_speed=tcp_speed
        )

        if not safe:
            logger.warning(
                "RealRunner 步 %d: 安全检查失败，缓停并结束 episode", self._step_count
            )
            self._safety_failed = True
            self._done = True
            self._robot.slow_stop()
            return {
                "u_cmd": u_cmd,
                "safe": False,
                "step": self._step_count,
                "done": True,
                "reason": "safety_failed",
            }

        # 6. 发送指令
        self._robot.send_joint_command(u_cmd)
        self._safe_step_count += 1
        self._step_count += 1

        # 7. 达到最大步数 → 结束
        if self._step_count >= self._max_steps:
            self._done = True
            logger.info("RealRunner: 达到最大步数 %d，结束 episode", self._max_steps)

        return {
            "u_cmd": u_cmd,
            "safe": True,
            "step": self._step_count,
            "done": self._done,
            "ball_pos": self._last_ball_pos,
            "ball_vel": self._last_ball_vel,
            "buffer_remaining": max(0, len(self._U_buffer) - self._buffer_idx),
        }

    def _submit_replan(self) -> None:
        """提交异步重规划请求（非阻塞）。"""
        if self._last_ball_pos is None or self._last_ball_vel is None:
            return
        request = PlanRequest(
            x_current=self._x_current.copy(),
            ball_pos=self._last_ball_pos.copy(),
            ball_vel=self._last_ball_vel.copy(),
            step=self._step_count,
            k_hit_current=self._replan_state.k_hit_new,
            U_prev=self._replan_state.U_prev.copy(),
            p_hit_current=self._replan_state.p_hit_new.copy(),
            v_hit_desired=self._replan_cfg["v_hit_desired"],
            n_des_current=self._replan_state.current_n_des.copy(),
            is_first_plan=False,
        )
        self._replanner.submit(request)

    # ── 状态查询 ──

    @property
    def done(self) -> bool:
        """是否结束（max_steps / safety / unreachable）。"""
        return self._done

    @property
    def ball_unreachable(self) -> bool:
        """球是否被判不可达。"""
        return self._ball_unreachable

    @property
    def safety_failed(self) -> bool:
        """是否因安全检查失败而结束。"""
        return self._safety_failed

    # ── 清理 ──

    def stop(self) -> dict[str, Any]:
        """清理资源并返回 episode 指标。

        停止异步重规划线程、球传感器，断开机器人。

        Returns:
            metrics dict，含 total_steps / safe_steps / ball_unreachable 等键。
        """
        try:
            self._replanner.stop()
        except Exception as e:
            logger.warning("RealRunner.stop: 停止重规划线程异常: %s", e)

        try:
            self._perceiver.stop_sensor()
        except Exception as e:
            logger.warning("RealRunner.stop: 停止球传感器异常: %s", e)

        try:
            self._robot.disconnect()
        except Exception as e:
            logger.warning("RealRunner.stop: 断开机器人异常: %s", e)

        metrics: dict[str, Any] = {
            "total_steps": self._step_count,
            "safe_steps": self._safe_step_count,
            "ball_unreachable": self._ball_unreachable,
            "safety_failed": self._safety_failed,
            "done": self._done,
            "replan_submit_count": self._replanner.submit_count,
            "replan_complete_count": self._replanner.complete_count,
        }
        logger.info("RealRunner.stop: %s", metrics)
        return metrics
