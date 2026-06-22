# tests/ — 单元测试

> 332 tests，38 文件，~7000 行
> 运行需设 MuJoCo 库路径：`export LD_LIBRARY_PATH="$(python -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))'):$LD_LIBRARY_PATH"`

## 运行

```bash
# 全量测试
pytest tests/ -q

# 单文件
pytest tests/test_replay.py -v

# 按 keyword 筛选
pytest tests/ -k "safety" -v

# 显示警告
pytest tests/ -q -W default::RuntimeWarning
```

## 测试清单（按 src/ 结构分组）

### 基础模块（5 文件 / 41 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_kinematics.py` | 5 | `src/robot/kinematics.py` | FK / Jacobian 正确性 |
| `test_linearize.py` | 5 | `src/dynamics/linearize.py` | 动力学线性化（解析 vs 有限差分） |
| `test_jacobian_cache.py` | 8 | `src/ilqt/planning_env.py` | 雅可比缓存回归 |
| `test_noise.py` | 17 | `src/utils/noise.py` | 噪声注入（标量/per-axis/Z clamp） |
| `test_robot_env_protocol.py` | 1 | `src/ilqt/robot_env_protocol.py` | RobotEnv Protocol 验证 |

### iLQR + C++ 加速（3 文件 / 24 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_mpc.py` | 6 | `src/ilqt/solver.py` + `cost.py` | iLQR 基本求解 + 代价函数 |
| `test_cpp_forward_pass.py` | 14 | `src/cpp/forward_pass.cpp` | C++ 前向传递 vs Python 等效性 |
| `test_cpp_backward_pass.py` | 4 | `src/cpp/backward_pass.cpp` | C++ 后向传递 vs Python 等效性 |

### 管线架构（4 文件 / 21 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_episode_runner.py` | 6 | `episode_runner.py` + `step_context.py` + `strategy_config.py` | 管线循环 + hook 调用 + 策略 DI |
| `test_component_protocols.py` | 3 | `components/protocols.py` | 3 个 Protocol `@runtime_checkable` 验证 |
| `test_mpc_controller.py` | 4 | `mpc_controller.py` | MPCController 生命周期 |
| `test_sim_components.py` | 8 | `components/sim_*.py` | SimComponent + SimPerception + Safety 组件 |

### 策略模块（7 文件 / 51 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_strategies.py` | 10 | `strategies/phase_schedule.py` + `direction.py` | 阶段调度 + 方向策略 |
| `test_follow_through.py` | 11 | `strategies/follow_through.py` | PlannedFollowThrough（触发/控制/kp/kd） |
| `test_hit_point_refiner.py` | 4 | `strategies/hit_point_refiner.py` | HybridRefiner（安全/危险/锁定） |
| `test_replan_mode.py` | 11 | `strategies/replan_mode.py` | Sync/Async 重规划模式 |
| `test_replan_config.py` | 15 | `replan_config.py` | ReplanConfig 字段 + `from_mpc_config` + `to_dict` |
| `test_replan_core.py` | 2 | `replan_core.py` | do_replan 端到端 |
| `test_mpc_helpers.py` | 3 | `mpc_helpers.py` | JT 初始控制 + fix_joint5 |

### 仿真环境（4 文件 / 68 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_actuator_modes.py` | 46 | `sim/rm65_env.py` | 双模式执行器（力矩/位置）物理正确性 |
| `test_planning_env.py` | 10 | `planning_env.py` | PlanningEnv FK/Jacobian/step/IK |
| `test_planning_env_ball.py` | 2 | `planning_env.py` + `ball_predictor.py` | PlanningEnv 球轨迹预测 |
| `test_mpc.py` | 6 | （同 iLQR 组） | （跨组复用） |

### 回放 + 击打检测（2 文件 / 20 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_replay.py` | 14 | `sim/replay.py` | 碰撞窗口纯函数 + 弹性反弹物理 + 完整回放管线 |
| `test_hit_detection.py` | 6 | `sim/hit_detection.py` | 击球结果分类 + `determine_hit_from_type` |

### 安全（2 文件 / 7 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_safety_contract.py` | 4 | `predictive_safety.py` + `v12.py` | X-wall 常量一致性 + PredictiveSafetyFilter 拒绝不安全输入 |
| `test_safety_monitor.py` | 3 | `real/safety_monitor.py` | 关节位置/速度/TCP 三重检查 |

### 感知（7 文件 / 82 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_ball.py` | 9 | `tennis/ball.py` | 抛物线轨迹 + 弹跳 + serve_box 生成 |
| `test_ball_estimator.py` | 22 | `perception/ball_estimator.py` | 6D 卡尔曼滤波（预测/更新/弹跳/per-axis R/收敛） |
| `test_ball_obs_gate.py` | 21 | `perception/ball_obs_gate.py` | 观测门控（频率/噪声/KF 集成） |
| `test_ball_predictor.py` | 4 | `ilqt/ball_predictor.py` | 解析抛物线预测 |
| `test_ball_sensor.py` | 9 | `real/ball_sensor.py` | BallSensor ABC + SimulatedBallSensor |
| `test_estimator_pipeline.py` | 22 | `perception/` → `ilqt/` 端到端 | 噪声→KF→规划链路 |
| `test_noise.py` | 17 | （同基础模块组） | （跨组复用） |

### 真机部署（6 文件 / 25 tests）

| 文件 | tests | 覆盖模块 | 说明 |
|------|-------|---------|------|
| `test_real_runner.py` | 5 | `real/real_runner.py` | RealRunner Mock 闭环（start/step/stop） |
| `test_robot_interface.py` | 11 | `real/robot_interface.py` | Realman SDK Mock（连接/读关节/写角度/安全配置） |
| `test_fake_robot.py` | 2 | `real/fake_robot.py` | FakeRobot Mock |
| `test_torque_to_position.py` | 3 | `real/torque_to_position.py` | 力矩→位置积分器 |
| `test_adaptive_timer.py` | 3 | `real/adaptive_timer.py` | 自适应频率控制 |
| `test_async_replanner.py` | 1 | `ilqt/async_replanner.py` | 异步重规划器 |

## 编写规范

- **文件命名**：`test_<模块名>.py`，与 `src/` 结构对应
- **中文 docstring**：所有测试函数和类使用中文 docstring（Google 风格）
- **Mock 优先**：不依赖 MuJoCo 的测试用 Fake 类（如 `FakeMPC`、`FakeExecutor`、`FakeRobot`）
- **环境变量**：需 MuJoCo 的测试通过 fixture 提供 `RM65Env` 实例，不硬编码模型路径
- **TDD 纪律**：新功能先写 RED 测试，watch fail，再写 GREEN 实现

## Mock 基础设施

位于 `test_episode_runner.py` 中的 Fake 类可跨文件复用：

| Fake 类 | 模拟对象 | 关键行为 |
|---------|---------|---------|
| `FakeMPC` | MPCController | 第 N 步 done，返回固定 MPCStepResult |
| `FakePerception` | PerceptionComponent | 总返回 (zeros(3), zeros(3)) |
| `FakeExecutor` | ExecutorComponent | 记录所有 execute 调用 |
| `FakeSafety` | SafetyComponent | 总是放行 |
| `UnsafeSafety` | SafetyComponent | 总是拒绝 |

位于 `test_fake_robot.py` 的 `FakeRobot` 实现 `RobotArmInterface` Protocol，供真机模块测试使用。
