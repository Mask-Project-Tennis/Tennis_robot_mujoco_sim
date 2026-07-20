# 真机轨迹重演测试流程设计

**日期**: 2026-07-15
**状态**: 已批准
**作者**: Agent + 用户

## 背景与目标

在仿真中用位置模式生成能命中网球的挥拍轨迹，然后在真实 RM-65B 机械臂上重演。
本次测试以**空挥验证**为主（无真实网球），验证机械臂能否安全、准确地跟随仿真轨迹运动。

物理前提：
- 球拍已安装在右臂法兰，但 25mm FK 偏差（§14 TODO）尚未标定
- 空挥无需精确命中，偏差不影响运动形状验证
- 已完成 01-13 API 测试，但从未在真机运行过 MPC 轨迹

## 流程总览

```
Stage 0   : 仿真生成 3 条轨迹 (ball_speed=5, position-mode, 不同 seed)
Stage 0.5 : 预检 → 选最佳 1-2 条 (inspect_trajectory.py)
Stage 1   : Mock 验证管道完整性 (--mock --speed 0.1)
           → 物理安全清单 7 项确认 ←
Stage 2   : 真机极慢速 (--speed 0.05, --max-tcp-speed 0.3, 人手放急停按钮)
Stage 3   : 逐步加速 (0.1 → 0.25 → 0.5)，每级记录对比跟踪误差
Stage 4   : 分析跟踪误差，决定是否进入带球测试
```

## 各阶段详细设计

### Stage 0：仿真生成轨迹

生成 3 条温和的位置模式轨迹：

```bash
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 5 --position-mode \
    --dump-trajectory results/stage0_traj_sN.npz --seed N --no-plot
```

选取标准：
- `hit_type` = solid_hit 或 clean_hit（非 miss）
- 关节行程适中（不接近限位边界）
- 挥拍平滑（无大加速度跳变）

### Stage 0.5：轨迹预检

新建工具 `scripts/tools/inspect_trajectory.py`，功能：

1. **加载 .npz** → 打印基本信息（步数、dt、hit_step、metadata）
2. **关节限位检查**：6 关节 q_desired 范围 vs 真机限位，裕度 <10° 标注
3. **平滑性检查**：相邻步角速度 max(Δq/dt)，突跳 >30°/step 标注
4. **TCP 速度估计**：FK tcp_pos 差分速度，标注峰值
5. **绘图**：6 关节 q_desired 时间序列 + TCP xyz 轨迹

通过条件：
- 所有关节在限位内且有 >10° 裕度
- 无突跳（单步 >30°）
- TCP 峰值速度 < 2.0 m/s

### Stage 1：Mock 验证

```bash
python scripts/replay_trajectory.py \
    --trajectory results/stage0_traj_sN.npz --speed 0.1 --mock
```

通过条件：跑完全部步数，无异常退出，日志显示安全失败 False。

### 物理安全检查清单

| # | 检查项 | 确认方式 |
|---|--------|---------|
| 1 | 急停按钮位置已知，随手可及 | 物理确认 |
| 2 | workspace 半径 1m 内无障碍物 | 目视 |
| 3 | 球拍安装紧固，无松动 | 手动检查 |
| 4 | 线缆无缠绕、无干涉 | 目视 |
| 5 | 示教器网页可访问，无报警 | 浏览器 |
| 6 | rm_get_joint_degree 读数正常 | test 02 |
| 7 | 当前关节位置距 init_q < 30° | .npz init_q |

### Stage 2：真机极慢速（speed=0.05）

```bash
python scripts/replay_trajectory.py \
    --trajectory results/stage0_traj_sN.npz \
    --speed 0.05 --max-tcp-speed 0.3 \
    --record results/stage2_real_sN_speed005.npz
```

行为：pre_motion 移到 init_q → 1/20 速度重演 → 记录 q_actual。

通过条件：运动平滑、workspace 符合预期、SafetyMonitor 不触发。

失败处理：急停 → 断电 → 检查轨迹/配置 → 降速重试。

### Stage 3：逐步加速

| 级别 | speed | 通过条件 |
|------|-------|---------|
| Level 1 | 0.1 | max 跟踪误差 <2°，SafetyMonitor 不触发 |
| Level 2 | 0.25 | 同上 |
| Level 3 | 0.5 | 同上 |

升级原则：上一级完全通过才进入下一级。

### Stage 4：事后分析

对每条 --record 的 .npz 分析：
1. 跟踪误差曲线（q_desired vs q_actual 逐关节）
2. 误差 vs 速度对比
3. 结论：<1° → 可带球；1-2° → 先标定 FK；>2° → 调 kp/kd

## 需要新建的代码

| 文件 | 用途 |
|------|------|
| `scripts/tools/inspect_trajectory.py` | 轨迹预检（限位/平滑性/TCP速度/绘图） |

现有 `replay_trajectory.py` 和 `rm65_mpc_v12.py` 无需修改。

## 现有基础设施

完整 Source→Sink 重演管道已建好：
- `TrajectoryRecorder` 记录 + 保存 .npz
- `InterpolatingResampler` CubicSpline 时间拉伸
- `FileSource` / `ResampledSource` / `TcpSpeedLimiter` Source 链
- `RobotSink` / `RecorderSink` / `TeeSink` Sink 链
- `pre_motion()` 安全移到初始位置
- `SafetyMonitor` 关节位置/速度/TCP 实时检查
- `AdaptiveTimer` 100Hz 节拍控制
- `RobotInterface` rm_movej_follow 100Hz 下发
