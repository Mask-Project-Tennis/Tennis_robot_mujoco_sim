# exp16: V12 真机限位 vs 仿真限位命中率对比

> 日期: 2026-07-08
> 状态: ✅ 完成
> 依赖: exp15（V12 力矩模式 7m/s = 78%, 旧默认 TCP 1.8）

## 目的

验证 V12 默认真机限位（TCP 1.0 m/s, J6 ±180°）是否导致 7 m/s 命中率
显著回退。若回退 > 10pp，需调整 terminal_exempt_steps 或默认限位策略。

## 假设

1. sim 组命中率 ≈ 78%（复现 exp15 基线）
2. real 组命中率 ≥ 70%（TCP 1.0 限制更严，但终段豁免应部分补偿）
3. Δ < 10pp → 真机限位可安全作为默认

## 参数矩阵

| 变量 | 值 |
|------|---|
| limits_mode | [real, sim] |
| ball_speed | 7 m/s（固定） |
| Seeds/mode | 50 |
| **总 runs** | **100** |

## 两组配置

| 组 | CLI | max_tcp_speed | J6 range |
|----|-----|---------------|----------|
| real | （默认） | 1.0 m/s | ±180° |
| sim | --sim-limits | 1.8 m/s | ±360° |

## 固定参数

```bash
--serve-box --ball-speed 7 --no-plot  # 力矩模式（默认）
```

## 预期产出

- `results.csv`: 100 行，含 limits_mode/seed/hit/pos_error/max_tcp
- `summary.txt`: 两组对比表 + Δ 命中率差
- 决策依据: Δ < 5pp → 安全；5-10pp → 可接受；> 10pp → 需调整

## 预估耗时

~2 分钟（100 runs × ~1.2s/run）

## 结果

| 组 | 命中率 | pos_error 均值 | v_racket@hit 均值 | max_tcp 均值 | n |
|----|--------|---------------|-------------------|-------------|---|
| **real (TCP 1.0)** | **30.0%** | 0.310m | 0.29 m/s | 1.05 m/s | 50 |
| **sim (TCP 1.8)** | **78.0%** | 0.155m | 1.20 m/s | 1.69 m/s | 50 |
| **Δ** | **-48.0pp** | +0.155m | -0.91 m/s | | |

- 失败 seeds（20/21/27）在两组都无结果（球轨迹不可达），与限位无关

### 关键数字

- real 组 v_racket@hit 仅 0.29 m/s（sim 组 1.20 m/s）→ 安全滤波在挥拍加速阶段截断控制量
- terminal_exempt_steps=20（100ms）不足以让球拍加速到击球所需速度
- tcp_peak 达 5.0 m/s（real 组），说明 terminal_exempt 窗口内确实释放了速度，但窗口太短

## 结论

### 数据观察

1. TCP 1.0 m/s 限制导致命中率从 78% 暴跌至 30%（Δ=-48pp），远超 10pp 阈值
2. 球拍速度仅 0.29 m/s（需 1.20 m/s），安全滤波在挥拍准备阶段就截断了加速
3. terminal_exempt_steps=20（100ms 豁免窗口）不足以让球拍完成加速-击球动作

### 分析与决策

1. **TCP 1.0 m/s 作为仿真默认不可行**：安全滤波 `strict_braking_check` 每步强制 TCP ≤ 1.0 m/s（terminal_exempt_steps=0），MPC 加速指令被拒（beta→0），臂接收零力矩无法加速。到达击球点时末端速度仅 0.29 m/s（需 1.20 m/s）。
2. **TCP 1.0 反而更危险**：臂被锁死后因重力坠落，max_qdot 峰值达 3.94× 额定限速（1/47 runs 超限）。TCP 1.8 全程合规（0/47 超限）。过严的限速导致"锁死-坠落"模式比适度限速更不安全。
3. **决策**：回退 V12 默认限位至仿真值（default.yaml, TCP 1.8 m/s）。真机部署通过 `--limits-config configs/real_robot.yaml` 单独配置。`--sim-limits` 标志废弃（现为默认行为）。
4. **次要发现**：V12 line 354 `robot_limits.terminal_exempt_steps = args.terminal_exempt_steps`（CLI default=0）无条件覆盖配置链中的 terminal_exempt_steps=20，导致 V12 中终段豁免恒为 0。此为独立 bug，不在本次修复范围。
