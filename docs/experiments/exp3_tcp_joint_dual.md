# exp3: TCP + 关节双约束

## 目的
在关节约束基础上增加 TCP 线速度硬限制，测试 TCP 约束对性能的影响。

## 参数矩阵
| 变量 | 取值 |
|------|------|
| ball_speed | 7, 8, 9, 10 m/s |
| tcp_limit | 1.0, 1.5, 1.8, 2.0, 2.5 m/s |
| seeds | 0-19 |

详见 `experiment_data/exp3_tcp_joint_dual/config.yaml`。

## 脚本
- 主脚本: `scripts/exp/run_tcp_limit_experiment_v3.py`（内部 monkey-patch 安全滤波器）

## 额外 CSV 列
| 列名 | 说明 |
|------|------|
| max_tcp_limit | TCP 速度硬限制 (m/s) |
| actual_max_tcp | 实际最大 TCP 速度 |

## 状态
✅ 已完成
