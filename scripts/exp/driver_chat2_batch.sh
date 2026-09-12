#!/bin/bash
# chat2 审稿补跑批次（9/12 夜）：
# 1) E6 时序重录（30 sync + 10 async，含逐步延迟 deciles → ECDF 数据源）——独占机器先行
# 2) 敏感性 sweep exp18_sensitivity（2600 runs）
# 3) 豁免窗口 TCP 记录 exp18_tcp_exempt（200 runs，dump-trajectory）
set -e
cd /data/fxy/research/Tennis_robot_mujoco_sim
export MUJOCO_GL=egl
PY=/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python

echo "==== [1/3] E6 时序重录（机器空闲，保证时序真实） ===="
$PY scripts/exp/collect_fig_assets.py --timing-episodes 30 --async-episodes 10 --scan-cap 0 --skip-npz

echo "==== [2/3] 敏感性 sweep ===="
$PY scripts/exp/batch_runner.py exp18_sensitivity --workers 16

echo "==== [3/3] 豁免窗口 TCP ===="
$PY scripts/exp/run_exemption_tcp.py --seeds 50 --workers 8

echo "ALL DONE $(date)"
