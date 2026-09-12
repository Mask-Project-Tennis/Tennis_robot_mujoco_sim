#!/usr/bin/env bash
# ============================================================================
# run_exp17i.sh —— exp17i_limits_ablation 配对实验启动器（幂等）
#
# 结构: batch_runner exp17i_limits_ablation（TCP 1.0 vs 1.8 × 4 档 × 400 seeds
#       @7 m/s = 3,200 runs, ~7 min）
#
# 幂等: 已有 _.COMPLETE 直接退出; batch_runner --resume 跳过已完成 (config, seed)。
# 用法:
#   bash scripts/exp/run_exp17i.sh
#   tmux new-session -d -s exp17i "bash scripts/exp/run_exp17i.sh"
# ============================================================================
set -u
set -o pipefail
PROJECT="/data/fxy/research/Tennis_robot_mujoco_sim"
PY="/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python"
export OMP_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$PROJECT"

LOG_DIR="$PROJECT/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%m%d-%H%M)"
LOG="$LOG_DIR/exp17i-$TS.log"

EXP="exp17i_limits_ablation"
DATA_DIR="experiment_data/$EXP"

if [ -f "$DATA_DIR/_.COMPLETE" ]; then
  echo "==> $EXP 已完成, 跳过（幂等）" | tee -a "$LOG"
  exit 0
fi

# ----------------------------------------------------------- mujoco 库路径
MUJOCO_LIB="$("$PY" -c 'import site, pathlib; p = pathlib.Path(site.getsitepackages()[0]) / "mujoco"; print(p if any(p.glob("libmujoco.so*")) else "")')"
if [ -n "$MUJOCO_LIB" ]; then
  export LD_LIBRARY_PATH="$MUJOCO_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "==> LD_LIBRARY_PATH 注入: $MUJOCO_LIB" | tee -a "$LOG"
fi

echo "════════ [$(date +%H:%M:%S)] $EXP 开始 (3200 runs) ════════" | tee -a "$LOG"
if "$PY" scripts/exp/batch_runner.py "$EXP" --workers 16 2>&1 | tee -a "$LOG"; then
  touch "$DATA_DIR/_.COMPLETE"
  echo "==> $EXP COMPLETE $(date +%H:%M:%S)" | tee -a "$LOG"
  "$PY" scripts/extract/extract_exp17i_results.py 2>&1 | tee -a "$LOG"
else
  touch "$DATA_DIR/_.FAILED"
  echo "!!! $EXP 失败（--resume 可续跑, 详见 $LOG）" | tee -a "$LOG"
  exit 1
fi
