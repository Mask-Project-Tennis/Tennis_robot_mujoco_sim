#!/usr/bin/env bash
# 一键安装脚本 / One-command setup：
#   conda 环境 → Python 依赖 → 可选 C++ 扩展 → 单集冒烟测试。
#
# 用法 / Usage:   bash install.sh
# 可覆盖项 / Overrides:  ENV_NAME=<name> PY_VER=<3.11> bash install.sh
#
# 幂等：已存在的环境直接复用，依赖已装则秒过；可重复运行。
# 幽默提示没有：本脚本只为复现论文结果服务，装完即用。

set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-mujoco_tennis}"
PY_VER="${PY_VER:-3.11}"

log() { printf '[install] %s\n' "$*"; }
die() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

# ── 0) 前置检查：conda 必须可用 ─────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || \
  die "conda not found — install Miniconda/Miniforge first (https://docs.conda.io)"

# ── 1) conda 环境（已存在则复用）────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  log "conda env '${ENV_NAME}' already exists — reusing."
else
  log "creating conda env '${ENV_NAME}' (python ${PY_VER}) ..."
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
PYBIN="$(conda info --base)/envs/${ENV_NAME}/bin/python"
[ -x "${PYBIN}" ] || die "python not found in env '${ENV_NAME}'"

# ── 2) Python 依赖 ──────────────────────────────────────────────────────────
log "installing python dependencies (requirements.txt) ..."
"${PYBIN}" -m pip install -r requirements.txt

# ── 3) 可选 C++ 扩展（失败不阻塞：求解器有 NumPy 回退，只是慢 ~1.5×）────────
log "building optional C++ extension (NumPy fallback if this fails) ..."
if ! "${PYBIN}" setup.py build_ext --inplace >/dev/null 2>&1; then
  log "WARNING: C++ build failed — continuing with the NumPy fallback (slower iLQR)."
fi

# ── 4) 冒烟测试：单集仿真，期望 __RESULT__ 行 hit_type=active（README 同款命令）──
log "smoke test: one simulated episode (~1.4 s on a workstation) ..."
if OUT="$(MUJOCO_GL=egl "${PYBIN}" scripts/rm65_mpc_v12.py \
          --serve-box --ball-speed 7 --seed 1 --no-plot 2>&1)"; then
  RESULT="$(printf '%s\n' "${OUT}" | grep "__RESULT__" || true)"
  if [ -n "${RESULT}" ]; then
    printf '%s\n' "${RESULT}" | sed 's/^/[smoke] /'
  else
    log "WARNING: no __RESULT__ line printed — tail of the run output:"
    printf '%s\n' "${OUT}" | tail -10
  fi
else
  printf '%s\n' "${OUT}" | tail -20 >&2
  die "smoke test failed — see output above"
fi

# ── 完成：打印复现论文结果的下一步 ───────────────────────────────────────────
log "DONE — environment '${ENV_NAME}' is ready (${PYBIN})."
log "next steps to reproduce the paper results (no simulation re-runs needed):"
log "  ${PYBIN} scripts/extract/paired_stats.py   # per-episode paired statistics"
log "  ${PYBIN} scripts/extract/aux_stats.py      # sensitivity sweep + TCP diagnostics"
log "  ${PYBIN} scripts/plot/paper_figs.py --fig 1hero 3alt 4 5 6 7 8 table --out repro_outputs"
log "tip (viewer only): export LD_LIBRARY_PATH=\"\$(${PYBIN} -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))'):\$LD_LIBRARY_PATH\""
