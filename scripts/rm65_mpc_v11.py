#!/usr/bin/env python3
"""RM-65 V11 — 已重构为 V12 薄壳（EpisodeRunner 管线架构）。

旧 V11 的 2582 行内联 MPC 循环已被 MPCController + EpisodeRunner 架构替代。
行为与 V12 一致，可视化函数已提取到 src/sim/v11_visuals.py。
旧 V11 行为已通过 exp13（403 runs, 42.9% vs 85.7%）捕获并归档。

用法（与旧 V11 CLI 完全一致）:
    python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --viewer
    python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --position-mode
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录和 scripts/ 到 sys.path
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "scripts"))

from rm65_mpc_v12 import main  # noqa: E402


if __name__ == "__main__":
    main()
