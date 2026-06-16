# C++ iLQR 热路径加速模块 — 构建与使用指南

## 一、文件结构

```
src/cpp/
├── __init__.py          # Python 包入口
├── solver_cpp.py        # C++ 加速版 iLQR 求解器（自动回退 + _backward_pass_numpy 参考实现）
├── core_ext.cpp         # pybind11 绑定 + Unity Build（linearize + forward_pass + backward_pass）
├── types.h              # 共享类型定义、工具函数（to_model/to_data/set_arm_forward）
├── mujoco_utils.h       # sim_step（含位置模式裁剪+前馈补偿+碰撞禁用）
├── cost_params.h        # StepCheckParams + check_step 约束检查
├── linearize.cpp        # 解析动力学线性化（批量 + invert_6x6）
├── forward_pass.cpp     # 前向传递（单步 + 线搜索，含碰撞禁用+limits+check_step）
└── backward_pass.cpp    # 后向传递（纯代数 Riccati，栈上小矩阵高斯消元）
setup.py                 # 构建脚本
```

## 二、依赖安装

### 2.1 Python 包

```bash
pip install pybind11 numpy mujoco
```

### 2.2 C++ 编译器

| 平台 | 编译器 | 安装方式 |
|------|--------|---------|
| Windows | MSVC 2019+ | 安装 Visual Studio 2019/2022，勾选 "Desktop development with C++" |
| Linux | GCC 9+ 或 Clang 10+ | `sudo apt install build-essential` |
| macOS | Clang (Xcode) | `xcode-select --install` |

验证编译器可用：

```bash
# Windows (Developer PowerShell)
cl /?

# Linux / macOS
g++ --version
```

## 三、编译

```bash
# 在项目根目录下执行：

# 方式 1：开发模式安装（推荐）
pip install -e .

# 方式 2：仅编译（不安装）
python setup.py build_ext --inplace
```

编译成功后将生成 `src/cpp/iLQR_Core.cp310-win_amd64.pyd`（或对应平台的 `.so`）。

## 四、使用方式

### 4.1 自动集成（无需改代码）

`rm65_mpc_v11.py` 已自动支持 C++ 加速：

```bash
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7
```

脚本启动时会打印：
- `iLQR C++ 加速模块已加载` — C++ 加速生效
- `C++ 加速模块未找到，使用纯 Python iLQR` — 自动回退

### 4.2 手动使用

```python
from src.cpp.solver_cpp import ILQTSolver

solver = ILQTSolver(ilqt_config, use_analytical=True)
X, U, costs = solver.solve_few_iters(
    env, cost_fn, x0, U_warm,
    max_iter=8,
    skip_linesearch=True,
)
```

### 4.3 检查 C++ 模块是否可用

```python
from src.cpp import is_available
if is_available():
    print("C++ 加速已启用")
```

## 五、性能对比

> 基准: N=60 horizon, max_iter=20, fast_lin=True（V11 生产路径），RM-65B 单臂

| 操作 | Python | C++ 加速后 | 说明 |
|------|--------|-----------|------|
| 线性化 (linearize_fast_trajectory) | — | ~18ms/步 | mj_forward + mj_step 物理地板 |
| 前向传递 (forward_pass_single) | — | ~0.75ms/步 | 含碰撞禁用+limits+check_step |
| 后向传递 (backward_pass) | ~18ms | ~1.2ms | 纯代数 Riccati（栈上高斯消元） |
| **单次 solve_few_iters** | **185.5ms** | **123.3ms** | **累计 1.50× 加速** |

优化演进路线（2026-06-16）:

| commit | 优化 | solve_few_iters |
|--------|------|-----------------|
| 原始 | — | 185.5ms |
| `e12f4ad` | np.eye 预分配 | 178.3ms |
| `f88760c` | linearize 去冗余 mj_forward | 148.0ms |
| `f74c953` | C++ backward_pass | **123.3ms** |

当前热点分布（123.3ms 总计）:
- mj_forward 16.5% + mj_step 16.4% = 33%（MuJoCo C 物理地板，不可压缩）
- forward_pass(C++) 12.2%
- smoothness 导数 7.7%
- 其余分散在 running_derivatives / linearize Python 封装 / np.stack 等

## 六、注意事项

1. **仅加速解析线性化**：有限差分模式（`--fd`）不走 C++，保持原速
2. **仅加速右臂 6-DOF**：关节数固定为 6
3. **MuJoCo 版本**：需 `mujoco>=3.0.0`，推荐 `>=3.2.0`
4. **回退安全**：C++ 编译失败只影响性能，不影响功能
5. **后向传递已 C++ 化**：纯代数 Riccati（`backward_pass.cpp`），栈上 6×6 高斯消元，无 MuJoCo 依赖
6. **代价导数仍在 Python**：V11 默认代价配置无 FK 项（`Q_p_running=0`），导数轻量，C++ 化 ROI 极低（profiling 仅 ~5%）
