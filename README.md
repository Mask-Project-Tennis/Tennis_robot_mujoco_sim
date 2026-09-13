# RM-65 Tennis Robot — MPC + iLQR + Tube Hitting Simulation and Real-Robot Deployment

**English** | [简体中文](README.zh-CN.md)

A tennis-hitting robot project on the dual-arm RM-65B manipulator. **MPC (model
predictive control)** forms the outer closed loop, **iLQR (iterative linear
quadratic regulator)** is the inner trajectory optimizer, a **spatial-corridor
("Tube") terminal cost** provides spatio-temporal robustness under prediction
error, and a **multi-layer safety filter** enforces the joint/TCP constraints.

Two control modes are supported:
- **Torque mode (default)**: MPC outputs joint torques, applied directly in MuJoCo
- **Position mode (`--position-mode`)**: MPC outputs joint angles; MuJoCo PD
  actuators emulate the real controller and iLQR plans the q_desired trajectory

Real-robot deployment uses angle control through the Realman SDK (IP
communication, `rm_movej_follow`) with a three-layer safety architecture
(controller firmware + software monitor + emergency stop).

---

## Environment Setup

```bash
# Create the conda environment
conda create -n mujoco_tennis python=3.11
conda activate mujoco_tennis
pip install -r requirements.txt

# Build the C++ acceleration module (iLQR linearization + forward/backward passes, 1.50x speedup)
python setup.py build_ext --inplace

# Linux: point MuJoCo at its shared libraries
export LD_LIBRARY_PATH="$(python -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))'):$LD_LIBRARY_PATH"
```

Dependencies: `mujoco>=3.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `pyyaml>=6.0`

---

## Quick Start

### Simulation

```bash
# V12 — current main script (EpisodeRunner pipeline, torque mode by default)
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --viewer

# Position mode (emulates real-robot angle control)
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --position-mode --viewer

# Fixed random seed
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --viewer

# Offline run (no rendering, faster)
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 9 --no-plot

# Joint viewer (slider control of every joint)
python scripts/tools/rm65_joint_viewer.py
```

### Real-Robot Deployment

```bash
# 1. Edit the real-robot configuration (IP address, safety limits, PD gains)
vim configs/real_robot.yaml

# 2. Verify the SDK interfaces one by one (read-only first, motion later)
python scripts/tools/test_real_robot/01_connect_disconnect.py    # connect / disconnect
python scripts/tools/test_real_robot/02_read_joints.py           # read joint angles
python scripts/tools/test_real_robot/04_send_zero_pose.py        # move to zero pose
# ...see scripts/tools/test_real_robot/README.md for the full list
```

### Tests

```bash
# Run the full test suite (662 tests)
pytest tests/

# Linting
ruff check src/ tests/ scripts/
```

---

## Reproducing the Paper Results

The statistics and figures of the accompanying manuscript are regenerated from
the tracked per-episode results by three scripts; no simulation re-runs are
required (the single-episode run below is an optional sanity check).

### 1. Setup

Follow [Environment Setup](#environment-setup). The C++ module is optional (a
NumPy fallback exists), but it is used for the numbers reported here.

### 2. Sanity check: one simulated episode

```bash
MUJOCO_GL=egl python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 1 --no-plot
```

Expected: a final `__RESULT__` line with `hit_type=active` (about 1.4 s on a
single workstation; exact error values may vary slightly across platforms).

### 3. Reproduce all statistics

```bash
python scripts/extract/paired_stats.py   # per-episode paired analysis -> statistics JSON
python scripts/extract/aux_stats.py      # sensitivity sweep + TCP-cap diagnostics
```

`paired_stats.py` writes `experiment_data/paper_stats_active.json` — the exact
statistics JSON used by the manuscript (tracked reference copy, sha256
`4cddd8aa87cab34b724b273ff26a9960d59af757394869faedcb1b80b8f0cd4c`).
`aux_stats.py` writes `experiment_data/aux_stats.json` and reproduces the
sensitivity sweep (93.9–94.9 %; corridor-only 84.8 %) and the TCP-cap
diagnostics (19.2 % of episodes exceed 1.0 m/s, mean 1.09, p90 1.01).

### 4. Reproduce figures and tables

```bash
python scripts/plot/paper_figs.py --fig 1hero 3alt 4 5 6 7 8 table --out repro_outputs
```

Writes `repro_outputs/*.pdf` (Fig. 1 and Fig. 3–8) and
`repro_outputs/table_data/table1_comparison.tex` + `table2_ablation.tex`
(Tables I–II), identical to the manuscript versions.

### 5. Expected outputs

| Quantity | Expected |
|---|---|
| `paper_stats_active.json` sha256 | `4cddd8aa87cab34b724b273ff26a9960d59af757394869faedcb1b80b8f0cd4c` |
| Table I, nominal 7 m/s, full | 83.8 % |
| Table I, nominal 9 m/s, full | 95.9 % |
| E1 speed sweep, 7 m/s | 84.3 % |
| E2 TCP 1.0 m/s cap, 7 m/s | 35.9 % (vs 84.4 % uncapped) |
| Sensitivity sweep (9 m/s) | 93.9–94.9 % (corridor-only 84.8 %) |
| TCP-cap exceedance | 19.2 % of episodes (mean 1.09, p90 1.01 m/s) |

### 6. Data map

| Experiment | Directory | Feeds |
|---|---|---|
| E1 capability sweep | `experiment_data/exp15_speed_v2` | Fig. 5, text |
| E2 limit cost | `experiment_data/exp16_limits_v2` | Fig. 5, TCP diagnostics |
| E3 nominal four-tier | `experiment_data/exp17d_mechanism` | Fig. 6(a), Table I |
| E4 perturbation grid | `experiment_data/exp17b_perturb`, `exp17f_mechanism_perturb` | Fig. 6(b,c), Table I |
| E5 perception sweep | `experiment_data/exp17a_noise`, `exp17c_obsfreq` | artifact only |
| E7 high-power corners | `experiment_data/exp17g_spatial_power`, `exp17h_extreme` | Fig. 6(d), Table I |
| E8 limit x mechanism | `experiment_data/exp17i_limits_ablation` | Table II, text |
| E6 real-time budget | `experiment_data/exp18_fig_assets/timing.json` | Fig. 8 |
| Trajectory assets | `experiment_data/exp18_fig_assets/raw/*.npz` | Fig. 1, 3, 4, 7 |
| Sensitivity sweep | `experiment_data/exp18_sensitivity` | text (Sec. VI) |
| TCP exemption | `experiment_data/exp18_tcp_exempt` | text (Sec. VI) |

---

## Dual-Mode Actuator

Both modes share the same MPC + iLQR framework; the difference is what the
controller outputs and how the linearization treats the actuator.

### Torque Mode (default)

MPC outputs joint torques `u = tau(6)`; MuJoCo `motor` actuators apply them
directly to the joints.

```bash
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7
```

### Position Mode (`--position-mode`)

MPC outputs desired joint angles `u = q_desired(6)`; MuJoCo `general` actuators
close the PD loop internally:

```
tau = Kp * (q_desired - q) - Kd * qdot
```

iLQR plans the optimal q_desired trajectory under the PD-actuator dynamics,
with feedforward compensation of gravity and Coriolis terms.

```bash
python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --position-mode
```

**The real robot uses position mode** — the Realman SDK `rm_movej_follow` accepts
angle commands, and `PlanningEnv` models the real controller with MuJoCo PD
actuators.

### Key Differences

| Property | Torque mode | Position mode |
|----------|-------------|---------------|
| Control input u | tau(6) torques | q_desired(6) angles |
| B matrix | `dt * M^-1` | `dt * M^-1 * diag(Kp)` |
| Safety-filter beta scaling | `beta * u` (torque) | `q + beta*(u - q)` (position) |
| Emergency braking | `u = -20*qdot` damping torque | hold current angles |
| Follow-through | `J^T * F` PD controller | `solve_ik()` inverse kinematics |
| Real-robot deployment | no (SDK has no torque API) | yes (`rm_movej_follow`) |

---

## Real-Robot Deployment Architecture

```
                    Real-robot control pipeline
+---------------------------------------------------+
|  Every control tick:                              |
|                                                   |
|  RobotInterface.get_arm_state()   <- joint angles |
|  BallPerceiver.get_latest_filtered() <- mocap+KF  |
|           |                                       |
|  PlanningEnv (pure MuJoCo computation)            |
|    * set_arm_state(x_real)  <- inject real state  |
|    * iLQR plans the optimal trajectory            |
|    * step_from_state(x, u)  <- try control inputs |
|           |                                       |
|  RobotInterface.send_joint_command(q_desired)     |
|    -> rm_movej_follow -> real robot               |
+---------------------------------------------------+
```

| Module | File | Responsibility |
|--------|------|----------------|
| **PlanningEnv** | `src/ilqt/planning_env.py` | Pure MuJoCo computation (FK/Jacobian/forward simulation), no hardware |
| **RobotInterface** | `src/real/robot_interface.py` | Realman SDK wrapper (angle control + controller safety at connect) |
| **BallPerceiver** | `src/real/ball_perceiver.py` | Ball perception (sensor -> KF filtering -> pos/vel) |
| **SafetyMonitor** | `src/real/safety_monitor.py` | Software safety checks (joint/TCP limits -> stop) |
| **Configuration** | `configs/real_robot.yaml` | 7 sections (connection/control/safety/PD/perception) with comments |

### Three-Layer Safety Architecture

```
Layer 1: Controller firmware (configured at connect time)
  rm_set_collision_state / rm_set_self_collision_enable / rm_set_controller_torque_limit

Layer 2: SafetyMonitor (software check every tick)
  joint position / velocity / TCP speed violation -> slow_stop()

Layer 3: Emergency stop (last resort)
  rm_set_arm_stop() (not recoverable) / hardware e-stop button
```

### Interface Test Tools

SDK APIs are verified one at a time before being integrated. Located in
`scripts/tools/test_real_robot/`:

| Script | Risk | Purpose |
|--------|------|---------|
| `01_connect_disconnect.py` | none | connect -> configure safety -> read angles -> disconnect |
| `02_read_joints.py` | none | continuous table of angles/velocities |
| `03_read_temperature.py` | none | continuous temperature/voltage/current readout |
| `04_send_zero_pose.py` | low | streamed interpolation back to zero pose |
| `05_send_joint_command.py` | medium | send arbitrary angles (`--deg`/`--rad`/interactive) |
| `06_safety_config_verify.py` | none | read back and verify safety parameters |
| `07_emergency_stop.py` | medium | slow-stop and emergency-stop tests |
| `08_full_motion_test.py` | medium | sinusoidal motion test |

See [`scripts/tools/test_real_robot/README.md`](scripts/tools/test_real_robot/README.md) (Chinese).

---

## Core Algorithms

### MPC + iLQR Closed Loop

```
Replan every N steps (controlled by replan-interval):
  1. Observe the current ball position and velocity
  2. find_hitting_point_physics -> physics-based hit-point prediction
  3. Softmin multi-terminal cost -> allow hitting anywhere in the candidate time window
  4. Generate a warm-start control sequence (backswing trajectory PD)
  5. solve_few_iters -> iLQR trajectory optimization
  6. Safety filtering (joint / TCP / half-space constraints)
  7. Execute the first control command (torque or angle)
  8. Repeat at the next time step
```

### Spatial-Corridor Robust Hitting ("Tube")

The robustness layer does not model a time-varying uncertainty tube; it shapes
the terminal cost so that a band of feasible contact states is accepted:

```
Candidate window: centered on the predicted best hit step best_k,
                  half-width window_half_ms (default 50 ms)

Spatial corridor (perpendicular hinge loss, no time-space correspondence):
  cost = 0.5 * s_k * Q_p_tube * max(0, ||P_perp (p_ee - p_ball,k)|| - r_racket)^2
  half-width = racket radius (0.12 m), zero cost inside the corridor

Softmin terminal aggregation over candidate ball states (sharpness beta):
  l_N^sm = softmin_i ( c_i ),  all candidates evaluated at the shared terminal state
```

Velocity-direction and racket-normal terms are carried by the terminal cost, not
by the corridor, which keeps the corridor a pure spatial relaxation. The
`Tube*` module names are historical; in the manuscript this relaxation is called
the spatial corridor.

### Multi-Layer Safety Filter

```
1. Joint constraints: position / velocity / acceleration / torque
2. TCP speed hard limit (default 1.8 m/s in simulation)
3. X-plane wall: the arm must stay at X >= -0.1 (no crossing the body midline)
4. Stepwise safety filter: beta descent [0.8, 0.6, 0.4, 0.2, 0.0];
   if every beta fails -> emergency braking
```

### Perception: BallEstimator 6D Kalman Filter

Estimates the ball's true position and velocity under noisy observations:

```
State:        x = [px, py, pz, vx, vy, vz]  (6D position + velocity)
Process:      constant velocity + gravity (F accounts for g=9.81 on Vz)
Observation:  H = I6 (full-state direct observation)
Bounce guard: for Z < 0.01 m the position is slammed and the velocity is
              kept on bounce / zeroed on landing
```

Three-layer perception architecture: simulation ground truth -> noise injection
(`add_observation_noise`) -> Kalman filtering (`BallEstimator`) -> planner
consumption.

---

## V12 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--serve-box` | flag | — | box-shaped serve region |
| `--ball-speed` | float | None | horizontal ball speed at the hit point (m/s) |
| `--position-mode` | flag | — | position mode (default: torque) |
| `--seed` | int | None | random seed |
| `--viewer` | flag | — | MuJoCo viewer replay |
| `--no-plot` | flag | — | disable matplotlib visualization |
| `--horizon` | int | None | short horizon length (steps) |
| `--iter` | int | None | iLQR iterations per replan |
| `--replan-interval` | int | None | replanning interval (steps) |
| `--window-ms` | float | 50.0 | candidate time-window half-width (ms) |
| `--softmin-beta` | float | 5.0 | softmin sharpness |
| `--corridor-radius` | float | 0.12 | corridor half-width (m) |
| `--max-tcp` | float | None | TCP speed hard limit (m/s) |
| `--terminal-exempt-steps` | int | None | final-segment qdot/TCP exemption steps |

> V11 is retired; use V12. Full argument list: `python scripts/rm65_mpc_v12.py --help`

---

## Repository Layout

```
.
├── README.md                             # this file (English)
├── README.zh-CN.md                       # Chinese version
├── AGENTS.md                             # detailed project conventions (Chinese)
├── requirements.txt
├── setup.py                              # C++ extension build (pybind11)
│
├── configs/
│   ├── default.yaml                      # base simulation parameters
│   ├── mpc.yaml                          # MPC-specific parameters
│   ├── v5_active_hit.yaml                # V5 active-hit configuration
│   └── real_robot.yaml                   # real-robot configuration (7 sections)
│
├── scripts/
│   ├── rm65_mpc_v12.py                   # current main simulation script (EpisodeRunner pipeline)
│   ├── rm65_mpc_v11.py                   # V11 shim (delegates to V12)
│   ├── archive/                          # archived scripts (V6-V10 + tube + old experiments)
│   ├── tools/
│   │   ├── rm65_joint_viewer.py          # joint-slider viewer
│   │   ├── test_real_robot/              # interface test tools (01-08)
│   │   └── ...
│   ├── exp/                              # batch-experiment infrastructure
│   ├── extract/                          # result extraction (paired_stats, aux_stats, ...)
│   └── plot/                             # paper figures and tables (paper_figs.py)
│
├── src/
│   ├── robot/
│   │   └── rm65_model.xml                # MuJoCo model (dual arm 12 DOF + racket + ball)
│   ├── sim/                              # MuJoCo simulation + replay + hit detection
│   ├── ilqt/                             # iLQR + MPC + pipeline + strategies + components
│   │   ├── solver.py                     # iLQR backward-forward iteration
│   │   ├── cost.py                       # cost terms (corridor + softmin)
│   │   ├── planning_env.py               # PlanningEnv planning environment
│   │   ├── mpc_controller.py             # MPCController (strategy injection + MPCConfig)
│   │   ├── episode_runner.py             # EpisodeRunner (4 components + 5 hooks)
│   │   ├── strategies/                   # pluggable strategies
│   │   └── components/                   # composable pipeline components
│   ├── real/                             # real-robot deployment modules
│   ├── dynamics/                         # dynamics linearization
│   ├── perception/                       # Kalman filter
│   ├── tennis/                           # ball trajectory prediction + hit-point computation
│   ├── cpp/                              # C++ acceleration (pybind11)
│   └── utils/                            # shared utilities (model loading / noise / math)
│
├── tests/                                # unit tests (662)
├── experiment_data/                      # per-episode results, logs, figure assets (tracked subset)
├── docs/                                 # technical documentation (Chinese)
└── paper/                                # manuscript LaTeX project (local only, not tracked)
```

> Module details: [`src/README.md`](src/README.md); script index: [`scripts/README.md`](scripts/README.md) (both Chinese).

---

## Configuration

### `configs/default.yaml` — simulation parameters

```yaml
sim:
  dt: 0.005              # simulation step (s)

cost:
  Q_p: [50000, 50000, 50000]   # terminal position cost weights
  Q_v: [200, 200, 200]         # terminal velocity cost weights
  R: 0.0001                    # control cost weight

hitting:
  racket_speed: 5.0            # nominal racket speed (m/s)
  workspace_radius: 0.85       # reachable workspace radius (m)
```

### `configs/real_robot.yaml` — real-robot configuration

```yaml
robot:
  ip: "192.168.1.18"             # arm IP address
  port: 8080

control:
  control_mode: "ip"             # "ip" (rm_movej_follow) | "canfd" (rm_movej_canfd)
  dt: 0.005                      # MPC planning step

safety:
  collision_stage: 5             # collision sensitivity 0-8 (start at 5; lower later)
  torque_limit: [50, 50, 50, 30, 30, 30]  # N*m
  max_tcp_speed: 1.0             # TCP speed limit (m/s)

position_mode:
  kp: [200, 200, 100, 50, 50, 20]   # PD position gains
  kd: [20, 20, 10, 5, 5, 2]         # PD velocity gains
  enable_feedforward: true           # gravity + Coriolis feedforward

perception:
  sensor_type: "simulated"       # "simulated" / "optitrack" / "realsense"
  pos_noise_std: 0.005           # position noise (m)
```

> Full parameter documentation: comments inside `configs/real_robot.yaml` and
> [`AGENTS.md`](AGENTS.md) (Chinese).
