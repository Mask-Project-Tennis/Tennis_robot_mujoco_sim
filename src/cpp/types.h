// Common types and utility functions for C++ accelerated iLQR
// No Chinese comments to avoid MSVC encoding issues

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

namespace py = pybind11;

constexpr int kNQ = 6;
constexpr int kNX = 12;
constexpr int kNU = 6;

// Cast uintptr_t from Python to MuJoCo struct pointers
inline mjModel* to_model(uintptr_t ptr) { return reinterpret_cast<mjModel*>(ptr); }
inline mjData* to_data(uintptr_t ptr) { return reinterpret_cast<mjData*>(ptr); }

// Clamp x to [lo, hi]
inline double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Set arm state (right arm q/qdot, left arm fixed) and run mj_forward
inline void set_arm_forward(mjModel* m, mjData* d,
                             const double* q, const double* qdot,
                             const double* init_q_left) {
    std::memcpy(d->qpos, q, kNQ * sizeof(double));
    std::memcpy(d->qvel, qdot, kNQ * sizeof(double));
    std::memcpy(d->qpos + kNQ, init_q_left, kNQ * sizeof(double));
    std::memset(d->qvel + kNQ, 0, kNQ * sizeof(double));
    mj_forward(m, d);
}

// sim_step moved to mujoco_utils.h (with actuator mode + FF + qfrc support)
#include "mujoco_utils.h"
