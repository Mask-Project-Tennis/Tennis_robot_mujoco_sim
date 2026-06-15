// MuJoCo simulation utilities (fixed sim_step with actuator mode + FF + qfrc)
// No Chinese comments to avoid MSVC encoding issues

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <cstring>
#include <cmath>

namespace py = pybind11;

// Single simulation step with full actuator mode support
// Fixes vs old types.h:sim_step:
//   1. Uses set_arm_forward (includes mj_forward)
//   2. Position mode clipping (|u-q| <= torque_max/Kp)
//   3. Feedforward compensation (mj_rne bias force)
//   4. qfrc_applied management (zero or bias)
//
// Parameters:
//   actuator_mode: 0=torque (u=tau), 1=position (u=q_desired)
//   kp, kd: (6,) gain arrays, nullptr when actuator_mode==0
//   torque_max: (6,) max torque for position error clipping, nullptr when mode==0
//   use_feedforward: only effective when actuator_mode==1
inline void sim_step(mjModel* m, mjData* d,
                     const double* q, const double* qdot, const double* u,
                     const double* init_q_left,
                     const double* ctrl_lo, const double* ctrl_hi,
                     int actuator_mode,
                     const double* kp, const double* kd,
                     bool use_feedforward,
                     const double* torque_max,
                     double* q_out, double* qdot_out) {
    // 1. Set state + mj_forward (fix: old version lacked mj_forward)
    std::memcpy(d->qpos, q, kNQ * sizeof(double));
    std::memcpy(d->qvel, qdot, kNQ * sizeof(double));
    std::memcpy(d->qpos + kNQ, init_q_left, kNQ * sizeof(double));
    std::memset(d->qvel + kNQ, 0, kNQ * sizeof(double));
    mj_forward(m, d);

    // 2. Right arm control clipping
    for (int i = 0; i < kNU; ++i)
        d->ctrl[i] = clip(u[i], ctrl_lo[i], ctrl_hi[i]);

    // 3. Position mode clipping (fix: was completely missing)
    if (actuator_mode == 1 && kp != nullptr && torque_max != nullptr) {
        for (int i = 0; i < kNU; ++i) {
            double max_err = std::abs(torque_max[i]) / (kp[i] + 1e-12);
            double err = d->ctrl[i] - q[i];
            d->ctrl[i] = q[i] + clip(err, -max_err, max_err);
        }
    }

    // 4. Left arm PD hold (unchanged from old version)
    for (int i = 0; i < kNU; ++i) {
        double err_q = init_q_left[i] - d->qpos[kNQ + i];
        double err_qd = -d->qvel[kNQ + i];
        d->ctrl[kNQ + i] = clip(200.0 * err_q - 20.0 * err_qd, ctrl_lo[i], ctrl_hi[i]);
    }

    // 5. qfrc_applied management (fix: was completely missing)
    if (actuator_mode == 1 && use_feedforward) {
        // qacc=0 ensures mj_rne returns only bias force (C*qdot + g)
        mju_zero(d->qacc, m->nv);
        mjtNum tau_bias[36];
        mj_rne(m, d, 0, tau_bias);
        mju_copy(d->qfrc_applied, tau_bias, kNQ);
    } else {
        mju_zero(d->qfrc_applied, kNQ);
    }

    // 6. Integrate
    mj_step(m, d);

    // 7. Output
    mju_copy(q_out, d->qpos, kNQ);
    mju_copy(qdot_out, d->qvel, kNQ);
}
