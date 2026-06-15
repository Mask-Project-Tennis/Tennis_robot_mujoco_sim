// Step check parameters and constraint checking (C++ port of robot_limits.check_step_feasibility)
// No Chinese comments to avoid MSVC encoding issues

#pragma once

#include <cstring>
#include <cmath>
#include <cstdio>

// Static constraint parameters (constant for entire forward pass)
struct StepCheckParams {
    const double* q_lo;        // (6,) lower q bound, nullptr=skip
    const double* q_hi;        // (6,) upper q bound, nullptr=skip
    const double* qd_max;      // (6,) joint speed limit
    const double* u_lo;        // (6,) torque lower, nullptr=skip
    const double* u_hi;        // (6,) torque upper, nullptr=skip
    const double* qdd_max;     // (6,) joint accel limit
    double margin;             // tolerance factor (forward_pass_margin)
    double fp_q_tol;           // extra q tolerance (rad)
    int actuator_mode;         // 0=torque, 1=position
    int qdd_window;            // sliding window size
    double dt;                 // timestep
    bool qdd_hard_reject;      // hard reject on qddot violation
};

struct StepCheckResult {
    bool feasible;
    char reason[128];
};

// Check single transition (x_prev -> x_next) with control u
// Dynamic state (qdot history) passed explicitly by forward pass loop
//   qdot_hist: (hist_len, 6) ring buffer, newest at end
//   hist_len: current number of entries
inline StepCheckResult check_step(
    const double* x_prev,      // (12,)
    const double* x_next,      // (12,)
    const double* u_try,       // (6,)
    const double* qdot_hist,   // (hist_len * 6,)
    int hist_len,
    const StepCheckParams& p)
{
    StepCheckResult res;
    res.feasible = true;
    res.reason[0] = '\0';

    const double* q_next = x_next;
    const double* qdot_next = x_next + 6;
    const double* qdot_prev = x_prev + 6;

    // 1. q bounds (with fp_q_tol relaxation)
    if (p.q_lo) {
        for (int j = 0; j < 6; ++j) {
            if (q_next[j] < p.q_lo[j] - p.fp_q_tol) {
                res.feasible = false;
                std::snprintf(res.reason, sizeof(res.reason),
                             "q lower bound, j=%d", j);
                return res;
            }
        }
    }
    if (p.q_hi) {
        for (int j = 0; j < 6; ++j) {
            if (q_next[j] > p.q_hi[j] + p.fp_q_tol) {
                res.feasible = false;
                std::snprintf(res.reason, sizeof(res.reason),
                             "q upper bound, j=%d", j);
                return res;
            }
        }
    }

    // 2. qdot braking-aware: reject only if overspeed + accelerating
    for (int j = 0; j < 6; ++j) {
        double abs_qd = std::abs(qdot_next[j]);
        double lim = p.qd_max[j] * p.margin;
        if (abs_qd > lim) {
            double qddot = (qdot_next[j] - qdot_prev[j]) / p.dt;
            double sign = (qdot_next[j] >= 0.0) ? 1.0 : -1.0;
            if (qddot * sign > 0.0) {
                res.feasible = false;
                std::snprintf(res.reason, sizeof(res.reason),
                             "qdot overspeed+accel, j=%d", j);
                return res;
            }
        }
    }

    // 3. u bounds (torque mode only; position mode skips)
    if (p.actuator_mode == 0) {
        if (p.u_lo) {
            for (int j = 0; j < 6; ++j) {
                if (u_try[j] < p.u_lo[j] * p.margin) {
                    res.feasible = false;
                    std::snprintf(res.reason, sizeof(res.reason),
                                 "u lower, j=%d", j);
                    return res;
                }
            }
        }
        if (p.u_hi) {
            for (int j = 0; j < 6; ++j) {
                if (u_try[j] > p.u_hi[j] * p.margin) {
                    res.feasible = false;
                    std::snprintf(res.reason, sizeof(res.reason),
                                 "u upper, j=%d", j);
                    return res;
                }
            }
        }
    }

    // 4. qddot sliding window (compute_qddot_filtered equivalent)
    if (hist_len >= 2) {
        int eff_len = hist_len < (p.qdd_window + 1) ? hist_len : (p.qdd_window + 1);
        const double* newest = qdot_hist + (hist_len - 1) * 6;
        const double* oldest = qdot_hist + (hist_len - eff_len) * 6;
        double eff_dt = (eff_len - 1) * p.dt;
        for (int j = 0; j < 6; ++j) {
            double qddot_j = (newest[j] - oldest[j]) / eff_dt;
            if (std::abs(qddot_j) > p.qdd_max[j] * p.margin) {
                if (p.qdd_hard_reject) {
                    res.feasible = false;
                    std::snprintf(res.reason, sizeof(res.reason),
                                 "qddot limit, j=%d", j);
                    return res;
                }
            }
        }
    }

    return res;
}
