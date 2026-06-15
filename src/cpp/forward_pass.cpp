// Forward pass: single step and linesearch variants

#include "types.h"

namespace fwd {

/// Single forward pass (MPC mode, fixed alpha=0.5)
/// Returns true if trajectory is valid (all finite)
/// NOTE: This is a minimal update for A1. Full rewrite (with collision disable,
/// actuator mode, limits) comes in B1.
bool single(
    py::array_t<double> X_new_a, py::array_t<double> U_new_a,
    py::array_t<double> X_nom_a, py::array_t<double> U_nom_a,
    py::array_t<double> Ks_a,   py::array_t<double> ks_a,
    uintptr_t model_ptr, uintptr_t data_ptr,
    const double* init_q_left,
    const double* ctrl_lo, const double* ctrl_hi,
    double alpha,
    int actuator_mode,
    const double* kp, const double* kd,
    bool use_feedforward,
    const double* torque_max,
    int ball_geom_start,
    bool disable_collision,
    const StepCheckParams* check_params,
    char* reason_out)
{
    mjModel* m = to_model(model_ptr);
    mjData* d = to_data(data_ptr);

    // Collision save + disable (detail 2)
    std::vector<int> contype_save;
    std::vector<int> conaffinity_save;
    if (disable_collision && ball_geom_start > 0) {
        contype_save.assign(m->geom_contype, m->geom_contype + ball_geom_start);
        conaffinity_save.assign(m->geom_conaffinity, m->geom_conaffinity + ball_geom_start);
        for (int i = 0; i < ball_geom_start; ++i) {
            m->geom_contype[i] = 0;
            m->geom_conaffinity[i] = 0;
        }
    }

    int N = static_cast<int>(U_nom_a.shape(0));

    double* X_new = X_new_a.mutable_data();
    double* U_new = U_new_a.mutable_data();
    const double* X_nom = X_nom_a.data();
    const double* U_nom = U_nom_a.data();
    const double* Ks   = Ks_a.data();
    const double* ks   = ks_a.data();

    // qdot history ring buffer for check_step (detail 3)
    // Buffer capacity must be >= qdd_window + 1 (pre-seed 1 + sliding window)
    constexpr int kMaxQddBuffer = 32;
    double qdot_hist[kMaxQddBuffer][6];
    int hist_count = 0;

    std::memcpy(X_new, X_nom, kNX * sizeof(double));

    // Pre-seed qdot history with initial state
    if (check_params) {
        assert(check_params->qdd_window + 1 <= kMaxQddBuffer
               && "qdd_window exceeds qdot_hist buffer capacity");
        std::memcpy(qdot_hist[0], X_new + kNQ, 6 * sizeof(double));
        hist_count = 1;
    }

    for (int k = 0; k < N; ++k) {
        double dx[12];
        for (int i = 0; i < kNX; ++i)
            dx[i] = X_new[k * kNX + i] - X_nom[k * kNX + i];

        const double* K_k = Ks + k * (kNX * kNU);
        const double* k_k = ks + k * kNU;
        for (int i = 0; i < kNU; ++i) {
            double sum = 0.0;
            for (int j = 0; j < kNX; ++j)
                sum += K_k[i * kNX + j] * dx[j];
            U_new[k * kNU + i] = U_nom[k * kNU + i] + alpha * k_k[i] + sum;
            U_new[k * kNU + i] = clip(U_new[k * kNU + i], ctrl_lo[i], ctrl_hi[i]);
        }

        sim_step(m, d,
                 X_new + k * kNX, X_new + k * kNX + kNQ,
                 U_new + k * kNU,
                 init_q_left, ctrl_lo, ctrl_hi,
                 actuator_mode, kp, kd, use_feedforward, torque_max,
                 X_new + (k + 1) * kNX, X_new + (k + 1) * kNX + kNQ);

        for (int i = 0; i < kNX; ++i) {
            if (!std::isfinite(X_new[(k + 1) * kNX + i])) {
                if (reason_out)
                    std::snprintf(reason_out, 128, "NaN in state at k=%d", k);
                if (disable_collision && ball_geom_start > 0) {
                    std::memcpy(m->geom_contype, contype_save.data(),
                                ball_geom_start * sizeof(int));
                    std::memcpy(m->geom_conaffinity, conaffinity_save.data(),
                                ball_geom_start * sizeof(int));
                }
                return false;
            }
        }

        // check_step (if limits enabled)
        if (check_params) {
            // Push new qdot to ring buffer
            if (hist_count >= check_params->qdd_window + 1) {
                for (int i = 0; i < hist_count - 1; ++i)
                    std::memcpy(qdot_hist[i], qdot_hist[i+1], 6*sizeof(double));
                hist_count--;
            }
            std::memcpy(qdot_hist[hist_count],
                       X_new + (k+1)*kNX + kNQ, 6*sizeof(double));
            hist_count++;

            StepCheckResult res = check_step(
                X_new + k * kNX, X_new + (k+1) * kNX,
                U_new + k * kNU,
                &qdot_hist[0][0], hist_count,
                *check_params);
            if (!res.feasible) {
                if (reason_out)
                    std::snprintf(reason_out, 128, "%s", res.reason);
                if (disable_collision && ball_geom_start > 0) {
                    std::memcpy(m->geom_contype, contype_save.data(),
                                ball_geom_start * sizeof(int));
                    std::memcpy(m->geom_conaffinity, conaffinity_save.data(),
                                ball_geom_start * sizeof(int));
                }
                return false;
            }
        }
    }

    // Collision restore
    if (disable_collision && ball_geom_start > 0) {
        std::memcpy(m->geom_contype, contype_save.data(),
                    ball_geom_start * sizeof(int));
        std::memcpy(m->geom_conaffinity, conaffinity_save.data(),
                    ball_geom_start * sizeof(int));
    }
    return true;
}

/// Returns: py::tuple of (accepted: bool, X_out, U_out, cost_out: float)
py::tuple linesearch(
    py::array_t<double> X_nom_a, py::array_t<double> U_nom_a,
    py::array_t<double> Ks_a,   py::array_t<double> ks_a,
    py::array_t<double> alpha_list_a,
    double cost_old,
    uintptr_t model_ptr, uintptr_t data_ptr,
    const double* init_q_left,
    const double* ctrl_lo, const double* ctrl_hi,
    py::object cost_fn,
    int actuator_mode,
    const double* kp, const double* kd,
    bool use_feedforward,
    const double* torque_max)
{
    int N = static_cast<int>(U_nom_a.shape(0));
    int n_alpha = static_cast<int>(alpha_list_a.size());

    // Pre-allocate temp buffers
    std::vector<py::ssize_t> shape_X = {static_cast<py::ssize_t>(N + 1), kNX};
    std::vector<py::ssize_t> shape_U = {static_cast<py::ssize_t>(N), kNU};
    py::array_t<double> X_tmp(shape_X);
    py::array_t<double> U_tmp(shape_U);

    const double* alpha_list = alpha_list_a.data();

    py::array_t<double> X_best = X_nom_a;
    py::array_t<double> U_best = U_nom_a;
    double cost_best = cost_old;

    for (int ia = 0; ia < n_alpha; ++ia) {
        double alpha = alpha_list[ia];
        bool ok = single(X_tmp, U_tmp, X_nom_a, U_nom_a,
                         Ks_a, ks_a,
                         model_ptr, data_ptr,
                         init_q_left, ctrl_lo, ctrl_hi, alpha,
                         actuator_mode, kp, kd, use_feedforward, torque_max,
                         0, false, nullptr, nullptr);  // no collision, no limits, no reason in linesearch
        if (!ok) continue;

        double cost_new;
        try {
            cost_new = cost_fn(X_tmp, U_tmp).cast<double>();
        } catch (py::error_already_set&) {
            PyErr_Clear();
            continue;
        }

        if (cost_new < cost_best) {
            X_best = X_tmp;
            U_best = U_tmp;
            cost_best = cost_new;
            return py::make_tuple(true, X_best, U_best, cost_best);
        }
    }
    return py::make_tuple(false, X_best, U_best, cost_best);
}

}  // namespace fwd
