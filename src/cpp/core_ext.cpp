// pybind11 bindings for C++ accelerated iLQR hot-path

#include "types.h"
#include "cost_params.h"
#include "linearize.cpp"
#include "forward_pass.cpp"

// Batch analytical linearization (public API)
void linearize_analytical_batch(
    py::array_t<double> A_all_a, py::array_t<double> B_all_a,
    py::array_t<double> x_next_all_a,
    py::array_t<double> X_a, py::array_t<double> U_a,
    uintptr_t model_ptr, uintptr_t data_ptr,
    const double* init_q_left,
    double eps, double dt,
    int actuator_mode,
    const double* kp, const double* kd,
    bool use_feedforward)
{
    mjModel* m = to_model(model_ptr);
    mjData* d = to_data(data_ptr);

    int N = static_cast<int>(U_a.shape(0));
    const double* X = X_a.data();
    const double* U = U_a.data();
    double* A_all = A_all_a.mutable_data();
    double* B_all = B_all_a.mutable_data();
    double* x_next_all = x_next_all_a.mutable_data();

    for (int k = 0; k < N; ++k) {
        linearize_analytical_single(
            m, d,
            X + k * kNX, U + k * kNU,
            init_q_left,
            eps, dt,
            actuator_mode, kp, kd,
            use_feedforward,
            A_all + k * kNX * kNX,
            B_all + k * kNX * kNU,
            x_next_all + k * kNX);
    }
}

// Module definition
PYBIND11_MODULE(iLQR_Core, m) {
    m.doc() = "C++ accelerated iLQR hot-path (analytical linearize + forward pass)";

    m.def("linearize_analytical_batch",
        [](py::array_t<double> A_all, py::array_t<double> B_all,
           py::array_t<double> x_next_all,
           py::array_t<double> X, py::array_t<double> U,
           uintptr_t model_ptr, uintptr_t data_ptr,
           py::array_t<double> init_q_left_a,
           double eps, double dt,
           int actuator_mode,
           py::object kp_obj, py::object kd_obj,
           bool use_feedforward)
        {
            const double* kp_ptr = nullptr;
            const double* kd_ptr = nullptr;
            if (!kp_obj.is_none()) {
                kp_ptr = py::array_t<double>(kp_obj).data();
            }
            if (!kd_obj.is_none()) {
                kd_ptr = py::array_t<double>(kd_obj).data();
            }
            linearize_analytical_batch(
                A_all, B_all, x_next_all, X, U,
                model_ptr, data_ptr,
                init_q_left_a.data(), eps, dt,
                actuator_mode, kp_ptr, kd_ptr,
                use_feedforward);
        },
        py::arg("A_all"), py::arg("B_all"), py::arg("x_next_all"),
        py::arg("X"), py::arg("U"),
        py::arg("model_ptr"), py::arg("data_ptr"),
        py::arg("init_q_left"),
        py::arg("eps") = 1e-5, py::arg("dt") = 0.005,
        py::arg("actuator_mode") = 0,
        py::arg("kp") = py::none(), py::arg("kd") = py::none(),
        py::arg("use_feedforward") = false,
        "Batch analytical linearization along trajectory. "
        "Output: A_all(N,12,12), B_all(N,12,6), x_next_all(N,12)");

    m.def("forward_pass_single",
        [](py::array_t<double> X_new, py::array_t<double> U_new,
           py::array_t<double> X_nom, py::array_t<double> U_nom,
           py::array_t<double> Ks, py::array_t<double> ks,
           uintptr_t model_ptr, uintptr_t data_ptr,
           py::array_t<double> init_q_left_a,
           py::array_t<double> ctrl_lo_a, py::array_t<double> ctrl_hi_a,
           double alpha,
           int actuator_mode,
           py::object kp_obj, py::object kd_obj,
           bool use_feedforward,
           py::object torque_max_obj,
           int ball_geom_start,
           bool disable_collision,
           py::object check_params_obj)
        {
            const double* kp_ptr = nullptr;
            const double* kd_ptr = nullptr;
            const double* torque_max_ptr = nullptr;
            if (!kp_obj.is_none()) kp_ptr = py::array_t<double>(kp_obj).data();
            if (!kd_obj.is_none()) kd_ptr = py::array_t<double>(kd_obj).data();
            if (!torque_max_obj.is_none())
                torque_max_ptr = py::array_t<double>(torque_max_obj).data();

            // Construct StepCheckParams if provided
            StepCheckParams params;
            const StepCheckParams* params_ptr = nullptr;
            // Local scope to keep numpy array refs alive while pointers are used
            py::object q_lo_arr, q_hi_arr, qd_max_arr, u_lo_arr, u_hi_arr, qdd_max_arr;
            if (!check_params_obj.is_none()) {
                py::dict cp = check_params_obj;
                q_lo_arr = cp["q_lo"]; q_hi_arr = cp["q_hi"];
                qd_max_arr = cp["qd_max"]; u_lo_arr = cp["u_lo"];
                u_hi_arr = cp["u_hi"]; qdd_max_arr = cp["qdd_max"];

                params.q_lo = q_lo_arr.is_none() ? nullptr :
                              py::array_t<double>(q_lo_arr).data();
                params.q_hi = q_hi_arr.is_none() ? nullptr :
                              py::array_t<double>(q_hi_arr).data();
                params.qd_max = py::array_t<double>(qd_max_arr).data();
                params.u_lo = u_lo_arr.is_none() ? nullptr :
                              py::array_t<double>(u_lo_arr).data();
                params.u_hi = u_hi_arr.is_none() ? nullptr :
                              py::array_t<double>(u_hi_arr).data();
                params.qdd_max = py::array_t<double>(qdd_max_arr).data();
                params.margin = py::float_(cp["margin"]);
                params.fp_q_tol = py::float_(cp["fp_q_tol"]);
                params.actuator_mode = py::int_(cp["actuator_mode"]);
                params.qdd_window = py::int_(cp["qdd_window"]);
                params.dt = py::float_(cp["dt"]);
                params.qdd_hard_reject = py::bool_(cp["qdd_hard_reject"]);
                params_ptr = &params;
            }

            return fwd::single(
                X_new, U_new, X_nom, U_nom, Ks, ks,
                model_ptr, data_ptr,
                init_q_left_a.data(),
                ctrl_lo_a.data(), ctrl_hi_a.data(),
                alpha,
                actuator_mode, kp_ptr, kd_ptr, use_feedforward, torque_max_ptr,
                ball_geom_start, disable_collision,
                params_ptr);
        },
        py::arg("X_new"), py::arg("U_new"),
        py::arg("X_nom"), py::arg("U_nom"),
        py::arg("Ks"), py::arg("ks"),
        py::arg("model_ptr"), py::arg("data_ptr"),
        py::arg("init_q_left"),
        py::arg("ctrl_lo"), py::arg("ctrl_hi"),
        py::arg("alpha") = 0.5,
        py::arg("actuator_mode") = 0,
        py::arg("kp") = py::none(), py::arg("kd") = py::none(),
        py::arg("use_feedforward") = false,
        py::arg("torque_max") = py::none(),
        py::arg("ball_geom_start") = 0,
        py::arg("disable_collision") = false,
        py::arg("check_params") = py::none(),
        "Single forward pass (MPC mode). Returns True if valid");

    m.def("forward_pass_linesearch",
        [](py::array_t<double> X_nom, py::array_t<double> U_nom,
           py::array_t<double> Ks, py::array_t<double> ks,
           py::array_t<double> alpha_list,
           double cost_old,
           uintptr_t model_ptr, uintptr_t data_ptr,
           py::array_t<double> init_q_left_a,
           py::array_t<double> ctrl_lo_a, py::array_t<double> ctrl_hi_a,
           py::object cost_fn,
           int actuator_mode,
           py::object kp_obj, py::object kd_obj,
           bool use_feedforward,
           py::object torque_max_obj)
        {
            const double* kp_ptr = nullptr;
            const double* kd_ptr = nullptr;
            const double* torque_max_ptr = nullptr;
            if (!kp_obj.is_none()) kp_ptr = py::array_t<double>(kp_obj).data();
            if (!kd_obj.is_none()) kd_ptr = py::array_t<double>(kd_obj).data();
            if (!torque_max_obj.is_none())
                torque_max_ptr = py::array_t<double>(torque_max_obj).data();

            return fwd::linesearch(
                X_nom, U_nom, Ks, ks,
                alpha_list, cost_old,
                model_ptr, data_ptr,
                init_q_left_a.data(),
                ctrl_lo_a.data(), ctrl_hi_a.data(),
                cost_fn,
                actuator_mode, kp_ptr, kd_ptr, use_feedforward, torque_max_ptr);
        },
        py::arg("X_nom"), py::arg("U_nom"),
        py::arg("Ks"), py::arg("ks"),
        py::arg("alpha_list"),
        py::arg("cost_old"),
        py::arg("model_ptr"), py::arg("data_ptr"),
        py::arg("init_q_left"),
        py::arg("ctrl_lo"), py::arg("ctrl_hi"),
        py::arg("cost_fn"),
        py::arg("actuator_mode") = 0,
        py::arg("kp") = py::none(), py::arg("kd") = py::none(),
        py::arg("use_feedforward") = false,
        py::arg("torque_max") = py::none(),
        "Forward pass with linesearch. Returns (accepted, X_out, U_out, cost_out)");

    // sim_step: single physics step (for direct testing)
    m.def("sim_step",
        [](py::array_t<double> x_a, py::array_t<double> u_a,
           uintptr_t model_ptr, uintptr_t data_ptr,
           py::array_t<double> init_q_left_a,
           py::array_t<double> ctrl_lo_a, py::array_t<double> ctrl_hi_a,
           int actuator_mode,
           py::object kp_obj, py::object kd_obj,
           bool use_feedforward,
           py::object torque_max_obj)
        {
            const double* kp_ptr = nullptr;
            const double* kd_ptr = nullptr;
            const double* torque_max_ptr = nullptr;
            if (!kp_obj.is_none()) kp_ptr = py::array_t<double>(kp_obj).data();
            if (!kd_obj.is_none()) kd_ptr = py::array_t<double>(kd_obj).data();
            if (!torque_max_obj.is_none())
                torque_max_ptr = py::array_t<double>(torque_max_obj).data();

            const double* x = x_a.data();
            const double* u = u_a.data();
            double x_next[12];
            sim_step(
                to_model(model_ptr), to_data(data_ptr),
                x, x + kNQ, u,
                init_q_left_a.data(),
                ctrl_lo_a.data(), ctrl_hi_a.data(),
                actuator_mode, kp_ptr, kd_ptr, use_feedforward, torque_max_ptr,
                x_next, x_next + kNQ);

            return py::array_t<double>(kNX, x_next);
        },
        py::arg("x"), py::arg("u"),
        py::arg("model_ptr"), py::arg("data_ptr"),
        py::arg("init_q_left"),
        py::arg("ctrl_lo"), py::arg("ctrl_hi"),
        py::arg("actuator_mode") = 0,
        py::arg("kp") = py::none(), py::arg("kd") = py::none(),
        py::arg("use_feedforward") = false,
        py::arg("torque_max") = py::none(),
        "Single sim step. x=(q,qdot) (12,), u=(6,) -> x_next (12,)");

    // check_step: constraint checking for single transition
    m.def("check_step",
        [](py::array_t<double> x_prev_a, py::array_t<double> x_next_a,
           py::array_t<double> u_a,
           py::array_t<double> qdot_hist_a,
           int hist_len,
           py::dict params)
        {
            StepCheckParams p;
            p.q_lo = params["q_lo"].is_none() ? nullptr :
                     py::array_t<double>(params["q_lo"]).data();
            p.q_hi = params["q_hi"].is_none() ? nullptr :
                     py::array_t<double>(params["q_hi"]).data();
            p.qd_max = py::array_t<double>(params["qd_max"]).data();
            p.u_lo = params["u_lo"].is_none() ? nullptr :
                      py::array_t<double>(params["u_lo"]).data();
            p.u_hi = params["u_hi"].is_none() ? nullptr :
                      py::array_t<double>(params["u_hi"]).data();
            p.qdd_max = py::array_t<double>(params["qdd_max"]).data();
            p.margin = py::float_(params["margin"]);
            p.fp_q_tol = py::float_(params["fp_q_tol"]);
            p.actuator_mode = py::int_(params["actuator_mode"]);
            p.qdd_window = py::int_(params["qdd_window"]);
            p.dt = py::float_(params["dt"]);
            p.qdd_hard_reject = py::bool_(params["qdd_hard_reject"]);

            StepCheckResult res = check_step(
                x_prev_a.data(), x_next_a.data(), u_a.data(),
                qdot_hist_a.data(), hist_len, p);

            return py::make_tuple(res.feasible, std::string(res.reason));
        },
        py::arg("x_prev"), py::arg("x_next"), py::arg("u"),
        py::arg("qdot_hist"), py::arg("hist_len"),
        py::arg("params"),
        "Check step feasibility. Returns (feasible: bool, reason: str)");
}
