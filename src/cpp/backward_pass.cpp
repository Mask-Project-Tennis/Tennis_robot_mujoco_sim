// Riccati backward pass (batch, pure algebra — no MuJoCo dependency)
// Port of solver_cpp._backward_pass_numpy: per-step Q-terms + 6x6 solve + V update.
// No Chinese comments to avoid MSVC encoding issues.

#include "types.h"
#include <cstring>
#include <cmath>

namespace backward_pass_ns {

// Solve M(6x6) * X(6,nb) = B(6,nb) via Gaussian elimination with partial pivoting.
// Returns false if M is singular (max pivot < 1e-12).
// A/B/X are row-major; nb <= 14 expected (12 for Q_ux + 1 for Q_u packed = 13).
inline bool solve_6x_n(const double* M, const double* B, int nb, double* X) {
    double A[6][20] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) A[i][j] = M[i * 6 + j];
        for (int j = 0; j < nb; ++j) A[i][6 + j] = B[i * nb + j];
    }
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        double mx = std::fabs(A[col][col]);
        for (int r = col + 1; r < 6; ++r) {
            double v = std::fabs(A[r][col]);
            if (v > mx) { mx = v; piv = r; }
        }
        if (mx < 1e-12) return false;
        if (piv != col) {
            for (int j = 0; j < 6 + nb; ++j) std::swap(A[col][j], A[piv][j]);
        }
        double inv = 1.0 / A[col][col];
        for (int j = 0; j < 6 + nb; ++j) A[col][j] *= inv;
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            double f = A[r][col];
            for (int j = 0; j < 6 + nb; ++j) A[r][j] -= f * A[col][j];
        }
    }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < nb; ++j)
            X[i * nb + j] = A[i][6 + j];
    return true;
}

// C(n,m) = A(n,k) * B(k,m), row-major
template<int n, int m, int k>
inline void matmul(const double* A, const double* B, double* C) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int t = 0; t < k; ++t) s += A[i * k + t] * B[t * m + j];
            C[i * m + j] = s;
        }
}

// Riccati backward pass.
// All matrix inputs are row-major contiguous:
//   A_all (N,12,12), B_all (N,12,6), l_x (N,12), l_u (N,6),
//   l_xx (N,12,12), l_ux (N,6,12), l_uu (N,6,6), l_x_N (12,), l_xx_N (12,12)
// Outputs: Ks_out (N,6,12), ks_out (N,6)
// Returns false if any Q_uu_reg is singular.
inline bool run(
    int N,
    const double* A_all, const double* B_all,
    const double* l_x, const double* l_u,
    const double* l_xx, const double* l_ux, const double* l_uu,
    const double* l_x_N, const double* l_xx_N,
    double mu,
    double* Ks_out, double* ks_out)
{
    double V_x[12], V_xx[144];
    std::memcpy(V_x, l_x_N, 12 * sizeof(double));
    std::memcpy(V_xx, l_xx_N, 144 * sizeof(double));

    for (int k = N - 1; k >= 0; --k) {
        const double* A = A_all + k * 144;  // (12,12)
        const double* B = B_all + k * 72;   // (12,6)

        // Q_x = l_x[k] + A^T @ V_x  (12,)   [A^T[i,j]=A[j,i]]
        double Q_x[12];
        for (int i = 0; i < 12; ++i) {
            double s = l_x[k * 12 + i];
            for (int j = 0; j < 12; ++j) s += A[j * 12 + i] * V_x[j];
            Q_x[i] = s;
        }

        // Q_u = l_u[k] + B^T @ V_x  (6,)     [B^T[i,j]=B[j,i], B is (12,6)]
        double Q_u[6];
        for (int i = 0; i < 6; ++i) {
            double s = l_u[k * 6 + i];
            for (int j = 0; j < 12; ++j) s += B[j * 6 + i] * V_x[j];
            Q_u[i] = s;
        }

        // tmp = V_xx @ A  (12,12)
        double tmp[144];
        matmul<12, 12, 12>(V_xx, A, tmp);
        // Q_xx = l_xx[k] + A^T @ tmp  (12,12)
        double Q_xx[144];
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j) {
                double s = l_xx[k * 144 + i * 12 + j];
                for (int t = 0; t < 12; ++t) s += A[t * 12 + i] * tmp[t * 12 + j];
                Q_xx[i * 12 + j] = s;
            }

        // BtVxx = B^T @ V_xx  (6,12)
        double BtVxx[72];
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 12; ++j) {
                double s = 0.0;
                for (int t = 0; t < 12; ++t) s += B[t * 6 + i] * V_xx[t * 12 + j];
                BtVxx[i * 12 + j] = s;
            }

        // Q_ux = l_ux[k] + BtVxx @ A  (6,12)
        double Q_ux[72];
        matmul<6, 12, 12>(BtVxx, A, Q_ux);
        for (int i = 0; i < 72; ++i) Q_ux[i] += l_ux[k * 72 + i];

        // Q_uu = l_uu[k] + BtVxx @ B  (6,6)
        double Q_uu[36];
        matmul<6, 6, 12>(BtVxx, B, Q_uu);
        for (int i = 0; i < 36; ++i) Q_uu[i] += l_uu[k * 36 + i];

        // Q_uu_reg = Q_uu + mu*I
        double Q_uu_reg[36];
        std::memcpy(Q_uu_reg, Q_uu, 36 * sizeof(double));
        for (int i = 0; i < 6; ++i) Q_uu_reg[i * 6 + i] += mu;

        // Pack RHS [Q_ux | Q_u] as (6, 13) and solve Q_uu_reg * sol = rhs
        // sol[:, 0:12] = Q_uu_inv @ Q_ux, sol[:, 12] = Q_uu_inv @ Q_u
        double rhs[6 * 13];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 12; ++j) rhs[i * 13 + j] = Q_ux[i * 12 + j];
            rhs[i * 13 + 12] = Q_u[i];
        }
        double sol[6 * 13];
        if (!solve_6x_n(Q_uu_reg, rhs, 13, sol)) return false;

        // K_k = -Q_uu_inv @ Q_ux ; k_k = -Q_uu_inv @ Q_u
        double* K_k = Ks_out + k * 72;
        double* k_k = ks_out + k * 6;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 12; ++j) K_k[i * 12 + j] = -sol[i * 13 + j];
            k_k[i] = -sol[i * 13 + 12];
        }

        // V_x = Q_x - Q_ux^T @ (Q_uu_inv @ Q_u)
        double Quuinv_Qu[6];
        for (int i = 0; i < 6; ++i) Quuinv_Qu[i] = sol[i * 13 + 12];
        for (int i = 0; i < 12; ++i) {
            double s = 0.0;
            for (int t = 0; t < 6; ++t) s += Q_ux[t * 12 + i] * Quuinv_Qu[t];
            V_x[i] = Q_x[i] - s;
        }

        // V_xx = Q_xx - Q_ux^T @ (Q_uu_inv @ Q_ux)
        // Q_uu_inv @ Q_ux = sol[:, 0:12] with stride 13
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j) {
                double s = 0.0;
                for (int t = 0; t < 6; ++t) s += Q_ux[t * 12 + i] * sol[t * 13 + j];
                V_xx[i * 12 + j] = Q_xx[i * 12 + j] - s;
            }
        // Symmetrize: V_xx = 0.5 * (V_xx + V_xx^T)
        for (int i = 0; i < 12; ++i)
            for (int j = i + 1; j < 12; ++j) {
                double avg = 0.5 * (V_xx[i * 12 + j] + V_xx[j * 12 + i]);
                V_xx[i * 12 + j] = avg;
                V_xx[j * 12 + i] = avg;
            }
    }
    return true;
}

}  // namespace backward_pass_ns
