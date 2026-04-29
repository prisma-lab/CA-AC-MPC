/**
 * Projected-Newton box-QP solver for 4x4 problems (PNQP).
 *
 * Pure __device__, header-only. Mirrors the algorithm in
 * acmpc_public-master/mpc.pytorch/mpc/pnqp.py but is specialized to NU=4 with
 * everything in registers / thread-local arrays. Used by the iLQR forward
 * Riccati to enforce u_min <= u_t + du <= u_max accurately (active-set
 * detection via gradient sign instead of post-hoc clamp tolerance).
 *
 * Convergence parameters match mpc.pytorch defaults (gamma=0.1, decay=0.1,
 * max line-search count 10, ridge 1e-11, tol 1e-4, n_iter=10).
 */

#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace drone_kernels {

// Solve a 4x4 linear system in-place via Gaussian elimination with partial
// pivoting (mutates A and b; result in b). Used inside pnqp4 each iteration.
__device__ __forceinline__ void pnqp_solve4x4(float A[4][4], float b[4]) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int max_row = k;
        float max_val = fabsf(A[k][k]);
        #pragma unroll
        for (int i = k + 1; i < 4; i++) {
            float v = fabsf(A[i][k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_row != k) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float t = A[k][j]; A[k][j] = A[max_row][j]; A[max_row][j] = t;
            }
            float t = b[k]; b[k] = b[max_row]; b[max_row] = t;
        }
        float p = A[k][k];
        if (fabsf(p) < 1e-10f) p = (p < 0 ? -1e-10f : 1e-10f);
        #pragma unroll
        for (int i = k + 1; i < 4; i++) {
            float f = A[i][k] / p;
            #pragma unroll
            for (int j = k + 1; j < 4; j++) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
            A[i][k] = 0.0f;
        }
    }
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        float s = b[i];
        #pragma unroll
        for (int j = i + 1; j < 4; j++) s -= A[i][j] * b[j];
        b[i] = s / A[i][i];
    }
}

// Quadratic objective at x: 0.5 x^T H x + q^T x.
__device__ __forceinline__ float pnqp_obj4(const float H[4][4], const float q[4],
                                           const float x[4]) {
    float Hx[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) s += H[i][j] * x[j];
        Hx[i] = s;
    }
    float v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) v += 0.5f * x[i] * Hx[i] + q[i] * x[i];
    return v;
}

/**
 * Solve box-QP min 0.5 x^T H x + q^T x s.t. lower <= x <= upper for x in R^4.
 *
 * Inputs:
 *   H[4][4], q[4], lower[4], upper[4]
 *   x_io[4]:  warm start; if all zeros, the routine seeds with -H^{-1} q.
 *
 * Outputs:
 *   x_io[4]:       optimum x*
 *   free_mask[4]:  1.0 for free dims, 0.0 for active-bound dims at the optimum
 *
 * Algorithm parameters (compile-time constants below) are aligned with
 * mpc.pytorch.pnqp defaults.
 */
__device__ __forceinline__ void pnqp4(
    const float H[4][4],
    const float q[4],
    const float lower[4],
    const float upper[4],
    float x_io[4],
    float free_mask[4]
) {
    constexpr int N_ITER = 10;
    constexpr int N_LS   = 10;
    constexpr float GAMMA = 0.1f;
    constexpr float DECAY = 0.1f;
    constexpr float RIDGE = 1e-11f;
    constexpr float TOL   = 1e-4f;
    constexpr float BOUND_EPS = 1e-8f;

    // Warm start: if x_io is all-zero, init from unconstrained solve.
    bool warm_zero = true;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (x_io[i] != 0.0f) { warm_zero = false; }
    if (warm_zero) {
        float A[4][4]; float b[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) A[i][j] = H[i][j];
            b[i] = -q[i];
        }
        pnqp_solve4x4(A, b);
        #pragma unroll
        for (int i = 0; i < 4; i++) x_io[i] = b[i];
    }
    // Project warm start to feasible region.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        x_io[i] = fminf(fmaxf(x_io[i], lower[i]), upper[i]);
    }

    for (int it = 0; it < N_ITER; it++) {
        // Gradient g = H x + q.
        float g[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float s = q[i];
            #pragma unroll
            for (int j = 0; j < 4; j++) s += H[i][j] * x_io[j];
            g[i] = s;
        }

        // Active set: clamped iff at bound AND gradient pushes further out.
        float Ic[4];  // 1 if clamped, 0 if free
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            bool at_lo = (x_io[i] <= lower[i] + BOUND_EPS) && (g[i] > 0.0f);
            bool at_hi = (x_io[i] >= upper[i] - BOUND_EPS) && (g[i] < 0.0f);
            Ic[i] = (at_lo || at_hi) ? 1.0f : 0.0f;
        }

        // Build masked Hessian H_: zero out rows/cols touching clamped dims, add ridge.
        float H_[4][4]; float g_[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float Ifi = 1.0f - Ic[i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float Ifj = 1.0f - Ic[j];
                H_[i][j] = H[i][j] * Ifi * Ifj;
            }
            // Diagonal ridge for invertibility on clamped dims.
            H_[i][i] += RIDGE;
            g_[i] = -g[i] * Ifi;  // negate here so dx = H_^{-1} g_ directly
        }
        pnqp_solve4x4(H_, g_);  // g_ now holds dx

        // Convergence: ||dx||_2.
        float dx_sqnorm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) dx_sqnorm += g_[i] * g_[i];
        if (dx_sqnorm < TOL * TOL) break;

        // Armijo line search.
        float obj_x = pnqp_obj4(H, q, x_io);
        float alpha = 1.0f;
        float maybe_x[4];
        bool ok = false;
        for (int ls = 0; ls < N_LS && !ok; ls++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                maybe_x[i] = fminf(fmaxf(x_io[i] + alpha * g_[i], lower[i]), upper[i]);
            }
            float obj_m = pnqp_obj4(H, q, maybe_x);
            float den = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) den += g[i] * (x_io[i] - maybe_x[i]);
            float armijo = (obj_x - obj_m) / (fabsf(den) > 1e-12f ? den : 1e-12f);
            if (armijo > GAMMA) ok = true;
            else alpha *= DECAY;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) x_io[i] = maybe_x[i];
    }

    // Final active-set determination.
    float g[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float s = q[i];
        #pragma unroll
        for (int j = 0; j < 4; j++) s += H[i][j] * x_io[j];
        g[i] = s;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        bool at_lo = (x_io[i] <= lower[i] + BOUND_EPS) && (g[i] > 0.0f);
        bool at_hi = (x_io[i] >= upper[i] - BOUND_EPS) && (g[i] < 0.0f);
        free_mask[i] = (at_lo || at_hi) ? 0.0f : 1.0f;
    }
}

}  // namespace drone_kernels
