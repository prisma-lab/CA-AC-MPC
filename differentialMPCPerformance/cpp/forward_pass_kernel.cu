/**
 * Fused CUDA kernel for iLQR forward pass
 */

#include "forward_pass_kernel.cuh"
#include <math.h>

namespace drone_kernels {

// ============================================================================
// Inline drone dynamics (avoid function call overhead)
// ============================================================================

__device__ __forceinline__ void quat_mult_inline(
    float w1, float x1, float y1, float z1,
    float w2, float x2, float y2, float z2,
    float& wo, float& xo, float& yo, float& zo
) {
    wo = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    xo = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    yo = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    zo = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

__device__ __forceinline__ void exp_map_inline(
    float wx, float wy, float wz, float dt,
    float& qw, float& qx, float& qy, float& qz
) {
    float w_norm = sqrtf(wx*wx + wy*wy + wz*wz) + 1e-8f;
    float half_angle = w_norm * (dt * 0.5f);
    float s = sinf(half_angle);
    float c = cosf(half_angle);
    float inv_norm = 1.0f / w_norm;
    qw = c;
    qx = wx * inv_norm * s;
    qy = wy * inv_norm * s;
    qz = wz * inv_norm * s;
}

__device__ __forceinline__ void rotate_quat_inline(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float& ox, float& oy, float& oz
) {
    // v_rotated = q * [0, v] * q^{-1}
    // Using the formula: v' = v + 2*q_w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))

    // q_xyz x v
    float tx = qy * vz - qz * vy;
    float ty = qz * vx - qx * vz;
    float tz = qx * vy - qy * vx;

    // q_xyz x (q_xyz x v)
    float ttx = qy * tz - qz * ty;
    float tty = qz * tx - qx * tz;
    float ttz = qx * ty - qy * tx;

    ox = vx + 2.0f * (qw * tx + ttx);
    oy = vy + 2.0f * (qw * ty + tty);
    oz = vz + 2.0f * (qw * tz + ttz);
}

// Drone step: x_next = f(x, u)
// x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz]
// u = [thrust, wx, wy, wz]
__device__ __forceinline__ void drone_step_inline(
    const float* x, const float* u, float dt, float* x_next
) {
    // Unpack state
    float px = x[0], py = x[1], pz = x[2];
    float qw = x[3], qx = x[4], qy = x[5], qz = x[6];
    float vx = x[7], vy = x[8], vz = x[9];

    // Unpack control
    float fc = u[0];
    float wx = u[1], wy = u[2], wz = u[3];

    // Position update: p_new = p + v * dt
    float new_px = px + vx * dt;
    float new_py = py + vy * dt;
    float new_pz = pz + vz * dt;

    // Quaternion update: q_new = q * exp_map(w, dt)
    float eq_w, eq_x, eq_y, eq_z;
    exp_map_inline(wx, wy, wz, dt, eq_w, eq_x, eq_y, eq_z);

    float new_qw, new_qx, new_qy, new_qz;
    quat_mult_inline(qw, qx, qy, qz, eq_w, eq_x, eq_y, eq_z,
                     new_qw, new_qx, new_qy, new_qz);

    // Normalize quaternion
    float qnorm = sqrtf(new_qw*new_qw + new_qx*new_qx + new_qy*new_qy + new_qz*new_qz) + 1e-8f;
    new_qw /= qnorm;
    new_qx /= qnorm;
    new_qy /= qnorm;
    new_qz /= qnorm;

    // Velocity update: v_new = v + (R(q) * [0,0,fc]/m + g) * dt
    // Thrust in body frame is [0, 0, fc]
    float ax, ay, az;
    rotate_quat_inline(qw, qx, qy, qz, 0.0f, 0.0f, fc, ax, ay, az);
    ax /= FP_MASS;
    ay /= FP_MASS;
    az /= FP_MASS;
    az += FP_G_VAL;  // Add gravity

    float new_vx = vx + ax * dt;
    float new_vy = vy + ay * dt;
    float new_vz = vz + az * dt;

    // Pack output
    x_next[0] = new_px; x_next[1] = new_py; x_next[2] = new_pz;
    x_next[3] = new_qw; x_next[4] = new_qx; x_next[5] = new_qy; x_next[6] = new_qz;
    x_next[7] = new_vx; x_next[8] = new_vy; x_next[9] = new_vz;
}

// ============================================================================
// Forward pass kernel
// ============================================================================

// Only thread 0 of each block runs the rollout; (32,4) tells the compiler to
// pack at least 4 such blocks per SM rather than reserving registers for a
// hypothetical fully-occupied 32-thread kernel.
__global__ __launch_bounds__(32, 4) void forward_pass_fused_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ X_nom,
    const float* __restrict__ U_nom,
    const float* __restrict__ K,
    const float* __restrict__ k,
    const float* __restrict__ alphas,
    const float* __restrict__ u_min,   // device ptr or nullptr
    const float* __restrict__ u_max,   // device ptr or nullptr
    float dt,
    int batch_size,
    int num_alphas,
    int T,
    bool has_bounds,
    float* __restrict__ X_out,
    float* __restrict__ U_out
) {
    // Each block handles one (batch, alpha) pair
    int idx = blockIdx.x;
    int total_trajectories = batch_size * num_alphas;
    if (idx >= total_trajectories) return;

    int b = idx / num_alphas;      // Batch index
    int a = idx % num_alphas;      // Alpha index
    float alpha = alphas[a];

    // Compute offsets
    int x0_offset = b * NX;
    int X_nom_offset = b * (T + 1) * NX;
    int U_nom_offset = b * T * NU;
    int K_offset = b * T * NU * NX;
    int k_offset = b * T * NU;
    int X_out_offset = idx * (T + 1) * NX;
    int U_out_offset = idx * T * NU;

    // Thread-local state
    float x_curr[NX];
    float x_next[NX];
    float u_t[NU];
    float dx[NX];

    // Initialize x_curr = x0
    if (threadIdx.x == 0) {
        for (int i = 0; i < NX; i++) {
            x_curr[i] = x0[x0_offset + i];
            X_out[X_out_offset + i] = x_curr[i];  // Store X[0]
        }

        // Forward rollout
        for (int t = 0; t < T; t++) {
            // Compute dx = x_curr - X_nom[t]
            int X_nom_t_offset = X_nom_offset + t * NX;
            for (int i = 0; i < NX; i++) {
                dx[i] = x_curr[i] - X_nom[X_nom_t_offset + i];
            }

            // Compute du = alpha * k[t] + K[t] @ dx
            int k_t_offset = k_offset + t * NU;
            int K_t_offset = K_offset + t * NU * NX;
            for (int i = 0; i < NU; i++) {
                float du_i = alpha * k[k_t_offset + i];
                for (int j = 0; j < NX; j++) {
                    du_i += K[K_t_offset + i * NX + j] * dx[j];
                }
                u_t[i] = U_nom[U_nom_offset + t * NU + i] + du_i;
            }

            // Clamp controls if needed (bounds read directly from device memory).
            if (has_bounds) {
                u_t[0] = fminf(fmaxf(u_t[0], __ldg(u_min + 0)), __ldg(u_max + 0));
                u_t[1] = fminf(fmaxf(u_t[1], __ldg(u_min + 1)), __ldg(u_max + 1));
                u_t[2] = fminf(fmaxf(u_t[2], __ldg(u_min + 2)), __ldg(u_max + 2));
                u_t[3] = fminf(fmaxf(u_t[3], __ldg(u_min + 3)), __ldg(u_max + 3));
            }

            // Store u_t
            for (int i = 0; i < NU; i++) {
                U_out[U_out_offset + t * NU + i] = u_t[i];
            }

            // Step dynamics
            drone_step_inline(x_curr, u_t, dt, x_next);

            // Copy x_next to x_curr and store
            for (int i = 0; i < NX; i++) {
                x_curr[i] = x_next[i];
                X_out[X_out_offset + (t + 1) * NX + i] = x_next[i];
            }
        }
    }
}

// ============================================================================
// Launch function
// ============================================================================

void launch_forward_pass_fused(
    const float* x0,
    const float* X_nom,
    const float* U_nom,
    const float* K,
    const float* k,
    const float* alphas,
    float dt,
    int batch_size,
    int num_alphas,
    int T,
    bool has_bounds,
    const float* u_min,
    const float* u_max,
    float* X_out,
    float* U_out,
    cudaStream_t stream
) {
    const int total_trajectories = batch_size * num_alphas;
    const bool effective_bounds = has_bounds && u_min && u_max;

    forward_pass_fused_kernel<<<total_trajectories, 32, 0, stream>>>(
        x0, X_nom, U_nom, K, k, alphas, u_min, u_max,
        dt, batch_size, num_alphas, T, effective_bounds,
        X_out, U_out
    );
}

} // namespace drone_kernels
