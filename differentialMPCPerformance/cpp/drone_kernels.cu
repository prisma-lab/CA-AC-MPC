/**
 * CUDA Kernels for Drone ILQR Solver
 * Pure CUDA implementation without PyTorch headers
 */

#include "drone_kernels.cuh"
#include <cmath>

namespace drone_kernels {

// Constants
constexpr float MASS = 0.752f;
constexpr float G_VAL = -9.8066f;

// ============================================================================
// Device helper functions
// ============================================================================

__device__ __forceinline__ void quat_mult_d(
    float w1, float x1, float y1, float z1,
    float w2, float x2, float y2, float z2,
    float& wo, float& xo, float& yo, float& zo
) {
    wo = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    xo = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    yo = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    zo = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

__device__ __forceinline__ void exp_map_d(
    float wx, float wy, float wz, float dt,
    float& qw, float& qx, float& qy, float& qz
) {
    float w_norm = sqrtf(wx*wx + wy*wy + wz*wz) + 1e-8f;
    float half_angle = w_norm * dt * 0.5f;
    float s = sinf(half_angle);
    float c = cosf(half_angle);
    float w_inv = 1.0f / w_norm;

    qw = c;
    qx = wx * w_inv * s;
    qy = wy * w_inv * s;
    qz = wz * w_inv * s;
}

__device__ __forceinline__ void rotate_by_quat_d(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float& rx, float& ry, float& rz
) {
    float q00 = qw*qw, q11 = qx*qx, q22 = qy*qy, q33 = qz*qz;
    float q01 = qw*qx, q02 = qw*qy, q03 = qw*qz;
    float q12 = qx*qy, q13 = qx*qz, q23 = qy*qz;

    rx = (q00 + q11 - q22 - q33)*vx + 2.0f*(q12 - q03)*vy + 2.0f*(q13 + q02)*vz;
    ry = 2.0f*(q12 + q03)*vx + (q00 - q11 + q22 - q33)*vy + 2.0f*(q23 - q01)*vz;
    rz = 2.0f*(q13 - q02)*vx + 2.0f*(q23 + q01)*vy + (q00 - q11 - q22 + q33)*vz;
}

// ============================================================================
// Fused Dynamics + Jacobian Kernel
// ============================================================================

__global__ void drone_step_jacobian_kernel(
    const float* __restrict__ x,
    const float* __restrict__ u,
    float* __restrict__ x_next,
    float* __restrict__ A,
    float* __restrict__ B_mat,
    const float dt,
    const int batch_size
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const float* xi = x + b * 10;
    const float* ui = u + b * 4;
    float* xo = x_next + b * 10;
    float* Ai = A + b * 100;
    float* Bi = B_mat + b * 40;

    float px = xi[0], py = xi[1], pz = xi[2];
    float qw = xi[3], qx = xi[4], qy = xi[5], qz = xi[6];
    float vx = xi[7], vy = xi[8], vz = xi[9];

    float fc = ui[0];
    float wx = ui[1], wy = ui[2], wz = ui[3];

    // Forward dynamics
    float px_new = px + vx * dt;
    float py_new = py + vy * dt;
    float pz_new = pz + vz * dt;

    float eq_w, eq_x, eq_y, eq_z;
    exp_map_d(wx, wy, wz, dt, eq_w, eq_x, eq_y, eq_z);

    float qw_new, qx_new, qy_new, qz_new;
    quat_mult_d(qw, qx, qy, qz, eq_w, eq_x, eq_y, eq_z, qw_new, qx_new, qy_new, qz_new);

    float q_norm = sqrtf(qw_new*qw_new + qx_new*qx_new + qy_new*qy_new + qz_new*qz_new) + 1e-8f;
    qw_new /= q_norm;
    qx_new /= q_norm;
    qy_new /= q_norm;
    qz_new /= q_norm;

    float ax, ay, az;
    rotate_by_quat_d(qw, qx, qy, qz, 0.0f, 0.0f, fc, ax, ay, az);
    ax /= MASS;
    ay /= MASS;
    az = az / MASS + G_VAL;

    float vx_new = vx + ax * dt;
    float vy_new = vy + ay * dt;
    float vz_new = vz + az * dt;

    xo[0] = px_new; xo[1] = py_new; xo[2] = pz_new;
    xo[3] = qw_new; xo[4] = qx_new; xo[5] = qy_new; xo[6] = qz_new;
    xo[7] = vx_new; xo[8] = vy_new; xo[9] = vz_new;

    // Initialize A to identity, B to zero
    for (int i = 0; i < 100; ++i) Ai[i] = 0.0f;
    for (int i = 0; i < 10; ++i) Ai[i * 10 + i] = 1.0f;
    for (int i = 0; i < 40; ++i) Bi[i] = 0.0f;

    // Position derivatives
    Ai[0*10 + 7] = dt;
    Ai[1*10 + 8] = dt;
    Ai[2*10 + 9] = dt;

    // Quaternion derivatives
    float half_dt = 0.5f * dt;

    Ai[3*10 + 4] -= half_dt * wx;
    Ai[3*10 + 5] -= half_dt * wy;
    Ai[3*10 + 6] -= half_dt * wz;

    Ai[4*10 + 3] += half_dt * wx;
    Ai[4*10 + 5] += half_dt * wz;
    Ai[4*10 + 6] -= half_dt * wy;

    Ai[5*10 + 3] += half_dt * wy;
    Ai[5*10 + 4] -= half_dt * wz;
    Ai[5*10 + 6] += half_dt * wx;

    Ai[6*10 + 3] += half_dt * wz;
    Ai[6*10 + 4] += half_dt * wy;
    Ai[6*10 + 5] -= half_dt * wx;

    Bi[3*4 + 1] = -half_dt * qx;
    Bi[3*4 + 2] = -half_dt * qy;
    Bi[3*4 + 3] = -half_dt * qz;

    Bi[4*4 + 1] = half_dt * qw;
    Bi[4*4 + 2] = -half_dt * qz;
    Bi[4*4 + 3] = half_dt * qy;

    Bi[5*4 + 1] = half_dt * qz;
    Bi[5*4 + 2] = half_dt * qw;
    Bi[5*4 + 3] = -half_dt * qx;

    Bi[6*4 + 1] = -half_dt * qy;
    Bi[6*4 + 2] = half_dt * qx;
    Bi[6*4 + 3] = half_dt * qw;

    // Velocity derivatives
    float fc_m = fc / MASS;
    float two_dt = 2.0f * dt;
    float q0_f = qw * fc_m, q1_f = qx * fc_m, q2_f = qy * fc_m, q3_f = qz * fc_m;

    Ai[7*10 + 3] = two_dt * q2_f;
    Ai[7*10 + 4] = two_dt * q3_f;
    Ai[7*10 + 5] = two_dt * q0_f;
    Ai[7*10 + 6] = two_dt * q1_f;

    Ai[8*10 + 3] = -two_dt * q1_f;
    Ai[8*10 + 4] = -two_dt * q0_f;
    Ai[8*10 + 5] = two_dt * q3_f;
    Ai[8*10 + 6] = two_dt * q2_f;

    Ai[9*10 + 3] = two_dt * q0_f;
    Ai[9*10 + 4] = -two_dt * q1_f;
    Ai[9*10 + 5] = -two_dt * q2_f;
    Ai[9*10 + 6] = two_dt * q3_f;

    float dt_m = dt / MASS;
    Bi[7*4 + 0] = 2.0f * (qx*qz + qw*qy) * dt_m;
    Bi[8*4 + 0] = 2.0f * (qy*qz - qw*qx) * dt_m;
    Bi[9*4 + 0] = (qw*qw - qx*qx - qy*qy + qz*qz) * dt_m;
}

// ============================================================================
// Batched Rollout Kernel
// ============================================================================

__global__ void batched_rollout_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ U,
    float* __restrict__ X,
    const float dt,
    const int batch_size,
    const int horizon
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const int nx = 10;
    const int nu = 4;

    for (int i = 0; i < nx; ++i) {
        X[b * (horizon + 1) * nx + i] = x0[b * nx + i];
    }

    float state[10];
    for (int i = 0; i < nx; ++i) state[i] = x0[b * nx + i];

    for (int t = 0; t < horizon; ++t) {
        const float* ut = U + b * horizon * nu + t * nu;
        float fc = ut[0], wx = ut[1], wy = ut[2], wz = ut[3];

        float px = state[0], py = state[1], pz = state[2];
        float qw = state[3], qx = state[4], qy = state[5], qz = state[6];
        float vx = state[7], vy = state[8], vz = state[9];

        state[0] = px + vx * dt;
        state[1] = py + vy * dt;
        state[2] = pz + vz * dt;

        float eq_w, eq_x, eq_y, eq_z;
        exp_map_d(wx, wy, wz, dt, eq_w, eq_x, eq_y, eq_z);

        float qw_new, qx_new, qy_new, qz_new;
        quat_mult_d(qw, qx, qy, qz, eq_w, eq_x, eq_y, eq_z, qw_new, qx_new, qy_new, qz_new);

        float q_norm = sqrtf(qw_new*qw_new + qx_new*qx_new + qy_new*qy_new + qz_new*qz_new) + 1e-8f;
        state[3] = qw_new / q_norm;
        state[4] = qx_new / q_norm;
        state[5] = qy_new / q_norm;
        state[6] = qz_new / q_norm;

        float ax, ay, az;
        rotate_by_quat_d(qw, qx, qy, qz, 0.0f, 0.0f, fc, ax, ay, az);
        state[7] = vx + (ax / MASS) * dt;
        state[8] = vy + (ay / MASS) * dt;
        state[9] = vz + (az / MASS + G_VAL) * dt;

        float* xt1 = X + b * (horizon + 1) * nx + (t + 1) * nx;
        for (int i = 0; i < nx; ++i) xt1[i] = state[i];
    }
}

// ============================================================================
// Launch functions
// ============================================================================

void launch_drone_step_jacobian(
    const float* x,
    const float* u,
    float* x_next,
    float* A,
    float* B_mat,
    float dt,
    int batch_size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    drone_step_jacobian_kernel<<<blocks, threads, 0, stream>>>(
        x, u, x_next, A, B_mat, dt, batch_size
    );
}

void launch_batched_rollout(
    const float* x0,
    const float* U,
    float* X,
    float dt,
    int batch_size,
    int horizon,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    batched_rollout_kernel<<<blocks, threads, 0, stream>>>(
        x0, U, X, dt, batch_size, horizon
    );
}

} // namespace drone_kernels
