/**
 * Fused CUDA kernel for iLQR forward pass (line search rollout with feedback)
 *
 * This kernel fuses the entire forward pass rollout into a single kernel launch,
 * eliminating kernel launch overhead for small batch sizes.
 */

#pragma once

#include <cuda_runtime.h>

namespace drone_kernels {

// Constants - use from backward_pass_kernel.cuh if already defined
#ifndef DRONE_KERNELS_CONSTANTS_DEFINED
#define DRONE_KERNELS_CONSTANTS_DEFINED
constexpr int NX = 10;
constexpr int NU = 4;
#endif

// Drone physical constants for forward pass
constexpr float FP_MASS = 0.752f;
constexpr float FP_G_VAL = -9.8066f;

/**
 * Fused forward pass kernel with feedback control law
 *
 * Performs rollout: x_{t+1} = f(x_t, u_t)
 * where u_t = U_nom[t] + alpha * k[t] + K[t] @ (x_t - X_nom[t])
 *
 * Grid: (num_alphas * batch_size, 1, 1)
 * Block: (32, 1, 1) - one thread does all work for one trajectory
 */
__global__ void forward_pass_fused_kernel(
    // Inputs
    const float* __restrict__ x0,          // [B, NX] - initial state
    const float* __restrict__ X_nom,       // [B, T+1, NX] - nominal trajectory
    const float* __restrict__ U_nom,       // [B, T, NU] - nominal controls
    const float* __restrict__ K,           // [B, T, NU, NX] - feedback gains
    const float* __restrict__ k,           // [B, T, NU] - feedforward gains
    const float* __restrict__ alphas,      // [num_alphas] - line search step sizes
    const float* __restrict__ u_min,       // [NU] or nullptr - read inline via __ldg
    const float* __restrict__ u_max,       // [NU] or nullptr
    float dt,
    int batch_size,
    int num_alphas,
    int T,
    bool has_bounds,
    // Outputs
    float* __restrict__ X_out,             // [B*num_alphas, T+1, NX]
    float* __restrict__ U_out              // [B*num_alphas, T, NU]
);

/**
 * Launch the fused forward pass kernel
 */
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
    const float* u_min,  // [NU] or nullptr
    const float* u_max,  // [NU] or nullptr
    float* X_out,
    float* U_out,
    cudaStream_t stream
);

} // namespace drone_kernels
