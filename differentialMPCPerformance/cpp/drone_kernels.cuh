/**
 * CUDA Kernels Header for Drone ILQR Solver
 */
#pragma once

#include <cuda_runtime.h>

namespace drone_kernels {

// Launch fused step + jacobian kernel
void launch_drone_step_jacobian(
    const float* x,
    const float* u,
    float* x_next,
    float* A,
    float* B_mat,
    float dt,
    int batch_size,
    cudaStream_t stream
);

// Launch batched rollout kernel
void launch_batched_rollout(
    const float* x0,
    const float* U,
    float* X,
    float dt,
    int batch_size,
    int horizon,
    cudaStream_t stream
);

} // namespace drone_kernels
