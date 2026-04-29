/**
 * Header for CUDA wrapper functions
 * Used by drone_ilqr_ext.cpp to call fused CUDA kernels
 */
#pragma once

#include <ATen/ATen.h>
#include <tuple>

// Fused step + jacobian computation using CUDA kernel
// x: [B, 10], u: [B, 4], dt: float
// Returns: (x_next [B, 10], A [B, 10, 10], B [B, 10, 4])
std::tuple<at::Tensor, at::Tensor, at::Tensor> drone_step_jacobian_cuda(
    const at::Tensor& x,
    const at::Tensor& u,
    double dt
);

// Batched rollout using CUDA kernel
// x0: [B, 10], U: [B, T, 4], dt: float
// Returns: X [B, T+1, 10]
at::Tensor batched_rollout_cuda(
    const at::Tensor& x0,
    const at::Tensor& U,
    double dt
);
