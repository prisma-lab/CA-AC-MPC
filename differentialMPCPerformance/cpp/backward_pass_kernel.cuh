/**
 * Fused CUDA kernel for iLQR backward pass
 *
 * This kernel fuses the entire backward pass into a single kernel launch,
 * eliminating the kernel launch overhead that dominates small batch performance.
 *
 * For B=1, T=25, this reduces from ~250 kernel launches to 1.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace drone_kernels {

// Constants
#ifndef DRONE_KERNELS_CONSTANTS_DEFINED
#define DRONE_KERNELS_CONSTANTS_DEFINED
constexpr int NX = 10;
constexpr int NU = 4;
#endif
constexpr int N_TOTAL = NX + NU;  // 14

/**
 * Fused backward pass kernel for single batch (B=1)
 *
 * This kernel processes the entire backward pass sequentially on the GPU,
 * keeping all data in registers/shared memory to avoid global memory latency.
 *
 * Grid: (1, 1, 1)
 * Block: (32, 1, 1) or (64, 1, 1) - use warp for matrix ops
 */
__global__ void backward_pass_fused_kernel_b1(
    // Inputs [T, ...]
    const float* __restrict__ A,       // [T, NX, NX]
    const float* __restrict__ B,       // [T, NX, NU]
    const float* __restrict__ l_xx,    // [T, NX, NX]
    const float* __restrict__ l_uu,    // [T, NU, NU]
    const float* __restrict__ l_xu,    // [T, NX, NU]
    const float* __restrict__ l_x,     // [T, NX]
    const float* __restrict__ l_u,     // [T, NU]
    const float* __restrict__ l_xxN,   // [NX, NX] - terminal cost
    const float* __restrict__ l_xN,    // [NX] - terminal cost
    const bool* __restrict__ tight_mask, // [T, NU] or nullptr
    float reg_eps,
    int T,
    // Outputs [T, ...]
    float* __restrict__ K_out,         // [T, NU, NX]
    float* __restrict__ k_out,         // [T, NU]
    float* __restrict__ v0_out         // [NX] or nullptr
);

/**
 * Launch the fused backward pass kernel
 */
void launch_backward_pass_fused_b1(
    const float* A,
    const float* B,
    const float* l_xx,
    const float* l_uu,
    const float* l_xu,
    const float* l_x,
    const float* l_u,
    const float* l_xxN,
    const float* l_xN,
    const bool* tight_mask,
    float reg_eps,
    int T,
    float* K_out,
    float* k_out,
    float* v0_out,
    cudaStream_t stream
);

/**
 * Batched backward pass using parallel execution across batch dimension
 * Each block handles one batch element
 *
 * Grid: (B, 1, 1)
 * Block: (64, 1, 1) or (128, 1, 1)
 */
__global__ void backward_pass_batched_kernel(
    const float* __restrict__ A,       // [B, T, NX, NX]
    const float* __restrict__ B,       // [B, T, NX, NU]
    const float* __restrict__ l_xx,    // [B, T, NX, NX]
    const float* __restrict__ l_uu,    // [B, T, NU, NU]
    const float* __restrict__ l_xu,    // [B, T, NX, NU]
    const float* __restrict__ l_x,     // [B, T, NX]
    const float* __restrict__ l_u,     // [B, T, NU]
    const float* __restrict__ l_xxN,   // [B, NX, NX]
    const float* __restrict__ l_xN,    // [B, NX]
    const bool* __restrict__ tight_mask, // [B, T, NU] or nullptr
    float reg_eps,
    int batch_size,
    int T,
    float* __restrict__ K_out,         // [B, T, NU, NX]
    float* __restrict__ k_out,         // [B, T, NU]
    float* __restrict__ v0_out         // [B, NX] or nullptr
);

void launch_backward_pass_batched(
    const float* A,
    const float* B,
    const float* l_xx,
    const float* l_uu,
    const float* l_xu,
    const float* l_x,
    const float* l_u,
    const float* l_xxN,
    const float* l_xN,
    const bool* tight_mask,
    float reg_eps,
    int batch_size,
    int T,
    float* K_out,
    float* k_out,
    float* v0_out,
    cudaStream_t stream
);

/**
 * PNQP-aware batched backward pass.
 *
 * Identical to launch_backward_pass_batched but additionally takes the
 * current control trajectory U (so per-step bounds are u_min - U[t] /
 * u_max - U[t]) and emits a free_mask. Each Riccati step solves the box-QP
 *   min 0.5 du^T Q_uu du + q_u^T du   s.t. u_min - u_t <= du <= u_max - u_t
 * via pnqp4, then K_t / k_t are produced from the free-set restricted system.
 *
 * Pass nullptr for u_min/u_max to disable bounds (falls back to plain Riccati).
 *
 * Outputs:
 *   K_out, k_out, v0_out: same as the unconstrained version
 *   free_mask_out [B, T, NU]: 1.0 for free dims, 0.0 for clamped at the optimum
 */
void launch_backward_pass_batched_pnqp(
    const float* A,
    const float* B,
    const float* l_xx,
    const float* l_uu,
    const float* l_xu,
    const float* l_x,
    const float* l_u,
    const float* l_xxN,
    const float* l_xN,
    const float* U,         // [B, T, NU] current controls
    const float* u_min,     // [NU] or nullptr
    const float* u_max,     // [NU] or nullptr
    float reg_eps,
    int batch_size,
    int T,
    float* K_out,
    float* k_out,
    float* v0_out,
    float* free_mask_out,   // [B, T, NU] or nullptr
    cudaStream_t stream
);

/**
 * Standalone batched pnqp4 launcher (used by tests/parity checks).
 *
 * Solves N independent 4x4 box-QPs in parallel.
 *   x_init: optional warm start (pass nullptr for cold start)
 */
void launch_pnqp4_batch(
    const float* H,
    const float* q,
    const float* lower,
    const float* upper,
    const float* x_init,
    float* x_out,
    float* free_mask_out,
    int N,
    cudaStream_t stream
);

/**
 * Fused implicit-diff assembly kernel.
 *
 * Consumes the Riccati-pass outputs (K, k, v0) plus the linearization and the
 * primal trajectories, runs the forward sweep dX[t+1] = A dX[t] + B dU[t],
 * dU[t] = K[t] dX[t] + k[t], and emits all gradients required by the
 * DroneMPCFunction.backward in one kernel — eliminating ~1.3 ms of Python
 * autograd overhead on B=64,T=20 problems.
 *
 *   grad_C   [B, T, N, N]   = -0.5 (d_tau outer tau + tau outer d_tau)
 *   grad_c   [B, T, N]      = -d_tau
 *   grad_x0  [B, NX]        = -v0
 *   grad_xref[B, T+1, NX]   = dX (optional, pass nullptr to skip)
 *   grad_uref[B, T,   NU]   = dU (optional, pass nullptr to skip)
 *   grad_C_f [B, N, N]      = -0.5 (d_tauN outer tauN + tauN outer d_tauN)  (optional)
 *   grad_c_f [B, N]         = -d_tauN  (optional)
 *
 * where N = NX + NU = 14.
 *
 * tau[t]   = [X[t] - x_ref[t]; U[t] - u_ref[t]]
 * d_tau[t] = [dX[t]; dU[t]]
 * tauN     = [X[T] - x_ref[T]; 0]
 * d_tauN   = [dX[T]; 0]
 */
void launch_implicit_diff_assembly(
    const float* K_seq,
    const float* k_seq,
    const float* v0,
    const float* A_lin,
    const float* B_lin,
    const float* X,
    const float* U,
    const float* x_ref,
    const float* u_ref,
    float* grad_C,
    float* grad_c,
    float* grad_x0,
    float* grad_xref,
    float* grad_uref,
    float* grad_C_final,
    float* grad_c_final,
    int batch_size,
    int T,
    cudaStream_t stream
);

} // namespace drone_kernels
