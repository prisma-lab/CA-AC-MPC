/**
 * C++ Extension for Drone ILQR Solver (10 state, 4 control)
 *
 * Implements a high-performance iterative LQR solver for Drone dynamics.
 * Uses PyTorch's ATen library for automatic GPU acceleration.
 *
 * When compiled with DRONE_ILQR_USE_CUDA_KERNELS, uses fused CUDA kernels
 * for dynamics and jacobian computation (significant speedup).
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <mutex>

// Include CUDA kernel wrappers if available
#ifdef DRONE_ILQR_USE_CUDA_KERNELS
#include "drone_cuda_wrapper.h"
#endif

// Include fused backward pass kernel if available
#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
#include "backward_pass_kernel.cuh"
#endif

// Include fused forward pass kernel if available
#ifdef DRONE_ILQR_USE_FUSED_FORWARD
#include "forward_pass_kernel.cuh"
#endif

// CUDA Graph support for B=1 inference optimization
#ifdef DRONE_ILQR_USE_CUDA_KERNELS
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

namespace cuda_graph_cache {

// Cache structure for CUDA graphs
struct GraphCache {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t capture_stream = nullptr;  // Dedicated stream for graph capture
    int64_t cached_B = -1;
    int64_t cached_T = -1;
    bool valid = false;

    // Input/output buffers for graph replay
    at::Tensor input_x;
    at::Tensor input_u;
    at::Tensor output_x_next;
    at::Tensor output_A;
    at::Tensor output_B;

    void reset() {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        // Keep the stream, it's reusable
        valid = false;
        cached_B = -1;
        cached_T = -1;
    }

    void init_stream() {
        if (capture_stream == nullptr) {
            cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
        }
    }

    ~GraphCache() {
        reset();
        if (capture_stream) {
            cudaStreamDestroy(capture_stream);
            capture_stream = nullptr;
        }
    }
};

// Global cache for linearization graph (thread-local for safety)
static thread_local GraphCache linearization_cache;
static std::mutex cache_mutex;

// Global flag to enable/disable CUDA graph capture (disabled by default due to PyTorch 2.x conflicts)
static bool g_cuda_graph_enabled = false;

// Check if we should use CUDA graphs (only for small batch sizes)
inline bool should_use_cuda_graph(int64_t B) {
    return g_cuda_graph_enabled && B <= 4;  // Only use graphs for small batches when enabled
}

} // namespace cuda_graph_cache
#endif

namespace drone_ilqr_ext {

// Constants
constexpr double MASS = 0.752;

#ifdef DRONE_ILQR_USE_CUDA_KERNELS
// Debug flag - set to true to enable debug prints
static bool g_cuda_graph_debug = false;

/**
 * Linearization with CUDA Graph optimization for small batches.
 * On first call with given dimensions, captures operations into a CUDA graph.
 * On subsequent calls with same dimensions, replays the graph (lower overhead).
 *
 * CUDA Graphs reduce kernel launch overhead by capturing a sequence of GPU operations
 * and replaying them with a single CPU-side call. This is especially beneficial for
 * small batch sizes where kernel launch overhead dominates.
 */
std::tuple<at::Tensor, at::Tensor> linearize_with_cuda_graph(
    const at::Tensor& x_flat,  // [B*T, 10]
    const at::Tensor& u_flat,  // [B*T, 4]
    double dt,
    int64_t B,
    int64_t T
) {
    using namespace cuda_graph_cache;

    const int64_t total = B * T;
    const int64_t nx = 10;
    const int64_t nu = 4;
    auto options = x_flat.options();

    if (g_cuda_graph_debug) {
        printf("[CUDA Graph] linearize_with_cuda_graph called: B=%ld, T=%ld, total=%ld\n", B, T, total);
        printf("[CUDA Graph] should_use_cuda_graph(B)=%d, is_cuda=%d, is_float=%d\n",
               should_use_cuda_graph(B), x_flat.is_cuda(),
               x_flat.scalar_type() == at::kFloat);
    }

    // For very small problems, graph overhead might not be worth it
    // Also skip if not on CUDA or not float32
    if (!should_use_cuda_graph(B) || !x_flat.is_cuda() ||
        x_flat.scalar_type() != at::kFloat) {
        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Skipping graph, using direct call\n");
        }
        // Fall back to direct call
        auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
        return std::make_tuple(A_flat, B_flat);
    }

    auto& cache = linearization_cache;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Check if the current stream is already in capture mode
    // (can happen when PyTorch uses CUDA Graphs internally with torch.compile, etc.)
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream, &capture_status);
    if (capture_status != cudaStreamCaptureStatusNone) {
        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Stream already capturing, falling back to direct call\n");
        }
        auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
        return std::make_tuple(A_flat, B_flat);
    }

    if (g_cuda_graph_debug) {
        printf("[CUDA Graph] cache.valid=%d, cached_B=%ld, cached_T=%ld\n",
               cache.valid, cache.cached_B, cache.cached_T);
    }

    // Check if we can reuse cached graph
    if (cache.valid && cache.cached_B == B && cache.cached_T == T) {
        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Reusing cached graph\n");
        }
        // Copy inputs to cached buffers (async on stream)
        cache.input_x.copy_(x_flat, /*non_blocking=*/true);
        cache.input_u.copy_(u_flat, /*non_blocking=*/true);

        // Launch cached graph on the current stream
        cudaError_t err = cudaGraphLaunch(cache.graph_exec, stream);
        if (err != cudaSuccess) {
            // Graph launch failed, fall back to direct call
            cache.reset();
            auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
            return std::make_tuple(A_flat, B_flat);
        }

        // Clone outputs (this implicitly syncs the stream for the copy)
        return std::make_tuple(cache.output_A.clone(), cache.output_B.clone());
    }

    // Need to capture new graph
    if (g_cuda_graph_debug) {
        printf("[CUDA Graph] Need to capture new graph\n");
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex);

        // Double-check after acquiring lock (another thread may have captured)
        if (cache.valid && cache.cached_B == B && cache.cached_T == T) {
            // Another thread captured while we waited, use the cached graph
            cache.input_x.copy_(x_flat, /*non_blocking=*/true);
            cache.input_u.copy_(u_flat, /*non_blocking=*/true);
            cudaGraphLaunch(cache.graph_exec, stream);
            return std::make_tuple(cache.output_A.clone(), cache.output_B.clone());
        }

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Resetting old cache and allocating buffers\n");
        }

        // Reset old cache
        cache.reset();

        // Allocate persistent buffers for this graph
        cache.input_x = at::empty({total, nx}, options);
        cache.input_u = at::empty({total, nu}, options);
        cache.output_x_next = at::empty({total, nx}, options);
        cache.output_A = at::empty({total, nx, nx}, options);
        cache.output_B = at::empty({total, nx, nu}, options);

        // Copy initial data
        cache.input_x.copy_(x_flat);
        cache.input_u.copy_(u_flat);

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Buffers allocated, running warmup\n");
        }

        // Warm up - run once before capture to ensure CUDA is ready
        // This ensures any lazy initialization is done before capture
        {
            auto [x_next_warmup, A_warmup, B_warmup] = drone_step_jacobian_cuda(
                cache.input_x, cache.input_u, dt);
            cudaStreamSynchronize(stream);
        }

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Warmup done, beginning capture\n");
        }

        // Begin capture
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            if (g_cuda_graph_debug) {
                printf("[CUDA Graph] BeginCapture failed: %s\n", cudaGetErrorString(err));
            }
            // Capture failed, fall back to direct call
            auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
            return std::make_tuple(A_flat, B_flat);
        }

        // The operations we want to capture
        auto [x_next_cap, A_cap, B_cap] = drone_step_jacobian_cuda(
            cache.input_x, cache.input_u, dt);

        // Copy results to output buffers (these copies are captured too)
        cache.output_x_next.copy_(x_next_cap, /*non_blocking=*/true);
        cache.output_A.copy_(A_cap, /*non_blocking=*/true);
        cache.output_B.copy_(B_cap, /*non_blocking=*/true);

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Operations recorded, ending capture\n");
        }

        // End capture
        err = cudaStreamEndCapture(stream, &cache.graph);
        if (err != cudaSuccess || cache.graph == nullptr) {
            if (g_cuda_graph_debug) {
                printf("[CUDA Graph] EndCapture failed: %s (graph=%p)\n",
                       cudaGetErrorString(err), (void*)cache.graph);
            }
            // Capture failed
            cache.reset();
            auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
            return std::make_tuple(A_flat, B_flat);
        }

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Capture complete, instantiating graph\n");
        }

        // Instantiate executable graph
        err = cudaGraphInstantiate(&cache.graph_exec, cache.graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            if (g_cuda_graph_debug) {
                printf("[CUDA Graph] Instantiate failed: %s\n", cudaGetErrorString(err));
            }
            // Instantiation failed
            cache.reset();
            auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
            return std::make_tuple(A_flat, B_flat);
        }

        cache.cached_B = B;
        cache.cached_T = T;
        cache.valid = true;

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Graph instantiated, launching\n");
        }

        // Now launch the graph to actually compute results
        // (the capture run doesn't execute the operations)
        err = cudaGraphLaunch(cache.graph_exec, stream);
        if (err != cudaSuccess) {
            if (g_cuda_graph_debug) {
                printf("[CUDA Graph] Launch failed: %s\n", cudaGetErrorString(err));
            }
            // Graph launch failed after successful capture - shouldn't happen
            cache.reset();
            auto [x_next, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
            return std::make_tuple(A_flat, B_flat);
        }

        if (g_cuda_graph_debug) {
            printf("[CUDA Graph] Graph successfully captured and launched!\n");
        }
    }

    // Return the results (output buffers now contain results after graph launch)
    return std::make_tuple(cache.output_A.clone(), cache.output_B.clone());
}
#endif
const double G_VAL = -9.8066;
const std::vector<double> INERTIA = {0.0025, 0.0021, 0.0043};

// --- Quaternion Utils ---

// q: [4, B], q_out: [4, B]
at::Tensor quat_mult(const at::Tensor& q1, const at::Tensor& q2) {
    auto w1 = q1.select(0, 0); auto x1 = q1.select(0, 1); auto y1 = q1.select(0, 2); auto z1 = q1.select(0, 3);
    auto w2 = q2.select(0, 0); auto x2 = q2.select(0, 1); auto y2 = q2.select(0, 2); auto z2 = q2.select(0, 3);
    
    return at::stack({
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    }, 0);
}

// w: [3, B] -> [4, B]
at::Tensor exp_map(const at::Tensor& w, double dt) {
    auto w_norm = at::norm(w, 2, 0) + 1e-8;
    auto half_angle = w_norm * (dt * 0.5);
    auto s = at::sin(half_angle);
    auto c = at::cos(half_angle);
    auto w_n = w / w_norm;
    
    return at::stack({
        c,
        w_n.select(0, 0) * s,
        w_n.select(0, 1) * s,
        w_n.select(0, 2) * s
    }, 0);
}

// q: [4, B], v: [3, B] -> [3, B]
at::Tensor rotate_quat(const at::Tensor& q, const at::Tensor& v) {
    auto B = q.size(1);
    auto options = q.options();
    
    // v_aug: [4, B]
    auto v_aug = at::cat({at::zeros({1, B}, options), v}, 0);
    
    // q_inv: [4, B] (conjugate)
    auto q_inv = at::stack({q.select(0, 0), -q.select(0, 1), -q.select(0, 2), -q.select(0, 3)}, 0);
    
    auto tmp = quat_mult(q, v_aug);
    auto res = quat_mult(tmp, q_inv);
    
    return res.slice(0, 1, 4);
}


/**
 * Drone dynamics step.
 * x: [B, 10] (p, q, v)
 * u: [B, 4] (thrust, w)
 */
at::Tensor drone_step(
    const at::Tensor& x,
    const at::Tensor& u,
    double dt
) {
    // Transpose to [10, B] for easier component access
    auto x_t = x.t(); 
    auto u_t = u.t();
    
    auto p = x_t.slice(0, 0, 3); // [3, B]
    auto q = x_t.slice(0, 3, 7); // [4, B]
    auto v = x_t.slice(0, 7, 10); // [3, B]
    
    auto fc = u_t.select(0, 0); // [B]
    auto w = u_t.slice(0, 1, 4); // [3, B]
    
    // Position
    auto new_p = p + v * dt;
    
    // Quaternion
    auto em = exp_map(w, dt);
    auto new_q = quat_mult(q, em);
    // Normalize
    new_q = new_q / (at::norm(new_q, 2, 0, true) + 1e-8);
    
    // Velocity
    // Thrust in body: [0, 0, fc]
    auto B_size = x.size(0);
    auto options = x.options();
    auto zeros = at::zeros({1, B_size}, options);
    auto thrust_vec = at::stack({zeros.squeeze(0), zeros.squeeze(0), fc}, 0); // [3, B]
    
    auto acc = rotate_quat(q, thrust_vec) / MASS;
    // Add gravity: [0, 0, -9.81]
    auto g_vec = at::tensor({0.0, 0.0, G_VAL}, options).unsqueeze(1).expand({3, B_size});
    acc = acc + g_vec;
    
    auto new_v = v + acc * dt;
    
    // Stack and transpose back to [B, 10]
    return at::cat({new_p, new_q, new_v}, 0).t();
}

/**
 * Analytical Jacobians for Drone dynamics.
 * Optimized to minimize tensor allocations and stack operations.
 */
std::tuple<at::Tensor, at::Tensor> drone_dynamics_jacobian(
    const at::Tensor& x, // [Batch, 10]
    const at::Tensor& u, // [Batch, 4]
    double dt
) {
    const int64_t batch = x.size(0);
    auto options = x.options();
    
    // Unpack - using select which returns a view (no copy)
    auto q = x.slice(1, 3, 7);     // [B, 4]
    auto qw = q.select(1, 0); auto qx = q.select(1, 1);
    auto qy = q.select(1, 2); auto qz = q.select(1, 3);
    
    auto w = u.slice(1, 1, 4);     // [B, 3]
    auto wx = w.select(1, 0); auto wy = w.select(1, 1); auto wz = w.select(1, 2);
    
    auto fc = u.select(1, 0);      // [B]
    
    // --- Initialize R and S ---
    // R: [B, 10, 10], S: [B, 10, 4]
    auto R = at::eye(10, options).unsqueeze(0).expand({batch, 10, 10}).clone();
    auto S = at::zeros({batch, 10, 4}, options);
    
    // --- 1. Position Derivatives ---
    // p_dot = v => R[:, 0:3, 7:10] += dt * I
    // We can set this efficiently
    auto dt_tens = at::tensor(dt, options);
    R.select(2, 7).select(1, 0).fill_(dt);
    R.select(2, 8).select(1, 1).fill_(dt);
    R.select(2, 9).select(1, 2).fill_(dt);
    
    // --- 2. Quaternion Derivatives ---
    // q_dot = 0.5 * q * w
    // d(q_dot)/dq = 0.5 * G(w)
    // G(w) = [[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]]
    
    auto half_dt = 0.5 * dt;
    auto hdt_wx = half_dt * wx;
    auto hdt_wy = half_dt * wy;
    auto hdt_wz = half_dt * wz;
    
    // Fill block R[:, 3:7, 3:7] += dt * 0.5 * G(w)
    // Row 0 (qw_dot): 0, -wx, -wy, -wz
    R.select(1, 3).select(1, 4).sub_(hdt_wx);
    R.select(1, 3).select(1, 5).sub_(hdt_wy);
    R.select(1, 3).select(1, 6).sub_(hdt_wz);
    
    // Row 1 (qx_dot): wx, 0, wz, -wy
    R.select(1, 4).select(1, 3).add_(hdt_wx);
    R.select(1, 4).select(1, 5).add_(hdt_wz);
    R.select(1, 4).select(1, 6).sub_(hdt_wy);
    
    // Row 2 (qy_dot): wy, -wz, 0, wx
    R.select(1, 5).select(1, 3).add_(hdt_wy);
    R.select(1, 5).select(1, 4).sub_(hdt_wz);
    R.select(1, 5).select(1, 6).add_(hdt_wx);
    
    // Row 3 (qz_dot): wz, wy, -wx, 0
    R.select(1, 6).select(1, 3).add_(hdt_wz);
    R.select(1, 6).select(1, 4).add_(hdt_wy);
    R.select(1, 6).select(1, 5).sub_(hdt_wx);
    
    // d(q_dot)/dw = 0.5 * Q(q)
    // Q = [[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]]
    // Fill block S[:, 3:7, 1:4]
    
    auto hdt_qx = half_dt * qx;
    auto hdt_qy = half_dt * qy;
    auto hdt_qz = half_dt * qz;
    auto hdt_qw = half_dt * qw;
    
    // Row 0
    S.select(1, 3).select(1, 1).sub_(hdt_qx);
    S.select(1, 3).select(1, 2).sub_(hdt_qy);
    S.select(1, 3).select(1, 3).sub_(hdt_qz);
    
    // Row 1
    S.select(1, 4).select(1, 1).add_(hdt_qw);
    S.select(1, 4).select(1, 2).sub_(hdt_qz);
    S.select(1, 4).select(1, 3).add_(hdt_qy);
    
    // Row 2
    S.select(1, 5).select(1, 1).add_(hdt_qz);
    S.select(1, 5).select(1, 2).add_(hdt_qw);
    S.select(1, 5).select(1, 3).sub_(hdt_qx);
    
    // Row 3
    S.select(1, 6).select(1, 1).sub_(hdt_qy);
    S.select(1, 6).select(1, 2).add_(hdt_qx);
    S.select(1, 6).select(1, 3).add_(hdt_qw);
    
    // --- 3. Velocity Derivatives ---
    // d(v_dot)/d(q)
    // qfcm = q * fc / m
    auto mass_inv_fc = fc / MASS;
    auto q0_f = qw * mass_inv_fc;
    auto q1_f = qx * mass_inv_fc;
    auto q2_f = qy * mass_inv_fc;
    auto q3_f = qz * mass_inv_fc;
    
    auto two_dt = 2.0 * dt;
    
    // Helper lambda for scaling
    auto s = [&](const at::Tensor& t) { return t * two_dt; };
    
    // Row 0 (vx_dot): 2q2, 2q3, 2q0, 2q1
    R.select(1, 7).select(1, 3).add_(s(q2_f));
    R.select(1, 7).select(1, 4).add_(s(q3_f));
    R.select(1, 7).select(1, 5).add_(s(q0_f));
    R.select(1, 7).select(1, 6).add_(s(q1_f));
    
    // Row 1 (vy_dot): -2q1, -2q0, 2q3, 2q2
    R.select(1, 8).select(1, 3).sub_(s(q1_f));
    R.select(1, 8).select(1, 4).sub_(s(q0_f));
    R.select(1, 8).select(1, 5).add_(s(q3_f));
    R.select(1, 8).select(1, 6).add_(s(q2_f));
    
    // Row 2 (vz_dot): 2q0, -2q1, -2q2, 2q3
    R.select(1, 9).select(1, 3).add_(s(q0_f));
    R.select(1, 9).select(1, 4).sub_(s(q1_f));
    R.select(1, 9).select(1, 5).sub_(s(q2_f));
    R.select(1, 9).select(1, 6).add_(s(q3_f));
    
    // d(v_dot)/dfc
    // R(q) * [0,0,1]
    auto dt_mass_inv = dt / MASS;
    
    // x: 2(q1q3 + q0q2)
    auto dv_dfc_x = 2.0 * (qx * qz + qw * qy);
    S.select(1, 7).select(1, 0).copy_(dv_dfc_x * dt_mass_inv);
    
    // y: 2(q2q3 - q0q1)
    auto dv_dfc_y = 2.0 * (qy * qz - qw * qx);
    S.select(1, 8).select(1, 0).copy_(dv_dfc_y * dt_mass_inv);
    
    // z: q0^2 - q1^2 - q2^2 + q3^2
    auto dv_dfc_z = (qw * qw - qx * qx - qy * qy + qz * qz);
    S.select(1, 9).select(1, 0).copy_(dv_dfc_z * dt_mass_inv);
    
    return std::make_tuple(R, S);
}

/**
 * Rollout trajectory
 */
at::Tensor rollout(
    const at::Tensor& x0,      // [B, 10]
    const at::Tensor& U,       // [B, T, 4]
    double dt,
    const at::Tensor& u_min,   // [4] or empty
    const at::Tensor& u_max    // [4] or empty
) {
    const int64_t B = x0.size(0);
    const int64_t T = U.size(1);
    const int64_t nx = 10;

    bool has_bounds = u_min.numel() > 0 && u_max.numel() > 0;

    // Apply control bounds if needed
    at::Tensor U_clamped = U;
    if (has_bounds) {
        U_clamped = at::clamp(U, u_min, u_max);
    }

#ifdef DRONE_ILQR_USE_CUDA_KERNELS
    // Use fused CUDA kernel for batched rollout (faster)
    if (x0.is_cuda() && x0.scalar_type() == at::kFloat) {
        return batched_rollout_cuda(x0.contiguous(), U_clamped.contiguous(), dt);
    }
#endif

    // Fallback to ATen loop
    auto X = at::empty({B, T + 1, nx}, x0.options());
    X.select(1, 0).copy_(x0);

    auto x_curr = x0;
    for (int64_t t = 0; t < T; ++t) {
        auto u_t = U_clamped.select(1, t);
        x_curr = drone_step(x_curr, u_t, dt);
        X.select(1, t + 1).copy_(x_curr);
    }

    return X;
}

/**
 * Compute total cost.
 */
at::Tensor compute_cost(
    const at::Tensor& X,       // [B, T+1, nx]
    const at::Tensor& U,       // [B, T, nu]
    const at::Tensor& x_ref,   // [B, T+1, nx] (broadcasted)
    const at::Tensor& u_ref,   // [B, T, nu] (broadcasted)
    const at::Tensor& C,       // [B, T, n, n] (broadcasted)
    const at::Tensor& c,       // [B, T, n] (broadcasted)
    const at::Tensor& C_final, // [B, n, n] (broadcasted)
    const at::Tensor& c_final  // [B, n] (broadcasted)
) {
    const int64_t nx = 10;
    const int64_t nu = 4;
    const int64_t T = U.size(1);

    auto X_run = X.slice(1, 0, T).reshape({-1, nx});
    auto U_run = U.reshape({-1, nu});
    auto x_ref_run = x_ref.slice(1, 0, T).reshape({-1, nx});
    auto u_ref_run = u_ref.slice(1, 0, T).reshape({-1, nu});
    
    auto z_run = at::cat({X_run - x_ref_run, U_run - u_ref_run}, -1).unsqueeze(-1); // [B*T, n, 1]
    
    auto C_run = C.reshape({-1, nx+nu, nx+nu});
    auto c_run = c.reshape({-1, nx+nu}).unsqueeze(-1);
    
    auto z_T = z_run.transpose(-1, -2);
    auto quad = 0.5 * at::matmul(z_T, at::matmul(C_run, z_run)).squeeze(-1).squeeze(-1);
    auto lin = at::matmul(z_T, c_run).squeeze(-1).squeeze(-1);
    
    auto cost_run = (quad + lin).view({-1, T}).sum(1);
    
    // Terminal
    auto err_xN = (X.select(1, T) - x_ref.select(1, T)); // [B, nx]
    auto z_N = at::cat({err_xN, at::zeros_like(U.select(1, 0))}, -1).unsqueeze(-1);
    
    auto z_N_T = z_N.transpose(-1, -2);
    auto quad_N = 0.5 * at::matmul(z_N_T, at::matmul(C_final, z_N)).squeeze(-1).squeeze(-1);
    auto lin_N = at::matmul(z_N_T, c_final.unsqueeze(-1)).squeeze(-1).squeeze(-1);
    
    return cost_run + quad_N + lin_N;
}

/**
 * Full Drone ILQR solver.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool>
drone_ilqr_solve(
    const at::Tensor& x0,
    const at::Tensor& U_init,
    const at::Tensor& C,
    const at::Tensor& c,
    const at::Tensor& C_final,
    const at::Tensor& c_final,
    const at::Tensor& x_ref,
    const at::Tensor& u_ref,
    const at::Tensor& u_min,
    const at::Tensor& u_max,
    double dt,
    int64_t max_iter,
    double reg_eps,
    double converge_tol
) {
    // Disable autograd for inference - reduces dispatch overhead
    at::NoGradGuard no_grad;

    const int64_t B = x0.size(0);
    const int64_t T = U_init.size(1);
    const int64_t nx = 10;
    const int64_t nu = 4;
    const int64_t n = nx + nu;

    auto options = x0.options();
    bool has_bounds = u_min.numel() > 0 && u_max.numel() > 0;

    // Broadcasting
    auto x_ref_b = x_ref.size(0) == 1 ? x_ref.expand({B, -1, -1}) : x_ref;
    auto u_ref_b = u_ref.size(0) == 1 ? u_ref.expand({B, -1, -1}) : u_ref;
    
    // Expand time dimension if constant reference [B, 1, nx] -> [B, T+1, nx]
    if (x_ref_b.size(1) == 1) {
        x_ref_b = x_ref_b.expand({-1, T + 1, -1});
    }
    if (u_ref_b.size(1) == 1) {
        u_ref_b = u_ref_b.expand({-1, T, -1});
    }
    
    auto C_b = C.dim() == 3 ? C.unsqueeze(0).expand({B, -1, -1, -1}) : C;
    auto c_b = c.dim() == 2 ? c.unsqueeze(0).expand({B, -1, -1}) : c;
    auto C_f_b = C_final.dim() == 2 ? C_final.unsqueeze(0).expand({B, -1, -1}) : C_final;
    auto c_f_b = c_final.dim() == 1 ? c_final.unsqueeze(0).expand({B, -1}) : c_final;

    // Init
    auto U = U_init.clone();
    if (has_bounds) {
        U = at::clamp(U, u_min, u_max);
    }
    auto X = rollout(x0, U, dt, u_min, u_max);
    auto best_cost = compute_cost(X, U, x_ref_b, u_ref_b, C_b, c_b, C_f_b, c_f_b);

    auto eye_nu = at::eye(nu, options).unsqueeze(0).expand({B, nu, nu}) * reg_eps;
    bool converged = false;

    // Buffers
    auto l_x = at::empty({B, T, nx}, options);
    auto l_u = at::empty({B, T, nu}, options);
    auto l_xx = at::empty({B, T, nx, nx}, options);
    auto l_xu = at::empty({B, T, nx, nu}, options);
    auto l_uu = at::empty({B, T, nu, nu}, options);
    at::Tensor l_xxN, l_xN;
    
    auto K_seq = at::empty({B, T, nu, nx}, options);
    auto k_seq = at::empty({B, T, nu}, options);
    // PNQP free_mask buffer: filled by the PNQP-aware backward kernel each
    // iteration. The final iteration's value is the authoritative active set.
    auto free_mask_f = at::empty({B, T, nu}, options);
    
    std::vector<double> alphas_vec = {1.0, 0.5, 0.25, 0.1};
    const int64_t num_alphas = static_cast<int64_t>(alphas_vec.size());
    auto alphas_t = at::tensor(alphas_vec, options).view({1, num_alphas, 1}); // [1, num_alphas, 1]

    auto V = at::empty({B, nx, nx}, options);
    auto v = at::empty({B, nx}, options);

    // Convergence is checked device-side; the CPU readback happens every
    // CONVERGE_CHECK_PERIOD iters to amortize the cudaStreamSynchronize cost
    // that .item() forces.
    constexpr int64_t CONVERGE_CHECK_PERIOD = 4;

    // Pre-allocate buffers for backward pass (avoid allocations inside loop)
    auto V_A = at::empty({B, nx, nx}, options);
    auto V_B = at::empty({B, nx, nu}, options);
    auto Q_xx = at::empty({B, nx, nx}, options);
    auto Q_xu = at::empty({B, nx, nu}, options);
    auto Q_uu = at::empty({B, nu, nu}, options);
    auto Q_uu_reg = at::empty({B, nu, nu}, options);  // Not used anymore but keeping for potential future use
    auto q_x = at::empty({B, nx}, options);
    auto q_u = at::empty({B, nu}, options);
    auto K_t = at::empty({B, nu, nx}, options);
    auto k_t = at::empty({B, nu}, options);
    auto V_new = at::empty({B, nx, nx}, options);
    auto v_new = at::empty({B, nx}, options);
    auto temp_nu_nx = at::empty({B, nu, nx}, options);  // For Q_uu @ K
    auto temp_nu = at::empty({B, nu}, options);  // For intermediate v computation

    at::Tensor A_lin, Bm_lin;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        // 1. Linearize (Batched)
        auto x_flat = X.slice(1, 0, T).reshape({-1, nx}).contiguous();
        auto u_flat = U.reshape({-1, nu}).contiguous();

#ifdef DRONE_ILQR_USE_CUDA_KERNELS
        // Use CUDA Graph-accelerated linearization for small batches (reduces kernel launch overhead)
        if (x_flat.is_cuda() && x_flat.scalar_type() == at::kFloat) {
            if (cuda_graph_cache::should_use_cuda_graph(B)) {
                // Use CUDA Graph cached version for B<=4
                auto [A_flat, B_flat] = linearize_with_cuda_graph(x_flat, u_flat, dt, B, T);
                A_lin = A_flat.view({B, T, nx, nx});
                Bm_lin = B_flat.view({B, T, nx, nu});
            } else {
                // Direct CUDA kernel call for larger batches
                auto [x_next_unused, A_flat, B_flat] = drone_step_jacobian_cuda(x_flat, u_flat, dt);
                A_lin = A_flat.view({B, T, nx, nx});
                Bm_lin = B_flat.view({B, T, nx, nu});
            }
        } else {
            auto [A_flat, B_flat] = drone_dynamics_jacobian(x_flat, u_flat, dt);
            A_lin = A_flat.view({B, T, nx, nx});
            Bm_lin = B_flat.view({B, T, nx, nu});
        }
#else
        auto [A_flat, B_flat] = drone_dynamics_jacobian(x_flat, u_flat, dt);
        A_lin = A_flat.view({B, T, nx, nx});
        Bm_lin = B_flat.view({B, T, nx, nu});
#endif

        // 2. Quadraticize Cost
        auto C_b_flat = C_b.reshape({-1, n, n});
        auto c_b_flat = c_b.reshape({-1, n});
        
        auto l_xx_flat = C_b_flat.slice(1, 0, nx).slice(2, 0, nx);
        auto l_uu_flat = C_b_flat.slice(1, nx, n).slice(2, nx, n);
        auto l_xu_flat = C_b_flat.slice(1, 0, nx).slice(2, nx, n);
        
        auto err_x_flat = (X.slice(1, 0, T) - x_ref_b.slice(1, 0, T)).reshape({-1, nx});
        auto err_u_flat = (U - u_ref_b).reshape({-1, nu});
        
        auto l_x_flat = at::matmul(l_xx_flat, err_x_flat.unsqueeze(-1)).squeeze(-1) + 
                        at::matmul(l_xu_flat, err_u_flat.unsqueeze(-1)).squeeze(-1) + 
                        c_b_flat.slice(1, 0, nx);
                        
        auto l_u_flat = at::matmul(l_xu_flat.transpose(-1, -2), err_x_flat.unsqueeze(-1)).squeeze(-1) + 
                        at::matmul(l_uu_flat, err_u_flat.unsqueeze(-1)).squeeze(-1) + 
                        c_b_flat.slice(1, nx, n);
                        
        l_xx.copy_(l_xx_flat.view({B, T, nx, nx}));
        l_uu.copy_(l_uu_flat.view({B, T, nu, nu}));
        l_xu.copy_(l_xu_flat.view({B, T, nx, nu}));
        l_x.copy_(l_x_flat.view({B, T, nx}));
        l_u.copy_(l_u_flat.view({B, T, nu}));
        
        auto err_xN = X.select(1, T) - x_ref_b.select(1, T);
        l_xxN = C_f_b.slice(1, 0, nx).slice(2, 0, nx);
        l_xN = at::matmul(l_xxN, err_xN.unsqueeze(-1)).squeeze(-1) + c_f_b.slice(1, 0, nx);
        
        // 3. Backward Pass
#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
        if (x0.is_cuda() && x0.scalar_type() == at::kFloat) {
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#ifdef DRONE_ILQR_USE_PNQP
            if (has_bounds) {
                // PNQP-aware Riccati: solves the per-step box-QP with pnqp4
                // and writes the active-set free_mask in float form.
                drone_kernels::launch_backward_pass_batched_pnqp(
                    A_lin.data_ptr<float>(),
                    Bm_lin.data_ptr<float>(),
                    l_xx.data_ptr<float>(),
                    l_uu.data_ptr<float>(),
                    l_xu.data_ptr<float>(),
                    l_x.data_ptr<float>(),
                    l_u.data_ptr<float>(),
                    l_xxN.data_ptr<float>(),
                    l_xN.data_ptr<float>(),
                    U.data_ptr<float>(),
                    u_min.data_ptr<float>(),
                    u_max.data_ptr<float>(),
                    static_cast<float>(reg_eps),
                    B, T,
                    K_seq.data_ptr<float>(),
                    k_seq.data_ptr<float>(),
                    nullptr,
                    free_mask_f.data_ptr<float>(),
                    stream
                );
            } else
#endif
            {
                drone_kernels::launch_backward_pass_batched(
                    A_lin.data_ptr<float>(),
                    Bm_lin.data_ptr<float>(),
                    l_xx.data_ptr<float>(),
                    l_uu.data_ptr<float>(),
                    l_xu.data_ptr<float>(),
                    l_x.data_ptr<float>(),
                    l_u.data_ptr<float>(),
                    l_xxN.data_ptr<float>(),
                    l_xN.data_ptr<float>(),
                    nullptr,
                    static_cast<float>(reg_eps),
                    B, T,
                    K_seq.data_ptr<float>(),
                    k_seq.data_ptr<float>(),
                    nullptr,
                    stream
                );
            }
        } else {
            // Fallback to ATen implementation for non-CUDA or non-float32
#endif
        V.copy_(l_xxN);
        v.copy_(l_xN);

        // Precompute transposed Jacobians to avoid repeated transpose calls
        auto A_lin_T = A_lin.transpose(-1, -2).contiguous();  // [B, T, nx, nx]
        auto B_lin_T = Bm_lin.transpose(-1, -2).contiguous(); // [B, T, nu, nx]

        for (int64_t t = T - 1; t >= 0; --t) {
            auto A_t = A_lin.select(1, t);
            auto B_t = Bm_lin.select(1, t);
            auto A_t_T = A_lin_T.select(1, t);
            auto B_t_T = B_lin_T.select(1, t);

            // Use pre-allocated buffers with matmul_out where possible
            at::matmul_out(V_A, V, A_t);
            at::matmul_out(V_B, V, B_t);

            // Q_xx = l_xx[t] + A_t^T @ V_A
            at::matmul_out(Q_xx, A_t_T, V_A);
            Q_xx.add_(l_xx.select(1, t));

            // Q_xu = l_xu[t] + A_t^T @ V_B
            at::matmul_out(Q_xu, A_t_T, V_B);
            Q_xu.add_(l_xu.select(1, t));

            auto Q_ux = Q_xu.transpose(-1, -2);  // This is a view, no allocation

            // Q_uu = l_uu[t] + B_t^T @ V_B + eye_nu
            at::matmul_out(Q_uu, B_t_T, V_B);
            Q_uu.add_(l_uu.select(1, t));
            Q_uu.add_(eye_nu);

            // q_x = l_x[t] + A_t^T @ v
            auto v_col = v.unsqueeze(-1);
            q_x.copy_(l_x.select(1, t));
            q_x.add_(at::matmul(A_t_T, v_col).squeeze(-1));

            // q_u = l_u[t] + B_t^T @ v
            q_u.copy_(l_u.select(1, t));
            q_u.add_(at::matmul(B_t_T, v_col).squeeze(-1));

            // Solve K_t = -Q_uu^{-1} @ Q_ux and k_t = -Q_uu^{-1} @ q_u
            // Note: For small matrices (4x4), LU is actually faster than Cholesky on GPU
            // due to lower overhead. Cholesky's theoretical advantage only applies to larger matrices.
            K_t = -at::linalg_solve(Q_uu, Q_ux);
            k_t = -at::linalg_solve(Q_uu, q_u.unsqueeze(-1)).squeeze(-1);

            K_seq.select(1, t).copy_(K_t);
            k_seq.select(1, t).copy_(k_t);

            // Update V and v for next iteration (simplified formula, numerically stable)
            // V_new = Q_xx + Q_xu @ K
            V_new.copy_(Q_xx);
            V_new.add_(at::matmul(Q_xu, K_t));
            V.copy_(V_new);

            // v_new = q_x + Q_xu @ k
            auto k_col = k_t.unsqueeze(-1);
            v_new.copy_(q_x);
            v_new.add_(at::matmul(Q_xu, k_col).squeeze(-1));
            v.copy_(v_new);
        }
#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
        }
#endif
        
        // 4. Forward Pass (Batched Line Search)
        // The fused CUDA kernel reads non-expanded inputs and indexes via
        // b = idx / num_alphas, so we avoid the costly expand+reshape for the
        // common (CUDA float) path.
        auto X_cand = at::empty({B * num_alphas, T + 1, nx}, options);
        auto U_cand = at::empty({B * num_alphas, T, nu}, options);

#ifdef DRONE_ILQR_USE_FUSED_FORWARD
        if (x0.is_cuda() && x0.scalar_type() == at::kFloat) {
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
            auto alphas_device = at::tensor(alphas_vec, options);

            drone_kernels::launch_forward_pass_fused(
                x0.data_ptr<float>(),
                X.data_ptr<float>(),
                U.data_ptr<float>(),
                K_seq.data_ptr<float>(),
                k_seq.data_ptr<float>(),
                alphas_device.data_ptr<float>(),
                static_cast<float>(dt),
                B, num_alphas, T,
                has_bounds,
                has_bounds ? u_min.data_ptr<float>() : nullptr,
                has_bounds ? u_max.data_ptr<float>() : nullptr,
                X_cand.data_ptr<float>(),
                U_cand.data_ptr<float>(),
                stream
            );
        } else
#endif
        {
            // ATen fallback: only here do we materialize the expansion.
            auto x0_exp = x0.unsqueeze(1).expand({B, num_alphas, nx}).reshape({B * num_alphas, nx});
            auto X_exp = X.unsqueeze(1).expand({B, num_alphas, T + 1, nx}).reshape({B * num_alphas, T + 1, nx});
            auto U_exp = U.unsqueeze(1).expand({B, num_alphas, T, nu}).reshape({B * num_alphas, T, nu});
            auto K_exp = K_seq.unsqueeze(1).expand({B, num_alphas, T, nu, nx}).reshape({B * num_alphas, T, nu, nx});
            auto k_exp = k_seq.unsqueeze(1).expand({B, num_alphas, T, nu}).reshape({B * num_alphas, T, nu});
            auto alphas_batch = alphas_t.expand({B, num_alphas, 1}).reshape({B * num_alphas, 1});

            auto x_curr_exp = x0_exp.clone();
            X_cand.select(1, 0).copy_(x_curr_exp);
            for (int64_t t = 0; t < T; ++t) {
                auto dx = x_curr_exp - X_exp.select(1, t);
                auto du = alphas_batch * k_exp.select(1, t)
                          + at::matmul(K_exp.select(1, t), dx.unsqueeze(-1)).squeeze(-1);
                auto u_t = U_exp.select(1, t) + du;
                if (has_bounds) u_t = at::clamp(u_t, u_min, u_max);
                U_cand.select(1, t).copy_(u_t);
                x_curr_exp = drone_step(x_curr_exp, u_t, dt);
                X_cand.select(1, t + 1).copy_(x_curr_exp);
            }
        }

        // Cost computation broadcast: avoid materializing num_alphas-wide refs/cost matrices.
        auto Xv = X_cand.view({B, num_alphas, T + 1, nx});
        auto Uv = U_cand.view({B, num_alphas, T, nu});
        auto err_x = Xv.slice(2, 0, T) - x_ref_b.slice(1, 0, T).unsqueeze(1);  // [B,A,T,nx]
        auto err_u = Uv - u_ref_b.unsqueeze(1);                                  // [B,A,T,nu]
        auto z_run = at::cat({err_x, err_u}, -1);                                // [B,A,T,n]
        auto C_bcast = C_b.unsqueeze(1);                                         // [B,1,T,n,n]
        auto Cz = at::matmul(C_bcast, z_run.unsqueeze(-1)).squeeze(-1);          // [B,A,T,n]
        auto cost_run = ((0.5f * (z_run * Cz)).sum(-1) + (z_run * c_b.unsqueeze(1)).sum(-1)).sum(-1);

        auto err_xN_bA = Xv.select(2, T) - x_ref_b.select(1, T).unsqueeze(1);    // [B,A,nx]
        auto z_N = at::cat({err_xN_bA, at::zeros({B, num_alphas, nu}, options)}, -1);
        auto CfzN = at::matmul(C_f_b.unsqueeze(1), z_N.unsqueeze(-1)).squeeze(-1);
        auto cost_term = (0.5f * (z_N * CfzN)).sum(-1) + (z_N * c_f_b.unsqueeze(1)).sum(-1);
        auto costs_reshaped = cost_run + cost_term;                              // [B, num_alphas]

        // Find best alpha for each batch element
        auto [best_costs_new, best_idx] = costs_reshaped.min(1);  // [B], [B]

        // Reshape candidates back to [B, num_alphas, ...]
        auto X_cand_reshaped = X_cand.view({B, num_alphas, T + 1, nx});
        auto U_cand_reshaped = U_cand.view({B, num_alphas, T, nu});

        // Gather the best trajectory for each batch
        auto best_idx_X = best_idx.view({B, 1, 1, 1}).expand({B, 1, T + 1, nx});
        auto best_idx_U = best_idx.view({B, 1, 1, 1}).expand({B, 1, T, nu});

        auto X_best = X_cand_reshaped.gather(1, best_idx_X).squeeze(1);  // [B, T+1, nx]
        auto U_best = U_cand_reshaped.gather(1, best_idx_U).squeeze(1);  // [B, T, nu]

        // Only update where improvement occurred
        auto improved = best_costs_new < best_cost;
        auto imp_mask_X = improved.view({B, 1, 1}).expand_as(X_best);
        auto imp_mask_U = improved.view({B, 1, 1}).expand_as(U_best);

        auto X_new = at::where(imp_mask_X, X_best, X);
        auto U_new = at::where(imp_mask_U, U_best, U);
        auto best_alpha_cost = at::where(improved, best_costs_new, best_cost);
        
        auto improvement = at::abs(best_cost - best_alpha_cost).max();
        X = X_new;
        U = U_new;
        best_cost = best_alpha_cost;

        // Amortize the GPU->CPU sync: only read back every CONVERGE_CHECK_PERIOD iters.
        if ((iter % CONVERGE_CHECK_PERIOD) == (CONVERGE_CHECK_PERIOD - 1)
            || iter == max_iter - 1) {
            if (improvement.item<double>() < converge_tol) {
                converged = true;
                break;
            }
        }
    }
    
    // tight_mask comes directly from the PNQP-aware backward kernel: 1.0 in
    // free_mask_f means "free", so tight = (free_mask_f < 0.5). When bounds
    // are absent we never populated free_mask_f, so emit an empty bool tensor
    // (semantically equivalent to "no clamped dimensions").
    at::Tensor tight_mask;
    if (has_bounds) {
        tight_mask = (free_mask_f < 0.5f);
    } else {
        tight_mask = at::zeros_like(U, at::kBool);
    }

    return std::make_tuple(X, U, l_xx, l_uu, l_xu, l_xxN, A_lin, Bm_lin, tight_mask, converged);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
drone_backward_pass_batched(
    const at::Tensor& A_lin,      // [B, T, nx, nx]
    const at::Tensor& Bm_lin,     // [B, T, nx, nu]
    const at::Tensor& l_xx,       // [B, T, nx, nx]
    const at::Tensor& l_uu,       // [B, T, nu, nu]
    const at::Tensor& l_xu,       // [B, T, nx, nu]
    const at::Tensor& l_x,        // [B, T, nx]
    const at::Tensor& l_u,        // [B, T, nu]
    const at::Tensor& l_xxN,      // [B, nx, nx]
    const at::Tensor& l_xN,       // [B, nx]
    const at::Tensor& tight_mask, // [B, T, nu] bool, optional/empty allowed
    double reg_eps
) {
    TORCH_CHECK(A_lin.dim() == 4, "A_lin must be [B,T,nx,nx]");
    TORCH_CHECK(Bm_lin.dim() == 4, "Bm_lin must be [B,T,nx,nu]");
    TORCH_CHECK(l_xx.dim() == 4, "l_xx must be [B,T,nx,nx]");
    TORCH_CHECK(l_uu.dim() == 4, "l_uu must be [B,T,nu,nu]");
    TORCH_CHECK(l_xu.dim() == 4, "l_xu must be [B,T,nx,nu]");
    TORCH_CHECK(l_x.dim() == 3, "l_x must be [B,T,nx]");
    TORCH_CHECK(l_u.dim() == 3, "l_u must be [B,T,nu]");
    TORCH_CHECK(l_xxN.dim() == 3, "l_xxN must be [B,nx,nx]");
    TORCH_CHECK(l_xN.dim() == 2, "l_xN must be [B,nx]");

    const auto B = A_lin.size(0);
    const auto T = A_lin.size(1);
    const auto nx = A_lin.size(2);
    const auto nu = Bm_lin.size(3);
    auto options = A_lin.options();

    auto K_seq = at::zeros({B, T, nu, nx}, options);
    auto k_seq = at::zeros({B, T, nu}, options);
    auto v0 = at::zeros({B, nx}, options);

#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
    if (A_lin.is_cuda() && A_lin.scalar_type() == at::kFloat) {
        const bool use_mask = tight_mask.defined() && tight_mask.numel() > 0;
        const bool* mask_ptr = use_mask ? tight_mask.contiguous().data_ptr<bool>() : nullptr;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        drone_kernels::launch_backward_pass_batched(
            A_lin.contiguous().data_ptr<float>(),
            Bm_lin.contiguous().data_ptr<float>(),
            l_xx.contiguous().data_ptr<float>(),
            l_uu.contiguous().data_ptr<float>(),
            l_xu.contiguous().data_ptr<float>(),
            l_x.contiguous().data_ptr<float>(),
            l_u.contiguous().data_ptr<float>(),
            l_xxN.contiguous().data_ptr<float>(),
            l_xN.contiguous().data_ptr<float>(),
            mask_ptr,
            static_cast<float>(reg_eps),
            B,
            T,
            K_seq.data_ptr<float>(),
            k_seq.data_ptr<float>(),
            v0.data_ptr<float>(),
            stream
        );
        return std::make_tuple(K_seq, k_seq, v0);
    }
#endif

    // Fallback ATen implementation (CPU or non-float CUDA).
    auto V = l_xxN.clone(); // [B,nx,nx]
    auto v = l_xN.clone();  // [B,nx]
    auto eye_nu = at::eye(nu, options).unsqueeze(0).expand({B, nu, nu}) * reg_eps;

    for (int64_t t = T - 1; t >= 0; --t) {
        auto A_t = A_lin.select(1, t);   // [B,nx,nx]
        auto B_t = Bm_lin.select(1, t);  // [B,nx,nu]

        auto V_A = at::matmul(V, A_t);
        auto V_B = at::matmul(V, B_t);

        auto Q_xx = l_xx.select(1, t) + at::matmul(A_t.transpose(-1, -2), V_A);
        auto Q_xu = l_xu.select(1, t) + at::matmul(A_t.transpose(-1, -2), V_B);
        auto Q_uu = l_uu.select(1, t) + at::matmul(B_t.transpose(-1, -2), V_B) + eye_nu;

        auto q_x = l_x.select(1, t) + at::matmul(A_t.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1);
        auto q_u = l_u.select(1, t) + at::matmul(B_t.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1);

        if (tight_mask.defined() && tight_mask.numel() > 0) {
            auto I_tight = tight_mask.select(1, t).to(at::kBool);  // [B,nu]
            auto free = (~I_tight).to(options.dtype());
            auto free_outer = free.unsqueeze(-1) * free.unsqueeze(-2);
            Q_uu = Q_uu * free_outer
                   + at::diag_embed(I_tight.to(options.dtype()) * 1e-11) + eye_nu;
            Q_xu = Q_xu * free.unsqueeze(-2);
            q_u = q_u * free;
        }

        auto Q_ux = Q_xu.transpose(-1, -2);
        auto K_t = -at::linalg_solve(Q_uu, Q_ux);
        auto k_t = -at::linalg_solve(Q_uu, q_u.unsqueeze(-1)).squeeze(-1);

        K_seq.select(1, t).copy_(K_t);
        k_seq.select(1, t).copy_(k_t);

        auto K_t_T = K_t.transpose(-1, -2);
        V = Q_xx
            + at::matmul(K_t_T, at::matmul(Q_uu, K_t))
            + at::matmul(K_t_T, Q_ux)
            + at::matmul(Q_xu, K_t);
        auto q_u_aug = at::matmul(Q_uu, k_t.unsqueeze(-1)).squeeze(-1) + q_u;
        v = q_x
            + at::matmul(K_t_T, q_u_aug.unsqueeze(-1)).squeeze(-1)
            + at::matmul(Q_xu, k_t.unsqueeze(-1)).squeeze(-1);
    }

    v0.copy_(v);
    return std::make_tuple(K_seq, k_seq, v0);
}

#ifdef DRONE_ILQR_USE_CUDA_KERNELS
/**
 * Clear the CUDA graph cache. Useful for benchmarking or when changing problem dimensions.
 */
void clear_cuda_graph_cache() {
    cuda_graph_cache::linearization_cache.reset();
}

/**
 * Check if CUDA graphs are enabled and get cache status.
 */
std::tuple<bool, int64_t, int64_t> get_cuda_graph_status() {
    auto& cache = cuda_graph_cache::linearization_cache;
    return std::make_tuple(cache.valid, cache.cached_B, cache.cached_T);
}

/**
 * Enable or disable debug mode for CUDA graphs.
 */
void set_cuda_graph_debug(bool enable) {
    g_cuda_graph_debug = enable;
}

/**
 * Enable or disable CUDA graph capture (disabled by default due to PyTorch 2.x conflicts).
 */
void enable_cuda_graphs(bool enable) {
    cuda_graph_cache::g_cuda_graph_enabled = enable;
    if (!enable) {
        cuda_graph_cache::linearization_cache.reset();
    }
}
#endif

#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
// Fused implicit-diff entry point: takes the Riccati outputs (K, k, v0) plus
// the linearization and primal trajectories, runs the forward sweep AND the
// gradient assembly entirely on the GPU, returning the gradient tensors that
// DroneMPCFunction.backward needs to send back to autograd.
//
// Inputs are all CUDA float32. Pass `compute_xref/_uref/_C_final/_c_final` as
// false to skip the corresponding outputs (returned as empty tensors).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
drone_implicit_diff_assembly(
    const at::Tensor& K_seq,
    const at::Tensor& k_seq,
    const at::Tensor& v0,
    const at::Tensor& A_lin,
    const at::Tensor& Bm_lin,
    const at::Tensor& X,
    const at::Tensor& U,
    const at::Tensor& x_ref,
    const at::Tensor& u_ref,
    bool compute_xref,
    bool compute_uref,
    bool compute_C_final,
    bool compute_c_final
) {
    TORCH_CHECK(K_seq.is_cuda() && K_seq.scalar_type() == at::kFloat,
                "K_seq must be CUDA float32");
    const int64_t B = K_seq.size(0);
    const int64_t T = K_seq.size(1);
    const int64_t nx = 10;
    const int64_t nu = 4;
    const int64_t n  = nx + nu;
    auto opt = K_seq.options();

    auto grad_C  = at::empty({B, T, n, n}, opt);
    auto grad_c  = at::empty({B, T, n}, opt);
    auto grad_x0 = at::empty({B, nx}, opt);
    auto grad_xref   = compute_xref     ? at::empty({B, T + 1, nx}, opt) : at::Tensor();
    auto grad_uref   = compute_uref     ? at::empty({B, T, nu}, opt)     : at::Tensor();
    auto grad_C_f    = compute_C_final  ? at::empty({B, n, n}, opt)      : at::Tensor();
    auto grad_c_f    = compute_c_final  ? at::empty({B, n}, opt)         : at::Tensor();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    drone_kernels::launch_implicit_diff_assembly(
        K_seq.data_ptr<float>(),
        k_seq.data_ptr<float>(),
        v0.data_ptr<float>(),
        A_lin.data_ptr<float>(),
        Bm_lin.data_ptr<float>(),
        X.data_ptr<float>(),
        U.data_ptr<float>(),
        x_ref.data_ptr<float>(),
        u_ref.data_ptr<float>(),
        grad_C.data_ptr<float>(),
        grad_c.data_ptr<float>(),
        grad_x0.data_ptr<float>(),
        grad_xref.defined()  ? grad_xref.data_ptr<float>()  : nullptr,
        grad_uref.defined()  ? grad_uref.data_ptr<float>()  : nullptr,
        grad_C_f.defined()   ? grad_C_f.data_ptr<float>()   : nullptr,
        grad_c_f.defined()   ? grad_c_f.data_ptr<float>()   : nullptr,
        static_cast<int>(B), static_cast<int>(T),
        stream
    );
    return std::make_tuple(grad_C, grad_c, grad_x0, grad_xref, grad_uref,
                           grad_C_f, grad_c_f);
}
#endif

#ifdef DRONE_ILQR_USE_PNQP
// Test/parity entry point: solves N independent 4x4 box-QPs on the GPU and
// returns (x*, free_mask). Mirrors mpc.pytorch.pnqp's signature for parity
// testing. Invokes the device pnqp4 inline.
std::tuple<at::Tensor, at::Tensor> drone_pnqp4_batch(
    const at::Tensor& H,        // [N, 4, 4]
    const at::Tensor& q,        // [N, 4]
    const at::Tensor& lower,    // [N, 4]
    const at::Tensor& upper,    // [N, 4]
    const at::Tensor& x_init    // [N, 4] or empty
) {
    TORCH_CHECK(H.is_cuda() && H.scalar_type() == at::kFloat, "H must be CUDA float32");
    TORCH_CHECK(H.dim() == 3 && H.size(1) == 4 && H.size(2) == 4, "H must be [N,4,4]");
    const int64_t N = H.size(0);
    auto Hc = H.contiguous();
    auto qc = q.contiguous();
    auto lc = lower.contiguous();
    auto uc = upper.contiguous();
    auto x_out = at::empty({N, 4}, H.options());
    auto fm_out = at::empty({N, 4}, H.options());
    const float* x_init_ptr = nullptr;
    at::Tensor xi_c;
    if (x_init.defined() && x_init.numel() == N * 4) {
        xi_c = x_init.contiguous();
        x_init_ptr = xi_c.data_ptr<float>();
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    drone_kernels::launch_pnqp4_batch(
        Hc.data_ptr<float>(), qc.data_ptr<float>(),
        lc.data_ptr<float>(), uc.data_ptr<float>(),
        x_init_ptr,
        x_out.data_ptr<float>(), fm_out.data_ptr<float>(),
        static_cast<int>(N), stream
    );
    return std::make_tuple(x_out, fm_out);
}
#endif

} // namespace drone_ilqr_ext

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("drone_ilqr_solve", &drone_ilqr_ext::drone_ilqr_solve,
          "Drone ILQR solver",
          py::arg("x0"),
          py::arg("U_init"),
          py::arg("C"),
          py::arg("c"),
          py::arg("C_final"),
          py::arg("c_final"),
          py::arg("x_ref"),
          py::arg("u_ref"),
          py::arg("u_min"),
          py::arg("u_max"),
          py::arg("dt"),
          py::arg("max_iter"),
          py::arg("reg_eps"),
          py::arg("converge_tol"));

    m.def("drone_backward_pass_batched", &drone_ilqr_ext::drone_backward_pass_batched,
          "Adjoint backward Riccati pass (returns K_seq, k_seq, v0)",
          py::arg("A_lin"),
          py::arg("Bm_lin"),
          py::arg("l_xx"),
          py::arg("l_uu"),
          py::arg("l_xu"),
          py::arg("l_x"),
          py::arg("l_u"),
          py::arg("l_xxN"),
          py::arg("l_xN"),
          py::arg("tight_mask"),
          py::arg("reg_eps"));

#ifdef DRONE_ILQR_USE_FUSED_BACKWARD
    m.def("drone_implicit_diff_assembly", &drone_ilqr_ext::drone_implicit_diff_assembly,
          "Fused implicit-diff backward: run forward sweep + assemble grads in one kernel.",
          py::arg("K_seq"), py::arg("k_seq"), py::arg("v0"),
          py::arg("A_lin"), py::arg("Bm_lin"),
          py::arg("X"), py::arg("U"), py::arg("x_ref"), py::arg("u_ref"),
          py::arg("compute_xref") = false,
          py::arg("compute_uref") = false,
          py::arg("compute_C_final") = false,
          py::arg("compute_c_final") = false);
#endif

#ifdef DRONE_ILQR_USE_PNQP
    m.def("drone_pnqp4_batch", &drone_ilqr_ext::drone_pnqp4_batch,
          "Batched 4x4 box-QP solver (PNQP). Returns (x*, free_mask).",
          py::arg("H"), py::arg("q"), py::arg("lower"), py::arg("upper"),
          py::arg("x_init") = at::Tensor());
#endif

#ifdef DRONE_ILQR_USE_CUDA_KERNELS
    m.def("clear_cuda_graph_cache", &drone_ilqr_ext::clear_cuda_graph_cache,
          "Clear the CUDA graph cache for linearization");
    m.def("get_cuda_graph_status", &drone_ilqr_ext::get_cuda_graph_status,
          "Get CUDA graph cache status: (valid, cached_B, cached_T)");
    m.def("set_cuda_graph_debug", &drone_ilqr_ext::set_cuda_graph_debug,
          "Enable/disable debug prints for CUDA graph operations",
          py::arg("enable"));
    m.def("enable_cuda_graphs", &drone_ilqr_ext::enable_cuda_graphs,
          "Enable/disable CUDA graph capture (disabled by default due to PyTorch 2.x conflicts)",
          py::arg("enable"));
    m.attr("CUDA_GRAPHS_ENABLED") = true;
#else
    m.attr("CUDA_GRAPHS_ENABLED") = false;
#endif
}
