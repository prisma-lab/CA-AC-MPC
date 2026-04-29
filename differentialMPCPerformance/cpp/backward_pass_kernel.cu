/**
 * Fused CUDA kernels for iLQR backward pass
 */

#include "backward_pass_kernel.cuh"
#include "pnqp_kernel.cuh"
#include <stdio.h>

namespace drone_kernels {

// ============================================================================
// Helper device functions
// ============================================================================

// Matrix multiply: C[m,n] = A[m,k] @ B[k,n] (small matrices, single thread)
__device__ __forceinline__ void matmul_small(
    const float* A, const float* B, float* C,
    int m, int k, int n
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Matrix multiply with A transposed: C[m,n] = A^T[m,k] @ B[k,n]
__device__ __forceinline__ void matmul_At_B(
    const float* A, const float* B, float* C,
    int m, int k, int n  // A is [k,m], B is [k,n]
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (int l = 0; l < k; l++) {
                sum += A[l * m + i] * B[l * n + j];  // A^T[i,l] = A[l,i]
            }
            C[i * n + j] = sum;
        }
    }
}

// Matrix-vector multiply: y[m] = A[m,n] @ x[n]
__device__ __forceinline__ void matvec(
    const float* A, const float* x, float* y,
    int m, int n
) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// Transposed matrix-vector: y[m] = A^T[m,n] @ x[n], where A is [n,m]
__device__ __forceinline__ void matvec_At(
    const float* A, const float* x, float* y,
    int m, int n  // A is [n,m]
) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < n; j++) {
            sum += A[j * m + i] * x[j];  // A^T[i,j] = A[j,i]
        }
        y[i] = sum;
    }
}

// Solve 4x4 linear system: A @ x = b using Gaussian elimination with partial pivoting
// Overwrites A and b, result in b
__device__ __forceinline__ void solve4x4(float A[4][4], float b[4]) {
    // Gaussian elimination with partial pivoting
    for (int k = 0; k < 4; k++) {
        // Find pivot
        int max_row = k;
        float max_val = fabsf(A[k][k]);
        for (int i = k + 1; i < 4; i++) {
            float val = fabsf(A[i][k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // Swap rows if needed
        if (max_row != k) {
            for (int j = 0; j < 4; j++) {
                float tmp = A[k][j];
                A[k][j] = A[max_row][j];
                A[max_row][j] = tmp;
            }
            float tmp = b[k];
            b[k] = b[max_row];
            b[max_row] = tmp;
        }

        // Eliminate
        float pivot = A[k][k];
        if (fabsf(pivot) < 1e-10f) pivot = 1e-10f;  // Avoid division by zero

        for (int i = k + 1; i < 4; i++) {
            float factor = A[i][k] / pivot;
            for (int j = k + 1; j < 4; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
            A[i][k] = 0.0f;
        }
    }

    // Back substitution
    for (int i = 3; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < 4; j++) {
            sum -= A[i][j] * b[j];
        }
        b[i] = sum / A[i][i];
    }
}

// Solve A @ X = B where A is 4x4 and B is 4xn (n columns)
__device__ __forceinline__ void solve4x4_multi(float A[4][4], float B[4][10], int n) {
    // Gaussian elimination with partial pivoting
    for (int k = 0; k < 4; k++) {
        int max_row = k;
        float max_val = fabsf(A[k][k]);
        for (int i = k + 1; i < 4; i++) {
            float val = fabsf(A[i][k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        if (max_row != k) {
            for (int j = 0; j < 4; j++) {
                float tmp = A[k][j];
                A[k][j] = A[max_row][j];
                A[max_row][j] = tmp;
            }
            for (int j = 0; j < n; j++) {
                float tmp = B[k][j];
                B[k][j] = B[max_row][j];
                B[max_row][j] = tmp;
            }
        }

        float pivot = A[k][k];
        if (fabsf(pivot) < 1e-10f) pivot = 1e-10f;

        for (int i = k + 1; i < 4; i++) {
            float factor = A[i][k] / pivot;
            for (int j = k + 1; j < 4; j++) {
                A[i][j] -= factor * A[k][j];
            }
            for (int j = 0; j < n; j++) {
                B[i][j] -= factor * B[k][j];
            }
            A[i][k] = 0.0f;
        }
    }

    // Back substitution
    for (int i = 3; i >= 0; i--) {
        for (int j = 0; j < n; j++) {
            float sum = B[i][j];
            for (int l = i + 1; l < 4; l++) {
                sum -= A[i][l] * B[l][j];
            }
            B[i][j] = sum / A[i][i];
        }
    }
}

// ============================================================================
// Fused backward pass kernel for B=1
// ============================================================================

__global__ __launch_bounds__(32, 2) void backward_pass_fused_kernel_b1(
    const float* __restrict__ A_in,
    const float* __restrict__ B_in,
    const float* __restrict__ l_xx_in,
    const float* __restrict__ l_uu_in,
    const float* __restrict__ l_xu_in,
    const float* __restrict__ l_x_in,
    const float* __restrict__ l_u_in,
    const float* __restrict__ l_xxN_in,
    const float* __restrict__ l_xN_in,
    const bool* __restrict__ tight_mask_in,
    float reg_eps,
    int T,
    float* __restrict__ K_out,
    float* __restrict__ k_out,
    float* __restrict__ v0_out
) {
    // Use shared memory for V and v (they're updated each iteration)
    __shared__ float V[NX * NX];  // [10, 10] = 100 floats
    __shared__ float v[NX];       // [10] = 10 floats

    // Thread-local temporary storage
    float V_A[NX * NX];   // V @ A_t
    float V_B[NX * NU];   // V @ B_t
    float Q_xx[NX * NX];
    float Q_xu[NX * NU];
    float Q_uu[NU][NU];   // Keep as 2D for solve
    float q_x[NX];
    float q_u[NU];
    float K_t[NU][NX];    // Keep as 2D for solve
    float k_t[NU];

    // Local copies of A_t, B_t
    float A_t[NX * NX];
    float B_t[NX * NU];

    // Initialize V = l_xxN, v = l_xN
    if (threadIdx.x == 0) {
        for (int i = 0; i < NX * NX; i++) V[i] = l_xxN_in[i];
        for (int i = 0; i < NX; i++) v[i] = l_xN_in[i];
    }
    __syncthreads();

    // Process backward: t = T-1, T-2, ..., 0
    for (int t = T - 1; t >= 0; t--) {
        if (threadIdx.x == 0) {
            // Load A_t and B_t
            int offset_A = t * NX * NX;
            int offset_B = t * NX * NU;
            for (int i = 0; i < NX * NX; i++) A_t[i] = A_in[offset_A + i];
            for (int i = 0; i < NX * NU; i++) B_t[i] = B_in[offset_B + i];

            // V_A = V @ A_t
            matmul_small(V, A_t, V_A, NX, NX, NX);

            // V_B = V @ B_t
            matmul_small(V, B_t, V_B, NX, NX, NU);

            // Q_xx = l_xx[t] + A_t^T @ V_A
            int offset_lxx = t * NX * NX;
            matmul_At_B(A_t, V_A, Q_xx, NX, NX, NX);
            for (int i = 0; i < NX * NX; i++) Q_xx[i] += l_xx_in[offset_lxx + i];

            // Q_xu = l_xu[t] + A_t^T @ V_B
            int offset_lxu = t * NX * NU;
            matmul_At_B(A_t, V_B, Q_xu, NX, NX, NU);
            for (int i = 0; i < NX * NU; i++) Q_xu[i] += l_xu_in[offset_lxu + i];

            // Q_uu = l_uu[t] + B_t^T @ V_B + reg_eps * I
            int offset_luu = t * NU * NU;
            for (int i = 0; i < NU; i++) {
                for (int j = 0; j < NU; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < NX; k++) {
                        sum += B_t[k * NU + i] * V_B[k * NU + j];
                    }
                    Q_uu[i][j] = l_uu_in[offset_luu + i * NU + j] + sum;
                    if (i == j) Q_uu[i][j] += reg_eps;
                }
            }

            // q_x = l_x[t] + A_t^T @ v
            int offset_lx = t * NX;
            matvec_At(A_t, v, q_x, NX, NX);
            for (int i = 0; i < NX; i++) q_x[i] += l_x_in[offset_lx + i];

            // q_u = l_u[t] + B_t^T @ v
            int offset_lu = t * NU;
            matvec_At(B_t, v, q_u, NU, NX);
            for (int i = 0; i < NU; i++) q_u[i] += l_u_in[offset_lu + i];

            // Optional active-set mask handling (match Python adjoint semantics)
            if (tight_mask_in != nullptr) {
                int offset_mask = t * NU;
                float free_mask[NU];
                for (int i = 0; i < NU; i++) {
                    bool tight = tight_mask_in[offset_mask + i];
                    free_mask[i] = tight ? 0.0f : 1.0f;
                }
                for (int i = 0; i < NU; i++) {
                    for (int j = 0; j < NU; j++) {
                        Q_uu[i][j] *= free_mask[i] * free_mask[j];
                    }
                }
                for (int i = 0; i < NU; i++) {
                    if (free_mask[i] == 0.0f) {
                        Q_uu[i][i] += 1e-11f;  // matches mpc.pytorch.pnqp ridge
                    }
                }
                for (int i = 0; i < NU; i++) {
                    for (int r = 0; r < NX; r++) {
                        Q_xu[r * NU + i] *= free_mask[i];
                    }
                    q_u[i] *= free_mask[i];
                }
            }

            // Solve K_t = -Q_uu^{-1} @ Q_ux (Q_ux = Q_xu^T)
            // Copy Q_xu^T to K_t, then solve in place
            for (int i = 0; i < NU; i++) {
                for (int j = 0; j < NX; j++) {
                    K_t[i][j] = -Q_xu[j * NU + i];  // -Q_ux = -Q_xu^T
                }
            }
            float Q_uu_copy[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    Q_uu_copy[i][j] = Q_uu[i][j];
            solve4x4_multi(Q_uu_copy, K_t, NX);

            // Solve k_t = -Q_uu^{-1} @ q_u
            for (int i = 0; i < NU; i++) k_t[i] = -q_u[i];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    Q_uu_copy[i][j] = Q_uu[i][j];
            solve4x4(Q_uu_copy, k_t);

            // Store K_t and k_t
            int offset_K = t * NU * NX;
            int offset_k = t * NU;
            for (int i = 0; i < NU; i++) {
                for (int j = 0; j < NX; j++) {
                    K_out[offset_K + i * NX + j] = K_t[i][j];
                }
                k_out[offset_k + i] = k_t[i];
            }

            // Update V = Q_xx + Q_xu @ K  (simplified formula, numerically stable)
            // Note: The expanded formula V = Q_xx + K^T@Q_uu@K + K^T@Q_ux + Q_xu@K
            // is mathematically equivalent but numerically unstable because
            // K^T@Q_uu@K + K^T@Q_ux should cancel to 0 but doesn't exactly.

            // V_new = Q_xx + Q_xu @ K
            for (int i = 0; i < NX * NX; i++) V[i] = Q_xx[i];
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NX; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < NU; l++) {
                        sum += Q_xu[i * NU + l] * K_t[l][j];
                    }
                    V[i * NX + j] += sum;
                }
            }

            // Update v = q_x + Q_xu @ k  (simplified formula)
            for (int i = 0; i < NX; i++) {
                float sum = q_x[i];
                for (int l = 0; l < NU; l++) {
                    sum += Q_xu[i * NU + l] * k_t[l];
                }
                v[i] = sum;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && v0_out != nullptr) {
        for (int i = 0; i < NX; i++) v0_out[i] = v[i];
    }
}

// ============================================================================
// Launch functions
// ============================================================================

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
) {
    backward_pass_fused_kernel_b1<<<1, 32, 0, stream>>>(
        A, B, l_xx, l_uu, l_xu, l_x, l_u, l_xxN, l_xN, tight_mask,
        reg_eps, T, K_out, k_out, v0_out
    );
}

// ============================================================================
// Batched backward pass - NX-parallel
// ============================================================================
//
// Each block processes one batch element. Within a block, threads cooperate:
//   - threads 0..NX-1 (10) handle the NX-wide rows of Q_xx, Q_xu, V_A, V_B,
//     V update, q_x, and the parallel back-substitution for the 10 columns
//     of K_t.
//   - threads 0..NU-1 (4) handle the NU rows of Q_uu, q_u, and the K_t / k_t
//     output writes.
//   - thread 0 owns the LU factorization of Q_uu (4x4) and the k_t solve.
//
// All per-step state lives in shared memory so it stays consistent across
// threads without warp-shuffle gymnastics. Approximate occupancy on a 4070 Ti
// at __launch_bounds__(32, 2) is 2 blocks/SM; the working set fits comfortably
// in 2*~3KB shared memory.
//
// The unconstrained variant (this kernel) optionally applies a tight_mask
// supplied from the iLQR forward solver: for adjoint differentiation we need
// to follow the same active set as the primal solution.

namespace {

// LU factorization of a 4x4 matrix in-place with partial pivoting.
// pivot[k] holds the row swap performed at step k. Result: A holds L (below
// diagonal, unit diagonal not stored) and U (on/above diagonal).
__device__ __forceinline__ void lu_factor4x4(float A[4][4], int pivot[4]) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int max_row = k;
        float max_val = fabsf(A[k][k]);
        #pragma unroll
        for (int i = k + 1; i < 4; i++) {
            float v = fabsf(A[i][k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        pivot[k] = max_row;
        if (max_row != k) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float t = A[k][j]; A[k][j] = A[max_row][j]; A[max_row][j] = t;
            }
        }
        float p = A[k][k];
        if (fabsf(p) < 1e-10f) p = (p < 0 ? -1e-10f : 1e-10f);
        #pragma unroll
        for (int i = k + 1; i < 4; i++) {
            A[i][k] = A[i][k] / p;
            #pragma unroll
            for (int j = k + 1; j < 4; j++) A[i][j] -= A[i][k] * A[k][j];
        }
    }
}

// Apply the LU factor (from lu_factor4x4) to solve A x = b. b is overwritten.
__device__ __forceinline__ void lu_solve4x4(const float LU[4][4],
                                            const int pivot[4],
                                            float b[4]) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        if (pivot[k] != k) {
            float t = b[k]; b[k] = b[pivot[k]]; b[pivot[k]] = t;
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int i = k + 1; i < 4; i++) b[i] -= LU[i][k] * b[k];
    }
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        float s = b[i];
        #pragma unroll
        for (int j = i + 1; j < 4; j++) s -= LU[i][j] * b[j];
        b[i] = s / LU[i][i];
    }
}

}  // namespace

// (drone_kernels namespace from line 8 is still open here.)

// One Riccati timestep, NX-parallel. All per-step shared-memory buffers must
// be __syncthreads()-stable on entry. tight_mask_row points at NU bools (or
// nullptr); when supplied it suppresses the clamped dimensions.
__device__ __forceinline__ void riccati_step_nxparallel(
    int tid,
    int t,
    const float* __restrict__ A_in,
    const float* __restrict__ B_in,
    const float* __restrict__ l_xx_in,
    const float* __restrict__ l_uu_in,
    const float* __restrict__ l_xu_in,
    const float* __restrict__ l_x_in,
    const float* __restrict__ l_u_in,
    const bool* __restrict__ tight_mask_row,   // [NU] or nullptr
    float reg_eps,
    // shared state in/out (all sized NX*NX, NX*NU, etc.)
    float V[NX][NX], float v[NX],
    float A_t[NX][NX], float B_t[NX][NU],
    float V_A[NX][NX], float V_B[NX][NU],
    float Q_xx[NX][NX], float Q_xu[NX][NU], float Q_uu[NU][NU],
    float q_x[NX], float q_u[NU],
    float K_t[NU][NX], float k_t[NU],
    float Q_uu_lu[4][4], int Q_uu_piv[4],
    float free_mask_io[NU],   // optional output (caller-cleared)
    float* K_out_row,         // [NU][NX] write target for this t
    float* k_out_row,         // [NU]
    int A_off, int B_off,
    int lxx_off, int luu_off, int lxu_off,
    int lx_off, int lu_off
) {
    // ---- Load A_t, B_t, l_xx_t, l_xu_t, l_uu_t into shared memory ----
    int t_A = A_off + t * NX * NX;
    int t_B = B_off + t * NX * NU;
    int t_lxx = lxx_off + t * NX * NX;
    int t_lxu = lxu_off + t * NX * NU;
    int t_luu = luu_off + t * NU * NU;
    int t_lx  = lx_off + t * NX;
    int t_lu  = lu_off + t * NU;

    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) A_t[tid][j] = A_in[t_A + tid * NX + j];
        #pragma unroll
        for (int j = 0; j < NU; j++) B_t[tid][j] = B_in[t_B + tid * NU + j];
    }
    if (tid < NU) {
        #pragma unroll
        for (int j = 0; j < NU; j++) Q_uu[tid][j] = l_uu_in[t_luu + tid * NU + j];
    }
    __syncthreads();

    // ---- V_A = V @ A_t (10x10), V_B = V @ B_t (10x4) ----
    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < NX; k++) s += V[tid][k] * A_t[k][j];
            V_A[tid][j] = s;
        }
        #pragma unroll
        for (int j = 0; j < NU; j++) {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < NX; k++) s += V[tid][k] * B_t[k][j];
            V_B[tid][j] = s;
        }
    }
    __syncthreads();

    // ---- Q_xx[i,:] = l_xx[i,:] + (A_t^T)[i,:] V_A   (per-row by thread i<NX) ----
    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) {
            float s = l_xx_in[t_lxx + tid * NX + j];
            #pragma unroll
            for (int k = 0; k < NX; k++) s += A_t[k][tid] * V_A[k][j];
            Q_xx[tid][j] = s;
        }
        // Q_xu[i,:] = l_xu[i,:] + (A_t^T)[i,:] V_B
        #pragma unroll
        for (int j = 0; j < NU; j++) {
            float s = l_xu_in[t_lxu + tid * NU + j];
            #pragma unroll
            for (int k = 0; k < NX; k++) s += A_t[k][tid] * V_B[k][j];
            Q_xu[tid][j] = s;
        }
        // q_x[i] = l_x[i] + sum_k A_t[k,i] * v[k]
        float s_qx = l_x_in[t_lx + tid];
        #pragma unroll
        for (int k = 0; k < NX; k++) s_qx += A_t[k][tid] * v[k];
        q_x[tid] = s_qx;
    }
    if (tid < NU) {
        // Q_uu[i,j] += sum_k B_t[k,i] * V_B[k,j], plus reg on diagonal
        #pragma unroll
        for (int j = 0; j < NU; j++) {
            float s = Q_uu[tid][j];
            #pragma unroll
            for (int k = 0; k < NX; k++) s += B_t[k][tid] * V_B[k][j];
            if (tid == j) s += reg_eps;
            Q_uu[tid][j] = s;
        }
        // q_u[i] = l_u[i] + sum_k B_t[k,i] * v[k]
        float s_qu = l_u_in[t_lu + tid];
        #pragma unroll
        for (int k = 0; k < NX; k++) s_qu += B_t[k][tid] * v[k];
        q_u[tid] = s_qu;
    }
    __syncthreads();

    // ---- Optional tight_mask: zero rows/cols then add 1e-11 ridge for clamped dims ----
    if (tight_mask_row != nullptr) {
        if (tid < NU) {
            float fm = tight_mask_row[tid] ? 0.0f : 1.0f;
            free_mask_io[tid] = fm;
        }
        __syncthreads();
        if (tid < NU) {
            #pragma unroll
            for (int j = 0; j < NU; j++) Q_uu[tid][j] *= free_mask_io[tid] * free_mask_io[j];
            if (free_mask_io[tid] == 0.0f) Q_uu[tid][tid] += 1e-11f;
            for (int r = 0; r < NX; r++) Q_xu[r][tid] *= free_mask_io[tid];
            q_u[tid] *= free_mask_io[tid];
        }
        __syncthreads();
    }

    // ---- LU factorize Q_uu on thread 0 ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) Q_uu_lu[i][j] = Q_uu[i][j];
        }
        lu_factor4x4(Q_uu_lu, Q_uu_piv);
    }
    __syncthreads();

    // ---- Solve K_t = -Q_uu^{-1} @ Q_xu^T (NX columns, parallel over tid<NX) ----
    if (tid < NX) {
        float rhs[NU];
        #pragma unroll
        for (int i = 0; i < NU; i++) rhs[i] = -Q_xu[tid][i];
        lu_solve4x4(Q_uu_lu, Q_uu_piv, rhs);
        #pragma unroll
        for (int i = 0; i < NU; i++) K_t[i][tid] = rhs[i];
    }
    // ---- Solve k_t = -Q_uu^{-1} @ q_u (thread 0 only) ----
    if (tid == 0) {
        float rhs[NU];
        #pragma unroll
        for (int i = 0; i < NU; i++) rhs[i] = -q_u[i];
        lu_solve4x4(Q_uu_lu, Q_uu_piv, rhs);
        #pragma unroll
        for (int i = 0; i < NU; i++) k_t[i] = rhs[i];
    }
    __syncthreads();

    // ---- Store K_t and k_t to global ----
    if (tid < NU) {
        #pragma unroll
        for (int j = 0; j < NX; j++) {
            K_out_row[tid * NX + j] = K_t[tid][j];
        }
        k_out_row[tid] = k_t[tid];
    }

    // ---- Update V = Q_xx + Q_xu @ K   (tid<NX computes row tid) ----
    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) {
            float s = Q_xx[tid][j];
            #pragma unroll
            for (int l = 0; l < NU; l++) s += Q_xu[tid][l] * K_t[l][j];
            V[tid][j] = s;
        }
        // v[i] = q_x[i] + Q_xu[i,:] @ k_t
        float s_v = q_x[tid];
        #pragma unroll
        for (int l = 0; l < NU; l++) s_v += Q_xu[tid][l] * k_t[l];
        v[tid] = s_v;
    }
    __syncthreads();
}

__global__ __launch_bounds__(32, 2) void backward_pass_batched_kernel(
    const float* __restrict__ A_in,
    const float* __restrict__ B_in,
    const float* __restrict__ l_xx_in,
    const float* __restrict__ l_uu_in,
    const float* __restrict__ l_xu_in,
    const float* __restrict__ l_x_in,
    const float* __restrict__ l_u_in,
    const float* __restrict__ l_xxN_in,
    const float* __restrict__ l_xN_in,
    const bool* __restrict__ tight_mask_in,
    float reg_eps,
    int batch_size,
    int T,
    float* __restrict__ K_out,
    float* __restrict__ k_out,
    float* __restrict__ v0_out
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    // Per-batch global offsets.
    const int A_off  = b * T * NX * NX;
    const int B_off  = b * T * NX * NU;
    const int lxx_off = b * T * NX * NX;
    const int luu_off = b * T * NU * NU;
    const int lxu_off = b * T * NX * NU;
    const int lx_off  = b * T * NX;
    const int lu_off  = b * T * NU;
    const int K_off   = b * T * NU * NX;
    const int k_off   = b * T * NU;
    const int mask_off = b * T * NU;

    // Shared state.
    __shared__ float V[NX][NX];
    __shared__ float v[NX];
    __shared__ float A_t[NX][NX];
    __shared__ float B_t[NX][NU];
    __shared__ float V_A[NX][NX];
    __shared__ float V_B[NX][NU];
    __shared__ float Q_xx[NX][NX];
    __shared__ float Q_xu[NX][NU];
    __shared__ float Q_uu[NU][NU];
    __shared__ float Q_uu_lu[4][4];
    __shared__ int Q_uu_piv[4];
    __shared__ float q_x[NX];
    __shared__ float q_u[NU];
    __shared__ float K_t[NU][NX];
    __shared__ float k_t[NU];
    __shared__ float free_mask[NU];

    // Init terminal V, v.
    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) V[tid][j] = l_xxN_in[b * NX * NX + tid * NX + j];
        v[tid] = l_xN_in[b * NX + tid];
    }
    __syncthreads();

    for (int t = T - 1; t >= 0; t--) {
        const bool* mask_row = (tight_mask_in != nullptr)
            ? (tight_mask_in + mask_off + t * NU) : nullptr;
        riccati_step_nxparallel(
            tid, t,
            A_in, B_in, l_xx_in, l_uu_in, l_xu_in, l_x_in, l_u_in,
            mask_row, reg_eps,
            V, v, A_t, B_t, V_A, V_B,
            Q_xx, Q_xu, Q_uu, q_x, q_u,
            K_t, k_t, Q_uu_lu, Q_uu_piv, free_mask,
            K_out + K_off + t * NU * NX,
            k_out + k_off + t * NU,
            A_off, B_off, lxx_off, luu_off, lxu_off, lx_off, lu_off
        );
    }

    if (v0_out != nullptr && tid < NX) {
        v0_out[b * NX + tid] = v[tid];
    }
}

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
) {
    backward_pass_batched_kernel<<<batch_size, 32, 0, stream>>>(
        A, B, l_xx, l_uu, l_xu, l_x, l_u, l_xxN, l_xN, tight_mask,
        reg_eps, batch_size, T, K_out, k_out, v0_out
    );
}

// ============================================================================
// PNQP-aware batched backward pass
// ============================================================================
//
// For each Riccati step we solve the box-QP
//   min 0.5 du^T Q_uu du + q_u^T du   s.t. u_min - u_t <= du <= u_max - u_t
// using the device pnqp4 routine. The free_mask returned by pnqp4 is the
// authoritative active set used to (a) zero clamped rows/cols of Q_uu/Q_xu,
// (b) seed k_t with the PNQP optimum on clamped dims, and (c) feed back to
// the solver as the per-(b,t) tight_mask.
//
// Bounds u_min, u_max are device-resident NU-element vectors (read once via
// __ldg). Pass nullptr for both to disable the box-QP and fall back to
// the unconstrained Riccati semantics.

__global__ __launch_bounds__(32, 2) void backward_pass_pnqp_batched_kernel(
    const float* __restrict__ A_in,
    const float* __restrict__ B_in,
    const float* __restrict__ l_xx_in,
    const float* __restrict__ l_uu_in,
    const float* __restrict__ l_xu_in,
    const float* __restrict__ l_x_in,
    const float* __restrict__ l_u_in,
    const float* __restrict__ l_xxN_in,
    const float* __restrict__ l_xN_in,
    const float* __restrict__ U_in,
    const float* __restrict__ u_min_in,
    const float* __restrict__ u_max_in,
    float reg_eps,
    int batch_size,
    int T,
    float* __restrict__ K_out,
    float* __restrict__ k_out,
    float* __restrict__ v0_out,
    float* __restrict__ free_mask_out
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;
    const bool has_bounds = (u_min_in != nullptr) && (u_max_in != nullptr);

    const int A_off  = b * T * NX * NX;
    const int B_off  = b * T * NX * NU;
    const int lxx_off = b * T * NX * NX;
    const int luu_off = b * T * NU * NU;
    const int lxu_off = b * T * NX * NU;
    const int lx_off  = b * T * NX;
    const int lu_off  = b * T * NU;
    const int K_off   = b * T * NU * NX;
    const int k_off   = b * T * NU;
    const int U_off   = b * T * NU;
    const int fm_off  = b * T * NU;

    __shared__ float V[NX][NX];
    __shared__ float v[NX];
    __shared__ float A_t[NX][NX];
    __shared__ float B_t[NX][NU];
    __shared__ float V_A[NX][NX];
    __shared__ float V_B[NX][NU];
    __shared__ float Q_xx[NX][NX];
    __shared__ float Q_xu[NX][NU];
    __shared__ float Q_uu[NU][NU];
    __shared__ float Q_uu_lu[4][4];
    __shared__ int   Q_uu_piv[4];
    __shared__ float q_x[NX];
    __shared__ float q_u[NU];
    __shared__ float K_t[NU][NX];
    __shared__ float k_t[NU];
    __shared__ float free_mask[NU];
    __shared__ float du_star[NU];   // PNQP optimum at dx=0 (used as k_t[clamped])
    __shared__ float u_min_sh[NU];
    __shared__ float u_max_sh[NU];

    if (tid < NX) {
        #pragma unroll
        for (int j = 0; j < NX; j++) V[tid][j] = l_xxN_in[b * NX * NX + tid * NX + j];
        v[tid] = l_xN_in[b * NX + tid];
    }
    if (tid < NU) {
        u_min_sh[tid] = has_bounds ? u_min_in[tid] : -1e30f;
        u_max_sh[tid] = has_bounds ? u_max_in[tid] :  1e30f;
        free_mask[tid] = 1.0f;
    }
    __syncthreads();

    for (int t = T - 1; t >= 0; t--) {
        // ---- Compute Q_xx, Q_xu, Q_uu, q_x, q_u (NX-parallel) ----
        // We replicate the load+matmul phase from riccati_step_nxparallel up
        // to the point right before the tight_mask handling, then branch into
        // the PNQP solve.
        const int t_A   = A_off + t * NX * NX;
        const int t_B   = B_off + t * NX * NU;
        const int t_lxx = lxx_off + t * NX * NX;
        const int t_lxu = lxu_off + t * NX * NU;
        const int t_luu = luu_off + t * NU * NU;
        const int t_lx  = lx_off + t * NX;
        const int t_lu  = lu_off + t * NU;

        if (tid < NX) {
            #pragma unroll
            for (int j = 0; j < NX; j++) A_t[tid][j] = A_in[t_A + tid * NX + j];
            #pragma unroll
            for (int j = 0; j < NU; j++) B_t[tid][j] = B_in[t_B + tid * NU + j];
        }
        if (tid < NU) {
            #pragma unroll
            for (int j = 0; j < NU; j++) Q_uu[tid][j] = l_uu_in[t_luu + tid * NU + j];
        }
        __syncthreads();

        if (tid < NX) {
            #pragma unroll
            for (int j = 0; j < NX; j++) {
                float s = 0.0f;
                #pragma unroll
                for (int k = 0; k < NX; k++) s += V[tid][k] * A_t[k][j];
                V_A[tid][j] = s;
            }
            #pragma unroll
            for (int j = 0; j < NU; j++) {
                float s = 0.0f;
                #pragma unroll
                for (int k = 0; k < NX; k++) s += V[tid][k] * B_t[k][j];
                V_B[tid][j] = s;
            }
        }
        __syncthreads();

        if (tid < NX) {
            #pragma unroll
            for (int j = 0; j < NX; j++) {
                float s = l_xx_in[t_lxx + tid * NX + j];
                #pragma unroll
                for (int k = 0; k < NX; k++) s += A_t[k][tid] * V_A[k][j];
                Q_xx[tid][j] = s;
            }
            #pragma unroll
            for (int j = 0; j < NU; j++) {
                float s = l_xu_in[t_lxu + tid * NU + j];
                #pragma unroll
                for (int k = 0; k < NX; k++) s += A_t[k][tid] * V_B[k][j];
                Q_xu[tid][j] = s;
            }
            float s_qx = l_x_in[t_lx + tid];
            #pragma unroll
            for (int k = 0; k < NX; k++) s_qx += A_t[k][tid] * v[k];
            q_x[tid] = s_qx;
        }
        if (tid < NU) {
            #pragma unroll
            for (int j = 0; j < NU; j++) {
                float s = Q_uu[tid][j];
                #pragma unroll
                for (int k = 0; k < NX; k++) s += B_t[k][tid] * V_B[k][j];
                if (tid == j) s += reg_eps;
                Q_uu[tid][j] = s;
            }
            float s_qu = l_u_in[t_lu + tid];
            #pragma unroll
            for (int k = 0; k < NX; k++) s_qu += B_t[k][tid] * v[k];
            q_u[tid] = s_qu;
        }
        __syncthreads();

        // ---- PNQP step on thread 0: solve box-QP at dx=0, get free_mask ----
        if (tid == 0) {
            if (has_bounds) {
                float Hloc[4][4]; float qloc[4];
                float lo[4]; float hi[4]; float xio[4]; float fm_loc[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) Hloc[i][j] = Q_uu[i][j];
                    qloc[i] = q_u[i];
                    float u_t_i = U_in[U_off + t * NU + i];
                    lo[i] = u_min_sh[i] - u_t_i;
                    hi[i] = u_max_sh[i] - u_t_i;
                    xio[i] = 0.0f;  // cold start
                }
                pnqp4(Hloc, qloc, lo, hi, xio, fm_loc);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    free_mask[i] = fm_loc[i];
                    du_star[i] = xio[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    free_mask[i] = 1.0f;
                    du_star[i] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- Apply free_mask: zero rows/cols of Q_uu, columns of Q_xu, rows of q_u ----
        if (has_bounds) {
            if (tid < NU) {
                #pragma unroll
                for (int j = 0; j < NU; j++) Q_uu[tid][j] *= free_mask[tid] * free_mask[j];
                if (free_mask[tid] == 0.0f) Q_uu[tid][tid] += 1e-11f;
                for (int r = 0; r < NX; r++) Q_xu[r][tid] *= free_mask[tid];
                q_u[tid] *= free_mask[tid];
            }
            __syncthreads();
        }

        // ---- LU + parallel column back-sub for K_t (free dims only) ----
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) Q_uu_lu[i][j] = Q_uu[i][j];
            }
            lu_factor4x4(Q_uu_lu, Q_uu_piv);
        }
        __syncthreads();

        if (tid < NX) {
            float rhs[NU];
            #pragma unroll
            for (int i = 0; i < NU; i++) rhs[i] = -Q_xu[tid][i];
            lu_solve4x4(Q_uu_lu, Q_uu_piv, rhs);
            // Force clamped rows of K_t to zero (consistent with PNQP semantics).
            #pragma unroll
            for (int i = 0; i < NU; i++) K_t[i][tid] = has_bounds ? rhs[i] * free_mask[i] : rhs[i];
        }
        if (tid == 0) {
            float rhs[NU];
            #pragma unroll
            for (int i = 0; i < NU; i++) rhs[i] = -q_u[i];
            lu_solve4x4(Q_uu_lu, Q_uu_piv, rhs);
            // For free dims, k_t[i] = solve(-Q_uu_ff^{-1} q_u_f) which equals du_star[i]
            // up to numerical error. For clamped dims, k_t[i] = du_star[i] (the bound).
            #pragma unroll
            for (int i = 0; i < NU; i++) {
                if (has_bounds) {
                    k_t[i] = (free_mask[i] != 0.0f) ? rhs[i] : du_star[i];
                } else {
                    k_t[i] = rhs[i];
                }
            }
        }
        __syncthreads();

        // ---- Write K_t, k_t, free_mask ----
        if (tid < NU) {
            #pragma unroll
            for (int j = 0; j < NX; j++) K_out[K_off + t * NU * NX + tid * NX + j] = K_t[tid][j];
            k_out[k_off + t * NU + tid] = k_t[tid];
            if (free_mask_out != nullptr) {
                free_mask_out[fm_off + t * NU + tid] = free_mask[tid];
            }
        }

        // ---- Update V, v (NX-parallel) ----
        if (tid < NX) {
            #pragma unroll
            for (int j = 0; j < NX; j++) {
                float s = Q_xx[tid][j];
                #pragma unroll
                for (int l = 0; l < NU; l++) s += Q_xu[tid][l] * K_t[l][j];
                V[tid][j] = s;
            }
            float s_v = q_x[tid];
            #pragma unroll
            for (int l = 0; l < NU; l++) s_v += Q_xu[tid][l] * k_t[l];
            v[tid] = s_v;
        }
        __syncthreads();
    }

    if (v0_out != nullptr && tid < NX) {
        v0_out[b * NX + tid] = v[tid];
    }
}

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
    const float* U,
    const float* u_min,
    const float* u_max,
    float reg_eps,
    int batch_size,
    int T,
    float* K_out,
    float* k_out,
    float* v0_out,
    float* free_mask_out,
    cudaStream_t stream
) {
    backward_pass_pnqp_batched_kernel<<<batch_size, 32, 0, stream>>>(
        A, B, l_xx, l_uu, l_xu, l_x, l_u, l_xxN, l_xN,
        U, u_min, u_max,
        reg_eps, batch_size, T,
        K_out, k_out, v0_out, free_mask_out
    );
}

// ============================================================================
// Fused implicit-diff assembly kernel
// ============================================================================
//
// One block per batch element, 32 threads. Sequential outer loop over t
// because the forward sweep dX[t+1] = A dX[t] + B dU[t] is intrinsically
// sequential. Within a step we parallelize:
//   - dU[t] (NU rows) over threads 0..NU-1
//   - dX[t+1] (NX rows) over threads 0..NX-1
//   - tau[t]/d_tau[t] (N=14 entries) over threads 0..N-1
//   - grad_C[t] (N*N=196 entries) over all 32 threads (~6 entries each)
//   - grad_c[t], grad_xref[t], grad_uref[t] over the relevant subset

__global__ __launch_bounds__(32, 2) void implicit_diff_assembly_kernel(
    const float* __restrict__ K_seq,
    const float* __restrict__ k_seq,
    const float* __restrict__ v0_in,
    const float* __restrict__ A_lin,
    const float* __restrict__ Bm_lin,
    const float* __restrict__ X_in,
    const float* __restrict__ U_in,
    const float* __restrict__ x_ref_in,
    const float* __restrict__ u_ref_in,
    float* __restrict__ grad_C,
    float* __restrict__ grad_c,
    float* __restrict__ grad_x0,
    float* __restrict__ grad_xref,
    float* __restrict__ grad_uref,
    float* __restrict__ grad_C_final,
    float* __restrict__ grad_c_final,
    int batch_size,
    int T
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    constexpr int N = NX + NU;  // 14

    // Per-batch global offsets.
    const int K_off    = b * T * NU * NX;
    const int k_off    = b * T * NU;
    const int A_off    = b * T * NX * NX;
    const int B_off    = b * T * NX * NU;
    const int X_off    = b * (T + 1) * NX;
    const int U_off    = b * T * NU;
    const int xref_off = b * (T + 1) * NX;
    const int uref_off = b * T * NU;
    const int gC_off   = b * T * N * N;
    const int gc_off   = b * T * N;
    const int gx0_off  = b * NX;
    const int gxref_off = b * (T + 1) * NX;
    const int guref_off = b * T * NU;

    __shared__ float dX[NX];
    __shared__ float dX_next[NX];
    __shared__ float dU[NU];
    __shared__ float A_t[NX][NX];
    __shared__ float B_t[NX][NU];
    __shared__ float K_t[NU][NX];
    __shared__ float k_t[NU];
    __shared__ float tau[N];
    __shared__ float d_tau[N];

    // Init dX[0] = 0.
    if (tid < NX) dX[tid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Load K_t, k_t.
        if (tid < NU) {
            #pragma unroll
            for (int j = 0; j < NX; j++) {
                K_t[tid][j] = K_seq[K_off + t * NU * NX + tid * NX + j];
            }
            k_t[tid] = k_seq[k_off + t * NU + tid];
        }
        // Load A_t, B_t.
        if (tid < NX) {
            #pragma unroll
            for (int j = 0; j < NX; j++) A_t[tid][j] = A_lin[A_off + t * NX * NX + tid * NX + j];
            #pragma unroll
            for (int j = 0; j < NU; j++) B_t[tid][j] = Bm_lin[B_off + t * NX * NU + tid * NU + j];
        }
        __syncthreads();

        // dU[t] = K_t @ dX[t] + k_t (NU-parallel).
        if (tid < NU) {
            float s = k_t[tid];
            #pragma unroll
            for (int j = 0; j < NX; j++) s += K_t[tid][j] * dX[j];
            dU[tid] = s;
        }
        __syncthreads();

        // dX[t+1] = A_t @ dX[t] + B_t @ dU[t] (NX-parallel).
        if (tid < NX) {
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < NX; j++) s += A_t[tid][j] * dX[j];
            #pragma unroll
            for (int j = 0; j < NU; j++) s += B_t[tid][j] * dU[j];
            dX_next[tid] = s;
        }

        // Build tau[t], d_tau[t] in shared (N=14 entries; first NX are X-rel,
        // remaining NU are U-rel). Independent of dX_next, can run in parallel.
        if (tid < NX) {
            tau[tid]   = X_in[X_off + t * NX + tid] - x_ref_in[xref_off + t * NX + tid];
            d_tau[tid] = dX[tid];
        } else if (tid < N) {
            int u_idx = tid - NX;
            tau[tid]   = U_in[U_off + t * NU + u_idx] - u_ref_in[uref_off + t * NU + u_idx];
            d_tau[tid] = dU[u_idx];
        }
        __syncthreads();

        // grad_C[b,t,i,j] = -0.5 (d_tau[i] tau[j] + tau[i] d_tau[j])
        // 196 entries / 32 threads = 6.125 entries per thread.
        for (int idx = tid; idx < N * N; idx += 32) {
            int i = idx / N;
            int j = idx % N;
            grad_C[gC_off + t * N * N + idx] =
                -0.5f * (d_tau[i] * tau[j] + tau[i] * d_tau[j]);
        }
        // grad_c[b,t,i] = -d_tau[i]  (N entries).
        if (tid < N) grad_c[gc_off + t * N + tid] = -d_tau[tid];

        // grad_xref[b,t,:] = dX[t]  (NX entries).
        if (grad_xref != nullptr && tid < NX) {
            grad_xref[gxref_off + t * NX + tid] = dX[tid];
        }
        // grad_uref[b,t,:] = dU[t]  (NU entries).
        if (grad_uref != nullptr && tid < NU) {
            grad_uref[guref_off + t * NU + tid] = dU[tid];
        }

        // Roll dX_next -> dX for the next iteration.
        if (tid < NX) dX[tid] = dX_next[tid];
        __syncthreads();
    }

    // ---- Terminal step ----
    // grad_xref[b, T, :] = dX[T]
    if (grad_xref != nullptr && tid < NX) {
        grad_xref[gxref_off + T * NX + tid] = dX[tid];
    }

    // grad_x0 = -v0_in (the v at t=0 from the Riccati sweep).
    if (tid < NX) grad_x0[gx0_off + tid] = -v0_in[b * NX + tid];

    // grad_C_final and grad_c_final: tauN = [X[T]-x_ref[T]; 0]; d_tauN = [dX[T]; 0]
    if (grad_C_final != nullptr || grad_c_final != nullptr) {
        // Build tau, d_tau for the terminal step.
        if (tid < NX) {
            tau[tid]   = X_in[X_off + T * NX + tid] - x_ref_in[xref_off + T * NX + tid];
            d_tau[tid] = dX[tid];
        } else if (tid < N) {
            tau[tid]   = 0.0f;
            d_tau[tid] = 0.0f;
        }
        __syncthreads();

        if (grad_C_final != nullptr) {
            const int gCf_off = b * N * N;
            for (int idx = tid; idx < N * N; idx += 32) {
                int i = idx / N;
                int j = idx % N;
                grad_C_final[gCf_off + idx] =
                    -0.5f * (d_tau[i] * tau[j] + tau[i] * d_tau[j]);
            }
        }
        if (grad_c_final != nullptr && tid < N) {
            grad_c_final[b * N + tid] = -d_tau[tid];
        }
    }
}

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
) {
    implicit_diff_assembly_kernel<<<batch_size, 32, 0, stream>>>(
        K_seq, k_seq, v0, A_lin, B_lin, X, U, x_ref, u_ref,
        grad_C, grad_c, grad_x0, grad_xref, grad_uref,
        grad_C_final, grad_c_final,
        batch_size, T
    );
}

} // namespace drone_kernels

// ============================================================================
// PyTorch bindings (only when compiled standalone, not as part of main solver)
// ============================================================================

#ifndef DRONE_ILQR_USE_FUSED_BACKWARD  // Standalone build

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

std::tuple<torch::Tensor, torch::Tensor> backward_pass_fused_b1_torch(
    torch::Tensor A,       // [T, NX, NX]
    torch::Tensor B,       // [T, NX, NU]
    torch::Tensor l_xx,    // [T, NX, NX]
    torch::Tensor l_uu,    // [T, NU, NU]
    torch::Tensor l_xu,    // [T, NX, NU]
    torch::Tensor l_x,     // [T, NX]
    torch::Tensor l_u,     // [T, NU]
    torch::Tensor l_xxN,   // [NX, NX]
    torch::Tensor l_xN,    // [NX]
    float reg_eps
) {
    int T = A.size(0);

    auto options = A.options();
    auto K_out = torch::empty({T, drone_kernels::NU, drone_kernels::NX}, options);
    auto k_out = torch::empty({T, drone_kernels::NU}, options);
    auto v0_out = torch::empty({drone_kernels::NX}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    drone_kernels::launch_backward_pass_fused_b1(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        l_xx.data_ptr<float>(),
        l_uu.data_ptr<float>(),
        l_xu.data_ptr<float>(),
        l_x.data_ptr<float>(),
        l_u.data_ptr<float>(),
        l_xxN.data_ptr<float>(),
        l_xN.data_ptr<float>(),
        nullptr,
        reg_eps,
        T,
        K_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        v0_out.data_ptr<float>(),
        stream
    );

    return std::make_tuple(K_out, k_out);
}

std::tuple<torch::Tensor, torch::Tensor> backward_pass_batched_torch(
    torch::Tensor A,       // [B, T, NX, NX]
    torch::Tensor B,       // [B, T, NX, NU]
    torch::Tensor l_xx,    // [B, T, NX, NX]
    torch::Tensor l_uu,    // [B, T, NU, NU]
    torch::Tensor l_xu,    // [B, T, NX, NU]
    torch::Tensor l_x,     // [B, T, NX]
    torch::Tensor l_u,     // [B, T, NU]
    torch::Tensor l_xxN,   // [B, NX, NX]
    torch::Tensor l_xN,    // [B, NX]
    float reg_eps
) {
    int batch_size = A.size(0);
    int T = A.size(1);

    auto options = A.options();
    auto K_out = torch::empty({batch_size, T, drone_kernels::NU, drone_kernels::NX}, options);
    auto k_out = torch::empty({batch_size, T, drone_kernels::NU}, options);
    auto v0_out = torch::empty({batch_size, drone_kernels::NX}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    drone_kernels::launch_backward_pass_batched(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        l_xx.data_ptr<float>(),
        l_uu.data_ptr<float>(),
        l_xu.data_ptr<float>(),
        l_x.data_ptr<float>(),
        l_u.data_ptr<float>(),
        l_xxN.data_ptr<float>(),
        l_xN.data_ptr<float>(),
        nullptr,
        reg_eps,
        batch_size,
        T,
        K_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        v0_out.data_ptr<float>(),
        stream
    );

    return std::make_tuple(K_out, k_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_pass_fused_b1", &backward_pass_fused_b1_torch,
          "Fused backward pass for B=1 (single kernel launch)");
    m.def("backward_pass_batched", &backward_pass_batched_torch,
          "Batched backward pass (one block per batch element)");
}

#endif // DRONE_ILQR_USE_FUSED_BACKWARD
