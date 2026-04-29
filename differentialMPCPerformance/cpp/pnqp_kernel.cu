#include "pnqp_kernel.cuh"

namespace drone_kernels {

__global__ __launch_bounds__(32, 4) void pnqp4_batch_kernel(
    const float* __restrict__ H_in,        // [N, 4, 4]
    const float* __restrict__ q_in,        // [N, 4]
    const float* __restrict__ lower_in,    // [N, 4]
    const float* __restrict__ upper_in,    // [N, 4]
    const float* __restrict__ x_init_in,   // [N, 4] or nullptr (cold start)
    float* __restrict__ x_out,             // [N, 4]
    float* __restrict__ free_mask_out,     // [N, 4]
    int N
) {
    int n = blockIdx.x;
    if (n >= N) return;
    if (threadIdx.x != 0) return;

    float H[4][4];
    float q[4], lower[4], upper[4], x[4], free_mask[4];

    int H_off = n * 16;
    int v_off = n * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) H[i][j] = H_in[H_off + i * 4 + j];
        q[i] = q_in[v_off + i];
        lower[i] = lower_in[v_off + i];
        upper[i] = upper_in[v_off + i];
        x[i] = x_init_in ? x_init_in[v_off + i] : 0.0f;
    }

    pnqp4(H, q, lower, upper, x, free_mask);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        x_out[v_off + i] = x[i];
        free_mask_out[v_off + i] = free_mask[i];
    }
}

void launch_pnqp4_batch(
    const float* H, const float* q,
    const float* lower, const float* upper,
    const float* x_init,
    float* x_out, float* free_mask_out,
    int N, cudaStream_t stream
) {
    pnqp4_batch_kernel<<<N, 32, 0, stream>>>(
        H, q, lower, upper, x_init, x_out, free_mask_out, N
    );
}

}  // namespace drone_kernels
