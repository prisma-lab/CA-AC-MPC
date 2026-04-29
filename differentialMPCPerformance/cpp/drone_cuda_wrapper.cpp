/**
 * PyTorch C++ wrapper for CUDA kernels
 * With autograd support for differentiability
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "drone_kernels.cuh"

// ============================================================================
// C++ Wrapper functions that call CUDA kernels
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> drone_step_jacobian_cuda(
    const at::Tensor& x,
    const at::Tensor& u,
    double dt
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(u.is_cuda(), "u must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(u.scalar_type() == at::kFloat, "u must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 10, "x must be [B, 10]");
    TORCH_CHECK(u.dim() == 2 && u.size(1) == 4, "u must be [B, 4]");

    const int batch_size = x.size(0);
    auto options = x.options();

    auto x_next = at::empty({batch_size, 10}, options);
    auto A = at::empty({batch_size, 10, 10}, options);
    auto B_mat = at::empty({batch_size, 10, 4}, options);

    drone_kernels::launch_drone_step_jacobian(
        x.data_ptr<float>(),
        u.data_ptr<float>(),
        x_next.data_ptr<float>(),
        A.data_ptr<float>(),
        B_mat.data_ptr<float>(),
        static_cast<float>(dt),
        batch_size,
        c10::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(x_next, A, B_mat);
}

at::Tensor batched_rollout_cuda(
    const at::Tensor& x0,
    const at::Tensor& U,
    double dt
) {
    TORCH_CHECK(x0.is_cuda(), "x0 must be CUDA tensor");
    TORCH_CHECK(U.is_cuda(), "U must be CUDA tensor");
    TORCH_CHECK(x0.scalar_type() == at::kFloat, "x0 must be float32");
    TORCH_CHECK(U.scalar_type() == at::kFloat, "U must be float32");
    TORCH_CHECK(x0.dim() == 2 && x0.size(1) == 10, "x0 must be [B, 10]");
    TORCH_CHECK(U.dim() == 3 && U.size(2) == 4, "U must be [B, T, 4]");

    const int batch_size = x0.size(0);
    const int horizon = U.size(1);
    auto options = x0.options();

    auto X = at::empty({batch_size, horizon + 1, 10}, options);

    drone_kernels::launch_batched_rollout(
        x0.data_ptr<float>(),
        U.data_ptr<float>(),
        X.data_ptr<float>(),
        static_cast<float>(dt),
        batch_size,
        horizon,
        c10::cuda::getCurrentCUDAStream()
    );

    return X;
}

// ============================================================================
// Autograd Function for differentiability
// ============================================================================

class DroneStepJacobianFunction : public torch::autograd::Function<DroneStepJacobianFunction> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x,
        const at::Tensor& u,
        double dt
    ) {
        auto [x_next, A, B_mat] = drone_step_jacobian_cuda(x, u, dt);

        // Save for backward
        ctx->save_for_backward({x, u, A, B_mat});
        ctx->saved_data["dt"] = dt;

        return {x_next, A, B_mat};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto u = saved[1];
        auto A = saved[2];
        auto B_mat = saved[3];

        auto grad_x_next = grad_outputs[0];  // [B, 10]
        // grad_A and grad_B are typically not used in standard MPC

        at::Tensor grad_x, grad_u;

        if (grad_x_next.defined() && grad_x_next.numel() > 0) {
            // Backward through x_next = f(x, u)
            // dx_next/dx = A, dx_next/du = B
            // grad_x = A^T @ grad_x_next, grad_u = B^T @ grad_x_next

            grad_x = at::matmul(A.transpose(-1, -2), grad_x_next.unsqueeze(-1)).squeeze(-1);
            grad_u = at::matmul(B_mat.transpose(-1, -2), grad_x_next.unsqueeze(-1)).squeeze(-1);
        }

        return {grad_x, grad_u, at::Tensor()};  // dt has no gradient
    }
};

std::vector<at::Tensor> drone_step_jacobian_autograd(
    const at::Tensor& x,
    const at::Tensor& u,
    double dt
) {
    return DroneStepJacobianFunction::apply(x, u, dt);
}

// ============================================================================
// Module registration (only when compiled as standalone module)
// ============================================================================

#ifndef DRONE_ILQR_USE_CUDA_KERNELS
// Register as standalone Python module only when NOT being included in the main solver
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("drone_step_jacobian_cuda", &drone_step_jacobian_cuda,
          "Fused drone step + jacobian (CUDA)",
          py::arg("x"), py::arg("u"), py::arg("dt"));

    m.def("batched_rollout_cuda", &batched_rollout_cuda,
          "Batched rollout (CUDA)",
          py::arg("x0"), py::arg("U"), py::arg("dt"));

    m.def("drone_step_jacobian_autograd", &drone_step_jacobian_autograd,
          "Fused drone step + jacobian with autograd support (CUDA)",
          py::arg("x"), py::arg("u"), py::arg("dt"));
}
#endif
