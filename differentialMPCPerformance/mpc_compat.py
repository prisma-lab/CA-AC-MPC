"""
Compatibility wrapper to use dmpc_perf with mpc.pytorch-like interface.

Supports both differentiable (gradient-enabled) and inference-only modes:
- Differentiable: uses DroneMPCFunction (implicit differentiation via KKT)
- Inference: uses torch.inference_mode() for maximum speed

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, NamedTuple, Union
from enum import Enum

import torch
from torch import nn, Tensor

try:
    from .drone_ilqr import load_drone_ilqr_ext
except ImportError:
    from drone_ilqr import load_drone_ilqr_ext

# Optional: use mpc.pytorch's PNQP for the per-step Q_uu solve in the Python
# backward fallback. Off by default (the masked Cholesky path is sufficient
# given the C++ forward already provides a PNQP-style active set). Enable by
# setting FASTMPC_USE_PNQP=1 if you suspect the active set is unreliable.
_PNQP = None
def _get_pnqp():
    global _PNQP
    if _PNQP is None:
        import sys
        mpc_pytorch_root = (Path(__file__).resolve().parents[1]
                            / "acmpc_public-master" / "mpc.pytorch")
        if str(mpc_pytorch_root) not in sys.path:
            sys.path.insert(0, str(mpc_pytorch_root))
        from mpc.pnqp import pnqp as _pnqp_fn  # noqa: E402
        _PNQP = _pnqp_fn
    return _PNQP


# Match mpc.pytorch interface
class QuadCost(NamedTuple):
    """Quadratic cost: 0.5 * z^T C z + c^T z where z = [x; u]"""
    C: Tensor  # [T, B, n, n] or [T, n, n] or [n, n]
    c: Optional[Tensor] = None  # [T, B, n] or [T, n] or [n]


class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


# ---------------------------------------------------------------------------
# Differentiable MPC via implicit differentiation (KKT adjoint)
# ---------------------------------------------------------------------------

def _outer(x: Tensor, y: Tensor) -> Tensor:
    """Batched outer product: x[..., i] * y[..., j] -> [..., i, j]."""
    return x.unsqueeze(-1) * y.unsqueeze(-2)


# Per-(device, dtype, shape) cache of zero tensors reused across forward calls.
# Avoids allocating fresh ref/terminal-cost tensors every step of a training loop.
_ZERO_CACHE: dict[tuple, Tensor] = {}


def _zeros(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype, shape)
    t = _ZERO_CACHE.get(key)
    if t is None:
        t = torch.zeros(shape, device=device, dtype=dtype)
        _ZERO_CACHE[key] = t
    return t


def _normalize_cost(C: Tensor, c: Optional[Tensor], B: int, T: int, n: int,
                    device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Bring (C, c) to [B, T, n, n] / [B, T, n] in float32-contiguous layout."""
    if C.ndim == 2:
        C = C.unsqueeze(0).unsqueeze(0).expand(T, B, n, n)
    elif C.ndim == 3:
        C = C.unsqueeze(1).expand(T, B, n, n)
    if c is None:
        c = _zeros((T, B, n), device, dtype)
    elif c.ndim == 1:
        c = c.unsqueeze(0).unsqueeze(0).expand(T, B, n)
    elif c.ndim == 2:
        c = c.unsqueeze(1).expand(T, B, n)
    return (
        C.permute(1, 0, 2, 3).contiguous().float(),
        c.permute(1, 0, 2).contiguous().float(),
    )


def _normalize_bounds(u_lower: Optional[Tensor], u_upper: Optional[Tensor], nu: int,
                      device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if u_lower is None or u_upper is None:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty
    def _flatten(b: Tensor) -> Tensor:
        if b.ndim == 3:
            return b[0, 0, :].contiguous()
        if b.ndim == 1:
            return b.contiguous()
        return b.flatten()[:nu].contiguous()
    return _flatten(u_lower).float(), _flatten(u_upper).float()


def _normalize_init(u_init: Optional[Tensor], B: int, T: int, nu: int,
                    device: torch.device, dtype: torch.dtype) -> Tensor:
    if u_init is None:
        out = torch.zeros(B, T, nu, device=device, dtype=torch.float32)
        out[:, :, 0] = 0.752 * 9.8066  # hover thrust
        return out
    if u_init.ndim == 3:
        out = u_init.permute(1, 0, 2)
    elif u_init.shape == (nu, B):
        out = u_init.t().unsqueeze(1).expand(B, T, nu)
    else:
        out = u_init.unsqueeze(0).expand(B, T, nu)
    return out.contiguous().float()


class DroneMPCFunction(torch.autograd.Function):
    """
    Differentiable wrapper around the CUDA iLQR drone solver.

    Forward: calls the C++ solver (runs under automatic no_grad).
    Backward: implicit differentiation via adjoint Riccati recursion,
              identical to DifferentialMPC-Amos Like/controller.py ILQRSolve.
    """

    @staticmethod
    def forward(
        ctx,
        x0: Tensor,       # [B, 10]
        C_bt: Tensor,      # [B, T, n, n]  (requires_grad from policy net)
        c_bt: Tensor,      # [B, T, n]     (requires_grad from policy net)
        C_final: Tensor,   # [B, n, n]
        c_final: Tensor,   # [B, n]
        x_ref: Tensor,     # [B, T+1, nx]
        u_ref: Tensor,     # [B, T, nu]
        solver_cfg: dict,  # non-differentiable solver config
    ):
        ext = solver_cfg['ext']
        result = ext.drone_ilqr_solve(
            x0,
            solver_cfg['U_init'],
            C_bt,
            c_bt,
            C_final,
            c_final,
            x_ref,
            u_ref,
            solver_cfg['u_min'],
            solver_cfg['u_max'],
            solver_cfg['dt'],
            solver_cfg['lqr_iter'],
            solver_cfg['eps'],
            solver_cfg['converge_tol'],
        )

        X, U = result[0], result[1]        # [B, T+1, nx], [B, T, nu]

        # Save inputs/outputs via save_for_backward
        ctx.save_for_backward(X, U, x_ref, u_ref)

        # Save solver intermediates as ctx attributes
        ctx.l_xx = result[2]        # [B, T, nx, nx]
        ctx.l_uu = result[3]        # [B, T, nu, nu]
        ctx.l_xu = result[4]        # [B, T, nx, nu]
        ctx.l_xxN = result[5]       # [B, nx, nx]
        ctx.A_lin = result[6]       # [B, T, nx, nx]
        ctx.Bm_lin = result[7]      # [B, T, nx, nu]
        ctx.tight_mask = result[8]  # [B, T, nu]
        ctx.reg_eps = solver_cfg.get('reg_eps', 1e-6)
        ctx.nx = solver_cfg.get('nx', 10)
        ctx.nu = solver_cfg.get('nu', 4)
        ctx.ext = ext

        return X, U

    @staticmethod
    def backward(ctx, grad_X: Tensor, grad_U: Tensor):
        need_x0, need_C, need_c, need_C_final, need_c_final, need_xref, need_uref, _ = ctx.needs_input_grad

        X, U, x_ref, u_ref = ctx.saved_tensors
        l_xx = ctx.l_xx
        l_uu = ctx.l_uu
        l_xu = ctx.l_xu
        l_xxN = ctx.l_xxN
        A_lin = ctx.A_lin
        Bm_lin = ctx.Bm_lin
        tight_mask = ctx.tight_mask
        reg_eps = ctx.reg_eps
        nx = ctx.nx
        nu = ctx.nu

        B = X.shape[0]
        T = U.shape[1]
        dtype = X.dtype
        device = X.device

        # If one output is unused, autograd can pass None.
        if grad_X is None:
            grad_X = torch.zeros_like(X)
        if grad_U is None:
            grad_U = torch.zeros_like(U)

        # ------------------------------------------------------------------
        # Adjoint Riccati backward sweep
        # Reference: DifferentialMPC-Amos Like/controller.py:1196-1312
        # ------------------------------------------------------------------
        eye_nu = torch.eye(nu, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        reg_I = eye_nu * reg_eps

        # Try fused C++/CUDA adjoint backward pass first.
        K_seq = None
        k_seq = None
        v = None
        ext = getattr(ctx, "ext", None)
        backward_impl = os.environ.get("FASTMPC_BACKWARD_IMPL", "auto").strip().lower()
        if backward_impl not in {"auto", "cpp", "python"}:
            backward_impl = "auto"

        can_use_fused = (
            ext is not None
            and hasattr(ext, "drone_backward_pass_batched")
            and X.is_cuda
            and X.dtype == torch.float32
            and A_lin.dtype == torch.float32
            and Bm_lin.dtype == torch.float32
            and l_xx.dtype == torch.float32
            and l_uu.dtype == torch.float32
            and l_xu.dtype == torch.float32
            and backward_impl in {"auto", "cpp"}
        )
        if can_use_fused:
            try:
                # Saved C++ outputs (A_lin, Bm_lin, l_xx, l_uu, l_xu, l_xxN, tight_mask)
                # are already contiguous; only the autograd-supplied gradients need it.
                l_x = (-grad_X[:, :T]).contiguous()
                l_u = (-grad_U).contiguous()
                l_xN = (-grad_X[:, -1]).contiguous()
                if isinstance(tight_mask, torch.Tensor):
                    tight_mask_in = tight_mask
                else:
                    tight_mask_in = torch.empty(0, device=device, dtype=torch.bool)
                K_seq, k_seq, v = ext.drone_backward_pass_batched(
                    A_lin, Bm_lin, l_xx, l_uu, l_xu,
                    l_x, l_u, l_xxN, l_xN,
                    tight_mask_in, float(reg_eps),
                )

                # Fused implicit-diff assembly: forward sweep + all grad outputs
                # produced by a single CUDA kernel (no Python autograd loop).
                if hasattr(ext, "drone_implicit_diff_assembly") and os.environ.get(
                    "FASTMPC_FUSED_BACKWARD", "1"
                ) == "1":
                    g_C, g_c, g_x0, g_xref, g_uref, g_C_f, g_c_f = ext.drone_implicit_diff_assembly(
                        K_seq, k_seq, v,
                        A_lin, Bm_lin, X, U, x_ref, u_ref,
                        bool(need_xref), bool(need_uref),
                        bool(need_C_final), bool(need_c_final),
                    )
                    return (
                        -v if need_x0 else None,
                        g_C if need_C else None,
                        g_c if need_c else None,
                        g_C_f if need_C_final else None,
                        g_c_f if need_c_final else None,
                        g_xref if need_xref else None,
                        g_uref if need_uref else None,
                        None,  # solver_cfg
                    )
            except Exception:
                K_seq = None
                k_seq = None
                v = None

        if K_seq is None or k_seq is None or v is None:
            # Terminal conditions (note sign convention from DiffMPC lines 79, 82)
            V = l_xxN                         # [B, nx, nx]
            v = -grad_X[:, -1]               # [B, nx]

            K_seq_list = [None] * T
            k_seq_list = [None] * T

            for t in reversed(range(T)):
                A_t = A_lin[:, t]             # [B, nx, nx]
                B_t = Bm_lin[:, t]            # [B, nx, nu]

                # Q-function matrices
                V_A = torch.matmul(V, A_t)    # [B, nx, nx]
                V_B = torch.matmul(V, B_t)    # [B, nx, nu]

                Q_xx = l_xx[:, t] + torch.matmul(A_t.transpose(-1, -2), V_A)
                Q_xu = l_xu[:, t] + torch.matmul(A_t.transpose(-1, -2), V_B)
                Q_uu = l_uu[:, t] + torch.matmul(B_t.transpose(-1, -2), V_B) + reg_I

                # Q-function linear terms (incoming gradients, negated as in DiffMPC line 82)
                q_x = -grad_X[:, t] + torch.matmul(
                    A_t.transpose(-1, -2), v.unsqueeze(-1)
                ).squeeze(-1)
                q_u = -grad_U[:, t] + torch.matmul(
                    B_t.transpose(-1, -2), v.unsqueeze(-1)
                ).squeeze(-1)

                # Constraint handling: zero out tight (active-bound) dimensions.
                # Reduced-space (Schur complement) solve over the free set,
                # matching the convention used by mpc.pytorch.pnqp (1e-11 ridge).
                if tight_mask is not None:
                    I_tight = tight_mask[:, t].to(dtype=torch.bool)   # [B, nu]
                    free = (~I_tight).to(dtype=dtype)                 # [B, nu]
                    free_outer = _outer(free, free)
                    Q_uu = Q_uu * free_outer + torch.diag_embed(I_tight.to(dtype=dtype) * 1e-11) + reg_I
                    Q_xu = Q_xu * free.unsqueeze(-2)
                    q_u = q_u * free

                Q_ux = Q_xu.transpose(-1, -2)

                # Solve for feedback gains K, k.
                # Optional PNQP path: when FASTMPC_USE_PNQP=1, use mpc.pytorch's
                # projected-Newton solver to factor Q_uu over the free set and
                # reuse that LU for both right-hand sides (Q_ux and q_u).
                if os.environ.get("FASTMPC_USE_PNQP", "0") == "1":
                    pnqp = _get_pnqp()
                    huge = 1e6 * torch.ones(B, nu, dtype=dtype, device=device)
                    _, Q_uu_lu, _, _ = pnqp(Q_uu, q_u, -huge, huge, n_iter=1)
                    if isinstance(Q_uu_lu, tuple):  # n>1 path returns LU
                        Kt = -torch.linalg.lu_solve(Q_uu_lu[0], Q_uu_lu[1], Q_ux)
                        kt = -torch.linalg.lu_solve(
                            Q_uu_lu[0], Q_uu_lu[1], q_u.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        Kt = -torch.linalg.solve(Q_uu, Q_ux)
                        kt = -torch.linalg.solve(Q_uu, q_u.unsqueeze(-1)).squeeze(-1)
                else:
                    try:
                        L = torch.linalg.cholesky(Q_uu)
                        Kt = -torch.cholesky_solve(Q_ux, L)
                        kt = -torch.cholesky_solve(q_u.unsqueeze(-1), L).squeeze(-1)
                    except Exception:
                        Kt = -torch.linalg.solve(Q_uu, Q_ux)
                        kt = -torch.linalg.solve(Q_uu, q_u.unsqueeze(-1)).squeeze(-1)

                K_seq_list[t] = Kt
                k_seq_list[t] = kt

                # Update value function (Reference: DiffMPC controller.py:1282-1291)
                Kt_T = Kt.transpose(-1, -2)
                V = (Q_xx
                     + torch.matmul(Kt_T, torch.matmul(Q_uu, Kt))
                     + torch.matmul(Kt_T, Q_ux)
                     + torch.matmul(Q_xu, Kt))

                q_u_aug = torch.matmul(Q_uu, kt.unsqueeze(-1)).squeeze(-1) + q_u
                v = (q_x
                     + torch.matmul(Kt_T, q_u_aug.unsqueeze(-1)).squeeze(-1)
                     + torch.matmul(Q_xu, kt.unsqueeze(-1)).squeeze(-1))

            K_seq = torch.stack(K_seq_list, dim=1)  # [B, T, nu, nx]
            k_seq = torch.stack(k_seq_list, dim=1)  # [B, T, nu]

        # ------------------------------------------------------------------
        # Forward sweep: compute perturbations dX, dU
        # Reference: DiffMPC controller.py:1296-1310
        # ------------------------------------------------------------------
        dX = torch.zeros(B, T + 1, nx, dtype=dtype, device=device)
        dU = torch.zeros(B, T, nu, dtype=dtype, device=device)

        for t in range(T):
            dx_t = dX[:, t]
            du_t = (torch.matmul(
                K_seq[:, t], dx_t.unsqueeze(-1)
            ).squeeze(-1) + k_seq[:, t])
            dU[:, t] = du_t
            dX[:, t + 1] = (
                torch.matmul(A_lin[:, t], dx_t.unsqueeze(-1)).squeeze(-1)
                + torch.matmul(Bm_lin[:, t], du_t.unsqueeze(-1)).squeeze(-1)
            )

        # ------------------------------------------------------------------
        # Compute gradients w.r.t. cost parameters
        # Reference: DiffMPC controller.py:88-127 (ILQRSolve.backward)
        # ------------------------------------------------------------------

        # tau = [x - x_ref; u - u_ref], d_tau = [dX; dU]
        tau = torch.cat(
            [X[:, :T] - x_ref[:, :T], U - u_ref], dim=-1
        )                                                     # [B, T, nx+nu]
        d_tau = torch.cat([dX[:, :T], dU], dim=-1)            # [B, T, nx+nu]

        grad_C = None
        if need_C:
            grad_C = -0.5 * (_outer(d_tau, tau) + _outer(tau, d_tau))  # [B, T, n, n]

        grad_c = None
        if need_c:
            grad_c = -d_tau                                            # [B, T, n]

        # Terminal cost gradients
        grad_C_final = None
        grad_c_final = None
        if need_C_final or need_c_final:
            tauN = torch.cat([
                X[:, -1] - x_ref[:, -1],
                torch.zeros(B, nu, device=device, dtype=dtype),
            ], dim=-1)                                            # [B, nx+nu]
            d_tauN = torch.cat([
                dX[:, -1],
                torch.zeros(B, nu, device=device, dtype=dtype),
            ], dim=-1)                                            # [B, nx+nu]
            if need_C_final:
                grad_C_final = -0.5 * (_outer(d_tauN, tauN) + _outer(tauN, d_tauN))
            if need_c_final:
                grad_c_final = -d_tauN

        # Gradient w.r.t. initial state (co-state at t=0)
        grad_x0 = -v if need_x0 else None                    # [B, nx]

        # Gradients w.r.t. references
        # Reference: DiffMPC controller.py:103-119
        grad_xref = None
        if need_xref:
            # grad_xref depends on d_tau, regardless of whether grad_c is requested.
            grad_xref = torch.zeros_like(x_ref)
            grad_xref[:, :T, :] = d_tau[..., :nx]
            grad_xref[:, -1, :] = dX[:, -1]

        grad_uref = None
        if need_uref:
            grad_uref = d_tau[..., nx:]                       # [B, T, nu]

        # Return gradients for each forward() input
        # (x0, C_bt, c_bt, C_final, c_final, x_ref, u_ref, solver_cfg)
        return (
            grad_x0,
            grad_C,
            grad_c,
            grad_C_final,
            grad_c_final,
            grad_xref,
            grad_uref,
            None,  # solver_cfg (non-differentiable)
        )


class FastMPC(nn.Module):
    """
    Drop-in replacement for mpc.pytorch's MPC class using dmpc_perf solver.

    Supports two modes:
    - backprop=True + grad enabled: differentiable via implicit KKT differentiation
    - backprop=False or no_grad: inference_mode for maximum speed

    Usage:
        # Original mpc.pytorch usage:
        # x, u, objs = mpc.MPC(n_state, n_ctrl, T, ...)(x_init, cost, dx)

        # With FastMPC:
        # x, u, objs = FastMPC(n_state, n_ctrl, T, ...)(x_init, cost, dx)
    """

    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        T: int,
        u_lower: Optional[Tensor] = None,
        u_upper: Optional[Tensor] = None,
        u_init: Optional[Tensor] = None,
        lqr_iter: int = 20,
        verbose: int = -1,
        exit_unconverged: bool = False,
        detach_unconverged: bool = True,
        linesearch_decay: float = 0.2,
        max_linesearch_iter: int = 10,
        grad_method: GradMethods = GradMethods.ANALYTIC,
        eps: float = 1e-6,
        n_batch: Optional[int] = None,
        backprop: bool = True,
        slew_rate_penalty: Optional[float] = None,
        prev_ctrl: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.u_init = u_init
        self.lqr_iter = lqr_iter
        self.verbose = verbose
        self.eps = eps
        self.n_batch = n_batch
        self.backprop = backprop

        # Load the C++ extension
        self._ext = None
        self._dt = 0.02  # Default dt, will be set from dynamics

    def _get_ext(self):
        if self._ext is None:
            self._ext = load_drone_ilqr_ext()
        return self._ext

    def forward(
        self,
        x_init: Tensor,
        cost: Union[QuadCost, nn.Module],
        dx: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Run MPC optimization.

        Args:
            x_init: Initial state [B, n_state]
            cost: QuadCost(C, c) where C is [T, B, n, n] and c is [T, B, n]
            dx: Dynamics module (used to get dt)

        Returns:
            x: State trajectory [T+1, B, n_state]
            u: Control trajectory [T, B, n_ctrl]
            objs: Objectives [T, B] (placeholder)
        """
        assert isinstance(cost, QuadCost), "Only QuadCost supported"

        B = x_init.size(0)
        T = self.T
        nx = self.n_state
        nu = self.n_ctrl
        n = nx + nu
        device = x_init.device
        dtype = x_init.dtype

        dt = getattr(dx, 'dt', 0.02)

        C_bt, c_bt = _normalize_cost(cost.C, cost.c, B, T, n, device, dtype)
        # mpc.pytorch convention: V_T = 0 (no terminal cost on x_T).
        C_final = _zeros((B, n, n), device, torch.float32)
        c_final = _zeros((B, n), device, torch.float32)
        x_ref = _zeros((B, T + 1, nx), device, torch.float32)
        u_ref = _zeros((B, T, nu), device, torch.float32)
        u_min, u_max = _normalize_bounds(self.u_lower, self.u_upper, nu, device, dtype)
        U_init = _normalize_init(self.u_init, B, T, nu, device, dtype)
        x0 = x_init.float().contiguous()

        ext = self._get_ext()

        if torch.is_grad_enabled() and self.backprop:
            # ---- DIFFERENTIABLE PATH ----
            # Use DroneMPCFunction for implicit differentiation via KKT.
            # Note: autograd.Function.forward() runs under no_grad automatically,
            # so the C++ solver has no autograd overhead.
            # 'unroll' mode is NOT supported by this backend (solver is C++/CUDA).
            solver_cfg = {
                'ext': ext,
                'U_init': U_init,
                'u_min': u_min,
                'u_max': u_max,
                'dt': float(dt),
                'lqr_iter': self.lqr_iter,
                'eps': self.eps,
                'converge_tol': 1e-6,
                'reg_eps': self.eps,
                'nx': nx,
                'nu': nu,
            }
            X, U = DroneMPCFunction.apply(
                x0, C_bt, c_bt, C_final, c_final, x_ref, u_ref, solver_cfg
            )
        else:
            # ---- INFERENCE PATH ----
            # Maximum speed: inference_mode skips all autograd bookkeeping.
            with torch.inference_mode():
                result = ext.drone_ilqr_solve(
                    x0,
                    U_init,
                    C_bt,
                    c_bt,
                    C_final,
                    c_final,
                    x_ref,
                    u_ref,
                    u_min,
                    u_max,
                    float(dt),
                    self.lqr_iter,
                    self.eps,
                    1e-6,  # converge_tol
                )
            X, U = result[0], result[1]  # [B, T+1, nx], [B, T, nu]

        # Convert back from [B, T, ...] to [T, B, ...] for mpc.pytorch compatibility
        x_out = X.permute(1, 0, 2).contiguous()  # [T+1, B, nx]
        u_out = U.permute(1, 0, 2).contiguous()  # [T, B, nu]

        # Cast back to original dtype if needed
        if dtype != torch.float32:
            x_out = x_out.to(dtype)
            u_out = u_out.to(dtype)

        # Placeholder for objectives (dmpc_perf doesn't return per-step costs)
        objs = torch.zeros(T, B, device=device, dtype=dtype)

        return x_out, u_out, objs


# Alias for drop-in replacement
MPC = FastMPC
