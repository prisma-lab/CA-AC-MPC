# cost.py

from __future__ import annotations
from typing import Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F


class BarrierConfig:
    def __init__(self, mu: float = 1.0, eps: float = 1e-6, delta: float = 1.0):
        self.mu = mu
        self.eps = eps
        self.delta = delta


class BarrierMuSchedule:
    def __init__(
        self,
        *,
        initial_mu: float,
        final_mu: float,
        decay_steps: int,
        mode: str = "exp",
    ):
        if decay_steps <= 0:
            raise ValueError("decay_steps must be > 0.")
        if final_mu <= 0.0 or initial_mu <= 0.0:
            raise ValueError("mu values must be > 0.")
        self.initial_mu = float(initial_mu)
        self.final_mu = float(final_mu)
        self.decay_steps = int(decay_steps)
        self.mode = mode

        if mode not in ("exp", "linear"):
            raise ValueError("mode must be 'exp' or 'linear'.")

        if mode == "exp":
            self._decay = (self.final_mu / self.initial_mu) ** (1.0 / self.decay_steps)

    def __call__(self, step: int) -> float:
        step = max(0, int(step))
        if step >= self.decay_steps:
            return self.final_mu
        if self.mode == "linear":
            frac = step / self.decay_steps
            return self.initial_mu + frac * (self.final_mu - self.initial_mu)
        return max(self.final_mu, self.initial_mu * (self._decay ** step))


class GeneralQuadCost(torch.nn.Module):
    """
    Represents a general quadratic cost of the form:
    L(x, u) = 0.5 * τ'Cτ + c'τ
    where τ = [x, u] is the augmented state-control vector.
    """

    uses_barrier = False

    def __init__(self, nx: int, nu: int, C: Tensor, c: Tensor, C_final: Tensor, c_final: Tensor, **kwargs):
        super().__init__()
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.nx, self.nu = nx, nu
        self.C, self.c = C.to(self.device), c.to(self.device)
        self.C_final, self.c_final = C_final.to(self.device), c_final.to(self.device)

        # Initialize default references
        T_ref = C.shape[0] if C.ndim == 3 else (C.shape[1] if C.ndim == 4 else 0)
        x_ref_default = torch.zeros(1, T_ref + 1, nx, device=self.device)
        u_ref_default = torch.zeros(1, T_ref, nu, device=self.device)
        self.register_buffer("x_ref", kwargs.get("x_ref", x_ref_default))
        self.register_buffer("u_ref", kwargs.get("u_ref", u_ref_default))
        self.set_reference(self.x_ref, self.u_ref)

    def set_reference(self, x_ref: Optional[Tensor] = None, u_ref: Optional[Tensor] = None):
        """Sets the reference trajectories for the state and control."""
        if x_ref is not None:
            if x_ref.ndim == 2: x_ref = x_ref.unsqueeze(0)
            self.x_ref = x_ref.to(self.device)
        if u_ref is not None:
            if u_ref.ndim == 2: u_ref = u_ref.unsqueeze(0)
            self.u_ref = u_ref.to(self.device)

    def _prepare_costs(self, B: int):
        """Prepares cost tensors for batching."""
        C, c, C_final, c_final = self.C, self.c, self.C_final, self.c_final
        if C.ndim == 3: C, c = C.unsqueeze(0), c.unsqueeze(0)
        if C_final.ndim == 2: C_final, c_final = C_final.unsqueeze(0), c_final.unsqueeze(0)
        if C.shape[0] == 1 and B > 1: C, c = C.expand(B, -1, -1, -1), c.expand(B, -1, -1)
        if C_final.shape[0] == 1 and B > 1: C_final, c_final = C_final.expand(B, -1, -1), c_final.expand(B, -1)
        return C, c, C_final, c_final

    def objective(self, X, U, *, x_ref_override=None, u_ref_override=None, **kwargs):
        """
        Calculates the total cost for the given trajectories.
        """
        _x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        _u_ref = u_ref_override if u_ref_override is not None else self.u_ref
        was_unbatched = X.ndim == 2

        if was_unbatched:
            X, U = X.unsqueeze(0), U.unsqueeze(0)
            if _x_ref.ndim == 2: _x_ref = _x_ref.unsqueeze(0)
            if _u_ref.ndim == 2: _u_ref = _u_ref.unsqueeze(0)

        B, T = X.shape[0], U.shape[1]
        if _x_ref.shape[0] == 1 and B > 1: _x_ref = _x_ref.expand(B, -1, -1)
        if _u_ref.shape[0] == 1 and B > 1: _u_ref = _u_ref.expand(B, -1, -1)

        C_run, c_run, C_final, c_final = self._prepare_costs(B)

        dx, du = X[:, :T] - _x_ref[:, :T], U - _u_ref
        tau = torch.cat((dx, du), dim=-1)

        # cost_run is a tensor of shape [B]
        cost_run = 0.5 * torch.einsum("bti,btij,btj->b", tau, C_run, tau) + torch.einsum("bti,bti->b", c_run, tau)

        dxN = X[:, -1] - _x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=X.device, dtype=X.dtype)], dim=-1)

        # cost_final is a tensor of shape [B]
        cost_final = 0.5 * torch.einsum("bi,bij,bj->b", tauN, C_final, tauN) + torch.einsum("bi,bi->b", c_final, tauN)

        # The cost should be a tensor of shape [B] (one cost value per batch element).
        total_cost_per_batch = cost_run + cost_final

        # If the original input was unbatched, return a single tensor. Otherwise, return the batch.
        return total_cost_per_batch.squeeze(0) if was_unbatched else total_cost_per_batch

    def quadraticize(self, X, U, *, x_ref_override=None, u_ref_override=None, **kwargs):
        """
        Calculates the quadratic approximations (gradient and Hessian) of the cost.
        """
        _x_ref = x_ref_override if x_ref_override is not None else self.x_ref
        _u_ref = u_ref_override if u_ref_override is not None else self.u_ref
        was_unbatched = X.ndim == 2

        if was_unbatched:
            X, U = X.unsqueeze(0), U.unsqueeze(0)
            if _x_ref.ndim == 2: _x_ref = _x_ref.unsqueeze(0)
            if _u_ref.ndim == 2: _u_ref = _u_ref.unsqueeze(0)

        B, T = X.shape[0], U.shape[1]
        if _x_ref.shape[0] == 1 and B > 1: _x_ref = _x_ref.expand(B, -1, -1)
        if _u_ref.shape[0] == 1 and B > 1: _u_ref = _u_ref.expand(B, -1, -1)

        C_run, c_run, C_final, c_final = self._prepare_costs(B)

        dx, du = X[:, :T] - _x_ref[:, :T], U - _u_ref
        tau = torch.cat((dx, du), dim=-1)

        # Gradient (∇f = Cτ + c) and Hessian (∇²f = C) for the form 0.5τ'Cτ + c'τ
        l_tau = torch.einsum("btij,btj->bti", C_run, tau) + c_run
        H_tau = C_run

        dxN = X[:, -1] - _x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, self.nu, device=X.device, dtype=X.dtype)], dim=-1)

        lN = torch.einsum("bij,bj->bi", C_final, tauN) + c_final
        HN = C_final

        l_x, l_u = l_tau[..., :self.nx], l_tau[..., self.nx:]
        l_xx, l_uu = H_tau[..., :self.nx, :self.nx], H_tau[..., self.nx:, self.nx:]
        l_xu = H_tau[..., :self.nx, self.nx:]
        l_xN, l_xxN = lN[..., :self.nx], HN[..., :self.nx, :self.nx]

        if was_unbatched:
            return l_x.squeeze(0), l_u.squeeze(0), l_xx.squeeze(0), l_xu.squeeze(0), l_uu.squeeze(0), l_xN.squeeze(
                0), l_xxN.squeeze(0)

        return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN

    @staticmethod
    def objective_pure(X, U, x_ref, u_ref, C, c, C_final, c_final, nx: int, nu: int):
        """
      objective that takes cost matrices as parameters.
        Args:
            X: State trajectory [B, T+1, nx] or [T+1, nx]
            U: Control trajectory [B, T, nu] or [T, nu]  
            x_ref: Reference state trajectory [B, T+1, nx] or [T+1, nx]
            u_ref: Reference control trajectory [B, T, nu] or [T, nu]
            C: Running cost matrices [B, T, nx+nu, nx+nu] or [T, nx+nu, nx+nu]
            c: Running cost vectors [B, T, nx+nu] or [T, nx+nu]
            C_final: Terminal cost matrix [B, nx+nu, nx+nu] or [nx+nu, nx+nu]
            c_final: Terminal cost vector [B, nx+nu] or [nx+nu]
            nx: Number of state dimensions
            nu: Number of control dimensions
            
        Returns:
            total_cost: Total cost [B] or scalar
        """
        was_unbatched = X.ndim == 2
        
        if was_unbatched:
            X, U = X.unsqueeze(0), U.unsqueeze(0)
            if x_ref.ndim == 2: x_ref = x_ref.unsqueeze(0)
            if u_ref.ndim == 2: u_ref = u_ref.unsqueeze(0)
            if C.ndim == 3: C = C.unsqueeze(0)
            if c.ndim == 2: c = c.unsqueeze(0)
            if C_final.ndim == 2: C_final = C_final.unsqueeze(0)
            if c_final.ndim == 1: c_final = c_final.unsqueeze(0)
        
        B, T = X.shape[0], U.shape[1]
        
        # Expand references to batch size if needed
        if x_ref.shape[0] == 1 and B > 1: x_ref = x_ref.expand(B, -1, -1)
        if u_ref.shape[0] == 1 and B > 1: u_ref = u_ref.expand(B, -1, -1)
        
        # Expand cost matrices to batch size if needed (handles fixed costs case)
        if C.shape[0] == 1 and B > 1: C = C.expand(B, -1, -1, -1)
        if c.shape[0] == 1 and B > 1: c = c.expand(B, -1, -1)
        if C_final.shape[0] == 1 and B > 1: C_final = C_final.expand(B, -1, -1)
        if c_final.shape[0] == 1 and B > 1: c_final = c_final.expand(B, -1)
        
        # Running cost calculation
        dx, du = X[:, :T] - x_ref[:, :T], U - u_ref
        tau = torch.cat((dx, du), dim=-1)
        
        # cost_run is a tensor of shape [B]
        cost_run = 0.5 * torch.einsum("bti,btij,btj->b", tau, C, tau) + torch.einsum("bti,bti->b", c, tau)
        
        # Terminal cost calculation
        dxN = X[:, -1] - x_ref[:, -1]
        tauN = torch.cat([dxN, torch.zeros(B, nu, device=X.device, dtype=X.dtype)], dim=-1)
        
        # cost_final is a tensor of shape [B]
        cost_final = 0.5 * torch.einsum("bi,bij,bj->b", tauN, C_final, tauN) + torch.einsum("bi,bi->b", c_final, tauN)
        
        # Total cost
        total_cost_per_batch = cost_run + cost_final
        
        # Return scalar if input was unbatched
        return total_cost_per_batch.squeeze(0) if was_unbatched else total_cost_per_batch


class QuadCostWithBarrier(GeneralQuadCost):
    """
    Quadratic cost augmented with a smooth log-barrier on state constraints g(x,t) <= 0.
    The constraint module is evaluated per time-step: g_t = g(x_t, params_t).
    """

    uses_barrier = True

    def __init__(
            self,
            nx: int,
            nu: int,
            C: Tensor,
            c: Tensor,
            C_final: Tensor,
            c_final: Tensor,
            *,
            constraint_module: Optional[torch.nn.Module] = None,
            constraint_params: Optional[Tensor] = None,
            barrier_cfg: Optional[BarrierConfig] = None,
            mu_schedule: Optional[BarrierMuSchedule] = None,
            include_terminal: bool = True,
            hess_approx: str = "gn",
            **kwargs
    ):
        super().__init__(nx, nu, C, c, C_final, c_final, **kwargs)
        self.constraint_module = constraint_module
        self.constraint_params = constraint_params
        self.barrier_cfg = barrier_cfg or BarrierConfig()
        self.mu_schedule = mu_schedule
        self.include_terminal = include_terminal
        self.hess_approx = hess_approx
        self.uses_barrier = constraint_module is not None

    def set_constraint_params(self, params: Optional[Tensor]) -> None:
        self.constraint_params = params

    def set_mu_schedule(self, schedule: Optional[BarrierMuSchedule]) -> None:
        self.mu_schedule = schedule

    def step_barrier_schedule(self, step: int) -> Optional[float]:
        if self.mu_schedule is None:
            return None
        self.barrier_cfg.mu = float(self.mu_schedule(step))
        return self.barrier_cfg.mu

    def _broadcast_constraint_params(self, params: Optional[Tensor], B: int, T1: int) -> Optional[Tensor]:
        if params is None:
            return None
        if params.ndim < 2:
            raise ValueError("constraint_params must include a time dimension [T+1, ...] or [B, T+1, ...].")
        # If missing batch dim (params.shape[0] == T1), add it.
        if params.shape[0] == T1:
            params = params.unsqueeze(0)
        if params.shape[0] == 1:
            params = params.expand(B, *params.shape[1:])
        elif params.shape[0] != B:
            # Allow repeat_interleave when evaluating candidate batches (B * n_alpha)
            if B % params.shape[0] == 0:
                repeat = B // params.shape[0]
                params = params.repeat_interleave(repeat, dim=0)
            else:
                raise ValueError(f"constraint_params batch dimension {params.shape[0]} does not match B={B}.")
        if params.shape[1] != T1:
            raise ValueError(f"constraint_params time dimension {params.shape[1]} does not match T+1={T1}.")
        return params

    def _call_constraint(self, x: Tensor, p: Optional[Tensor]) -> Tensor:
        if self.constraint_module is None:
            raise RuntimeError("constraint_module is not set.")
        if p is None:
            return self.constraint_module(x)
        try:
            return self.constraint_module(x, p)
        except TypeError:
            return self.constraint_module(x)

    def _evaluate_constraints(self, X: Tensor, params: Optional[Tensor]) -> Tensor:
        B, T1, _ = X.shape
        g_list = []
        for t in range(T1):
            x_t = X[:, t]
            p_t = params[:, t] if params is not None else None
            g_t = self._call_constraint(x_t, p_t)
            if g_t.ndim == 1:
                if g_t.shape[0] == B:
                    g_t = g_t.unsqueeze(-1)
                else:
                    g_t = g_t.unsqueeze(0)
            elif g_t.ndim > 2:
                raise ValueError("constraint_module must return shape [B, m] per time-step.")
            g_list.append(g_t)
        return torch.stack(g_list, dim=1)

    def _barrier_terms(self, g: Tensor) -> tuple[Tensor, Tensor]:
        mu, eps, delta = self.barrier_cfg.mu, self.barrier_cfg.eps, self.barrier_cfg.delta
        sp = F.softplus(-g / delta)
        b = sp * delta + eps
        phi = -mu * torch.log(b)

        s = torch.sigmoid(-g / delta)
        s_prime = -(s * (1.0 - s)) / delta
        d2 = mu * (s_prime * b + s * s) / (b * b)
        return phi, d2

    def _barrier_objective(self, X: Tensor, params: Optional[Tensor]) -> Tensor:
        was_unbatched = X.ndim == 2
        if was_unbatched:
            X = X.unsqueeze(0)
        B, T1, _ = X.shape
        params_b = self._broadcast_constraint_params(params, B, T1)
        g = self._evaluate_constraints(X, params_b)
        if not self.include_terminal:
            g = g[:, :-1]
        phi, _ = self._barrier_terms(g)
        barrier_cost = phi.sum(dim=(-1, -2))
        return barrier_cost.squeeze(0) if was_unbatched else barrier_cost

    def objective(self, X, U, *, x_ref_override=None, u_ref_override=None, constraint_params_override=None, **kwargs):
        base_cost = super().objective(X, U, x_ref_override=x_ref_override, u_ref_override=u_ref_override)
        if self.constraint_module is None:
            return base_cost
        params = constraint_params_override if constraint_params_override is not None else self.constraint_params
        barrier_cost = self._barrier_objective(X, params)
        return base_cost + barrier_cost

    def _constraint_jacobian_step(self, x_t: Tensor, p_t: Optional[Tensor], m: int) -> Tensor:
        B, nx = x_t.shape
        J_out = torch.zeros(B, m, nx, device=x_t.device, dtype=x_t.dtype)
        for b in range(B):
            x_b = x_t[b].detach().requires_grad_(True)
            p_b = p_t[b] if p_t is not None else None

            def g_fn(x_in):
                x_in = x_in.unsqueeze(0)
                if p_b is None:
                    g_out = self._call_constraint(x_in, None)
                else:
                    g_out = self._call_constraint(x_in, p_b.unsqueeze(0))
                return g_out.squeeze(0)

            J_b = torch.autograd.functional.jacobian(g_fn, x_b, create_graph=False)
            J_out[b] = J_b
        return J_out

    def quadraticize(self, X, U, *, x_ref_override=None, u_ref_override=None, constraint_params_override=None, **kwargs):
        was_unbatched = X.ndim == 2
        if was_unbatched:
            X = X.unsqueeze(0)
            U = U.unsqueeze(0)
            if x_ref_override is not None and x_ref_override.ndim == 2:
                x_ref_override = x_ref_override.unsqueeze(0)
            if u_ref_override is not None and u_ref_override.ndim == 2:
                u_ref_override = u_ref_override.unsqueeze(0)

        l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN = super().quadraticize(
            X, U, x_ref_override=x_ref_override, u_ref_override=u_ref_override
        )

        if self.constraint_module is None:
            if was_unbatched:
                return l_x.squeeze(0), l_u.squeeze(0), l_xx.squeeze(0), l_xu.squeeze(0), l_uu.squeeze(0), l_xN.squeeze(
                    0), l_xxN.squeeze(0)
            return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN

        B, T, nx = X.shape[0], U.shape[1], self.nx
        params = constraint_params_override if constraint_params_override is not None else self.constraint_params
        params_b = self._broadcast_constraint_params(params, B, T + 1)

        with torch.enable_grad():
            X_req = X.detach().requires_grad_(True)
            g = self._evaluate_constraints(X_req, params_b)
            if not self.include_terminal:
                g_use = g[:, :-1]
            else:
                g_use = g

            phi, d2 = self._barrier_terms(g_use)
            barrier_cost = phi.sum(dim=(-1, -2))
            grad_X = torch.autograd.grad(barrier_cost.sum(), X_req, create_graph=False, retain_graph=False)[0]

        l_x_bar = grad_X[:, :T]
        l_xN_bar = grad_X[:, -1] if self.include_terminal else torch.zeros_like(grad_X[:, -1])

        l_xx_bar = torch.zeros(B, T, nx, nx, device=X.device, dtype=X.dtype)
        l_xxN_bar = torch.zeros(B, nx, nx, device=X.device, dtype=X.dtype)

        if self.hess_approx == "gn":
            d2 = torch.clamp(d2, min=0.0)
            T_use = g_use.shape[1]
            for t in range(T_use):
                x_t = X[:, t]
                p_t = params_b[:, t] if params_b is not None else None
                J_t = self._constraint_jacobian_step(x_t, p_t, g_use.shape[-1])
                w_t = d2[:, t]
                H_t = torch.einsum("bmi,bm,bmj->bij", J_t, w_t, J_t)
                if t < T:
                    l_xx_bar[:, t] = H_t
                else:
                    l_xxN_bar = H_t

        l_x = l_x + l_x_bar
        l_xN = l_xN + l_xN_bar
        l_xx = l_xx + l_xx_bar
        l_xxN = l_xxN + l_xxN_bar

        if was_unbatched:
            return l_x.squeeze(0), l_u.squeeze(0), l_xx.squeeze(0), l_xu.squeeze(0), l_uu.squeeze(0), l_xN.squeeze(
                0), l_xxN.squeeze(0)

        return l_x, l_u, l_xx, l_xu, l_uu, l_xN, l_xxN
