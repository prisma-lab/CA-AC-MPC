import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch as th
from torch import nn

_HERE = Path(__file__).resolve()
_ACMPC_ROOT = _HERE.parents[1]
_REPO_ROOT = _ACMPC_ROOT.parent
sys.path.insert(0, str(_ACMPC_ROOT / "diff_mpc_drones"))
sys.path.insert(0, str(_ACMPC_ROOT / "mpc.pytorch"))
sys.path.insert(0, str(_REPO_ROOT / "differentialMPCPerformance"))
sys.path.insert(0, str(_REPO_ROOT))

import drone  # noqa: E402
from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod  # noqa: E402
from mpc_compat import FastMPC, QuadCost as FastQuadCost  # noqa: E402

from gym import spaces

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
except ModuleNotFoundError:
    class ActorCriticPolicy(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "stable_baselines3 is required to instantiate MlpMpcPolicyDiffMPC."
            )


class DroneDxDiffMPCWrapper:
    def __init__(self, dx_model: drone.DroneDx):
        self.dx_model = dx_model
        self.dt = dx_model.dt

    def f_dyn(self, x: th.Tensor, u: th.Tensor, _dt: float) -> th.Tensor:
        return self.dx_model(x, u)

    def f_dyn_jac(self, x: th.Tensor, u: th.Tensor, _dt: float) -> tuple[th.Tensor, th.Tensor]:
        return self.dx_model.grad_input(x, u)


class CustomNetworkDiffMPC(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.T = int(os.environ["ACMPC_T"])
        self.n_o = 28
        self.n_output = self.n_o * self.T

        self.mpc_device = self._resolve_mpc_device()
        self.mpc_backend = self._resolve_mpc_backend()
        self.mpc_grad_mode = self._resolve_mpc_grad_mode()
        self.predictions = None
        self.last_cost_vectors = None
        self.last_backend_used = None
        self.last_mpc_metrics: Dict[str, float] = {}
        self._prev_tight_mask: th.Tensor | None = None

        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
        )

        self.dx = drone.DroneDx(device=str(self.mpc_device))
        self.dx_wrapper = DroneDxDiffMPCWrapper(self.dx)
        self.mpc_controller = self._build_mpc_controller()
        self.fast_mpc_solver = self._build_fast_mpc_solver()
        self.u_prev = None

    @staticmethod
    def _resolve_mpc_device() -> th.device:
        preferred = os.environ.get("ACMPC_MPC_DEVICE")
        if preferred is None:
            preferred = "cuda:0" if th.cuda.is_available() else "cpu"
        if preferred.startswith("cuda") and not th.cuda.is_available():
            return th.device("cpu")
        return th.device(preferred)

    @staticmethod
    def _resolve_mpc_backend() -> str:
        backend = os.environ.get("ACMPC_MPC_BACKEND", "diffmpc").strip().lower()
        allowed = {"diffmpc", "fast"}
        if backend not in allowed:
            raise ValueError(f"Unsupported ACMPC_MPC_BACKEND={backend}. Allowed: {sorted(allowed)}")
        return backend

    @staticmethod
    def _resolve_mpc_grad_mode() -> str:
        mode = os.environ.get("ACMPC_MPC_GRAD_MODE", "diff").strip().lower()
        allowed = {"diff", "stop", "unroll"}
        if mode not in allowed:
            raise ValueError(f"Unsupported ACMPC_MPC_GRAD_MODE={mode}. Allowed: {sorted(allowed)}")
        return mode

    def _build_mpc_controller(self) -> DifferentiableMPCController:
        nx, nu = self.dx.n_state, self.dx.n_ctrl
        n_tau = nx + nu
        C0 = th.zeros(1, self.T, n_tau, n_tau, device=self.mpc_device)
        c0 = th.zeros(1, self.T, n_tau, device=self.mpc_device)
        C0_final = th.zeros(1, n_tau, n_tau, device=self.mpc_device)
        c0_final = th.zeros(1, n_tau, device=self.mpc_device)

        cost_module = GeneralQuadCost(
            nx=nx,
            nu=nu,
            C=C0,
            c=c0,
            C_final=C0_final,
            c_final=c0_final,
            device=str(self.mpc_device),
        )

        omega_max = th.tensor([10.0, 10.0, 4.0], device=self.mpc_device)
        u_min = th.tensor([0.0, -omega_max[0], -omega_max[1], -omega_max[2]], device=self.mpc_device)
        u_max = th.tensor([8.5 * 4, omega_max[0], omega_max[1], omega_max[2]], device=self.mpc_device)

        max_iter = int(os.environ.get("ACMPC_MPC_MAX_ITER", "1"))

        # DifferentialMPC fa tutto in un solo rollout quando fa la linesearch
        alphas = None
        if os.environ.get("ACMPC_DIFFMPC_COMPAT_ALPHAS", "0") == "1":
            alphas = tuple(float(self.dx.linesearch_decay) ** i for i in range(int(self.dx.max_linesearch_iter)))
        alphas_env = os.environ.get("ACMPC_DIFFMPC_ALPHAS")
        if alphas_env:
            alphas = tuple(float(x.strip()) for x in alphas_env.split(",") if x.strip())
        return DifferentiableMPCController(
            f_dyn=self.dx_wrapper.f_dyn,
            f_dyn_jac=self.dx_wrapper.f_dyn_jac,
            total_time=self.T * self.dx.dt,
            step_size=self.dx.dt,
            horizon=self.T,
            cost_module=cost_module,
            u_min=u_min,
            u_max=u_max,
            grad_method=GradMethod.ANALYTIC,
            max_iter=max_iter,
            exit_unconverged=False,
            detach_unconverged=False,
            device=str(self.mpc_device),
            verbose=0,
            use_vmap_line_search=False,
            alphas=alphas,
        )

    def _build_fast_mpc_solver(self) -> FastMPC:
        horizon = self.T
        nu = self.dx.n_ctrl
        max_iter = int(os.environ.get("ACMPC_MPC_MAX_ITER", "1"))
        u_lower = th.zeros(horizon, 1, nu, device=self.mpc_device)
        u_upper = th.zeros(horizon, 1, nu, device=self.mpc_device)
        u_lower[:, :, 0] = self.dx.thrust_min * 4
        u_lower[:, :, 1] = -self.dx.omega_max[0]
        u_lower[:, :, 2] = -self.dx.omega_max[1]
        u_lower[:, :, 3] = -self.dx.omega_max[2]
        u_upper[:, :, 0] = self.dx.thrust_max * 4
        u_upper[:, :, 1] = self.dx.omega_max[0]
        u_upper[:, :, 2] = self.dx.omega_max[1]
        u_upper[:, :, 3] = self.dx.omega_max[2]
        return FastMPC(
            n_state=self.dx.n_state,
            n_ctrl=self.dx.n_ctrl,
            T=horizon,
            u_lower=u_lower,
            u_upper=u_upper,
            lqr_iter=max_iter,
            eps=float(self.dx.mpc_eps),
            n_batch=1,
            backprop=True,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=float(self.dx.linesearch_decay),
            max_linesearch_iter=int(self.dx.max_linesearch_iter),
        )

    def _build_cost_tensors(self, sigmoid_cost: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        batch_size = sigmoid_cost.shape[0]
        horizon = self.T
        n_tau = self.dx.n_state + self.dx.n_ctrl

        epsilon = 0.1
        range_q = 100000.0
        range_p = 100000.0
        range_p_t = range_q * self.dx.mass * 9.806

        x_q = sigmoid_cost[:, :14 * horizon] * range_q + epsilon
        x_p = sigmoid_cost[:, 14 * horizon:]

        q_p = x_q[:, :3 * horizon]
        q_q = x_q[:, 3 * horizon:7 * horizon]
        q_v = x_q[:, 7 * horizon:10 * horizon]
        q_w = x_q[:, 10 * horizon:13 * horizon]
        q_t = x_q[:, 13 * horizon:14 * horizon]

        p_p = (x_p[:, :3 * horizon] - 0.5) * range_p
        p_q = (x_p[:, 3 * horizon:7 * horizon] - 0.5) * range_p
        p_v = (x_p[:, 7 * horizon:10 * horizon] - 0.5) * range_p
        p_w = (x_p[:, 10 * horizon:13 * horizon] - 0.5) * range_p
        p_t = x_p[:, 13 * horizon:14 * horizon] * range_p_t + epsilon

        C = th.zeros(batch_size, horizon, n_tau, n_tau, device=self.mpc_device)
        c = th.zeros(batch_size, horizon, n_tau, device=self.mpc_device)

        for t in range(horizon):
            diag_vals = th.cat(
                [
                    q_p[:, t * 3:t * 3 + 3],
                    q_q[:, t * 4:t * 4 + 4],
                    q_v[:, t * 3:t * 3 + 3],
                    q_t[:, t].unsqueeze(1),
                    q_w[:, t * 3:t * 3 + 3],
                ],
                dim=1,
            )
            C[:, t] = th.diag_embed(diag_vals)
            c[:, t] = th.cat(
                [
                    p_p[:, t * 3:t * 3 + 3],
                    p_q[:, t * 4:t * 4 + 4],
                    p_v[:, t * 3:t * 3 + 3],
                    -p_t[:, t].unsqueeze(1),
                    p_w[:, t * 3:t * 3 + 3],
                ],
                dim=1,
            )

        C_final = th.zeros(batch_size, n_tau, n_tau, device=self.mpc_device)
        c_final = th.zeros(batch_size, n_tau, device=self.mpc_device)
        return C, c, C_final, c_final

    def _hover_u_init(self, batch_size: int) -> th.Tensor:
        U_init = th.zeros(batch_size, self.T, self.dx.n_ctrl, device=self.mpc_device)
        U_init[:, :, 0] = self.dx.mass * 9.8066
        return U_init

    def _select_effective_backend(self) -> str:
        # Fast backend now supports differentiable mode via DroneMPCFunction.
        # No fallback needed; 'unroll' mode still requires diffmpc backend.
        if self.mpc_backend == "fast" and self.mpc_grad_mode == "unroll" and th.is_grad_enabled():
            return "diffmpc"
        return self.mpc_backend

    def _solve_chunk_diffmpc(
        self,
        states_chunk: th.Tensor,
        C: th.Tensor,
        c: th.Tensor,
        C_final: th.Tensor,
        c_final: th.Tensor,
        U_init: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        n_chunk = states_chunk.shape[0]
        x_ref = th.zeros(n_chunk, self.T + 1, self.dx.n_state, device=self.mpc_device)
        u_ref = th.zeros(n_chunk, self.T, self.dx.n_ctrl, device=self.mpc_device)
        self.mpc_controller.cost_module.C = C
        self.mpc_controller.cost_module.c = c
        self.mpc_controller.cost_module.C_final = C_final
        self.mpc_controller.cost_module.c_final = c_final
        self.mpc_controller.cost_module.set_reference(x_ref, u_ref)
        # Unrolled gradients derive the actual solver iterations (surrogate when not at fixed-point).
        self.mpc_controller.preserve_gradients = bool(self.mpc_grad_mode == "unroll" and th.is_grad_enabled())
        return self.mpc_controller(states_chunk, U_init=U_init)

    def _solve_chunk_fast(
        self,
        states_chunk: th.Tensor,
        C: th.Tensor,
        c: th.Tensor,
        U_init: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        self.fast_mpc_solver.u_init = U_init.permute(1, 0, 2).contiguous()
        cost = FastQuadCost(
            C=C.permute(1, 0, 2, 3).contiguous(),
            c=c.permute(1, 0, 2).contiguous(),
        )
        X_tbn, U_tbn, _ = self.fast_mpc_solver(states_chunk, cost, self.dx)
        X_chunk = X_tbn.permute(1, 0, 2).contiguous().to(states_chunk.dtype)
        U_chunk = U_tbn.permute(1, 0, 2).contiguous().to(states_chunk.dtype)
        return X_chunk, U_chunk

    def forward(self, features: th.Tensor, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: th.Tensor) -> th.Tensor:
        policy_device = features.device
        states = states.float()
        if states.ndimension() == 1:
            states = states.unsqueeze(0)
        states = states[:, 0:10]

        features_in = features[:, :self.features_in_dim]
        sigmoid_cost_all = self.policy_net(features_in)
        self.last_cost_vectors = sigmoid_cost_all.detach().to("cpu")

        # [DIAG] Log sigmoid output statistics
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 1000 == 0:
            with th.no_grad():
                print(f"[DIAG-DiffMPC] sigmoid_out: mean={sigmoid_cost_all.mean():.4f}, "
                      f"std={sigmoid_cost_all.std():.4f}, "
                      f"min={sigmoid_cost_all.min():.4f}, max={sigmoid_cost_all.max():.4f}")

        n_batch = features.shape[0]
        chunk_length = int(os.environ.get("ACMPC_MPC_CHUNK", "1024"))
        sat_vals = []
        flip_vals = []
        converged_vals = []
        naninf_count = 0

        use_warmstart = not th.is_grad_enabled()
        if use_warmstart and (self.u_prev is None or self.u_prev.shape[0] != n_batch):
            self.u_prev = th.zeros(n_batch, self.dx.n_ctrl, device=self.mpc_device)
            self.u_prev[:, 0] = self.dx.mass * 9.8066

        nom_x = th.zeros((n_batch, self.T, self.dx.n_state), device=self.mpc_device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl), device=self.mpc_device)

        idx_start = 0
        for sigmoid_cost in th.split(sigmoid_cost_all, chunk_length, dim=0):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk

            states_chunk = states[idx_start:idx_end].to(self.mpc_device)
            C, c, C_final, c_final = self._build_cost_tensors(sigmoid_cost.to(self.mpc_device))

            if use_warmstart:
                U_init = self.u_prev[idx_start:idx_end].unsqueeze(1).expand(-1, self.T, -1).clone()
            else:
                U_init = self._hover_u_init(n_chunk)

            effective_backend = self._select_effective_backend()
            self.last_backend_used = effective_backend
            if effective_backend == "fast":
                # Set backprop flag based on grad mode:
                # "diff" → differentiable via DroneMPCFunction (implicit KKT)
                # "stop" → run solver but detach output (no grad through MPC)
                # no_grad context → inference_mode path automatically
                self.fast_mpc_solver.backprop = (self.mpc_grad_mode == "diff")

                solve_ctx = th.no_grad() if (self.mpc_grad_mode == "stop" and th.is_grad_enabled()) else nullcontext()
                with solve_ctx:
                    X_chunk, U_chunk = self._solve_chunk_fast(
                        states_chunk=states_chunk.float(),
                        C=C.float(),
                        c=c.float(),
                        U_init=U_init.float(),
                    )
                if (self.mpc_grad_mode == "stop") and th.is_grad_enabled():
                    X_chunk = X_chunk.detach()
                    U_chunk = U_chunk.detach()
            else:
                # stop-grad ablation: the actor does not receive gradient through MPC.
                solve_ctx = th.no_grad() if (self.mpc_grad_mode == "stop" and th.is_grad_enabled()) else nullcontext()
                with solve_ctx:
                    X_chunk, U_chunk = self._solve_chunk_diffmpc(
                        states_chunk=states_chunk,
                        C=C,
                        c=c,
                        C_final=C_final,
                        c_final=c_final,
                        U_init=U_init,
                    )
                if (self.mpc_grad_mode == "stop") and th.is_grad_enabled():
                    X_chunk = X_chunk.detach()
                    U_chunk = U_chunk.detach()

                tight_mask = getattr(self.mpc_controller, "tight_mask_last", None)
                if isinstance(tight_mask, th.Tensor):
                    mask_now = tight_mask.detach()
                    sat_vals.append(float(mask_now.float().mean().cpu().item()))
                    if self._prev_tight_mask is not None and self._prev_tight_mask.shape == mask_now.shape:
                        flips = th.logical_xor(mask_now, self._prev_tight_mask).float().mean()
                        flip_vals.append(float(flips.cpu().item()))
                    self._prev_tight_mask = mask_now
                converged_vals.append(float(1.0 if bool(getattr(self.mpc_controller, "converged", False)) else 0.0))
            nom_x[idx_start:idx_end] = X_chunk[:, :self.T, :]
            nom_u[idx_start:idx_end] = U_chunk
            naninf_count += int((~th.isfinite(U_chunk)).any().detach().cpu().item())
            idx_start = idx_end

        if use_warmstart:
            self.u_prev = nom_u[:, 0, :].detach()

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()

        thrust = nom_u[:, 0, 0] / self.dx.mass
        omegas = nom_u[:, 0, 1:4]

        normalization_max = float(self.dx.thrust_max)
        max_acc = (normalization_max * 4) / float(self.dx.mass)
        force_mean = 9.8066
        force_std = max(1e-6, max_acc - force_mean)
        thrust_normalized = (thrust - force_mean) / force_std

        omega_max = self.dx.omega_max.to(device=self.mpc_device, dtype=omegas.dtype)
        omegas_normalized = omegas / omega_max
        inputs_normalized = th.cat((thrust_normalized.unsqueeze(1), omegas_normalized), dim=1)

        # [DIAG] Log normalized actions
        if self._diag_counter % 1000 == 0:
            with th.no_grad():
                print(f"[DIAG-DiffMPC] actions_norm: mean={inputs_normalized.mean():.4f}, "
                      f"min={inputs_normalized.min():.4f}, max={inputs_normalized.max():.4f}")

        self.last_mpc_metrics = {
            "sat_rate_mean": float(sum(sat_vals) / max(1, len(sat_vals))) if sat_vals else float("nan"),
            "mask_flip_mean": float(sum(flip_vals) / max(1, len(flip_vals))) if flip_vals else float("nan"),
            "solver_converged_frac": float(sum(converged_vals) / max(1, len(converged_vals))) if converged_vals else float("nan"),
            "u_naninf_chunks": float(naninf_count),
            "backend_is_fast": float(1.0 if self.last_backend_used == "fast" else 0.0),
            "grad_mode_stop": float(1.0 if self.mpc_grad_mode == "stop" else 0.0),
            "grad_mode_unroll": float(1.0 if self.mpc_grad_mode == "unroll" else 0.0),
        }

        return inputs_normalized.to(policy_device)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.value_net(features_in)


class MlpMpcPolicyDiffMPC(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            distr_identity=True,
            *args,
            **kwargs,
        )

        # NOTE: Changed gain from 0.01 to 0.5 to allow wider exploration of Q/R values.
        # With gain=0.01 and 0.1, outputs were stuck near 0.5 (median).
        last_linear_layer = self.mlp_extractor.policy_net[6]
        nn.init.orthogonal_(last_linear_layer.weight, gain=0.5)
        nn.init.constant_(last_linear_layer.bias, 0.0)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkDiffMPC(self.features_dim)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Keep older checkpoints loadable when saved reference tensors used a different batch size.
        """
        patched = dict(state_dict)
        x_key = "mlp_extractor.mpc_controller.cost_module.x_ref"
        u_key = "mlp_extractor.mpc_controller.cost_module.u_ref"

        if x_key in patched and hasattr(self.mlp_extractor.mpc_controller.cost_module, "x_ref"):
            saved = patched[x_key]
            current = self.mlp_extractor.mpc_controller.cost_module.x_ref
            if isinstance(saved, th.Tensor) and isinstance(current, th.Tensor):
                if saved.ndim >= 1 and current.ndim >= 1 and saved.shape[0] != current.shape[0]:
                    patched[x_key] = saved[: current.shape[0]].contiguous()

        if u_key in patched and hasattr(self.mlp_extractor.mpc_controller.cost_module, "u_ref"):
            saved = patched[u_key]
            current = self.mlp_extractor.mpc_controller.cost_module.u_ref
            if isinstance(saved, th.Tensor) and isinstance(current, th.Tensor):
                if saved.ndim >= 1 and current.ndim >= 1 and saved.shape[0] != current.shape[0]:
                    patched[u_key] = saved[: current.shape[0]].contiguous()

        return super().load_state_dict(patched, strict=strict)
