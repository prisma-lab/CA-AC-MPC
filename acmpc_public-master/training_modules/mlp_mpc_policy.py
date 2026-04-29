import os
import sys
from pathlib import Path

# Make this module runnable without relying on the caller mutating sys.path.
_HERE = Path(__file__).resolve()
_ACMPC_ROOT = _HERE.parents[1]  # acmpc_public-master/
sys.path.insert(0, str(_ACMPC_ROOT / "mpc.pytorch"))
sys.path.insert(0, str(_ACMPC_ROOT / "diff_mpc_drones"))

import drone  # noqa: E402
import il_env  # noqa: E402

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.policies import ActorCriticPolicy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allows importing CustomNetwork without SB3 installed
    PPO = None  # type: ignore

    class ActorCriticPolicy(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "stable_baselines3 is required to instantiate MlpMpcPolicy."
            )


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim;

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        # MPC horizon length (paper uses short horizons; e.g. T=2 in several experiments/ablations).
        self.T = int(os.environ["ACMPC_T"])
        self.n_o = 28
        self.n_output = self.n_o * self.T
        # Run the differentiable MPC on CPU by default (more robust with this repo's drone model).
        # We move data between devices as needed.
        self.mpc_device = th.device("cpu")
        self.predictions = None
        self.last_cost_vectors = None

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU()
        )


        self.mpc_env = il_env.IL_Env("drone", mpc_T=self.T)
        self.dx = drone.DroneDx(device=str(self.mpc_device))
        self.u_prev = None

        print(self.policy_net)
        print(self.value_net)

    def forward(self, features: th.Tensor, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """ 
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: th.Tensor) -> th.Tensor:
        policy_device = features.device
        states = states.float()
        features_in = features[:, :self.features_in_dim]
        if (states.ndimension() == 1):
            states = th.unsqueeze(states, dim=0)

        # [p, q, v]:
        states = states[:, 0:10]

        # Forward MLP to get cost function for MPC
        sigmoid_cost_all = self.policy_net(features_in)
        # Expose for logging/analysis (stored on CPU to avoid leaking GPU memory).
        self.last_cost_vectors = sigmoid_cost_all.detach().to("cpu")

        # [DIAG] Log sigmoid output statistics
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 1000 == 0:
            with th.no_grad():
                print(f"[DIAG] sigmoid_out: mean={sigmoid_cost_all.mean():.4f}, "
                      f"std={sigmoid_cost_all.std():.4f}, "
                      f"min={sigmoid_cost_all.min():.4f}, max={sigmoid_cost_all.max():.4f}")

        # Solve optimization in smaller batches
        n_batch = features.shape[0]

        chunk_length = 1024
        # n_chunks = n_batch // chunk_length + 1

        chunks = th.split(sigmoid_cost_all, chunk_length, dim=0)
        epsilon = 0.1
        range_Q = 100000.0
        range_p = 100000.0
        range_p_t = 2 * range_Q / 2 * self.dx.mass * 9.806
        n_tau = 14


        # Warm-start is meaningful during rollouts (no_grad) but not during PPO minibatch updates
        # (batches are shuffled, so "previous control" does not correspond to an environment anymore).
        use_warmstart = not th.is_grad_enabled()
        if use_warmstart and (self.u_prev is None or self.u_prev.shape[0] != n_batch):
            # Store as (batch, n_ctrl) to match mpc.pytorch's expected u_init layout.
            self.u_prev = th.zeros(n_batch, 4, device=self.mpc_device)
            self.u_prev[:, 0] = self.dx.mass * 9.806


        # Containers for full solution
        nom_x = th.zeros((n_batch, self.T, self.dx.n_state), device=self.mpc_device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl), device=self.mpc_device)
        idx_start = 0

        for idx, sigmoid_cost in enumerate(chunks):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            # Move to MPC device
            x_Q = sigmoid_cost[:, :14*self.T].to(device=self.mpc_device)  # these are between 0 and 1 right now
            x_p = sigmoid_cost[:, 14*self.T:].to(device=self.mpc_device)  # these are between 0 and 1 right now

            q_p = x_Q[:, :3*self.T] * range_Q + epsilon
            q_q = x_Q[:, 3*self.T:7*self.T] * range_Q + epsilon
            q_v = x_Q[:, 7*self.T:10*self.T] * range_Q + epsilon
            q_w = x_Q[:, 10*self.T:13*self.T] * range_Q + epsilon
            q_t = x_Q[:, 13*self.T:14*self.T] * range_Q + epsilon

            p_p = (x_p[:, :3*self.T] - 0.5) * range_p
            p_q = (x_p[:, 3*self.T:7*self.T] - 0.5) * range_p
            p_v = (x_p[:, 7*self.T:10*self.T] - 0.5) * range_p
            p_w = (x_p[:, 10*self.T:13*self.T] - 0.5) * range_p
            p_t = x_p[:, 13*self.T:14*self.T] * range_p_t + epsilon

            u_prev_chunk = None
            if use_warmstart:
                # mpc.pytorch expects u_init as either (T, n_ctrl) or (T, batch, n_ctrl).
                # We warm-start by repeating the previous control across the horizon.
                u0 = self.u_prev[idx_start:idx_end, :]  # (batch, n_ctrl)
                u_prev_chunk = u0.unsqueeze(0).expand(self.T, n_chunk, -1).clone()

            _Q = th.zeros(self.T, n_chunk, n_tau, n_tau, device=self.mpc_device)
            _p = th.zeros(self.T, n_chunk, n_tau, device=self.mpc_device)


            states_chunk = states[idx_start:idx_end, :].to(self.mpc_device)

            for i in range(self.T):

                Q_diag_embed_i = th.diag_embed(th.cat([q_p[:, i*3:i*3+3],
                                                         q_q[:, i*4:i*4+4],
                                                         q_v[:,i*3:i*3+3],
                                                         q_t[:,i].unsqueeze(1),
                                                         q_w[:,i*3:i*3+3]], dim=1))


                p_i = th.cat([p_p[:, i*3:i*3+3],
                             p_q[:, i*4:i*4+4],
                             p_v[:, i*3:i*3+3],
                             -p_t[:,i].unsqueeze(1),
                             p_w[:,i*3:i*3+3],
                             ], dim=1)


                _Q[i, :,:,:] = Q_diag_embed_i
                _p[i, :, :] = p_i


            # Run MPC
            lqr_iter_override = int(os.environ.get("ACMPC_MPC_MAX_ITER", "1"))
            nom_x_chunk, nom_u_chunk = self.mpc_env.mpc(
                self.dx, states_chunk, _Q, _p,
                # u_init=train_warmstart[idxs].transpose(0,1),
                u_init=u_prev_chunk,
                # eps_override=0.1,
                lqr_iter_override=lqr_iter_override,
            )

            nom_x[idx_start:idx_end, :, :] = nom_x_chunk.transpose(0,1)
            nom_u[idx_start:idx_end, :, :] = nom_u_chunk.transpose(0,1)
            idx_start = idx_end


        if use_warmstart:
            self.u_prev = nom_u[:, 0, :].detach()

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()


        # Return actions from MPC. These actions will be taken into account to create a gaussian distribution.
        # Units of first control input are thrust normalized by mass
        thrust = nom_u[:, 0, 0]/self.dx.mass
        # The other 3 control inputs are the body rates, in rad/s
        omegas = nom_u[:,0,1:4]

        # Normalize thrust so that u0=0 corresponds to hover (c=g).
        normalization_max = float(self.dx.thrust_max)  # max thrust per rotor (N)
        max_acc = (normalization_max * 4) / float(self.dx.mass)
        force_mean = 9.8066
        force_std = max(1e-6, max_acc - force_mean)
        thrust_normalized = (thrust  - force_mean) / force_std

        # print("normalized_thrust_origin")
        # print(thrust_normalized)

        omega_max = self.dx.omega_max.to(device=self.mpc_device, dtype=omegas.dtype)

        omegas_normalized = th.div(omegas, omega_max).to(device=self.mpc_device)

        inputs_normalized = th.cat((thrust_normalized.unsqueeze(1), omegas_normalized), dim=1).to(self.mpc_device)

        # [DIAG] Log normalized actions
        if self._diag_counter % 1000 == 0:
            with th.no_grad():
                print(f"[DIAG] actions_norm: mean={inputs_normalized.mean():.4f}, "
                      f"min={inputs_normalized.min():.4f}, max={inputs_normalized.max():.4f}")

        # Move back to policy device if needed (e.g. when the rest of the policy runs on GPU).
        return inputs_normalized.to(policy_device)


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.value_net(features_in)


class MlpMpcPolicy(ActorCriticPolicy):
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
            distr_identity = True, # We have a distribution identity such that there is no extra neural network after the MPC output
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # Re-initialize the last layer of the policy network.
        # The policy_net structure is [Linear, GELU, Linear, GELU, Linear, GELU, Linear, Sigmoid].
        # The last Linear layer is at index 6.
        # NOTE: Changed gain from 0.01 to 0.1 to allow wider exploration of Q/R values.
        # With gain=0.01, outputs were stuck near 0.5 (median) due to vanishing gradients.
        last_linear_layer = self.mlp_extractor.policy_net[6]
        nn.init.orthogonal_(last_linear_layer.weight, gain=0.1)
        nn.init.constant_(last_linear_layer.bias, 0.0)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
