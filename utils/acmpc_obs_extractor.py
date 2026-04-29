from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gym
import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


@dataclass(frozen=True)
class AcmpcObsLayout:
    """
    Observation layout used in this repo to match AC-MPC (arXiv:2306.09852):

    - First `mpc_state_dim` entries: raw state for MPC (p, q, v) -> 10 dims
    - Remaining entries: policy observation o_t = [v_t, R_t, o_track] -> 36 dims (for N=2 gates)

    The policy MLP should only see the "policy observation" tail, not the MPC state prefix.
    """

    mpc_state_dim: int = 10


class AcmpcPolicyObsExtractor(BaseFeaturesExtractor):
    """
    Slice a flat Box observation to keep only the policy observation tail.
    This lets the environment provide a single flat vector that contains:
    - state for the differentiable MPC (prefix)
    - observation for the neural cost network (suffix)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        mpc_state_dim: int = 10,
        policy_obs_dim: Optional[int] = None,
    ):
        assert isinstance(observation_space, gym.spaces.Box), "Only Box observations are supported."
        assert len(observation_space.shape) == 1, "Only 1D Box observations are supported."

        self.mpc_state_dim = int(mpc_state_dim)
        full_dim = int(observation_space.shape[0])
        assert 0 < self.mpc_state_dim < full_dim, "Invalid mpc_state_dim for the given observation_space."

        if policy_obs_dim is None:
            self.policy_obs_dim = full_dim - self.mpc_state_dim
        else:
            self.policy_obs_dim = int(policy_obs_dim)
            assert self.mpc_state_dim + self.policy_obs_dim <= full_dim, "policy_obs_dim out of bounds."

        super().__init__(observation_space, features_dim=self.policy_obs_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (batch, full_dim)
        start = self.mpc_state_dim
        end = start + self.policy_obs_dim
        return observations[..., start:end]

