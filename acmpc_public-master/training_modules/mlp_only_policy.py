import sys
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomNetworkMlp(nn.Module):
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
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim;

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # self.device = "cuda:0"

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )


    def forward(self, features: th.Tensor, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.policy_net(features_in)


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.value_net(features_in)


class MlpOnlyPolicy(ActorCriticPolicy):
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
            distr_identity=False,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkMlp(self.features_dim)
        
