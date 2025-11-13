import torch
import torch.nn as nn
from drl_wizard.algorithms.algos.ppo_algo.network.act_layer import ACTLayer
from drl_wizard.algorithms.algos.ppo_algo.network.cnn import CNNLayer
from drl_wizard.algorithms.utils import check
from drl_wizard.algorithms.algos.ppo_algo.network.mlp import MLPLayer
from drl_wizard.configs import PPOConfig
from drl_wizard.algorithms.utils import get_shape_from_obs_space


class Actor(nn.Module):
    """Actor network for the PPO algorithm that outputs actions and their log probabilities.

    This class implements a neural network that takes observations as input and outputs
    actions along with their log probabilities. It supports both CNN and MLP architectures
    based on the observation space dimensionality.

    Args:
        obs_space: Observation space of the environment
        action_space: Action space of the environment
        cfg (PPOConfig): Configuration object containing network parameters

    Attributes:
        alg_cfg (PPOConfig): Configuration parameters for the algorithm
        base (nn.Module): Base network (CNN or MLP) for processing observations
        act (ACTLayer): Action layer for computing actions and probabilities
    """

    def __init__(self, obs_space, action_space, cfg: PPOConfig):
        super(Actor, self).__init__()
        self.alg_cfg: PPOConfig = cfg
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.base = CNNLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        else:
            self.base = MLPLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        self.act = ACTLayer(action_space, self.alg_cfg.fc_hidden, self.alg_cfg.use_orthogonal, self.alg_cfg.gain)
        # self.device = cfg.resolved_device

    def forward(self, obs, available_actions=None, deterministic=False):
        # obs = check(obs, torch.float, self.device)
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs

    def evaluate_actions(self, obs, action, available_actions=None):
        # obs = check(obs, torch.float, self.device)
        # action = check(action, torch.float, self.device)
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """Critic network for the PPO algorithm that estimates state values.

    This class implements a neural network that estimates the value function
    for given observations. It supports both CNN and MLP architectures based
    on the observation space dimensionality.

    Args:
        shared_obs_space: Shared observation space of the environment
        cfg (PPOConfig): Configuration object containing network parameters

    Attributes:
        alg_cfg (PPOConfig): Configuration parameters for the algorithm
        base (nn.Module): Base network (CNN or MLP) for processing observations
        v_out (MLPLayer): Output layer for computing state values
    """

    def __init__(self, shared_obs_space, cfg: PPOConfig):
        super(Critic, self).__init__()
        self.alg_cfg = cfg
        shared_obs_shape = get_shape_from_obs_space(shared_obs_space)
        if len(shared_obs_shape) == 3:
            self.base = CNNLayer(shared_obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        else:
            self.base = MLPLayer(shared_obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        self.v_out = MLPLayer((self.alg_cfg.fc_hidden,), 1, self.alg_cfg, norm_last_layer=False)
        # self.device = cfg.resolved_device

    def forward(self, shared_ob):
        critic_features = self.base(shared_ob)
        values = self.v_out(critic_features)
        return values
