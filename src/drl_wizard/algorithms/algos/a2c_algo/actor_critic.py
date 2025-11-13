import torch
import torch.nn as nn

from drl_wizard.algorithms.algos.a2c_algo.network.act_layer import ACTLayer
from drl_wizard.algorithms.algos.a2c_algo.network.cnn import CNNLayer
from drl_wizard.algorithms.algos.a2c_algo.network.mlp import MLPLayer
from drl_wizard.algorithms.utils import check
from drl_wizard.configs import A2CConfig
from drl_wizard.algorithms.utils import get_shape_from_obs_space


class Actor(nn.Module):
    """
    Actor network class for A2C algorithm that predicts actions given observations.

    Args:
        obs_space: Observation space object defining the shape of input observations
        action_space: Action space object defining possible actions
        cfg (A2CConfig): Configuration object containing network parameters

    The network consists of:
        1. A base network (CNN or MLP) that processes observations
        2. An action layer that outputs actions and their log probabilities

    Methods:
        forward: Produces actions and their log probabilities given observations
        evaluate_actions: Evaluates given actions and returns their log probabilities and entropy
    """

    def __init__(self, obs_space, action_space, cfg: A2CConfig):
        super(Actor, self).__init__()
        self.alg_cfg: A2CConfig = cfg
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.base = CNNLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        else:
            self.base = MLPLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        self.act = ACTLayer(action_space, self.alg_cfg.fc_hidden, self.alg_cfg.use_orthogonal, self.alg_cfg.gain)

    def forward(self, obs, available_actions=None, deterministic=False):
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs

    def evaluate_actions(self, obs, action, available_actions=None):
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for A2C algorithm that estimates state values.

    Args:
        shared_obs_space: Observation space object defining the shape of input observations
        cfg (A2CConfig): Configuration object containing network parameters

    The network consists of:
        1. A base network (CNN or MLP) that processes observations
        2. A value output layer that estimates state values

    Methods:
        forward: Produces value estimates given observations
    """

    def __init__(self, shared_obs_space, cfg: A2CConfig):
        super(Critic, self).__init__()
        self.alg_cfg = cfg
        shared_obs_shape = get_shape_from_obs_space(shared_obs_space)
        if len(shared_obs_shape) == 3:
            self.base = CNNLayer(shared_obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        else:
            self.base = MLPLayer(shared_obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        self.v_out = MLPLayer((self.alg_cfg.fc_hidden,), 1, self.alg_cfg, norm_last_layer=False)

    def forward(self, shared_ob):
        critic_features = self.base(shared_ob)
        values = self.v_out(critic_features)
        return values
