import torch
import torch.nn as nn
from drl_wizard.algorithms.algos.trpo_algo.network.act_layer import ACTLayer
from drl_wizard.algorithms.algos.trpo_algo.network.cnn import CNNLayer
from drl_wizard.algorithms.utils import check
from drl_wizard.algorithms.algos.trpo_algo.network.mlp import MLPLayer
from drl_wizard.configs import TRPOConfig
from drl_wizard.algorithms.utils import get_shape_from_obs_space


class Actor(nn.Module):
    def __init__(self, obs_space, action_space, cfg: TRPOConfig):
        super(Actor, self).__init__()
        self.alg_cfg: TRPOConfig = cfg
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

    def get_dist(self, obs, available_actions=None):
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        return self.act.dist(actor_features, available_actions)

    def evaluate_actions(self, obs, action, available_actions=None):
        # obs = check(obs, torch.float, self.device)
        # action = check(action, torch.float, self.device)
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64, self.device)
        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(self, shared_obs_space, cfg: TRPOConfig):
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
