import torch
import torch.nn as nn

from drl_wizard.algorithms.algos.sac_algo.network.act_layer import ACTLayer
from drl_wizard.algorithms.algos.sac_algo.network.cnn import CNNLayer
from drl_wizard.algorithms.algos.sac_algo.network.mlp import MLPLayer
from drl_wizard.algorithms.utils import check, get_num_actions, get_len_from_act_space
from drl_wizard.algorithms.utils import get_shape_from_obs_space
from drl_wizard.configs import SACConfig


class Actor(nn.Module):
    def __init__(self, obs_space, action_space, cfg: SACConfig):
        super(Actor, self).__init__()
        self.alg_cfg: SACConfig = cfg
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
            available_actions = check(available_actions, torch.int64)
        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs

    def evaluate_actions(self, obs, action, available_actions=None):
        # obs = check(obs, torch.float, self.device)
        # action = check(action, torch.float, self.device)
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64)
        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_log_probs, dist_entropy


class QNetwork(nn.Module):
    def __init__(self, shared_obs_space,action_space, cfg: SACConfig):
        super(QNetwork, self).__init__()
        self.alg_cfg = cfg
        shared_obs_shape = get_shape_from_obs_space(shared_obs_space)
        num_actions = get_len_from_act_space(action_space)
        if len(shared_obs_shape) == 3:
            self.base = CNNLayer(shared_obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
            self.v_out = MLPLayer((self.alg_cfg.fc_hidden + num_actions,), 1, self.alg_cfg, norm_last_layer=False)
            self.is_cnn=True
        else:
            q_input=(shared_obs_shape[0]+ num_actions,)
            self.base = MLPLayer(q_input, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
            self.v_out = MLPLayer((self.alg_cfg.fc_hidden,), 1, self.alg_cfg, norm_last_layer=False)
            self.is_cnn=False

        # self.device = cfg.resolved_device

    def forward(self, shared_ob,actions):
        if self.is_cnn:
            critic_features = self.base(shared_ob)
            value_input = torch.cat([critic_features, actions], dim=-1)
            values = self.v_out(value_input)
        else:
            inputs = torch.cat([shared_ob,actions],dim=-1)
            value_input = self.base(inputs)
            values = self.v_out(value_input)
        return values
