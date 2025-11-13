import torch
import torch.nn as nn

from drl_wizard.algorithms.algos.dqn_algo.network.act_layer import ACTLayer
from drl_wizard.algorithms.algos.dqn_algo.network.cnn import CNNLayer
from drl_wizard.algorithms.algos.dqn_algo.network.mlp import MLPLayer
from drl_wizard.algorithms.utils import check
from drl_wizard.algorithms.utils import get_shape_from_obs_space
from drl_wizard.configs import DQNConfig


class QActor(nn.Module):
    """
    Q-Network Actor implementing Deep Q-Network (DQN) architecture.

    This class implements a neural network that approximates the Q-function,
    mapping states to action-values. It supports both CNN and MLP base networks
    depending on the observation space, and can use dueling network architecture.

    The network consists of:
        1. A base network (either CNN or MLP) that processes observations
        2. An action layer that outputs Q-values for each action
    """

    def __init__(self, obs_space, action_space, cfg: DQNConfig):
        """
        Initialize the Q-Network actor.

        Args:
            obs_space: Observation space object defining input dimensions
            action_space: Action space object defining output dimensions
            cfg (DQNConfig): Configuration object containing network parameters
        """
        super(QActor, self).__init__()
        self.alg_cfg: DQNConfig = cfg
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.base = CNNLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        else:
            self.base = MLPLayer(obs_shape, self.alg_cfg.fc_hidden, self.alg_cfg, norm_last_layer=True)
        self.act = ACTLayer(action_space, self.alg_cfg.fc_hidden,self.alg_cfg.use_dueling, self.alg_cfg.use_orthogonal, self.alg_cfg.gain)

    def forward(self, obs,epsilon, available_actions=None, deterministic=False):
        """
        Forward pass to select actions based on observations.

        Args:
            obs: Input observation tensor
            epsilon: Exploration rate for epsilon-greedy policy
            available_actions: Optional mask for valid actions
            deterministic: If True, selects best action without exploration

        Returns:
            Selected actions based on the current policy
        """
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64,obs.device)
        actor_features = self.base(obs)
        actions = self.act(actor_features,epsilon, available_actions, deterministic)
        return actions

    def evaluate_actions(self, obs, action, available_actions=None):
        """
        Evaluate Q-values for given observation-action pairs.

        Args:
            obs: Input observation tensor
            action: Actions to evaluate
            available_actions: Optional mask for valid actions

        Returns:
            Q-values for the given observation-action pairs
        """
        if available_actions is not None:
            available_actions = check(available_actions, torch.int64,obs.device)
        actor_features = self.base(obs)
        action_vals = self.act.evaluate_actions(actor_features, action, available_actions)
        return action_vals

    def get_vals(self, obs):
        """
        Get Q-values for all actions given observations.

        Args:
            obs: Input observation tensor

        Returns:
            Q-values for all actions in the action space
        """
        with torch.no_grad():
            actor_features = self.base(obs)
        q_val = self.act.get_vals(actor_features)
        return q_val


