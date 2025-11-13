import torch
import torch.nn as nn
from drl_wizard.algorithms.algos.dqn_algo.network.distributions import LinearDuelingDQN, LinearDQN


class ACTLayer(nn.Module):
    """
    Action Layer for DQN network that handles both discrete and multi-discrete action spaces.

    This layer implements Q-value based action selection with epsilon-greedy exploration.
    It supports both discrete and multi-discrete action spaces, optional dueling network
    architecture, and action masking through available_actions parameter.

    For discrete action spaces, it uses a single network to output Q-values for all actions.
    For multi-discrete action spaces, it maintains separate networks for each action dimension.

    Args:
        action_space: Gym space object (Discrete or MultiDiscrete)
        inputs_dim (int): Dimension of input features
        use_dueling (bool): Whether to use dueling network architecture
        use_orthogonal (bool): Whether to use orthogonal initialization
        gain (float): Gain factor for weight initialization

    Raises:
        ValueError: If action_space is not Discrete or MultiDiscrete
    """

    def __init__(self, action_space, inputs_dim, use_dueling,use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        dqn = LinearDuelingDQN if use_dueling else LinearDQN
        if action_space.__class__.__name__ == "Discrete":
            self.multi_discrete = False
            self.action_dims = action_space.n
            self.action_outs = dqn(inputs_dim, self.action_dims, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            self.action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in self.action_dims:
                self.action_outs.append(dqn(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:
            raise ValueError("Unknown action space: " + str(action_space))

    def forward(self, x, epsilon, available_actions=None, deterministic=False):
        """
        Select actions using epsilon-greedy policy.

        Args:
            x (torch.Tensor): Input state features
            epsilon (float): Exploration probability [0,1]
            available_actions (torch.Tensor, optional): Mask for valid actions
            deterministic (bool): Whether to act deterministically (not used)

        Returns:
            torch.Tensor: Selected actions of shape (batch_size, action_dims)
        """
        device = x.device  # ensure all tensors stay on the same device
        action = None
        if self.multi_discrete:
            actions = []
            if available_actions is not None:
                cur_limit = 0
                for idx, action_out in enumerate(self.action_outs):
                    next_limit = action_out.linear.out_features

                    if torch.rand(1, device=device) < epsilon:
                        # random action (torch equivalent of np.random.randint)
                        action = torch.randint(
                            low=0,
                            high=self.action_dims[idx],
                            size=(x.shape[0], 1),
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        with torch.no_grad():
                            action_val = action_out(x, available_actions[:, cur_limit:cur_limit + next_limit])
                            action = torch.argmax(action_val, dim=2, keepdim=True)
                    actions.append(action)
                    cur_limit += next_limit
            else:
                for idx, action_out in enumerate(self.action_outs):
                    if torch.rand(1, device=device) < epsilon:
                        action = torch.randint(
                            low=0,
                            high=self.action_dims[idx],
                            size=(x.shape[0], 1),
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        with torch.no_grad():
                            action_val = action_out(x)
                            action = torch.argmax(action_val, dim=1, keepdim=True)
                    actions.append(action)

            actions = torch.cat(actions, dim=-1)
            return actions

        else:
            rnd_numb=torch.rand(1, device=device)
            if  rnd_numb < epsilon:
                action = torch.randint(
                    low=0,
                    high=self.action_dims,
                    size=(x.shape[0], 1),
                    device=device,
                    dtype=torch.long,
                )
            else:
                with torch.no_grad():
                    q_values = self.action_outs(x, available_actions)
                    action = torch.argmax(q_values, dim=1, keepdim=True)
            return action

    def get_vals(self, x):
        """
        Get maximum Q-values for each state and action dimension.

        Args:
            x (torch.Tensor): Input state features

        Returns:
            torch.Tensor: Maximum Q-values for each state and action dimension
        """
        if self.multi_discrete:
            max_vals = []
            for action_out in self.action_outs:
                with torch.no_grad():
                    action_val = action_out(x).detach()
                    q_max = action_val.max(1)[0]
                    q_view = q_max.view(x.shape[0], 1)
                max_vals.append(q_view)

            action_vals = torch.cat(max_vals, dim=1)
            return action_vals
        else:
            with torch.no_grad():
                action_val = self.action_outs(x).detach()
                q_max = action_val.max(1)[0]
                q_view = q_max.view(x.shape[0], 1)
                return q_view

    def evaluate_actions(self, x, action, available_actions=None):
        """
        Compute Q-values for given state-action pairs.

        Args:
            x (torch.Tensor): Input state features
            action (torch.Tensor): Actions to evaluate
            available_actions (torch.Tensor, optional): Mask for valid actions

        Returns:
            torch.Tensor: Q-values for the given state-action pairs
        """
        if self.multi_discrete:
            action_vals = []
            if available_actions is not None:
                cur_limit = 0
                for idx, action_out in enumerate(self.action_outs):
                    next_limit = action_out.linear.out_features
                    act = action[:, idx].reshape((-1, 1))
                    action_val = action_out(x, available_actions[:, cur_limit:cur_limit + next_limit]).gather(1, act)
                    action_vals.append(action_val)
                    cur_limit += next_limit
            else:
                for idx, action_out in enumerate(self.action_outs):
                    act = action[:, idx].reshape((-1, 1))
                    action_val = action_out(x).gather(1, act)
                    action_vals.append(action_val)

            action_vals = torch.cat(action_vals, -1)
            return action_vals
        else:
            q_eval = self.action_outs(x, available_actions).gather(1, action)
            return q_eval