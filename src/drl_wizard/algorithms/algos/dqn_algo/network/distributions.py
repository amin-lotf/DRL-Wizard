import torch
import torch.nn as nn
from drl_wizard.algorithms.utils import init

class LinearDQN(nn.Module):
    """
    A simple linear Deep Q-Network (DQN) implementation.

    Args:
        num_inputs (int): Number of input features
        num_outputs (int): Number of output actions
        use_orthogonal (bool, optional): Whether to use orthogonal initialization. Defaults to True.
        gain (float, optional): Gain factor for weight initialization. Defaults to 0.01.

    The network consists of a single linear layer that maps inputs to Q-values for each action.
    """

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(LinearDQN, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return x


class LinearDuelingDQN(nn.Module):
    """
    A linear Dueling Deep Q-Network (DQN) implementation.

    Args:
        num_inputs (int): Number of input features
        num_outputs (int): Number of output actions
        use_orthogonal (bool, optional): Whether to use orthogonal initialization. Defaults to True.
        gain (float, optional): Gain factor for weight initialization. Defaults to 0.01.

    The network implements the dueling architecture which separates state value and
    action advantage streams.
    """

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(LinearDuelingDQN, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_adv = init_(nn.Linear(num_inputs, 1))

    def forward(self, x, available_actions=None):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
            available_actions (torch.Tensor, optional): Binary mask indicating available actions.
                If provided, unavailable actions will be masked with large negative values.

        Returns:
            torch.Tensor: Q-values for each action
        """
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
            available_actions (torch.Tensor, optional): Binary mask indicating available actions.
                If provided, unavailable actions will be masked with large negative values.

        Returns:
            torch.Tensor: Q-values for each action, computed as V(s) + A(s,a) - mean(A(s,a))
        """
        adv = self.fc_adv(x)
        val = self.linear(x)
        res = val + (adv - adv.mean(dim=1, keepdim=True))
        if available_actions is not None:
            res[available_actions == 0] = -1e10
        return res