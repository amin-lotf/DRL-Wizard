import torch
import torch.nn as nn
from drl_wizard.algorithms.utils import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""


#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    """A modified Categorical distribution for compatibility with the A2C codebase.
    Inherits from torch.distributions.Categorical and modifies some methods.
    """

    def sample(self, sample_shape=torch.Size()):
        """Sample from the categorical distribution and add a dimension.

        Args:
            sample_shape (torch.Size, optional): Shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: Sampled actions with an additional dimension
        """
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        """Calculate log probabilities of actions.

        Args:
            actions (torch.Tensor): Actions to calculate log probabilities for

        Returns:
            torch.Tensor: Log probabilities of the actions
        """
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        """Get the mode (most likely action) of the distribution.

        Returns:
            torch.Tensor: Mode of the distribution
        """
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    """A modified Normal distribution for compatibility with the A2C codebase.
    Inherits from torch.distributions.Normal and modifies some methods.
    """

    def log_probs(self, actions):
        """Calculate log probabilities of actions.

        Args:
            actions (torch.Tensor): Actions to calculate log probabilities for

        Returns:
            torch.Tensor: Log probabilities of the actions
        """
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        """Calculate the entropy of the distribution.

        Returns:
            torch.Tensor: Entropy of the distribution
        """
        return super().entropy().sum(-1)

    def mode(self):
        """Get the mode (mean) of the distribution.

        Returns:
            torch.Tensor: Mode (mean) of the distribution
        """
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    """A modified Bernoulli distribution for compatibility with the A2C codebase.
    Inherits from torch.distributions.Bernoulli and modifies some methods.
    """

    def log_probs(self, actions):
        """Calculate log probabilities of actions.

        Args:
            actions (torch.Tensor): Actions to calculate log probabilities for

        Returns:
            torch.Tensor: Log probabilities of the actions
        """
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        """Calculate the entropy of the distribution.

        Returns:
            torch.Tensor: Entropy of the distribution
        """
        return super().entropy().sum(-1)

    def mode(self):
        """Get the mode of the distribution.

        Returns:
            torch.Tensor: Mode of the distribution (1 if prob > 0.5, else 0)
        """
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    """Neural network module that outputs a categorical distribution."""

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """Initialize the Categorical network.

        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output categories
            use_orthogonal (bool, optional): Whether to use orthogonal initialization. Defaults to True.
            gain (float, optional): Gain for the initialization. Defaults to 0.01.
        """
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input features
            available_actions (torch.Tensor, optional): Mask for available actions. Defaults to None.

        Returns:
            FixedCategorical: Categorical distribution over actions
        """
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """Neural network module that outputs a diagonal Gaussian distribution."""

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """Initialize the DiagGaussian network.

        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output dimensions
            use_orthogonal (bool, optional): Whether to use orthogonal initialization. Defaults to True.
            gain (float, optional): Gain for the initialization. Defaults to 0.01.
        """
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input features

        Returns:
            FixedNormal: Diagonal Gaussian distribution
        """
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    """Neural network module that outputs a Bernoulli distribution."""

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """Initialize the Bernoulli network.

        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output dimensions
            use_orthogonal (bool, optional): Whether to use orthogonal initialization. Defaults to True.
            gain (float, optional): Gain for the initialization. Defaults to 0.01.
        """
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input features

        Returns:
            FixedBernoulli: Bernoulli distribution
        """
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    """Neural network module that adds a learnable bias to inputs."""

    def __init__(self, bias):
        """Initialize the AddBias module.

        Args:
            bias (torch.Tensor): Initial bias values
        """
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        """Forward pass through the module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Input tensor with added bias
        """
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
