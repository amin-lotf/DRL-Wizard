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
    """A modified version of PyTorch's Categorical distribution for RL usage.

    Extends the standard Categorical distribution with additional functionality
    needed for reinforcement learning algorithms.

    Methods:
        sample(): Returns sampled actions with an extra dimension
        log_probs(actions): Calculates log probabilities of given actions
        mode(): Returns the action with highest probability
    """

    def sample(self, sample_shape=torch.Size()):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """A modified version of PyTorch's Normal distribution with action scaling.

    Extends the standard Normal distribution by adding action scaling and bias
    to handle continuous action spaces.

    Args:
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        action_scale: Scaling factor for actions
        action_bias: Bias term for actions

    Methods:
        log_probs(actions): Computes log probabilities with action scaling
        mean_action(): Returns the mean action after scaling
        entropy(): Computes distribution entropy
        sample(): Returns scaled and transformed actions
    """

    def __init__(self,mean,std,action_scale,action_bias):
        super().__init__(mean,std)
        self.action_scale=action_scale
        self.action_bias=action_bias

    def log_probs(self,actions):
        eps=1e-6
        actions_unscaled=torch.clamp((actions-self.action_bias)/self.action_scale,-1.0+eps,1.0-eps)
        raw_actions=torch.atanh(actions_unscaled)
        log_probs=super().log_prob(raw_actions).sum(dim=-1,keepdim=True)
        log_probs-=torch.sum(torch.log(self.action_scale*(1-torch.tanh(raw_actions).pow(2))+eps),dim=-1,keepdim=True)
        return log_probs

    def mean_action(self):
        mean=self.mean
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def entropy(self):
        return super().entropy().sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        raw_action=super().rsample()
        action=torch.tanh(raw_action)*self.action_scale+self.action_bias
        return action


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    """A modified version of PyTorch's Bernoulli distribution for RL usage.

    Extends the standard Bernoulli distribution with additional functionality
    for reinforcement learning algorithms.

    Methods:
        log_probs(actions): Calculates log probabilities of given actions
        entropy(): Computes distribution entropy
        mode(): Returns most likely action (0 or 1)
    """

    def log_probs(self, actions):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    """Neural network module that outputs a categorical distribution.

    Takes input features and produces a categorical distribution over possible actions.

    Args:
        num_inputs (int): Size of input feature dimension
        num_outputs (int): Number of possible actions
        use_orthogonal (bool): Whether to use orthogonal initialization
        gain (float): Initialization gain parameter

    Methods:
        forward(x, available_actions): Returns a FixedCategorical distribution
    """

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """Neural network module that outputs a diagonal Gaussian distribution.

    Takes input features and produces a Gaussian distribution with diagonal covariance
    for continuous action spaces.

    Args:
        num_inputs (int): Size of input feature dimension
        num_outputs (int): Dimension of action space
        action_space: Action space object defining bounds
        use_orthogonal (bool): Whether to use orthogonal initialization
        gain (float): Initialization gain parameter

    Methods:
        forward(x): Returns a FixedNormal distribution
    """

    def __init__(self, num_inputs, num_outputs, action_space, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()
        self.action_scale = nn.Parameter(torch.FloatTensor((action_space.high - action_space.low) / 2.),
                                         requires_grad=False)
        self.action_bias = nn.Parameter(torch.FloatTensor((action_space.high + action_space.low) / 2.),
                                        requires_grad=False)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean,action_logstd.exp(),self.action_scale,self.action_bias)


class Bernoulli(nn.Module):
    """Neural network module that outputs a Bernoulli distribution.

    Takes input features and produces a Bernoulli distribution for binary actions.

    Args:
        num_inputs (int): Size of input feature dimension
        num_outputs (int): Number of binary actions
        use_orthogonal (bool): Whether to use orthogonal initialization
        gain (float): Initialization gain parameter

    Methods:
        forward(x): Returns a FixedBernoulli distribution
    """

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    """Neural network module that adds a learnable bias to inputs.

    A simple module that adds a learnable bias parameter to input tensors.

    Args:
        bias: Initial bias values to learn

    Methods:
        forward(x): Adds bias to input tensor
    """

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias