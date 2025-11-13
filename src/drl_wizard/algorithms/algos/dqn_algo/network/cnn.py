import torch
import torch.nn as nn

from drl_wizard.algorithms.utils import init
from drl_wizard.configs import DQNConfig


class CNNLayer(nn.Module):
    """
    CNN layer for processing observations in DQN algorithm.

    Args:
        obs_shape (tuple): Shape of the observation input (channels, height, width)
        output_dim (int): Dimension of the output features
        cfg (DQNConfig): Configuration object containing network parameters like:
            - use_relu: Whether to use ReLU or Tanh activation
            - use_orthogonal: Whether to use orthogonal initialization
            - cnn_hidden: Number of CNN hidden units
            - cnn_kernel_size: Size of CNN kernels
            - cnn_stride: Stride of CNN layers
            - fc_hidden: Number of hidden units in fully connected layers
        norm_last_layer (bool): Whether to apply layer normalization to the last layer

    The layer consists of:
        1. A CNN network that processes the raw observations
        2. A fully connected network that processes the CNN output
        3. Optional layer normalization on the output
    """

    def __init__(self, obs_shape, output_dim, cfg: DQNConfig, norm_last_layer):
        """
        Initialize the CNN layer.

        Args:
            obs_shape (tuple): Shape of the observation input
            output_dim (int): Dimension of the output features
            cfg (DQNConfig): Configuration object for network parameters
            norm_last_layer (bool): Whether to apply layer normalization to the last layer
        """
        super(CNNLayer, self).__init__()
        active_func = nn.ReLU() if cfg.use_relu else nn.Tanh()
        init_method = nn.init.orthogonal_ if cfg.use_orthogonal else nn.init.xavier_uniform_
        gain = nn.init.calculate_gain('relu' if cfg.use_relu else 'tanh')

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=cfg.cnn_hidden // 2,
                            kernel_size=cfg.cnn_kernel_size,
                            stride=cfg.cnn_stride)
                  ),
            active_func,
            nn.Flatten())

        cnn_out = self.cnn(torch.zeros(1, *obs_shape)).shape[1]

        self.fc_output = nn.Sequential(
            init_(nn.Linear(cnn_out, output_dim)), active_func,
            nn.LayerNorm(cfg.fc_hidden) if norm_last_layer else \
                init_(nn.Linear(cfg.fc_hidden, output_dim))
        )

    def forward(self, x):
        """
        Forward pass through the CNN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Processed output tensor after CNN and fully connected layers
        """
        x = x / 255.0
        x = self.cnn(x)
        x = self.fc_output(x)
        return x
