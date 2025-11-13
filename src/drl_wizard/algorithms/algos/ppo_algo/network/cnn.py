import torch
import torch.nn as nn

from drl_wizard.algorithms.utils import init
from drl_wizard.configs.algo_cfg import PPOConfig


class CNNLayer(nn.Module):
    """Convolutional Neural Network layer for processing image observations.

    This layer consists of a CNN followed by fully connected layers to process
    image inputs for reinforcement learning tasks.

    Args:
        obs_shape (tuple): Shape of the input observation (channels, height, width)
        output_dim (int): Dimension of the output features
        cfg (PPOConfig): Configuration object containing network parameters
        norm_last_layer (bool): Whether to apply layer normalization to the final layer

    Attributes:
        cnn (nn.Sequential): Convolutional layers for feature extraction
        fc_output (nn.Sequential): Fully connected layers for final output
    """

    def __init__(self, obs_shape,output_dim, cfg: PPOConfig,norm_last_layer):
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
        x = x / 255.0
        x = self.cnn(x)
        x = self.fc_output(x)
        return x
