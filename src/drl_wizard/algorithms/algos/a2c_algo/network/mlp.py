import torch.nn as nn
from drl_wizard.configs import A2CConfig
from drl_wizard.algorithms.utils import init


class MLPLayer(nn.Module):
    """
    Initialize a Multi-Layer Perceptron (MLP) network layer for processing inputs in A2C algorithm.

    Args:
        input_shape (tuple): Shape of the input features
        output_dim (int): Dimension of the output features
        cfg (A2CConfig): Configuration object containing network parameters like:
            - use_relu: Whether to use ReLU or Tanh activation
            - use_orthogonal: Whether to use orthogonal initialization
            - fc_hidden: Number of hidden units in fully connected layers
            - fc_num_hidden: Number of hidden layers
            - use_feature_normalization: Whether to use layer normalization on input features
        norm_last_layer (bool): Whether to apply layer normalization to the last layer

    The layer consists of:
        1. Optional feature normalization on input
        2. Input fully connected layer with activation and normalization
        3. Multiple hidden fully connected layers with activation and normalization
        4. Output fully connected layer with optional normalization
    """

    def __init__(self, input_shape, output_dim, cfg: A2CConfig, norm_last_layer):
        super(MLPLayer, self).__init__()
        input_dim = input_shape[0]
        active_func = nn.ReLU() if cfg.use_relu else nn.Tanh()
        init_method = nn.init.orthogonal_ if cfg.use_orthogonal else nn.init.xavier_uniform_
        gain = nn.init.calculate_gain('relu' if cfg.use_relu else 'tanh')
        self.num_hidden = cfg.fc_num_hidden
        self._use_feature_normalization = cfg.use_feature_normalization
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc_input = nn.Sequential(
            init_(nn.Linear(input_dim, cfg.fc_hidden)), active_func, nn.LayerNorm(cfg.fc_hidden))
        self.fc_hidden = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(cfg.fc_hidden, cfg.fc_hidden)), active_func, nn.LayerNorm(cfg.fc_hidden)) for _ in
            range(self.num_hidden)])
        if norm_last_layer:
            self.fc_output = nn.Sequential(
                init_(nn.Linear(cfg.fc_hidden, output_dim)),
                active_func,
                nn.LayerNorm(cfg.fc_hidden)
            )
        else:
            self.fc_output = nn.Sequential(
                    init_(nn.Linear(cfg.fc_hidden, output_dim))
            )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.fc_input(x)
        for i in range(self.num_hidden):
            x = self.fc_hidden[i](x)
        x=self.fc_output(x)
        return x
