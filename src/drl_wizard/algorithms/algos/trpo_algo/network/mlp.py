import torch.nn as nn
from drl_wizard.configs import TRPOConfig
from drl_wizard.algorithms.utils import init


class MLPLayer(nn.Module):
    def __init__(self, input_shape, output_dim, cfg: TRPOConfig, norm_last_layer):
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
