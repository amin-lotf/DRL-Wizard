import torch

from drl_wizard.algorithms.algos.ppo_algo.network.distributions import FixedNormal as PPOFixedNormal
from drl_wizard.algorithms.algos.sac_algo.network.distributions import FixedNormal as SACFixedNormal
from drl_wizard.algorithms.algos.trpo_algo.network.distributions import FixedNormal as TRPOFixedNormal


def _check_mode(dist_cls):
    mean = torch.tensor([[0.2, -0.4]], dtype=torch.float32)
    std = torch.ones_like(mean)
    action_scale = torch.tensor([2.0, 2.0], dtype=torch.float32)
    action_bias = torch.tensor([0.5, -0.5], dtype=torch.float32)

    dist = dist_cls(mean, std, action_scale, action_bias)
    expected = torch.tanh(mean) * action_scale + action_bias

    got = dist.mode()
    assert torch.allclose(got, expected)


def test_continuous_distributions_expose_callable_mode():
    for dist_cls in (PPOFixedNormal, SACFixedNormal, TRPOFixedNormal):
        _check_mode(dist_cls)
