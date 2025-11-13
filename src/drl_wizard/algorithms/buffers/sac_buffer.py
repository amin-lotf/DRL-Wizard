import numpy as np
import torch

from drl_wizard.common.types import AlgoType
from drl_wizard.configs import SACConfig
from drl_wizard.algorithms.utils.extras import get_len_from_act_space, get_shape_from_obs_space
from drl_wizard.configs.app_cfg import AppConfig


class SACBuffer:
    def __init__(self, cfg: AppConfig, obs_space, act_space, shared_obs_space=None, action_masking=False):
        if cfg.algo_cfg.algo_id != AlgoType.SAC:
            raise ValueError("SACBuffer only supports SAC")

        self.cfg: SACConfig = cfg.algo_cfg
        self.buffer_size = self.cfg.buffer_size
        self.batch_size = self.cfg.batch_size
        self.num_agents = getattr(self.cfg, "num_agents", 1)
        self.n_envs = cfg.n_envs
        self.device = cfg.resolved_device

        # shape helpers
        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.act_len = get_len_from_act_space(act_space)
        self.shared_obs_shape = get_shape_from_obs_space(shared_obs_space) if shared_obs_space is not None else None

        # main buffers (ring)
        self.obs = np.zeros((self.buffer_size, self.n_envs, self.num_agents, *self.obs_shape), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.num_agents, self.act_len), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs, self.num_agents, 1), dtype=np.float32)

        self.shared_obs = np.zeros(
            (self.buffer_size, self.n_envs, self.num_agents, *self.shared_obs_shape),
            dtype=np.float32
        ) if shared_obs_space is not None else None

        self.next_shared_obs = np.zeros_like(self.shared_obs) if self.shared_obs is not None else None

        self.masked_acts = np.zeros(
            (self.buffer_size, self.n_envs, self.num_agents, self.act_len),
            dtype=np.float32
        ) if action_masking else None

        self.next_masked_acts = np.zeros_like(self.masked_acts) if self.masked_acts is not None else None

        # pointers
        self.ptr = 0
        self.size = 0

    # ------------- INSERT -------------
    def insert(
        self,
        obs,
        act,
        reward,
        done,
        next_obs,
        shared_obs=None,
        next_shared_obs=None,
        masked_acts=None,
        next_masked_acts=None,
    ):
        idx = self.ptr

        self.obs[idx] = obs
        self.actions[idx] = act
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_obs[idx] = next_obs

        if shared_obs is not None and self.shared_obs is not None:
            self.shared_obs[idx] = shared_obs
        if next_shared_obs is not None and self.next_shared_obs is not None:
            self.next_shared_obs[idx] = next_shared_obs
        if masked_acts is not None and self.masked_acts is not None:
            self.masked_acts[idx] = masked_acts
        if next_masked_acts is not None and self.next_masked_acts is not None:
            self.next_masked_acts[idx] = next_masked_acts

        # ring buffer update
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    # ------------- SAMPLE GENERATOR -------------
    def feed_forward_generator(self):
        """
        Yields mini-batches just like PPOBuffer.feed_forward_generator
        """
        total = self.size * self.n_envs * self.num_agents
        mini_batch_size = self.cfg.minibatch_size

        if total < self.batch_size:
            raise ValueError("Not enough samples to create a batch")

        tot_batch_size = self.batch_size*self.n_envs*self.num_agents
        if tot_batch_size % mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        rnd = torch.randperm(total).numpy()
        num_batches = self.batch_size // mini_batch_size
        sampler = [rnd[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_batches)]

        # flatten all axes
        obs = self.obs[:self.size].reshape(-1, *self.obs_shape)
        next_obs = self.next_obs[:self.size].reshape(-1, *self.obs_shape)
        actions = self.actions[:self.size].reshape(-1, self.act_len)
        rewards = self.rewards[:self.size].reshape(-1, 1)
        dones = self.dones[:self.size].reshape(-1, 1)

        shared_obs = self.shared_obs[:self.size].reshape(-1, *self.shared_obs_shape) if self.shared_obs is not None else None
        next_shared_obs = self.next_shared_obs[:self.size].reshape(-1, *self.shared_obs_shape) if self.next_shared_obs is not None else None

        masked_acts = self.masked_acts[:self.size].reshape(-1, self.act_len) if self.masked_acts is not None else None
        next_masked_acts = self.next_masked_acts[:self.size].reshape(-1, self.act_len) if self.next_masked_acts is not None else None

        for indices in sampler:
            obs_b = obs[indices]
            next_obs_b = next_obs[indices]
            actions_b = actions[indices]
            rewards_b = rewards[indices]
            dones_b = dones[indices]
            shared_obs_b = shared_obs[indices] if shared_obs is not None else None
            next_shared_obs_b = next_shared_obs[indices] if next_shared_obs is not None else None
            masked_acts_b = masked_acts[indices] if masked_acts is not None else None
            next_masked_acts_b = next_masked_acts[indices] if next_masked_acts is not None else None

            yield (
                obs_b,
                next_obs_b,
                actions_b,
                rewards_b,
                dones_b,
                shared_obs_b,
                next_shared_obs_b,
                masked_acts_b,
                next_masked_acts_b,
            )
