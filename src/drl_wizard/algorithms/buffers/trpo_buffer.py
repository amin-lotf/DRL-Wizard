import torch
import numpy as np

from drl_wizard.common.types import AlgoType
from drl_wizard.configs import TRPOConfig
from drl_wizard.algorithms.utils.extras import get_len_from_act_space, get_shape_from_obs_space
from drl_wizard.configs.app_cfg import AppConfig


class TRPOBuffer:
    def __init__(self, cfg: AppConfig, obs_space, act_space, shared_obs_space=None, action_masking=False):
        if cfg.algo_cfg.algo_id != AlgoType.TRPO:
            raise ValueError("TRPOBuffer only supports PPO")
        self.cfg: TRPOConfig = cfg.algo_cfg
        self.episode_length = self.cfg.episode_length
        self.num_agents = self.cfg.num_agents
        self.n_envs = cfg.n_envs
        self.device = cfg.resolved_device
        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.act_len = get_len_from_act_space(act_space)
        self.shared_obs_shape = get_shape_from_obs_space(shared_obs_space) if shared_obs_space is not None else None
        self.obs = np.zeros((self.episode_length + 1, self.n_envs, self.num_agents, *self.obs_shape), dtype=np.float32)
        self.act = np.zeros((self.episode_length, self.n_envs, self.num_agents, self.act_len), dtype=np.float32)
        self.shared_obs = np.zeros((self.episode_length + 1, self.n_envs, self.num_agents, *self.shared_obs_shape),
                                   dtype=np.float32) if shared_obs_space is not None else None
        self.masked_acts = np.zeros((self.episode_length + 1, self.n_envs, self.num_agents, self.act_len),
                                    dtype=np.float32) if action_masking else None
        self.value_predicts = np.zeros((self.episode_length + 1, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((self.episode_length, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.episode_length, self.n_envs, self.num_agents, self.act_len), dtype=np.float32)
        self.dones = np.zeros((self.episode_length, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_envs, self.num_agents, 1), dtype=np.float32)
        self.step = 0

    @property
    def latest_shared_obs_batched(self):
        latest_share_obs= self.shared_obs[self.step] if self.shared_obs_shape is not None else self.obs[self.step]
        return np.concatenate(latest_share_obs, axis=0)

    @property
    def latest_obs_batched(self):
        latest_obs = self.obs[self.step]
        return np.concatenate(latest_obs, axis=0)

    def warmup(self,obs,shared_obs=None,masked_acts=None):
        self.obs[0]=obs
        if shared_obs is not None:
            self.shared_obs[0] = shared_obs
        if masked_acts is not None:
            self.masked_acts[0] = masked_acts

    def insert(self,obs,act,reward,done,value_predict,log_prob,shared_obs=None,masked_acts=None):
        self.obs[self.step+1] = obs
        self.act[self.step] = act
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        if shared_obs is not None:
            self.shared_obs[self.step+1] = shared_obs
        if masked_acts is not None:
            self.masked_acts[self.step+1] = masked_acts
        self.value_predicts[self.step] = value_predict
        self.log_probs[self.step] = log_prob
        self.step = self.step + 1
        if self.step > self.episode_length:
            raise ValueError("PPOBuffer episode length exceeded")


    def after_update(self):
        self.obs[0] = self.obs[-1]
        if self.shared_obs_shape is not None:
            self.shared_obs[0] = self.shared_obs[-1]
        if self.masked_acts is not None:
            self.masked_acts[0] = self.masked_acts[-1]
        self.step = 0


    def compute_advantage_returns(self,next_values):
        gamma = self.cfg.gamma
        gae_lambda = self.cfg.gae_lambda
        gae=0
        if self.cfg.use_gae:
            self.value_predicts[-1] = next_values
            for step in reversed(range(self.episode_length)):
                delta=self.rewards[step]+gamma*self.value_predicts[step+1]*(1-self.dones[step])-self.value_predicts[step]
                gae= delta+gamma*gae_lambda*(1-self.dones[step])*gae
                self.returns[step] = gae+self.value_predicts[step]
                self.advantages[step] = self.returns[step]-self.value_predicts[step]
        else:
            self.returns[-1] = next_values
            for step in reversed(range(self.episode_length)):
                self.returns[step]=self.rewards[step]+gamma*self.returns[step+1]*(1-self.dones[step])
                self.advantages[step] = self.returns[step] - self.value_predicts[step]



    def feed_forward_generator(self):
        batch_size = self.n_envs * self.episode_length * self.num_agents
        mini_batch_size=self.cfg.minibatch_size
        if batch_size % mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        num_mini_batches=int(np.floor(batch_size/mini_batch_size))
        rnd = torch.randperm(batch_size).numpy()
        sampler=[rnd[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batches)]
        obs=self.obs[:-1].reshape(-1,*self.obs_shape)
        shared_obs=self.shared_obs[:-1].reshape(-1,*self.shared_obs_shape) if self.shared_obs_shape is not None else None
        actions=self.act.reshape(-1,self.act_len)
        masked_actions=self.masked_acts[:-1].reshape(-1,*self.act_len) if self.masked_acts is not None else None
        returns=self.returns[:-1].reshape(-1,1)
        value_predicts=self.value_predicts[:-1].reshape(-1,1)
        advantages=self.advantages.reshape(-1, 1)
        mean_advantages = np.nanmean(advantages)
        std_advantages = np.nanstd(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        log_probs=self.log_probs.reshape(-1,self.act_len)
        for indices in sampler:
            obs_b=obs[indices]
            shared_obs_b=shared_obs[indices] if shared_obs is not None else None
            actions_b=actions[indices]
            masked_actions_b=masked_actions[indices] if masked_actions is not None else None
            returns_b=returns[indices]
            value_predicts_b=value_predicts[indices]
            advantages_b=advantages[indices]
            log_probs_b=log_probs[indices]
            yield obs_b,shared_obs_b,actions_b,masked_actions_b,returns_b,value_predicts_b,advantages_b,log_probs_b






