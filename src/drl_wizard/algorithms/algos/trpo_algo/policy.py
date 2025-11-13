import torch
from torch.optim.lr_scheduler import StepLR
from drl_wizard.algorithms.algos.trpo_algo.actor_critic import Actor, Critic
from drl_wizard.configs import  TRPOConfig
from drl_wizard.configs.app_cfg import AppConfig


class Policy:
    def __init__(self, actor_input_space, critic_input_space, action_space, cfg: AppConfig):
        self.algo_cfg: TRPOConfig = cfg.algo_cfg
        self.actor = Actor(actor_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.critic = Critic(critic_input_space, self.algo_cfg).to(cfg.resolved_device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.algo_cfg.critic_lr,
                                                 eps=self.algo_cfg.opti_eps,
                                                 weight_decay=self.algo_cfg.weight_decay)

        lr_decay_steps = cfg.total_steps // self.algo_cfg.episode_length


        self.critic_scheduler = StepLR(
            self.critic_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

    def lr_decay(self):
        self.critic_scheduler.step()

    def get_dist(self, obs, available_actions=None):
        dist = self.actor.get_dist(obs, available_actions)
        return dist

    def get_actions(self, obs, available_actions=None, deterministic=False):
        actions, action_log_probs = self.actor(obs, available_actions, deterministic)
        return actions, action_log_probs

    def get_values(self, obs):
        values = self.critic(obs)
        return values

    def evaluate_actions(self, obs, action, available_actions=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions)
        return action_log_probs, dist_entropy


