import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from drl_wizard.algorithms.algos.sac_algo.actor_q import Actor,QNetwork
from drl_wizard.algorithms.utils import get_num_actions
from drl_wizard.configs import  SACConfig
from drl_wizard.configs.app_cfg import AppConfig


class Policy:
    def __init__(self, actor_input_space, critic_input_space, action_space, cfg: AppConfig):
        self.algo_cfg: SACConfig = cfg.algo_cfg
        self.actor = Actor(actor_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.q_net1 = QNetwork(critic_input_space,action_space,self.algo_cfg).to(cfg.resolved_device)
        self.q_net2 = QNetwork(critic_input_space,action_space,self.algo_cfg).to(cfg.resolved_device)
        self.tgt_q_net1 = QNetwork(critic_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.tgt_q_net2 = QNetwork(critic_input_space,action_space,self.algo_cfg).to(cfg.resolved_device)
        self.tgt_q_net1.load_state_dict(self.q_net1.state_dict())
        self.tgt_q_net2.load_state_dict(self.q_net2.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(np.log(self.algo_cfg.alpha_init)), requires_grad=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.algo_cfg.actor_lr, eps=self.algo_cfg.opti_eps,
                                                weight_decay=self.algo_cfg.weight_decay)

        self.q_net1_optimizer = torch.optim.Adam(self.q_net1.parameters(),
                                                 lr=self.algo_cfg.q_lr,
                                                 eps=self.algo_cfg.opti_eps,
                                                 weight_decay=self.algo_cfg.weight_decay)
        self.q_net2_optimizer = torch.optim.Adam(self.q_net2.parameters(),
                                                 lr=self.algo_cfg.q_lr,
                                                 eps=self.algo_cfg.opti_eps,
                                                 weight_decay=self.algo_cfg.weight_decay)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.algo_cfg.alpha_lr)
        num_actions = get_num_actions(action_space)
        self.target_entropy = self.algo_cfg.target_entropy_scale * num_actions
        lr_decay_steps = max(1, cfg.total_steps // max(1, self.algo_cfg.batch_size))


        self.actor_scheduler = StepLR(
            self.actor_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

        self.q_net1_scheduler = StepLR(
            self.q_net1_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )
        self.q_net2_scheduler = StepLR(
            self.q_net2_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )
        self.alpha_scheduler = StepLR(
            self.alpha_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

    def lr_decay(self):
        self.actor_scheduler.step()
        self.q_net1_scheduler.step()
        self.q_net2_scheduler.step()
        self.alpha_scheduler.step()

    def get_actions(self, obs, available_actions=None, deterministic=False):
        actions, action_log_probs = self.actor(obs, available_actions, deterministic)
        return actions, action_log_probs

    def get_q_vals(self, obs,actions):
        q1 = self.q_net1(obs,actions)
        q2 = self.q_net2(obs,actions)
        return q1,q2

    def get_tgt_q_vals(self, obs,actions):
        q1 = self.tgt_q_net1(obs,actions)
        q2 = self.tgt_q_net2(obs,actions)
        return q1,q2

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def evaluate_actions(self, obs, action, available_actions=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions)
        return action_log_probs, dist_entropy


