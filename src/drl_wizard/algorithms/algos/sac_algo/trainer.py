import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_wizard.algorithms.algos.sac_algo.policy import Policy
from drl_wizard.algorithms.buffers.sac_buffer import SACBuffer
from drl_wizard.configs import SACConfig
from drl_wizard.algorithms.utils import check
from drl_wizard.configs.app_cfg import AppConfig


class Trainer:
    def __init__(self, policy: Policy, cfg: AppConfig):
        self.device = cfg.resolved_device
        self.policy = policy
        self.algo_cfg: SACConfig = cfg.algo_cfg

    def sac_update(self, sample):
        (obs_b, next_obs_b, actions_b, rewards_b, dones_b,
         shared_obs_b, next_shared_obs_b, masked_acts_b,
         next_masked_acts_b) = sample
        obs = check(obs_b, torch.float32, self.device)
        next_obs = check(next_obs_b, torch.float32, self.device)
        actions = check(actions_b, torch.float32, self.device)
        rewards = check(rewards_b, torch.float32, self.device)
        dones = check(dones_b, torch.float32, self.device)
        shared_obs = check(shared_obs_b, torch.float32, self.device) if shared_obs_b is not None else None
        next_shared_obs = check(next_shared_obs_b, torch.float32,
                                self.device) if next_shared_obs_b is not None else None
        masked_acts = check(masked_acts_b, torch.int64, self.device) if masked_acts_b is not None else None
        next_masked_acts = check(next_masked_acts_b, torch.int64,
                                 self.device) if next_masked_acts_b is not None else None

        if shared_obs is not None:
            q_input = shared_obs
            q_next_input = next_shared_obs
        else:
            q_input = obs
            q_next_input = next_obs
        with torch.no_grad():
            next_actions,next_action_log_probs=self.policy.get_actions(next_obs,next_masked_acts)
            tgt_q1,tgt_q2=self.policy.get_tgt_q_vals(q_next_input,next_actions)
            q_t = torch.min(tgt_q1, tgt_q2) - self.policy.alpha * next_action_log_probs
            y = rewards + (1 - dones) * self.algo_cfg.gamma * q_t
        max_grad_norm = self.algo_cfg.max_grad_norm if self.algo_cfg.max_grad_norm else 1e6
        q1,q2=self.policy.get_q_vals(q_input,actions)
        self.policy.q_net1_optimizer.zero_grad()
        self.policy.q_net2_optimizer.zero_grad()
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        q1_loss.backward()
        q2_loss.backward()
        q1_grad_norm = nn.utils.clip_grad_norm_(self.policy.q_net1.parameters(), max_grad_norm)
        q2_grad_norm = nn.utils.clip_grad_norm_(self.policy.q_net2.parameters(), max_grad_norm)
        self.policy.q_net1_optimizer.step()
        self.policy.q_net2_optimizer.step()
        new_actions,new_action_action_log_probs = self.policy.get_actions(obs,masked_acts)
        # with torch.no_grad():
        new_q1,new_q2 = self.policy.get_q_vals(q_input,new_actions)
        new_q = torch.min(new_q1, new_q2)
        self.policy.actor_optimizer.zero_grad()
        actor_loss = (self.policy.alpha.detach() * new_action_action_log_probs - new_q).mean()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_grad_norm)
        self.policy.actor_optimizer.step()
        self.policy.alpha_optimizer.zero_grad()
        alpha_loss = -(self.policy.log_alpha * (new_action_action_log_probs + self.policy.target_entropy).detach()).mean()
        alpha_loss.backward()
        alpha_grad_norm = nn.utils.clip_grad_norm_([self.policy.log_alpha], max_grad_norm)
        self.policy.alpha_optimizer.step()
        self.soft_update()
        return actor_loss, actor_grad_norm,q1_grad_norm,q2_grad_norm,alpha_grad_norm

    def soft_update(self):
        with torch.no_grad():
            for target_par,source_par in zip(self.policy.tgt_q_net1.parameters(),self.policy.q_net1.parameters()):
                target_par.lerp_(source_par,self.algo_cfg.tau)
            for target_par,source_par in zip(self.policy.tgt_q_net2.parameters(),self.policy.q_net2.parameters()):
                target_par.lerp_(source_par,self.algo_cfg.tau)

    def train(self, buffer: SACBuffer):
        train_info = {
            'actor_loss': 0,
            'actor_grad_norm': 0,
            'q1_grad_norm': 0,
            'q2_grad_norm': 0,
            'alpha_grad_norm': 0,
        }
        tot_updates = 0
        for _ in range(self.algo_cfg.num_epochs):
            data_generator = buffer.feed_forward_generator()
            for sample in data_generator:
                actor_loss, actor_grad_norm,q1_grad_norm,q2_grad_norm,alpha_grad_norm \
                    = self.sac_update(sample)
                train_info['actor_loss'] += actor_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                train_info['q1_grad_norm'] += q1_grad_norm.item()
                train_info['q2_grad_norm'] += q2_grad_norm.item()
                train_info['alpha_grad_norm'] += alpha_grad_norm.item()
                tot_updates += 1
        for k in train_info.keys():
            train_info[k] /= tot_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.q_net1.train()
        self.policy.q_net2.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.q_net1.eval()
        self.policy.q_net2.eval()
