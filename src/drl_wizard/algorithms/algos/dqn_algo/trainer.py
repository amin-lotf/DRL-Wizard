import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_wizard.algorithms.algos.dqn_algo.policy import Policy
from drl_wizard.algorithms.buffers.dqn_buffer import DQNBuffer
from drl_wizard.configs import DQNConfig
from drl_wizard.algorithms.utils import check
from drl_wizard.configs.app_cfg import AppConfig


class Trainer:
    """
    DQN (Deep Q-Network) trainer class that handles the training process of a DQN agent.
    Implements both standard DQN and Double DQN training algorithms with experience replay.
    """

    def __init__(self, policy: Policy, cfg: AppConfig):
        """
        Initialize the DQN trainer.

        Args:
            policy (Policy): The policy network to be trained
            cfg (AppConfig): Configuration object containing training parameters
        """
        self.device = cfg.resolved_device
        self.policy = policy
        self.algo_cfg: DQNConfig = cfg.algo_cfg

    def dqn_update(self, sample):
        """
        Perform a single DQN update step using a batch of experiences.

        Args:
            sample (tuple): A batch of experiences containing:
                (observations, next observations, actions, rewards, dones,
                shared observations, next shared observations, masked actions,
                next masked actions)

        Returns:
            tuple: (q_loss, actor_grad_norm) The Q-network loss and gradient norm
        """
        (obs_b, next_obs_b, actions_b, rewards_b, dones_b,
         shared_obs_b, next_shared_obs_b, masked_acts_b,
         next_masked_acts_b) = sample
        obs = check(obs_b, torch.float32, self.device)
        next_obs = check(next_obs_b, torch.float32, self.device)
        actions = check(actions_b, torch.int64, self.device)
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

        if self.algo_cfg.use_double_dqn:
            with torch.no_grad():
                next_actions = self.policy.get_actions(next_obs, next_masked_acts).to(torch.int64)
                q_t = self.policy.evaluate_tgt_actions(q_next_input, next_actions)
        else:
            q_t = self.policy.get_tgt_q_vals(q_next_input)
        y = rewards + (1 - dones) * self.algo_cfg.gamma * q_t
        max_grad_norm = self.algo_cfg.max_grad_norm if self.algo_cfg.max_grad_norm else 1e6
        q_eval = self.policy.evaluate_actions(q_input, actions, masked_acts)
        self.policy.actor.zero_grad()
        q_loss = F.mse_loss(q_eval, y)
        q_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_grad_norm)
        self.policy.actor_optimizer.step()
        self.soft_update()
        return q_loss, actor_grad_norm

    def soft_update(self):
        """
        Perform soft update of target network parameters using polyak averaging:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        with torch.no_grad():
            for target_par, source_par in zip(self.policy.tgt_actor.parameters(), self.policy.actor.parameters()):
                target_par.lerp_(source_par, self.algo_cfg.tau)

    def train(self, buffer: DQNBuffer):
        """
        Train the DQN network using experiences from the replay buffer.

        Args:
            buffer (DQNBuffer): Experience replay buffer containing collected transitions

        Returns:
            dict: Training metrics including average actor loss and gradient norm
        """
        train_info = {
            'actor_loss': 0,
            'actor_grad_norm': 0,
        }
        tot_updates = 0
        for _ in range(self.algo_cfg.num_epochs):
            data_generator = buffer.feed_forward_generator()
            for sample in data_generator:
                actor_loss, actor_grad_norm = self.dqn_update(sample)
                train_info['actor_loss'] += actor_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                tot_updates += 1
        for k in train_info.keys():
            train_info[k] /= tot_updates
        return train_info

    def prep_training(self):
        """
        Prepare the network for training by setting it to training mode.
        """
        self.policy.actor.train()

    def prep_rollout(self):
        """
        Prepare the network for evaluation by setting it to evaluation mode.
        """
        self.policy.actor.eval()
