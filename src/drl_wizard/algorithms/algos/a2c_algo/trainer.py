import torch
import torch.nn as nn

from drl_wizard.algorithms.algos.a2c_algo.policy import Policy
from drl_wizard.algorithms.buffers.a2c_buffer import A2CBuffer
from drl_wizard.configs import A2CConfig
from drl_wizard.algorithms.utils import check
from drl_wizard.configs.app_cfg import AppConfig


class Trainer:
    """
    A2C algorithm trainer that handles policy updates and optimization.

    This class manages the training process for the Advantage Actor-Critic (A2C) algorithm,
    including policy updates, value function estimation, and gradient calculations.
    """

    def __init__(self, policy: Policy, cfg: AppConfig):
        """
        Initialize the A2C trainer.

        Args:
            policy (Policy): The policy network to be trained
            cfg (AppConfig): Configuration containing device and algorithm settings
        """
        self.device = cfg.resolved_device
        self.policy = policy
        self.algo_cfg: A2CConfig = cfg.algo_cfg

    def cal_value_loss(self, values, value_predicts_batch, return_batch):
        """
        Calculate the value function loss with optional clipping.

        Args:
            values: Current value estimates
            value_predicts_batch: Previous value predictions
            return_batch: Target return values

        Returns:
            torch.Tensor: Mean value loss
        """
        value_pred_clipped = value_predicts_batch + (values - value_predicts_batch).clamp(-self.algo_cfg.clip_ratio,
                                                                                          self.algo_cfg.clip_ratio)
        value_loss_clipped = (return_batch - value_pred_clipped) ** 2 / 2
        value_loss_original = (return_batch - values) ** 2 / 2
        if self.algo_cfg.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        mean_value_loss = value_loss.mean()
        return mean_value_loss

    def a2c_update(self, sample):
        """
        Perform one A2C update step using the provided sample batch.

        Args:
            sample: Tuple containing (observations, shared_observations, actions, 
                   masked_actions, returns, value_predictions, advantages)

        Returns:
            tuple: (value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm)
        """
        obs, shared_obs, actions, masked_actions, returns, value_predicts, advantages = sample
        obs = check(obs, torch.float32, self.device)
        if shared_obs is not None:
            critic_input = check(shared_obs, torch.float32, self.device)
        else:
            critic_input=obs
        actions = check(actions, torch.float32, self.device)
        advantages = check(advantages, torch.float32, self.device)
        value_predicts = check(value_predicts, torch.float32, self.device)
        returns = check(returns, torch.float32, self.device)
        if masked_actions is not None:
            masked_actions = check(masked_actions, torch.int64, self.device)
        action_log_probs, dist_entropy = self.policy.evaluate_actions(obs, actions, masked_actions)
        values = self.policy.get_values(critic_input)
        policy_loss = (advantages*action_log_probs).mean()
        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.algo_cfg.ent_coef).backward()
        max_grad_norm = self.algo_cfg.max_grad_norm if self.algo_cfg.max_grad_norm else 1e6
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_grad_norm)
        self.policy.actor_optimizer.step()
        value_loss = self.cal_value_loss(values, value_predicts, returns)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.algo_cfg.vf_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_grad_norm)
        self.policy.critic_optimizer.step()
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm

    def train(self, buffer:A2CBuffer):
        """
        Train the policy using samples from the provided buffer.

        Args:
            buffer (A2CBuffer): Buffer containing collected experience data

        Returns:
            dict: Training metrics including losses and gradient norms
        """
        train_info = {
            'value_loss': 0,
            'policy_loss': 0,
            'dist_entropy': 0,
            'actor_grad_norm': 0,
            'critic_grad_norm': 0,
        }
        tot_updates=0
        for _ in range(self.algo_cfg.num_epochs):
            data_generator = buffer.feed_forward_generator()
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm \
                    = self.a2c_update(sample)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                train_info['critic_grad_norm'] += critic_grad_norm.item()
                tot_updates += 1
        for k in train_info.keys():
            train_info[k] /= tot_updates
        return train_info

    def prep_training(self):
        """
        Prepare networks for training by setting them to training mode.
        """
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """
        Prepare networks for rollout by setting them to evaluation mode.
        """
        self.policy.actor.eval()
        self.policy.critic.eval()
