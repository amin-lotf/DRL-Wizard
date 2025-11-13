import torch
from torch.optim.lr_scheduler import StepLR
from drl_wizard.algorithms.algos.a2c_algo.actor_critic import Actor, Critic
from drl_wizard.configs import A2CConfig
from drl_wizard.configs.app_cfg import AppConfig


class Policy:
    """
    A2C Policy class that manages actor and critic networks for reinforcement learning.

    This class handles the policy and value function approximation using separate networks
    for the actor (policy) and critic (value function). It manages the optimization of both
    networks including learning rate scheduling.
    """

    def __init__(self, actor_input_space, critic_input_space, action_space, cfg: AppConfig):
        """
        Initialize the Policy with actor and critic networks.

        Args:
            actor_input_space: Input space dimensions for the actor network
            critic_input_space: Input space dimensions for the critic network
            action_space: Action space dimensions for the actor network
            cfg (AppConfig): Configuration object containing algorithm and environment settings
        """
        self.algo_cfg: A2CConfig = cfg.algo_cfg
        self.actor = Actor(actor_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.critic = Critic(critic_input_space, self.algo_cfg).to(cfg.resolved_device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.algo_cfg.actor_lr, eps=self.algo_cfg.opti_eps,
                                                weight_decay=self.algo_cfg.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.algo_cfg.critic_lr,
                                                 eps=self.algo_cfg.opti_eps,
                                                 weight_decay=self.algo_cfg.weight_decay)

        lr_decay_steps = cfg.total_steps // self.algo_cfg.episode_length

        self.actor_scheduler = StepLR(
            self.actor_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

        self.critic_scheduler = StepLR(
            self.critic_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

    def lr_decay(self):
        """
        Step the learning rate schedulers for both actor and critic networks.
        This reduces the learning rate according to the scheduler configuration.
        """
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_actions(self, obs, available_actions=None, deterministic=False):
        """
        Get actions from the actor network for given observations.

        Args:
            obs: Current environment observations
            available_actions: Mask of available actions (optional)
            deterministic: If True, return deterministic actions instead of sampling

        Returns:
            tuple: (actions, action log probabilities)
        """
        actions, action_log_probs = self.actor(obs, available_actions, deterministic)
        return actions, action_log_probs

    def get_values(self, obs):
        """
        Get value estimates from the critic network for given observations.

        Args:
            obs: Current environment observations

        Returns:
            tensor: Estimated values for the given observations
        """
        values = self.critic(obs)
        return values

    def evaluate_actions(self, obs, action, available_actions=None):
        """
        Evaluate actions by computing log probabilities and entropy.

        Args:
            obs: Current environment observations
            action: Actions to evaluate
            available_actions: Mask of available actions (optional)

        Returns:
            tuple: (action log probabilities, distribution entropy)
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions)
        return action_log_probs, dist_entropy
