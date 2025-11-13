import torch
from torch.optim.lr_scheduler import StepLR
from drl_wizard.algorithms.algos.ppo_algo.actor_critic import Actor, Critic
from drl_wizard.configs import  PPOConfig
from drl_wizard.configs.app_cfg import AppConfig


class Policy:
    """Policy class that manages actor and critic networks for PPO algorithm.

    This class handles the policy and value function approximation, including network
    initialization, action selection, value estimation, and learning rate scheduling.

    Attributes:
        algo_cfg (PPOConfig): Configuration for the PPO algorithm
        actor (Actor): Neural network for the policy/actor
        critic (Critic): Neural network for the value function/critic
        actor_optimizer (Adam): Optimizer for the actor network
        critic_optimizer (Adam): Optimizer for the critic network
        actor_scheduler (StepLR): Learning rate scheduler for actor
        critic_scheduler (StepLR): Learning rate scheduler for critic
    """

    def __init__(self, actor_input_space, critic_input_space, action_space, cfg: AppConfig):
        """Initialize Policy with actor and critic networks.

        Args:
            actor_input_space: Input space dimensions for actor network
            critic_input_space: Input space dimensions for critic network
            action_space: Action space dimensions
            cfg (AppConfig): Application configuration containing algorithm settings
        """
        self.algo_cfg: PPOConfig = cfg.algo_cfg
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
        """Update learning rates for both actor and critic using their schedulers."""
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_actions(self, obs, available_actions=None, deterministic=False):
        """Get actions from the actor network for given observations.

        Args:
            obs: Current observations
            available_actions: Mask of available actions (optional)
            deterministic: If True, return deterministic actions instead of sampling

        Returns:
            tuple: (actions, action log probabilities)
        """
        actions, action_log_probs = self.actor(obs, available_actions, deterministic)
        return actions, action_log_probs

    def get_values(self, obs):
        """Get value estimates from the critic network.

        Args:
            obs: Current observations

        Returns:
            tensor: Estimated values for the observations
        """
        values = self.critic(obs)
        return values

    def evaluate_actions(self, obs, action, available_actions=None):
        """Evaluate actions by computing log probabilities and entropy.

        Args:
            obs: Current observations
            action: Actions to evaluate
            available_actions: Mask of available actions (optional)

        Returns:
            tuple: (action log probabilities, distribution entropy)
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions)
        return action_log_probs, dist_entropy


