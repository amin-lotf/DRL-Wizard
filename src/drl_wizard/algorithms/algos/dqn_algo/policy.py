import torch
from torch.optim.lr_scheduler import StepLR
from drl_wizard.algorithms.algos.dqn_algo.actor import QActor
from drl_wizard.configs import  DQNConfig
from drl_wizard.configs.app_cfg import AppConfig


class Policy:
    """A policy class for DQN algorithm that manages Q-network training and action selection.

    Args:
        actor_input_space: Input space dimensions for the actor network
        critic_input_space: Input space dimensions for the critic network
        action_space: Action space dimensions
        cfg (AppConfig): Application configuration containing algorithm parameters
    """

    def __init__(self, actor_input_space, critic_input_space, action_space, cfg: AppConfig):
        self.algo_cfg: DQNConfig = cfg.algo_cfg
        self.epsilon = self.algo_cfg.dqn_epsilon_start
        self.actor = QActor(actor_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.tgt_actor = QActor(actor_input_space, action_space, self.algo_cfg).to(cfg.resolved_device)
        self.tgt_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.algo_cfg.actor_lr, eps=self.algo_cfg.opti_eps,
                                                weight_decay=self.algo_cfg.weight_decay)
        lr_decay_steps = max(1, cfg.total_steps // max(1, self.algo_cfg.batch_size))
        self.actor_scheduler = StepLR(
            self.actor_optimizer, step_size=lr_decay_steps, gamma=self.algo_cfg.lr_gamma
        )

    def epsilon_decay(self,episode):
        """Update epsilon value for epsilon-greedy exploration based on episode number.

        Args:
            episode: Current episode number
        """
        self.epsilon = max(self.algo_cfg.dqn_epsilon_end, self.algo_cfg.dqn_epsilon_start -
                      episode / self.algo_cfg.dqn_epsilon_decay_last_episode)

    def lr_decay(self):
        """Decay learning rate according to the scheduler."""
        self.actor_scheduler.step()

    def get_actions(self, obs, available_actions=None, deterministic=False):
        """Get actions from the actor network for given observations.

        Args:
            obs: Current observations
            available_actions: Mask of available actions (optional)
            deterministic: Whether to use deterministic action selection

        Returns:
            Selected actions
        """
        actions = self.actor(obs,self.epsilon, available_actions, deterministic)
        return actions

    def get_q_vals(self, obs):
        """Get Q-values from the actor network for given observations.

        Args:
            obs: Current observations

        Returns:
            Q-values for all actions
        """
        actions_vals = self.actor.get_vals(obs)
        return actions_vals

    def get_tgt_q_vals(self, obs):
        """Get Q-values from the target actor network for given observations.

        Args:
            obs: Current observations

        Returns:
            Target Q-values for all actions
        """
        actions_vals = self.tgt_actor.get_vals(obs)
        return actions_vals

    torch.no_grad()
    def evaluate_actions(self, obs, action, available_actions=None):
        """Evaluate Q-values for specific actions using the actor network.

        Args:
            obs: Current observations
            action: Actions to evaluate
            available_actions: Mask of available actions (optional)

        Returns:
            Q-values for the given actions
        """
        action_vals = self.actor.evaluate_actions(obs, action, available_actions)
        return action_vals

    @torch.no_grad()
    def evaluate_tgt_actions(self, obs, action, available_actions=None):
        """Evaluate Q-values for specific actions using the target actor network.

        Args:
            obs: Current observations
            action: Actions to evaluate
            available_actions: Mask of available actions (optional)

        Returns:
            Target Q-values for the given actions
        """
        action_vals = self.tgt_actor.evaluate_actions(obs, action, available_actions)
        return action_vals





