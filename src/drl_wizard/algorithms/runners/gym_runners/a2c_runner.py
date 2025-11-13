import threading
import time
import numpy as np
import torch
from gymnasium.spaces import Discrete
from drl_wizard.algorithms.runners.base_runners.a2c_base_runner import Runner
from drl_wizard.algorithms.utils.extras import tensor_to_numpy, check
from drl_wizard.backend.services.logging.json_logger import SegmentedJsonlLogger


class A2CRunner(Runner):
    def __init__(self, config,logger:SegmentedJsonlLogger):
        super(A2CRunner, self).__init__(config, logger)

    def run(self,stop_event:threading.Event=None):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.algo_cfg.num_epochs // self.n_envs
        for episode in range(episodes):
            for step in range(self.algo_cfg.episode_length):
                if stop_event is not None and stop_event.is_set():
                    break
                actions, values = self.collect()
                obs, rewards, dones, infos = self.envs.step(actions.squeeze(axis=1))
                if isinstance(self.envs.observation_space, Discrete):
                    obs = np.expand_dims(obs, axis=(1,2))
                else:
                    obs = np.expand_dims(obs, axis=1)
                rewards = np.expand_dims(rewards, axis=(1, 2))
                dones = np.expand_dims(dones, axis=(1, 2))
                data = obs, actions, rewards, dones, values
                self.insert(data)
            if stop_event is not None and stop_event.is_set():
                break
            self.compute()
            train_infos = self.train()
            if self.algo_cfg.use_lr_decay:
                self.policy.lr_decay()
            cur_num_steps = (episode + 1) * self.algo_cfg.episode_length
            if episode > 0 and episode % self.app_cfg.save_interval == 0 or episode == episodes - 1:
                self.save()
            if episode > 0 and episode % self.app_cfg.log_interval == 0:
                end = time.time()
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards)
                m_reward = train_infos["average_episode_rewards"]
                print(
                    f"steps: {cur_num_steps}, episodes: {episode}, average episode rewards: {m_reward:.3f}, {(end - start) / cur_num_steps:.3f}Steps/second")
                self.log_train(train_infos, cur_num_steps)
            if self.app_cfg.use_eval and episode > 0 and episode % self.app_cfg.eval_interval == 0:
                self.eval(cur_num_steps)

    def warmup(self):
        obs, _ = self.envs.reset()
        if isinstance(self.envs.observation_space, Discrete):
            obs = np.expand_dims(obs, axis=(1, 2))
        else:
            obs = np.expand_dims(obs, axis=1)
        self.buffer.warmup(obs)

    @torch.no_grad()
    def collect(self):
        self.trainer.prep_rollout()
        obs_t = check(self.buffer.latest_obs_batched, torch.float32, device=self.device)
        shared_obs_t = check(self.buffer.latest_shared_obs_batched, torch.float32, device=self.device)
        actions_t, _ = self.policy.get_actions(obs_t)
        values_t = self.policy.get_values(shared_obs_t)
        actions = np.array(np.split(tensor_to_numpy(actions_t), self.n_envs))
        values = np.array(np.split(tensor_to_numpy(values_t), self.n_envs))
        return actions, values

    def insert(self, data):
        obs, act, reward, done, value_predict = data
        self.buffer.insert(obs, act, reward, done, value_predict)

    @torch.no_grad()
    def eval(self, total_num_steps:int):
        # only 1 env for eval
        eval_obs, _ = self.eval_envs.reset()
        self.trainer.prep_rollout()
        eval_tot_rewards=[]
        for _ in range(self.app_cfg.eval_episodes):
            eval_episode_rewards = []
            while True:
                eval_obs_t = check(eval_obs, torch.float32, device=self.device)
                actions_t, _ = self.policy.get_actions(eval_obs_t)
                actions = np.array(np.split(tensor_to_numpy(actions_t), self.app_cfg.n_eval_envs))
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(actions.squeeze(axis=1))
                eval_episode_rewards.append(eval_rewards)
                if np.any(eval_dones):
                    break
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_tot_rewards.append(np.sum(eval_episode_rewards))
        mean_eval_rewards = np.mean(eval_tot_rewards[-100:])
        if mean_eval_rewards > self.best_reward:
            self.save(is_best=True)
            self.best_reward=mean_eval_rewards
            self.best_reward=mean_eval_rewards
        eval_env_infos = {'eval_average_episode_rewards': mean_eval_rewards}
        print(f"eval average episode rewards: {mean_eval_rewards:.2f}")
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        render_obs, _ = self.render_env.reset()
        self.trainer.prep_rollout()
        while True:
            render_obs_t = check(render_obs, torch.float32, device=self.device).reshape(1, -1)
            actions_t, _ = self.policy.get_actions(render_obs_t)
            actions = np.array(np.split(tensor_to_numpy(actions_t), 1))
            eval_obs, eval_rewards, eval_dones, eval_infos = self.render_env.step(actions.squeeze(axis=1))
            frame= self.render_env.render()
            self.log_render(frame)
            if np.any(eval_dones):
                break
