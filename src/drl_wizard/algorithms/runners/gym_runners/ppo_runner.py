import threading
import time
import numpy as np
import torch
from gymnasium.spaces import Discrete
from drl_wizard.algorithms.runners.base_runners.ppo_base_runner import Runner
from drl_wizard.algorithms.utils.extras import tensor_to_numpy, check
from drl_wizard.backend.services.logging.json_logger import SegmentedJsonlLogger


class PPORunner(Runner):
    def __init__(self, config, logger: SegmentedJsonlLogger | None = None, checkpoint_dir=None, allow_saving: bool = True):
        super(PPORunner, self).__init__(
            config,
            logger,
            checkpoint_dir=checkpoint_dir,
            allow_saving=allow_saving,
        )

    def run(self,stop_event:threading.Event=None):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.algo_cfg.num_epochs // self.n_envs
        for episode in range(episodes):
            for step in range(self.algo_cfg.episode_length):
                if stop_event is not None and stop_event.is_set():
                    break
                actions, action_log_probs, values = self.collect()
                obs, rewards, dones, infos = self.envs.step(actions.squeeze(axis=1))
                if isinstance(self.envs.observation_space, Discrete):
                    obs = np.expand_dims(obs, axis=(1,2))
                else:
                    obs = np.expand_dims(obs, axis=1)
                rewards = np.expand_dims(rewards, axis=(1, 2))
                dones = np.expand_dims(dones, axis=(1, 2))
                data = obs, actions, rewards, dones, values, action_log_probs
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
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                m_reward = train_infos["average_step_rewards"]
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
        actions_t, action_log_probs_t = self.policy.get_actions(obs_t)
        values_t = self.policy.get_values(shared_obs_t)
        actions = np.array(np.split(tensor_to_numpy(actions_t), self.n_envs))
        action_log_probs = np.array(np.split(tensor_to_numpy(action_log_probs_t), self.n_envs))
        values = np.array(np.split(tensor_to_numpy(values_t), self.n_envs))
        return actions, action_log_probs, values

    def insert(self, data):
        obs, act, reward, done, value_predict, log_prob = data
        self.buffer.insert(obs, act, reward, done, value_predict, log_prob)

    @torch.no_grad()
    def eval(self, total_num_steps:int):
        metrics = self.evaluate_model(self.app_cfg.eval_episodes, deterministic=True)
        mean_eval_rewards = metrics["average_episode_reward"]
        if mean_eval_rewards > self.best_reward:
            self.save(is_best=True)
            self.best_reward=mean_eval_rewards
        eval_env_infos = {'eval_average_episode_rewards': mean_eval_rewards}
        print(f"eval average episode rewards: {mean_eval_rewards:.2f}")
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def evaluate_model(
        self,
        episodes: int,
        deterministic: bool = False,
        render_first_episode: bool = False,
    ):
        self.trainer.prep_rollout()
        eval_tot_rewards = []
        step_rewards = []
        rendered_frames = []
        render_warning = None
        render_fps = self._get_render_fps()
        remaining_episodes = episodes

        if render_first_episode:
            if self.render_env is None:
                render_warning = "Render environment is not available for this evaluation."
            else:
                try:
                    episode_reward, episode_step_rewards, rendered_frames = self._evaluate_episode(
                        self.render_env,
                        deterministic=deterministic,
                        capture_frames=True,
                    )
                    eval_tot_rewards.append(episode_reward)
                    step_rewards.extend(episode_step_rewards)
                    remaining_episodes -= 1
                    if not rendered_frames:
                        render_warning = (
                            "Rendering was requested, but the environment did not return any frames."
                        )
                except Exception as exc:
                    render_warning = f"Rendering failed: {exc}"

        for _ in range(remaining_episodes):
            episode_reward, episode_step_rewards, _ = self._evaluate_episode(
                self.eval_envs,
                deterministic=deterministic,
                capture_frames=False,
            )
            eval_tot_rewards.append(episode_reward)
            step_rewards.extend(episode_step_rewards)

        results = {
            "average_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "average_episode_reward": float(np.mean(eval_tot_rewards)) if eval_tot_rewards else 0.0,
        }
        if render_first_episode:
            results["rendered_frames"] = rendered_frames
            results["render_fps"] = render_fps
            results["render_warning"] = render_warning
        return results

    def _evaluate_episode(self, envs, deterministic: bool = False, capture_frames: bool = False):
        env_count = getattr(envs, "num_envs", self.n_eval_envs)
        eval_obs, _ = envs.reset()
        eval_episode_rewards = []
        episode_step_rewards = []
        rendered_frames = []

        if capture_frames:
            frame = envs.render()
            if frame is not None:
                rendered_frames.append(frame)

        while True:
            eval_obs_t = check(eval_obs, torch.float32, device=self.device)
            actions_t, _ = self.policy.get_actions(eval_obs_t, deterministic=deterministic)
            actions = np.array(np.split(tensor_to_numpy(actions_t), env_count))
            eval_obs, eval_rewards, eval_dones, _ = envs.step(actions.squeeze(axis=1))
            episode_step_rewards.append(float(np.mean(eval_rewards)))
            eval_episode_rewards.append(eval_rewards)
            if np.any(eval_dones):
                break
            if capture_frames:
                frame = envs.render()
                if frame is not None:
                    rendered_frames.append(frame)

        return float(np.sum(np.array(eval_episode_rewards))), episode_step_rewards, rendered_frames

    def _get_render_fps(self, default_fps: int = 30) -> int:
        if self.render_env is None:
            return default_fps
        try:
            metadata = self.render_env.get_attr("metadata")[0]
        except Exception:
            metadata = getattr(self.render_env, "metadata", None)
        render_fps = metadata.get("render_fps") if isinstance(metadata, dict) else None
        if isinstance(render_fps, (int, float)) and render_fps > 0:
            return int(round(render_fps))
        return default_fps

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
