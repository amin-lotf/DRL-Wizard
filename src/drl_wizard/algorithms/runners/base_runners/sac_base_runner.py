import threading
from pathlib import Path
import numpy as np
import torch

from drl_wizard.algorithms.algos.sac_algo.policy import Policy
from drl_wizard.algorithms.algos.sac_algo.trainer import Trainer
from drl_wizard.algorithms.buffers.sac_buffer import SACConfig, SACBuffer
from drl_wizard.backend.services.logging.json_logger import SegmentedJsonlLogger
from drl_wizard.common.types import ResultType
from drl_wizard.configs import SACConfig
from drl_wizard.algorithms.utils.extras import make_train_env, tensor_to_numpy, check
from drl_wizard.configs.app_cfg import AppConfig


class Runner(object):
    def __init__(self, config: AppConfig,logger:SegmentedJsonlLogger):
        self.app_cfg: AppConfig = config
        self.algo_cfg: SACConfig = config.algo_cfg
        self.envs = make_train_env(config, is_eval=False)
        self.eval_envs = make_train_env(config, is_eval=True)
        self.render_env = make_train_env(config, is_eval=True, is_render=True) if self.app_cfg.is_render else None
        self.device = config.resolved_device
        self.env_name = config.env_id
        self.num_env_steps = config.total_steps
        self.n_envs = config.n_envs
        self.n_eval_envs = config.n_eval_envs
        self.best_reward = -np.inf
        self.run_dir = Path(config.run_dir)
        self.logger = logger
        self.save_dir = self.run_dir / 'models'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.algo_cfg.num_agents == 1:
            q_input = self.envs.observation_space
            shared_obs_space = None
        else:
            raise NotImplementedError("Multi-agent PPO is not supported yet.")
        self.policy = Policy(self.envs.observation_space, q_input,
                             self.envs.action_space, config)
        self.trainer = Trainer(self.policy, config)
        self.buffer = SACBuffer(config,
                                self.envs.observation_space,
                                self.envs.action_space,
                                shared_obs_space
                                )

    def run(self,stop_event:threading.Event=None):
        raise NotImplementedError


    def collect(self,obs,masked_acts=None):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def eval(self,total_num_steps:int):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        return train_infos

    def save(self, is_best=False):
        prefix_str = "best_" if is_best else "latest_"
        path=self.logger.checkpoints_path
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(path / f"{prefix_str}actor.pt"))
        q1_net = self.trainer.policy.q_net1
        torch.save(q1_net.state_dict(), str(path / f"{prefix_str}q1_net.pt"))
        q2_net = self.trainer.policy.q_net2
        torch.save(q2_net.state_dict(), str(path / f"{prefix_str}q2_net.pt"))
        log_alpha = self.trainer.policy.log_alpha
        torch.save(log_alpha.detach().cpu(), str(path / f"{prefix_str}log_alpha.pt"))

    def restore(self, is_best=False):
        prefix_str = "best_" if is_best else "latest_"
        path = self.logger.checkpoints_path
        policy_actor_state_dict = torch.load(str(path / f"{prefix_str}actor.pt"))
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        q1_net_state_dict = torch.load(str(path / f"{prefix_str}q1_net.pt"))
        self.policy.q_net1.load_state_dict(q1_net_state_dict)
        q2_net_state_dict = torch.load(str(path / f"{prefix_str}q2_net.pt"))
        self.policy.q_net2.load_state_dict(q2_net_state_dict)
        log_alpha = torch.load(str(path / f"{prefix_str}log_alpha.pt"), map_location=self.app_cfg.resolved_device)
        self.policy.log_alpha.data.copy_(log_alpha)
        self.policy.tgt_q_net1.load_state_dict(q1_net_state_dict)
        self.policy.tgt_q_net2.load_state_dict(q2_net_state_dict)

    def log_train(self, train_infos, cur_steps):
        self.logger.log_data(train_infos, cur_steps, log_type=ResultType.TRAIN)

    def log_env(self, env_infos, cur_steps):
        self.logger.log_data(env_infos, cur_steps, log_type=ResultType.EVALUATE)

    def log_render(self,frame):
        self.logger.log_frame(frame)

