from __future__ import annotations

from pathlib import Path
from typing import Any

from drl_wizard.common.types import AlgoType
from drl_wizard.configs.app_cfg import AppConfig


def build_gym_runner(
    app_cfg: AppConfig,
    logger: Any | None = None,
    checkpoint_dir: str | Path | None = None,
    allow_saving: bool = True,
):
    runner_kwargs = {
        "config": app_cfg,
        "logger": logger,
        "checkpoint_dir": checkpoint_dir,
        "allow_saving": allow_saving,
    }

    if app_cfg.algo_cfg.algo_id == AlgoType.PPO:
        from drl_wizard.algorithms.runners.gym_runners.ppo_runner import PPORunner

        return PPORunner(**runner_kwargs)
    if app_cfg.algo_cfg.algo_id == AlgoType.SAC:
        from drl_wizard.algorithms.runners.gym_runners.sac_runner import SACRunner

        return SACRunner(**runner_kwargs)
    if app_cfg.algo_cfg.algo_id == AlgoType.DQN:
        from drl_wizard.algorithms.runners.gym_runners.dqn_runner import DQNRunner

        return DQNRunner(**runner_kwargs)
    if app_cfg.algo_cfg.algo_id == AlgoType.A2C:
        from drl_wizard.algorithms.runners.gym_runners.a2c_runner import A2CRunner

        return A2CRunner(**runner_kwargs)
    if app_cfg.algo_cfg.algo_id == AlgoType.TRPO:
        from drl_wizard.algorithms.runners.gym_runners.trpo_runner import TRPORunner

        return TRPORunner(**runner_kwargs)

    raise ValueError(f"Unsupported algo {app_cfg.algo_cfg.algo_id}")
