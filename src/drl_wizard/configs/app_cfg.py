from dataclasses import dataclass, field
from typing import Optional

from drl_wizard.configs.algo_cfg import BaseAlgoConfig, PPOConfig
from drl_wizard.configs.general_cfg import GeneralConfig
from drl_wizard.configs.log_cfg import LogConfig


@dataclass
class AppConfig(GeneralConfig):
    # keep domain type broad: BaseAlgoConfig
    algo_cfg: Optional[BaseAlgoConfig] = field(default_factory=lambda:None)
    log_cfg: Optional[LogConfig] = field(default_factory=lambda:None)
