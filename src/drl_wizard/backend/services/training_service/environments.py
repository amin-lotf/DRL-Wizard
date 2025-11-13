from argparse import Action
from dataclasses import dataclass
from typing import Optional

from drl_wizard.common.types import ActionType


@dataclass
class EnvironmentState:
    env_id: str
    env_name: str
    origin:str
    supported_action: ActionType




def create_environment(env_id: str, env_name: str, origin:str, supported_action: ActionType) -> EnvironmentState:
    return EnvironmentState(
        env_id=env_id,
        env_name=env_name,
        origin=origin,
        supported_action=supported_action
    )