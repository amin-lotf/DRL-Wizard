from dataclasses import dataclass
from typing import List

from drl_wizard.common.types import AlgoType, ActionType


@dataclass
class AlgoState:
    algo_id: AlgoType
    algo_name: str
    action_type: List[ActionType]



def create_algo(algo_id: AlgoType, algo_name: str, action_type: List[ActionType]) -> AlgoState:
    return AlgoState(algo_id=algo_id, algo_name=algo_name, action_type=action_type)