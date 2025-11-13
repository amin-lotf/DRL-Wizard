from typing import List
from pydantic import BaseModel, Field
from drl_wizard.common.types import ActionType, AlgoType


class AlgoResponse(BaseModel):
    algo_id: AlgoType
    algo_name: str
    action_type: List[ActionType]

    model_config = {
        "json_schema_extra": {
            "example": {
                "algo_id": AlgoType.PPO,
                "algo_name": "PPO",
                "action_type": [ActionType.CONTINUOUS.value, ActionType.DISCRETE.value, ActionType.MULTI_DISCRETE.value]
            }
        }
    }
