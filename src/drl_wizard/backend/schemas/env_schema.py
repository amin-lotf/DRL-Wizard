from pydantic import BaseModel

from drl_wizard.common.types import ActionType


class EnvResponse(BaseModel):
    env_id: str
    env_name: str
    origin:str
    supported_action: ActionType

    model_config = {
        "json_schema_extra" : {
            "example" : {
                "env_id" : "ALE/Pong-v5",
                "env_name" : "Ant",
                "origin" : "gymnasium",
                "supported_action" : ActionType.DISCRETE
            }
        }
    }