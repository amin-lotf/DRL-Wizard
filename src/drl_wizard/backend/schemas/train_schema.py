from datetime import  datetime
from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field

from drl_wizard.backend.schemas import EnvResponse
from drl_wizard.backend.schemas.algo_cfg_schema import BaseAlgoConfigSchema, PPOConfigSchema, AlgoConfigSchema
from drl_wizard.backend.schemas.algo_schema import AlgoResponse
from drl_wizard.backend.schemas.base_schema import GlobalModel
from drl_wizard.backend.schemas.general_cfg_schema import GeneralConfigSchema
from drl_wizard.backend.schemas.log_cfg_schema import LogConfigSchema
from drl_wizard.common.types import JobStatus, ResultType, AlgoType


class JobResultResponse(BaseModel):
    id:int
    type:ResultType
    created_at: datetime
    result: Dict[str, Any]
    model_config ={
        'from_attributes':True
    }


class JobResponse(GlobalModel):
    job_id: Optional[int] = Field(default=None, description="Present for async mode")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    env: Optional[EnvResponse] = None
    algo: Optional[AlgoResponse] = None
    created_at: Optional[datetime] =  None
    started_at: Optional[datetime] =  None
    finished_at: Optional[datetime] = None
    detail: Optional[str] = None
    result: Optional[List[JobResultResponse]] = None
    model_config = {
        "json_schema_extra" : {
            "examples" : [
                {
                    "job_id": "1234567890",
                    "status": "running",
                    "env": {
                        "env_id": "Ant-v5",
                        "env_name": "Ant",
                        "origin": "gymnasium",
                        "supported_action": "discrete"
                    },
                    "algo": {
                        "algo_id": "ppo",
                        "algo_name": "PPO",
                        "action_type": ["discrete", "multi_discrete"]
                    },
                    "created_at": "2023-01-01T00:00:00Z",
                    "started_at": "2023-01-01T00:00:00Z",
                    "finished_at": None
                }
            ]
        }}





class JobRequest(BaseModel):
    env_id: str
    algo_id: AlgoType
    general_cfg:GeneralConfigSchema
    log_cfg: LogConfigSchema
    algo_cfg: AlgoConfigSchema
    model_config = {
        "json_schema_extra" : {
            "examples" : [
                {
                    "env_id": "Ant-v5",
                    "algo_id": AlgoType.PPO,
                    "general_cfg":GeneralConfigSchema().model_dump(),
                    "log_cfg": LogConfigSchema().model_dump(),
                    "algo_cfg": PPOConfigSchema().model_dump()
                }
            ]
        }
    }