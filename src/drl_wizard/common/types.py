from enum import Enum
from typing import Mapping, Union, Any, Literal
from pydantic import BaseModel

ModelData = Mapping[str, Any]
ModelLike = Union[ModelData, BaseModel]

class DeviceType(str,Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
    AUTO = 'auto'

class AlgoType(str,Enum):
    TRPO = 'TRPO'
    A2C = 'A2C'
    PPO = 'PPO'
    SAC = 'SAC'
    DQN = 'DQN'


class ActionType(str,Enum):
    DISCRETE = 'Discrete'
    CONTINUOUS = 'Continuous'
    MULTI_DISCRETE = 'MultiDiscrete'

class ConfigType(str,Enum):
    GENERAL = 'General'
    LOG = 'Log'
    ALGO = 'Algo'


class ResultType(str,Enum):
    TRAIN = 'train'
    EVALUATE = 'evaluate'

class ResultName(str,Enum):
    LOSS = 'Loss'



class JobStatus(str, Enum):  # use `StrEnum` on 3.11+ if you like
    QUEUED   = "queued"
    RUNNING  = "running"
    STOPPED  = "stopped"
    STOPPING = "stopping"
    FAILED   = "failed"
    FINISHED = "finished"


STOPPED_STATUSES = {
    JobStatus.STOPPED,
    JobStatus.STOPPING,
    JobStatus.FAILED,
    JobStatus.FINISHED,
}

class JobActions(str, Enum):
    STOP = 'stop'
    DETAILS = 'details'
    DELETE = 'delete'
    DOWNLOAD = 'download'
