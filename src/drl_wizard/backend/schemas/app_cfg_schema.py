from pydantic import Field

from drl_wizard.backend.schemas.algo_cfg_schema import AlgoConfigSchema
from drl_wizard.backend.schemas.general_cfg_schema import GeneralConfigSchema
from drl_wizard.backend.schemas.log_cfg_schema import LogConfigSchema


class AppConfigSchema(GeneralConfigSchema):
    algo_cfg: AlgoConfigSchema
    log_cfg: LogConfigSchema = Field(default_factory=lambda:LogConfigSchema())
