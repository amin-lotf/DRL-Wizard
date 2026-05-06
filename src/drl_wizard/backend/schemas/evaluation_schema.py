from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from drl_wizard.common.types import AlgoType


class SavedRunSummarySchema(BaseModel):
    run_id: str
    run_dir: str
    env_id: str
    algo_id: AlgoType
    checkpoint_label: str
    checkpoint_path: str


class SavedRunDiscoveryResponse(BaseModel):
    runs: list[SavedRunSummarySchema]
    warnings: list[str] = Field(default_factory=list)


class SavedRunDetailsResponse(BaseModel):
    summary: SavedRunSummarySchema
    env_config: dict[str, Any]
    algo_config: dict[str, Any]
    log_config: dict[str, Any]
    raw_app_config: dict[str, Any]
    eval_episodes_default: int = Field(..., gt=0)


class SavedRunEvaluationRequest(BaseModel):
    env_id: str
    episodes: int | None = Field(default=None, gt=0)
    render: bool = False


class SavedRunEvaluationResponse(BaseModel):
    average_step_reward: float
    average_episode_reward: float
    rendered_video_base64: str | None = None
    rendered_video_mime_type: str | None = None
    render_warning: str | None = None
