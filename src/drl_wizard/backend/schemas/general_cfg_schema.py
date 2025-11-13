# api/schemas/app.py
from pathlib import Path
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field
from pathlib import Path

from drl_wizard.common.types import DeviceType


class GeneralConfigSchema(BaseModel):
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        title="Compute Device",
        description="Select whether to use cpu, cuda, auto."
    )
    env_id: str = Field(
        default="FrozenLake-v1",
        title="Environment ID",
        description="Name of the Gymnasium environment to train on."
    )
    seed: int = Field(
        default=0,
        ge=0,
        title="Random Seed",
        description="Seed for reproducibility."
    )
    n_envs: int = Field(default=1, gt=0, title="Number of Environments")
    n_eval_envs: int = Field(default=1, gt=0, title="Evaluation Environments")
    total_steps: int = Field(default=1_000_000, gt=0, title="Total Training Steps")
    run_dir: Path = Field(default=Path("./runs"), title="Run Directory")
    rescale_frames: bool = Field(default=True, title="Rescale Frames")
    save_interval: int = Field(default=100, gt=0, title="Save Interval")
    log_interval: int = Field(default=10, gt=0, title="Log Interval")
    use_eval: bool = Field(default=True, title="Enable Evaluation")
    eval_interval: int = Field(default=10, gt=0, title="Evaluation Interval")
    eval_episodes: int = Field(default=4, gt=0, title="Evaluation Episodes")



