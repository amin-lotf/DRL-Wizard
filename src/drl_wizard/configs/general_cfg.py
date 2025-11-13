from dataclasses import field, dataclass
from pathlib import Path
import torch

from drl_wizard.common.types import DeviceType


@dataclass
class GeneralConfig:
    device: DeviceType = field(default=DeviceType.AUTO)     # "cpu" | "cuda" | "auto"
    env_id: str = field(default="Ant-v5")
    seed: int = field(default=0)
    is_render: bool = field(default=False)
    n_envs: int = field(default=10)
    n_eval_envs: int = field(default=1)
    total_steps: int = field(default=1_000_000)
    run_dir: Path = field(default=Path("./runs"))
    save_interval: int = field(default=10)
    log_interval: int = field(default=10)
    rescale_frames:bool = field(default=True)
    use_eval: bool = field(default=True)
    eval_interval: int = field(default=10)
    eval_episodes: int = field(default=4)


    def __post_init__(self):
        if not self.env_id or not self.env_id.strip():
            raise ValueError("env_id cannot be empty")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be one of 'cpu','cuda','auto'")

    @property
    def resolved_device(self) -> str:
        if self.device == DeviceType.AUTO:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return str(self.device.value)