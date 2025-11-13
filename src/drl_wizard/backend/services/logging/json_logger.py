# backend/services/logging/json_logger.py
from __future__ import annotations
import tempfile
from dataclasses import asdict
from pathlib import Path
import json, threading
from typing import Any, Dict, Union, Optional, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from drl_wizard.backend.services.logging.log_manifest import Manifest
from drl_wizard.backend.services.training_service.service import TrainingService
from drl_wizard.backend.services.utils import json_default
from drl_wizard.common.types import ResultType
from drl_wizard.configs.app_cfg import AppConfig

try:
    import zstandard as zstd
except ImportError:
    zstd = None

Number = Union[int, float, np.number]

class SegmentedJsonlLogger:
    """A logger that writes training and evaluation data to segmented JSONL files.

    Handles logging of training metrics and evaluation results, with support for
    buffered writes, data compression, and TensorBoard integration.

    Args:
        svc (TrainingService): Training service instance
        app_cfg (AppConfig): Application configuration
        save_dir (Union[str, Path]): Directory to save logs
        job_id (int): Unique job identifier
    """

    def __init__(
        self,
        svc: TrainingService,
        app_cfg: AppConfig,
        save_dir: Union[str, Path],
        job_id: int,
    ):
        self.svc = svc
        self.job_id = job_id
        self.base = Path(save_dir) / str(job_id)
        self.log_dir = self.base / "log"
        (self.log_dir / ResultType.TRAIN).mkdir(parents=True, exist_ok=True)
        (self.log_dir / ResultType.EVALUATE).mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.base / "manifest.json"
        self.config_path = self.base / "configs" / "app.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = self.base / "checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._buf_train: list[dict] = []
        self._buf_env: list[dict] = []
        self.buffer_rows = app_cfg.log_cfg.buffer_rows
        self.segment_steps = app_cfg.log_cfg.segment_steps
        self.compress = bool(app_cfg.log_cfg.compress and zstd is not None)

        self.writer: Optional[SummaryWriter] = None
        if getattr(app_cfg.log_cfg, "tb_writer", False):
            self.writer = SummaryWriter(str(self.base / "tb"))

        if self.manifest_path.exists():
            try:
                self._manifest = Manifest.load(self.manifest_path)
            except Exception:
                self._manifest = Manifest(
                    job_id=job_id,
                    path=self.manifest_path,
                    log_path=self.log_dir,
                    configs_path=self.config_path,
                    checkpoints_path=self.checkpoints_path,
                )
                self._manifest.atomic_write()
        else:
            self._manifest = Manifest(
                job_id=job_id,
                path=self.manifest_path,
                log_path=self.log_dir,
                configs_path=self.config_path,
                checkpoints_path=self.checkpoints_path,
            )
            self.save_config(app_cfg)
            self._manifest.atomic_write()
        # NOTE: no DB calls here anymore

    async def register_tracks(self) -> None:
        """Call once before training starts; creates TRAIN/EVALUATE rows in DB."""
        await self.svc.add_job_results(
            self.job_id, ResultType.TRAIN, self.segment_steps, str(self.manifest_path)
        )
        await self.svc.add_job_results(
            self.job_id, ResultType.EVALUATE, self.segment_steps, str(self.manifest_path)
        )

    def save_config(self, app_cfg: AppConfig) -> None:
        """Save the application configuration to a JSON file.

        Args:
            app_cfg (AppConfig): Application configuration to save
        """
        dest = self.config_path
        body = asdict(app_cfg)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(dest.parent)) as tmp:
            json.dump(body, tmp, ensure_ascii=False, indent=2, default=json_default)
            tmp_path = Path(tmp.name)
        tmp_path.replace(dest)

    def log_data(self, infos: Dict[str, Any], step: int, log_type: ResultType) -> None:
        """Log training or evaluation metrics for a given step.

        Args:
            infos (Dict[str, Any]): Dictionary containing metrics to log
            step (int): Current training step
            log_type (ResultType): Type of log (TRAIN or EVALUATE)
        """
        if self.writer is not None:
            for k, v in infos.items():
                self.writer.add_scalars(k, {k: _to_scalar(v)}, step)

        row = {"general": {"step": int(step)}}
        for k, v in infos.items():
            if _is_array_like(v):
                arr = np.asarray(v)
                if arr.size == 0:
                    raise ValueError(f"Empty env_infos for key '{k}'")
                row[k] = {
                    "mean": float(np.mean(arr)),
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "std": float(np.std(arr)),
                }
            else:
                val = _to_scalar(v)
                row[k] = {"mean": float(val), "count": 1, "min": float(val), "max": float(val), "std": 0.0}
        self._append(log_type, row)

    def flush(self) -> None:
        """Force write all buffered data to disk."""
        with self._lock:
            self._flush_locked(ResultType.TRAIN)
            self._flush_locked(ResultType.EVALUATE)

    def close(self) -> None:
        """Close the logger, flushing all buffers, and closing TensorBoard writer if present."""
        self.flush()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    # ---- internals ----
    def _append(self, kind: ResultType, row: Dict[str, Any]) -> None:
        """Append a row of data to the appropriate buffer.

        Args:
            kind (ResultType): Type of data (TRAIN or EVALUATE)
            row (Dict[str, Any]): Data row to append
        """
        buf = self._buf_train if kind == ResultType.TRAIN else self._buf_env
        buf.append(row)
        if len(buf) >= self.buffer_rows:
            with self._lock:
                self._flush_locked(kind)

    def _ensure_segment(self, kind: ResultType, step: int) -> Path:
        """Ensure the log segment file exists for the given step.

        Args:
            kind (ResultType): Type of data (TRAIN or EVALUATE)
            step (int): Current training step

        Returns:
            Path: Path to the log segment file
        """
        return self._manifest.ensure_segment_for_step(
            kind=kind,
            step=step,
            segment_steps=self.segment_steps,
            base_dir=self.log_dir,
            compressed=self.compress,
        )

    def _flush_locked(self, kind: ResultType) -> None:
        """Flush buffered data to disk with thread safety.

        Args:
            kind (ResultType): Type of data (TRAIN or EVALUATE) to flush
        """
        buf = self._buf_train if kind == ResultType.TRAIN else self._buf_env
        if not buf:
            return
        max_step = max(r["general"]["step"] for r in buf)
        path = self._ensure_segment(kind, max_step)

        if self.compress:
            cctx = zstd.ZstdCompressor(level=3)
            data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in buf).encode("utf-8")
            with path.open("ab") as f:
                f.write(cctx.compress(data))
        else:
            with path.open("a", encoding="utf-8") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        buf.clear()


def _is_array_like(x: Any) -> bool:
    """Check if a value is array-like (numpy array, list, or tuple).

    Args:
        x (Any): Value to check

    Returns:
        bool: True if value is array-like, False otherwise
    """
    return isinstance(x, (np.ndarray, list, tuple))

def _to_scalar(x: Any) -> Number:
    """Convert a value to a scalar number.

    Args:
        x (Any): Value to convert

    Returns:
        Number: Scalar representation of the value

    Raises:
        ValueError: If x is an empty array-like value
    """
    if _is_array_like(x):
        arr = np.asarray(x)
        if arr.size == 0:
            raise ValueError("Empty array-like value")
        return float(np.mean(arr))
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, bool):
        return int(x)
    return x
