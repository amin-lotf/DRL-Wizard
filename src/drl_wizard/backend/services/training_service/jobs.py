from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any
from uuid import uuid4

from drl_wizard.common.types import JobStatus, AlgoType


@dataclass
class JobState:
    job_id: Optional[int] = None
    status: JobStatus = JobStatus.QUEUED
    env_id: Optional[str] = None
    algo_id: Optional[AlgoType] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    detail: Optional[str] = None
    stop_requested: bool = False


def create_job(env_id:str,algo_id:AlgoType) -> JobState:
    job=JobState(env_id=env_id,algo_id=algo_id)
    return job

def mark_stopped(job: JobState) -> None:
    job.status = JobStatus.STOPPED
    job.detail = "Training stopped by user."
    job.finished_at = datetime.now(timezone.utc)


def mark_stop_requested(job: JobState) -> None:
    job.stop_requested = True
    job.detail = "Training is being stopped."
    job.status = JobStatus.STOPPING


def mark_running(job: JobState) -> None:
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)


def mark_finished(job: JobState) -> None:
    job.status = JobStatus.FINISHED
    job.detail = "Training completed successfully."
    job.finished_at = datetime.now(timezone.utc)


def mark_failure(job: JobState, detail: str | None = None) -> None:
    job.status = JobStatus.FAILED
    job.detail = detail
    job.finished_at = datetime.now(timezone.utc)
