from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from drl_wizard.common.types import ResultType, ResultName


@dataclass
class JobResultState:
    job_id: int
    result_id: Optional[int] = None
    result_type: Optional[ResultType] = None
    manifest_uri: Optional[str] = None
    segment_steps: Optional[int] = None
    latest_step: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def create_job_result(job_id: int, result_type: ResultType, segment_steps: int,
                      uri: Optional[str] = None) -> JobResultState:
    return JobResultState(
        job_id=job_id,
        result_type=result_type,
        segment_steps=segment_steps,
        latest_step=0,
        manifest_uri=uri
    )
