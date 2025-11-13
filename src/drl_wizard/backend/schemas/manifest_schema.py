# manifests/schemas.py
from __future__ import annotations
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, field_serializer
from drl_wizard.common.types import ResultType

class SegmentSchema(BaseModel):
    path: str
    start: int
    end: int

    @field_validator("end")
    @classmethod
    def end_ge_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("segment.end must be >= segment.start")
        return v

class ManifestSchema(BaseModel):
    job_id: int
    path: Path
    log_path: Path
    configs_path: Path
    checkpoints_path: Path
    created_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: int = 1
    segments: Dict[ResultType, List[SegmentSchema]] = Field(
        default_factory=lambda: {ResultType.TRAIN: [], ResultType.EVALUATE: []}
    )

    # Accept incoming JSON with string keys ("train"/"evaluate") or enum names
    @field_validator("segments", mode="before")
    @classmethod
    def accept_str_keys(cls, v):
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if isinstance(k, ResultType):
                    out[k] = val
                elif isinstance(k, str):
                    # try value first, then name fallback
                    try:
                        out[ResultType(k)] = val          # "train" -> ResultType.TRAIN (value)
                    except Exception:
                        out[ResultType[k.upper()]] = val   # "TRAIN" -> ResultType.TRAIN (name)
                else:
                    out[ResultType(str(k))] = val
            return out
        return v

    # Emit JSON with string keys; tolerate keys already being strings
    @field_serializer("segments", when_used="json")
    def serialize_enum_keys(self, segs):
        def key_to_str(k):
            # if enum -> its value; if str -> keep; else -> str()
            return getattr(k, "value", k) if isinstance(k, str) else (
                k.value if hasattr(k, "value") else str(k)
            )
        return {key_to_str(k): v for k, v in segs.items()}
