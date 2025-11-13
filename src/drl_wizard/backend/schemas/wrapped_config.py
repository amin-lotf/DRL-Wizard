# schemas/wrapped_config.py
from __future__ import annotations
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, get_type_hints
from pydantic import BaseModel, Field


T = TypeVar("T")

class FieldMeta(BaseModel):
    label: str
    description: str = ""
    type: str = "str"
    required: bool = False
    default: Any = None
    # common constraints
    ge: float | int | None = None
    gt: float | int | None = None
    le: float | int | None = None
    lt: float | int | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    # UI helpers
    enum_choices: List[str] | None = None
    order: int | None = None  # field order in the model

class WrappedConfigSchema(BaseModel, Generic[T]):
    config: T
    meta: Dict[str, FieldMeta]
