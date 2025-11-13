# schemas/meta_utils.py
from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Type, get_type_hints

from enum import Enum
from typing import Literal, get_args, get_origin

from drl_wizard.backend.schemas.wrapped_config import FieldMeta


def _enum_choices_from_annotation(ann) -> list[str] | None:
    # Enum class
    if isinstance(ann, type) and issubclass(ann, Enum):
        return [e.value if hasattr(e, "value") else str(e) for e in ann]
    # Literal['a','b']
    if get_origin(ann) is Literal:
        return [str(x) for x in get_args(ann)]
    return None

def build_meta(model: Type[BaseModel]) -> Dict[str, FieldMeta]:
    hints = get_type_hints(model)
    meta: Dict[str, FieldMeta] = {}

    # Preserve field order as declared
    for idx, (name, f) in enumerate(model.model_fields.items()):
        ann = hints.get(name, Any)
        label = f.title or name.replace("_", " ").title()
        desc = f.description or ""
        py_type_name = getattr(ann, "__name__", str(ann))
        # constraints live on f; in v2, they’re mostly simple attrs
        m = FieldMeta(
            label=label,
            description=desc,
            type=py_type_name,
            required=f.is_required(),
            default=f.default,
            ge=getattr(f, "ge", None),
            gt=getattr(f, "gt", None),
            le=getattr(f, "le", None),
            lt=getattr(f, "lt", None),
            min_length=getattr(f, "min_length", None),
            max_length=getattr(f, "max_length", None),
            pattern=getattr(f, "pattern", None),
            enum_choices=_enum_choices_from_annotation(ann),
            order=idx,
        )
        meta[name] = m
    return meta
