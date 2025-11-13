from typing import Any
from drl_wizard.common.types import AlgoType

ALGO_REGISTRY: dict[AlgoType, type[Any]] = {}


def register_algo(cls: type) -> type:
    key = getattr(cls, "algo_id", None)
    if key is None:
        raise ValueError(f"{cls.__name__} must define class attribute algo_id")
    ALGO_REGISTRY[key] = cls
    return cls
