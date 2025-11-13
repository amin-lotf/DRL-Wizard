from typing import Any

import numpy as np

from drl_wizard.common.types import ModelLike

def squeeze_scalar(x: np.ndarray):
    """
    If x is a numpy array with shape (1,), (1,1), (1,1,1), ...
    return its scalar value (a Python number).
    Otherwise, return x itself.
    """
    x = np.asarray(x)
    if x.size == 1:
        return x.item()  # extract scalar
    return x


def get_field(env:ModelLike, name: str) -> Any:
    """Read a field from either a dict or a Pydantic model instance."""
    if isinstance(env, dict):
        return env.get(name)
    # Pydantic BaseModel (v1 or v2) behaves with attribute access
    return getattr(env, name)