import json
from datetime import datetime, timezone
from pathlib import Path
import torch





_next_gpu = 0  # trivial round-robin; replace with something smarter if needed
def pick_device_id() -> int | None:
    n = torch.cuda.device_count()
    if n == 0:
        return None  # CPU-only
    global _next_gpu
    did = _next_gpu % n
    _next_gpu += 1
    return did



def ensure_aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    # If DB returns naive but you *know* it's stored as UTC, attach UTC:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# -------- helpers for JSON ----------
def json_default(o):
    # robust encoder for dataclass -> JSON
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    try:
        # Enums (ResultType) -> their value (e.g. "train")
        import enum
        if isinstance(o, enum.Enum):
            return o.value
    except Exception:
        pass
    raise TypeError(f"Not JSON serializable: {type(o)}")

