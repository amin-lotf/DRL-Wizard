import threading
from typing import Dict

# Process-local, in-memory registry
_events: Dict[int, threading.Event] = {}

def ensure_event(job_id: int) -> threading.Event:
    evt = _events.get(job_id)
    if evt is None:
        evt = threading.Event()
        _events[job_id] = evt
    return evt

def set_stop_event(job_id: int) -> None:
    if job_id in _events:
        _events[job_id].set()

def clear_event(job_id: int) -> None:
    _events.pop(job_id, None)
