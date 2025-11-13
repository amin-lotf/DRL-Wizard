from datetime import datetime

import httpx
import streamlit as st
from drl_wizard.common.types import JobActions


def format_date_time(date_time: str) -> str:
    """Format a date/time string to a more readable format."""
    utc_time = datetime.fromisoformat(date_time)
    local_time = utc_time.astimezone()  # local timezone
    return local_time.strftime("%Y-%m-%d %H:%M:%S")

def unwrap_error(e: Exception) -> str:
    """Extract a human-readable error message from httpx exceptions."""
    if isinstance(e, httpx.HTTPStatusError):
        # The request reached the server, but the response had an error code
        try:
            data = e.response.json()
            detail = data.get("detail") or data
            return f"{e.response.status_code} {e.response.reason_phrase}: {detail}"
        except Exception:
            return f"{e.response.status_code} {e.response.reason_phrase}"

    elif isinstance(e, httpx.RequestError):
        # Connection errors, timeouts, DNS failures, etc.
        return f"Request failed: {e.__class__.__name__}: {e}"

    elif isinstance(e, httpx.TimeoutException):
        return "Request timed out."

    elif isinstance(e, httpx.ConnectError):
        return "Failed to connect to server. Is it running?"

    else:
        # Anything unexpected
        return str(e)


def handle_httpx_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(unwrap_error(e))
    return wrapper

def handle_ui_action(action:JobActions,job_id:int):
    st.session_state.ui_actions.append({'type':action,'job_id':job_id})