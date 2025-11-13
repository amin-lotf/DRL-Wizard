# frontend/streamlit_app/components/job_card.py
from typing import Dict, Any, Optional, Callable
import streamlit as st
from datetime import datetime, timezone
from drl_wizard.common.types import JobActions, JobStatus, STOPPED_STATUSES

def _nbsp(s: str) -> str:
    return s.replace(" ", "\u00A0")

def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _fmt_dt_iso(dt_str: Optional[str]) -> str:
    dt = _parse_iso(dt_str)
    if not dt:
        return "—"
    local = dt.astimezone()
    return _nbsp(local.strftime("%Y-%m-%d %H:%M"))

def _fmt_period(start: Optional[str], end: Optional[str]) -> str:
    s = _fmt_dt_iso(start)
    e = _fmt_dt_iso(end)
    return s if e == "—" else f"{s} → {e}"

def _duration_hms(start_str: Optional[str], end_str: Optional[str]) -> str:
    """Return duration as HH:MM:SS. If end_str is None, uses now()."""

    start = _parse_iso(start_str)
    if not start:
        return "—"
    end = _parse_iso(end_str) or _parse_iso(datetime.now(timezone.utc).isoformat())
    secs = max(0, int((end - start).total_seconds()))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _status_badge(status: JobStatus) -> str:
    s = status.value
    if status in {JobStatus.RUNNING, JobStatus.QUEUED}:
        return f"🟢 {_nbsp(s)}"
    if status in {JobStatus.STOPPING}:
        return f"🟡 {_nbsp(s)}"
    if status in STOPPED_STATUSES:
        return f"🔴 {_nbsp(s)}"
    return f"🔷 {_nbsp(s)}"

def job_general_card(
    job: Dict[str, Any],
    on_action: Optional[Callable[[JobActions, int], None]],
    prefix: str,
    is_summary: bool = True
):
    st.write("---")

    def k(name: str) -> str:
        return f"{prefix}:{job['job_id']}:{name}"

    status = JobStatus(job["status"])
    is_stopped_like = status in STOPPED_STATUSES
    is_active = not is_stopped_like
    job_id = job["job_id"]

    # ---- Top row: add 'Run time' column ----
    info_cols = st.columns([2, 2, 2, 1.6, 3.0, 1.4], gap="small")

    with info_cols[0]:
        st.markdown("**Job ID**")
        st.code(str(job_id), language=None)

    with info_cols[1]:
        st.markdown("**Environment**")
        env_name= f'{job["env"]["env_name"]} ({job["env"]["env_id"]})'
        st.write(_nbsp(env_name))

    with info_cols[2]:
        st.markdown("**Algorithm**")
        st.write(_nbsp(job["algo"]["algo_name"]))

    with info_cols[3]:
        st.markdown("**Status**")
        st.write(_status_badge(status))

    with info_cols[4]:
        st.markdown("**Period**")
        st.write(_fmt_period(job.get("started_at"), job.get("finished_at")))

    with info_cols[5]:
        st.markdown("**Run time**")
        st.write(
            _duration_hms(
                job.get("started_at"),
                job.get("finished_at") if is_stopped_like else None
            )
        )

    # ---- Buttons ----
    btn_cols = st.columns([1, 1, 1.6, 1, 1.2], gap="small")

    with btn_cols[0]:
        if st.button(
            "⏹ Stop",
            key=k("stop"),
            type="primary",
            width='stretch',
            disabled=not is_active,
            help="Stop the running job",
        ):
            if on_action:
                on_action(JobActions.STOP, job_id)

    with btn_cols[1]:
        if is_summary and st.button(
            "🔍 Details",
            key=k("details"),
            width='stretch',
            help="Open the full job view",
        ):
            if on_action:
                on_action(JobActions.DETAILS, job_id)

    with btn_cols[2]:
        if st.button(
            "⬇️ Prepare Zip File",
            key=k("download_trigger"),
            width='stretch',
            help="Bundle logs & artifacts to a ZIP on the server",
        ):
            if on_action:
                on_action(JobActions.DOWNLOAD, job_id)

    with btn_cols[3]:
        if st.button(
            "🗑️ Delete",
            key=k("delete"),
            width='stretch',
            disabled=not is_stopped_like,
            help="Delete this job and its data",
        ):
            if on_action:
                on_action(JobActions.DELETE, job_id)

    spare_slot = btn_cols[4].empty()
    return spare_slot
