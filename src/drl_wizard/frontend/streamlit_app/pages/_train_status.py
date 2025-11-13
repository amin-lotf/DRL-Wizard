# frontend/streamlit_app/pages/job_status.py
import time
import streamlit as st
from drl_wizard.frontend.streamlit_app.components.job_card import job_general_card
from drl_wizard.frontend.streamlit_app.components.plot_card import render_plot_grid, render_multi_plot_grid
from drl_wizard.frontend.streamlit_app.extras import handle_ui_action
from drl_wizard.frontend.streamlit_app.services.api import Api
from drl_wizard.frontend.streamlit_app.settings import BASE_URL
from drl_wizard.common.types import ResultType, JobActions
from pathlib import Path

api = Api(BASE_URL)


if "ui_actions" not in st.session_state:
    st.session_state.ui_actions = []
if "__zip_paths__" not in st.session_state:
    st.session_state["__zip_paths__"] = {}  # { job_id: str(path) }

# --- Title + Refresh controls ---
header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
with header_col1:
    st.title("Job Status")
with header_col2:
    if st.button("Refresh"):
        st.rerun()
with header_col3:
    st.checkbox("Automatic refresh (every 5s)", key="__auto_refresh__", value=False)

job_id = st.session_state.get("job_id")
download_slot = None  # will hold the placeholder returned by the card

if job_id:
    try:
        resp = api.get_job_status(job_id)
        # ⬇️ capture the slot next to the buttons
        download_slot = job_general_card(resp, handle_ui_action, job_id, is_summary=False)
        env_id = resp["env"]["env_id"]
    except RuntimeError as e:
        st.error(str(e))
        env_id = None

    # ----- Training + Evaluation (in expanders) -----

    # sensible defaults once per session
    if "__plot_per_row__" not in st.session_state:
        st.session_state["__plot_per_row__"] = 3
    if "__plot_height__" not in st.session_state:
        st.session_state["__plot_height__"] = 180
    if "__plot_smooth__" not in st.session_state:
        st.session_state["__plot_smooth__"] = 1

    # ---- Training Metrics (this job) ----
    with st.expander("Training Metrics (this job)", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.session_state["__plot_per_row__"] = st.selectbox(
                "Charts per row", [1, 2, 3, 4],
                index=[1, 2, 3, 4].index(st.session_state["__plot_per_row__"])
                if st.session_state["__plot_per_row__"] in [1, 2, 3, 4] else 2,
                key="__per_row_training__"
            )
        with c2:
            st.session_state["__plot_height__"] = st.slider(
                "Chart height", min_value=120, max_value=600,
                value=st.session_state["__plot_height__"], step=10, key="__height_training__"
            )
        with c3:
            st.session_state["__plot_smooth__"] = st.slider(
                "Smoothing (rolling window)", min_value=1, max_value=50,
                value=st.session_state["__plot_smooth__"], step=1, key="__smooth_training__"
            )

        try:
            df = api.get_job_results(str(job_id), ResultType.TRAIN)
            metrics_to_show = [c for c in df.columns if c != "step"][:12]
            render_plot_grid(
                df,
                metrics=metrics_to_show,
                per_row=st.session_state["__plot_per_row__"],
                height=st.session_state["__plot_height__"],
                smooth=st.session_state["__plot_smooth__"],
            )
        except RuntimeError as e:
            st.warning(f"Could not load training metrics: {e}")

    # ---- Evaluation Metrics (this job) ----
    with st.expander("Evaluation Metrics (this job)", expanded=True):
        try:
            eval_df = api.get_job_results(str(job_id), ResultType.EVALUATE)
            eval_metrics = [c for c in eval_df.columns if c != "step"][:12]
            render_plot_grid(
                eval_df,
                metrics=eval_metrics,
                per_row=st.session_state["__plot_per_row__"],
                height=st.session_state["__plot_height__"],
                smooth=st.session_state["__plot_smooth__"],
            )
        except RuntimeError:
            st.info("No evaluation metrics yet for this job.")

    # ---- Env-wide Evaluation Comparison ----
    if env_id:
        with st.expander("Env-wide Evaluation Comparison (all jobs in this env)", expanded=True):
            try:
                env_eval_df = api.get_env_results(env_id, ResultType.EVALUATE)
                if env_eval_df is None or env_eval_df.empty:
                    st.info("No env-wide evaluation data yet.")
                else:
                    available = list(env_eval_df["metric"].dropna().unique())
                    default = [m for m in available if "eval" in m] or available
                    selected = st.multiselect(
                        "Metrics to compare", available, default=default[:6],
                        key="__env_metrics_select__"
                    )

                    c1e, c2e = st.columns([1, 1])
                    with c1e:
                        per_row_env = st.selectbox(
                            "Charts per row (env-wide)", [1, 2, 3],
                            index=0, key="__per_row_env__"
                        )
                    with c2e:
                        height_env = st.slider(
                            "Chart height (env-wide)", 200, 800, 420, 10,
                            key="__height_env__"
                        )

                    render_multi_plot_grid(
                        env_eval_df,
                        metrics=selected,
                        per_row=per_row_env,
                        height=height_env,
                        smooth=st.session_state["__plot_smooth__"],  # reuse smoothing from training/eval
                    )
            except RuntimeError as e:
                st.warning(f"Could not load env-wide evaluation results: {e}")

# ---- Handle UI actions ----
need_rerun = False
while st.session_state.ui_actions:
    action = st.session_state.ui_actions.pop(0)
    try:
        if action["type"] == JobActions.STOP:
            ok = api.stop_job(action["job_id"])
            st.toast(
                f"Job {action['job_id']} stopped successfully"
                if ok else f"Error stopping job {action['job_id']}"
            )
            need_rerun = True

        elif action["type"] == JobActions.DELETE:
            ok = api.delete_job(action["job_id"])
            if ok:
                st.toast(f"Job {action['job_id']} deleted")
                # clear any stored zip path
                st.session_state["__zip_paths__"].pop(action["job_id"], None)
                if st.session_state.get("job_id") == action["job_id"]:
                    st.session_state["job_id"] = None
                need_rerun = True
            else:
                st.error(f"Failed to delete job {action['job_id']}")

        elif action["type"] == JobActions.DOWNLOAD:
            # 1) fetch to disk
            # need_rerun = True
            zip_path = api.download_job_zip(action["job_id"])
            st.session_state["__zip_paths__"][action["job_id"]] = str(zip_path)
            st.toast("Download prepared.")
            # 2) render the button *immediately* into the card's placeholder if available
            if download_slot is not None and st.session_state.get("job_id") == action["job_id"]:
                with download_slot:
                    p = Path(zip_path)
                    with open(p, "rb") as f:
                        st.download_button(
                            "Download ZIP",
                            data=f.read(),
                            file_name=p.name,
                            mime="application/zip",
                            key=f"dl_{action['job_id']}_{p.stat().st_mtime_ns}"
                        )
            # no rerun needed here; we already rendered the button
    except RuntimeError as e:
        st.error(str(e))
        need_rerun = False
        break

if need_rerun:
    st.rerun()

# ---- Automatic refresh every 5 seconds ----
if st.session_state.get("__auto_refresh__"):
    time.sleep(5)
    st.rerun()
