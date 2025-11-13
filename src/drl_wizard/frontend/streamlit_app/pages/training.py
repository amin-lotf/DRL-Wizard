# pages/train.py
import streamlit as st
from typing import Any

from drl_wizard.frontend.streamlit_app.components.dynamic_forms import (
    selector_action_env_algo,
    render_config_form,
)
from drl_wizard.frontend.streamlit_app.services.api import Api
from drl_wizard.frontend.streamlit_app.settings import BASE_URL

try:
    st.set_page_config(layout="wide")
except Exception:
    # It's okay if another page already called it
    pass


api = Api(BASE_URL)

WKEY = "train_wizard"  # session-state namespace


def _init_state():
    st.session_state.setdefault(
        WKEY,
        {
            "step": 1,          # 1..3  (1=Select, 2=Configure, 3=Review/Start)
            "env": None,
            "algo": None,
            "general_cfg": None,
            "log_cfg": None,
            "algo_cfg": None,
        },
    )


def _reset():
    st.session_state[WKEY] = {
        "step": 1,
        "env": None,
        "algo": None,
        "general_cfg": None,
        "log_cfg": None,
        "algo_cfg": None,
    }


def _load_default_cfgs(env_id: Any, algo_id: Any) -> None:
    state = st.session_state[WKEY]
    if state["general_cfg"] is None:
        state["general_cfg"] = api.get_general_config(env_id)
    if state["log_cfg"] is None:
        state["log_cfg"] = api.get_log_config()
    if state["algo_cfg"] is None:
        state["algo_cfg"] = api.get_algo_config(algo_id)


def _progress():
    state = st.session_state[WKEY]
    # 3 steps: 1->0%, 2->50%, 3->100%
    pct = int(((state["step"] - 1) / 2) * 100)
    st.progress(pct)


def _step_header():
    """Numbered step row + caption for 3 steps."""
    state = st.session_state[WKEY]
    step = state["step"]

    cols = st.columns(3)
    labels = ["Select", "Configure", "Start"]
    for i, c in enumerate(cols, start=1):
        with c:
            active = (i == step)
            prefix = f"**{i}. {labels[i-1]}**" if active else f"{i}. {labels[i-1]}"
            st.markdown(prefix)

    st.caption(f"Step {step} of 3")



st.title("Training Wizard")
_init_state()
state = st.session_state[WKEY]

# -- Header controls row
top_l, top_r = st.columns([6, 1])
with top_l:
    _step_header()
    _progress()
with top_r:
    if st.button("⟲ Reset", key="tw_reset_header"):
        _reset()
        st.rerun()

# Load lists once
envs = api.get_env_list()
algos = api.get_algo_list()

# --------------------------
# Step 1: Select (Env + Algo)
# --------------------------
if state["step"] == 1:
    st.subheader("1) Select Environment & Algorithm")
    left, right = st.columns([2, 1], gap="large")

    with left:
        chosen_env, chosen_algo = selector_action_env_algo(envs, algos, key_prefix="tw")

    with right:
        st.markdown("#### Current selection")
        if chosen_env:
            st.markdown(
                f"- **Environment:** {chosen_env.get('env_name')} "
                f"(`{chosen_env.get('env_id')}`)"
            )
            st.markdown(f"- **Action type:** {chosen_env.get('supported_action')}")
        else:
            st.caption("Pick an action type, then an environment.")

        if chosen_algo:
            st.markdown(
                f"- **Algorithm:** {chosen_algo.get('algo_name')} "
                f"(`{chosen_algo.get('algo_id')}`)"
            )
        else:
            st.caption("Next, choose a compatible algorithm.")

        st.divider()
        # Controls (top-right area)
        c1, c2 = st.columns([1, 1])
        with c2:
            if st.button("Next →", key="tw_next_select", type="primary", disabled=chosen_algo is None):
                state["env"] = chosen_env
                state["algo"] = chosen_algo
                state["step"] = 2
                st.rerun()
        with c1:
            if st.button("Reset", key="tw_reset_select"):
                _reset()
                st.rerun()

    st.stop()

# --------------------------
# Step 2: Configure
# --------------------------
if state["step"] == 2:
    st.subheader("2) Configure Training")
    if not (state["env"] and state["algo"]):
        st.warning("Please choose an environment and algorithm first.")
        state["step"] = 1
        st.rerun()

    algo_id = state["algo"].get("algo_id")
    env_id = state["env"].get("env_id")
    _load_default_cfgs(env_id, algo_id)

    # Compact selection summary top row
    sum_l, sum_r = st.columns([2, 2])
    with sum_l:
        st.markdown(f"**Environment:** {state['env'].get('env_name')} (`{env_id}`)")
    with sum_r:
        st.markdown(f"**Algorithm:** {state['algo'].get('algo_name')} (`{algo_id}`)")

    # Forms in tabs for a cleaner look
    tabs = st.tabs(["General", "Log", "Algo"])
    with tabs[0]:
        gen_cfg_updated, gen_sub = render_config_form(
            "General Config", state["general_cfg"]['config'], key_prefix="gen", schema_meta=state["general_cfg"]['meta']
        )
        if gen_sub:
            state["general_cfg"]['config'] = gen_cfg_updated
            st.success("General config saved.")

    with tabs[1]:
        log_cfg_updated, log_sub = render_config_form(
            "Log Config", state["log_cfg"]['config'], key_prefix="log", schema_meta=state["log_cfg"]['meta']
        )
        if log_sub:
            state["log_cfg"]['config'] = log_cfg_updated
            st.success("Log config saved.")

    with tabs[2]:
        algo_cfg_updated, algo_sub = render_config_form(
            f"Algo Config • {algo_id}", state["algo_cfg"]['config'], key_prefix="algo", schema_meta=state["algo_cfg"]['meta']
        )
        if algo_sub:
            state["algo_cfg"]['config'] = algo_cfg_updated
            st.success("Algo config saved.")

    st.divider()
    nav_l, nav_r = st.columns([1, 1])
    with nav_l:
        if st.button("← Back", key="tw_back_cfg"):
            state["step"] = 1
            st.rerun()
    with nav_r:
        if st.button("Proceed to Start →", key="tw_next_cfg", type="primary"):
            state["step"] = 3
            st.rerun()
    st.stop()
# --------------------------
# Step 3: Review & Start
# --------------------------
if state["step"] == 3:
    st.subheader("3) Review & Start")
    env = state["env"]
    algo = state["algo"]
    general_cfg = state["general_cfg"]['config']
    log_cfg = state["log_cfg"]['config']
    algo_cfg = state["algo_cfg"]['config']

    with st.expander("Summary", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Environment:** {env.get('env_name')} (`{env.get('env_id')}`)")
            st.markdown(f"**Action type:** {env.get('supported_action')}")
        with c2:
            st.markdown(f"**Algorithm:** {algo.get('algo_name')} (`{algo.get('algo_id')}`)")

    payload = {
        "env_id": env.get("env_id"),
        "algo_id": algo.get("algo_id"),
        "general_cfg": general_cfg,
        "log_cfg": log_cfg,
        "algo_cfg": algo_cfg,
    }

    # Buttons aligned right
    spacer, b_edit, b_start = st.columns([4, 1, 1])
    with b_edit:
        if st.button("← Edit", key="tw_edit_final"):
            state["step"] = 2
            st.rerun()
    with b_start:
        start = st.button("🚀 Start", key="tw_start", type="primary")

    if start:
        try:
            resp = api.start_training(payload)
            st.session_state.job_id = resp.get("job_id")
            st.success(f"Job started • status: {resp.get('status')}")
            try:
                st.switch_page("pages/_train_status.py")
            except Exception:
                st.info("No _train_status.py page found; staying here.")
        except Exception as e:
            st.error(f"Error: {e}")
