import streamlit as st

from drl_wizard.backend.services.training_service.evaluation_service import (
    discover_runs_for_environment,
    evaluate_saved_run,
    load_saved_run,
)
from drl_wizard.frontend.streamlit_app.components.dynamic_forms import selector_action_env
from drl_wizard.frontend.streamlit_app.services.api import Api
from drl_wizard.frontend.streamlit_app.settings import BASE_URL

try:
    st.set_page_config(layout="wide")
except Exception:
    pass


api = Api(BASE_URL)

st.title("Evaluate Model")
st.caption("Load a saved training run, inspect the saved configs, and run evaluation on a fresh environment instance.")

envs = api.get_env_list()
selected_env = selector_action_env(envs, key_prefix="eval_model")

if not selected_env:
    st.info("Select an action type and environment to load compatible runs.")
    st.stop()

selected_env_id = selected_env.get("env_id")
st.markdown(
    f"**Environment:** {selected_env.get('env_name')} (`{selected_env_id}`)  \n"
    f"**Action type:** {selected_env.get('supported_action')}"
)

discovery = discover_runs_for_environment(selected_env_id)
for warning in discovery.warnings:
    st.warning(warning)

if not discovery.runs:
    st.info("No saved runs match the selected environment.")
    st.stop()

selected_run = st.selectbox(
    "Saved run",
    options=discovery.runs,
    format_func=lambda run: (
        f"Run {run.run_id} • {run.algo_id.value} • {run.checkpoint_label} checkpoint"
    ),
)

if not selected_run:
    st.stop()

try:
    run_details = load_saved_run(selected_run.run_dir)
except Exception as exc:
    st.error(f"Failed to load run data: {exc}")
    st.stop()

episodes_default = int(run_details.app_config.eval_episodes or 5)
episodes = st.number_input(
    "Evaluation episodes",
    min_value=1,
    value=episodes_default,
    step=1,
    help="This is separate from the saved training config.",
)
render_video = st.checkbox(
    "Render evaluation video",
    value=False,
    help="Render the first evaluation episode and display it after evaluation.",
)
if render_video:
    st.info("Rendering may be slower.")

meta_left, meta_right = st.columns(2)
with meta_left:
    st.markdown(f"**Algorithm:** `{run_details.summary.algo_id.value}`")
    st.markdown(f"**Run directory:** `{run_details.summary.run_dir}`")
with meta_right:
    st.markdown(f"**Checkpoint selection:** `{run_details.summary.checkpoint_label}`")
    st.markdown(f"**Checkpoint path:** `{run_details.summary.checkpoint_path}`")

st.caption("Loaded training configs are read-only on this page.")

config_tabs = st.tabs(["Environment Config", "Agent Config"])
with config_tabs[0]:
    st.json(run_details.env_config, expanded=True)
with config_tabs[1]:
    st.json(run_details.algo_config, expanded=True)

if st.button("Start Evaluation", type="primary"):
    try:
        with st.spinner("Running evaluation..."):
            result = evaluate_saved_run(
                run_dir=run_details.summary.run_dir,
                env_id=selected_env_id,
                episodes=int(episodes),
                render=render_video,
            )
        result_left, result_right = st.columns(2)
        with result_left:
            st.metric("Average Step Reward", f"{result.average_step_reward:.4f}")
        with result_right:
            st.metric("Average Episode Reward", f"{result.average_episode_reward:.4f}")
        if render_video:
            st.markdown("**Rendered sample episode**")
            if result.rendered_video_bytes:
                st.video(
                    result.rendered_video_bytes,
                    format=result.rendered_video_mime_type or "video/mp4",
                )
                if result.render_warning:
                    st.warning(result.render_warning)
            else:
                reason = result.render_warning or "No video was generated."
                st.info(f"Rendered sample episode unavailable: {reason}")
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
