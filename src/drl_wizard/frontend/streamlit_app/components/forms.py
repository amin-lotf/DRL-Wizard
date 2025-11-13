from typing import Dict, Any, List, Union, Optional, Mapping
import streamlit as st
from pydantic import BaseModel

from drl_wizard.common.types import ModelLike
from drl_wizard.common.utils import get_field


def env_algo_selection_form(envs: List[ModelLike],algos: List[ModelLike], key_prefix: str) -> Optional[dict[str, Any]]:
    all_act_types = set([get_field(env, 'supported_action') for env in envs])
    chosen_type = st.radio(
        label="Action type",
        options=all_act_types,
        index=None,
        key=f'{key_prefix}_action_type',
        horizontal=True,
    )
    chosen_envs = [env for env in envs if get_field(env, 'supported_action') == chosen_type]
    chosen_env = st.selectbox(
        label="Environment",
        options=chosen_envs,
        format_func=lambda env: f"{get_field(env, 'env_name')} ({get_field(env, 'env_id')})",
        index=None,
        disabled=len(chosen_envs) == 0,
        key=f'{key_prefix}_env',
    )
    if chosen_env is not None:
        chosen_algos = [algo for algo in algos if get_field(chosen_env, 'supported_action') in get_field(algo, 'action_type')]
    else:
        chosen_algos = []
    chosen_algo = st.selectbox(
        label="Algorithm",
        options=chosen_algos,
        format_func=lambda algo: f"{get_field(algo, 'algo_name')} ({get_field(algo, 'algo_id')})",
        index=None,
        disabled=len(chosen_algos) == 0,
        key=f'{key_prefix}_algo',
    )
    with st.form(key=f'{key_prefix}_form', clear_on_submit=False):
        submitted = st.form_submit_button("Submit",disabled=chosen_algo is None)
        if submitted:
            training_payload={"env_id":get_field(chosen_env,"env_id"),"algo_id":get_field(chosen_algo,"algo_id")}
            return training_payload
        return None

