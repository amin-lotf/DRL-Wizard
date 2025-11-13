# frontend/streamlit_app/components/dynamic_forms.py
from typing import Any, Dict, Iterable, Optional, Tuple
from pathlib import Path
import json
import streamlit as st

# ---------- Small helpers ----------

def _is_int_like(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

def _is_float_like(x: Any) -> bool:
    # avoid true ints and bools
    return isinstance(x, float)

def _is_path_like(x: Any) -> bool:
    return isinstance(x, (Path,)) or (isinstance(x, str) and ("/" in x or "\\" in x))

def _coerce_path(v: Any) -> str:
    if isinstance(v, Path):
        return str(v)
    return str(v)

def _json_text_input(label: str, value: Any, key: str) -> Any:
    # pretty JSON editor; falls back to original value if parse fails
    text = st.text_area(label, json.dumps(value, indent=2), key=key, height=160)
    try:
        return json.loads(text) if text.strip() else value
    except Exception:
        st.caption("⚠️ Invalid JSON, keeping previous value.")
        return value

def _choices_from_literal_or_enum(meta: Optional[Dict[str, Any]]) -> Optional[Iterable[Any]]:
    # If your API returns JSON schema-ish hints like {"enum": ["cpu","cuda","auto"]}, we use them.
    if not meta:
        return None
    enum = meta.get("enum")
    if isinstance(enum, list) and enum:
        return enum
    return None

def _guess_meta_for_key(schema_meta: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    # schema_meta can be a flat dict of {field_name: {type, enum, title, description, ...}}
    if not schema_meta:
        return None
    return schema_meta.get(key)

def _field_help(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not meta:
        return None
    desc = meta.get("description")
    rng = []
    if "minimum" in meta: rng.append(f"min={meta['minimum']}")
    if "maximum" in meta: rng.append(f"max={meta['maximum']}")
    if rng: desc = (desc + " " if desc else "") + f"({', '.join(rng)})"
    return desc

# ---------- Core: dynamic config form ----------

def render_config_form(
    title: str,
    config: Dict[str, Any],
    key_prefix: str,
    schema_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Render a dynamic form for any dict-like config.
    - We infer widget types from the *current values*.
    - If schema_meta provides hints (enum choices, min/max), we use them.
    Returns (updated_config, submitted)
    """
    # make a shallow copy we’ll mutate

    current_config = dict(config)

    with st.form(key=f"{key_prefix}__form", clear_on_submit=False):
        st.subheader(title)

        cols = st.columns(2, gap="large")  # simple two-column layout
        left, right = cols

        # stable iteration order
        for i, (k, v) in enumerate(sorted(current_config.items(), key=lambda kv: kv[0])):
            meta = _guess_meta_for_key(schema_meta, k)
            help_text = _field_help(meta)
            choices = _choices_from_literal_or_enum(meta)
            label=meta.get('label',k)
            key = f"{key_prefix}__{k}"

            container = left if i % 2 == 0 else right

            # Branch by type / hints
            with container:
                if choices is not None:
                    # Enum/Literal-like
                    idx = None
                    if v in choices:
                        idx = list(choices).index(v)
                    sel = st.selectbox(label, options=list(choices), index=idx, key=key, help=help_text)
                    current_config[k] = sel

                elif isinstance(v, bool):
                    current_config[k] = st.checkbox(label, value=v, key=key, help=help_text)

                elif _is_int_like(v):
                    minv = meta.get("minimum") if meta else None
                    maxv = meta.get("maximum") if meta else None
                    step = 1
                    current_config[k] = st.number_input(
                        label, value=int(v), step=step,
                        min_value=int(minv) if isinstance(minv, int) else None,
                        max_value=int(maxv) if isinstance(maxv, int) else None,
                        key=key, help=help_text
                    )

                elif _is_float_like(v):
                    minv = meta.get("minimum") if meta else None
                    maxv = meta.get("maximum") if meta else None
                    step = 0.001
                    current_config[k] = st.number_input(
                        label, value=float(v), step=step,
                        min_value=float(minv) if isinstance(minv, (int, float)) else None,
                        max_value=float(maxv) if isinstance(maxv, (int, float)) else None,
                        key=key, help=help_text
                    )

                elif isinstance(v, (list, dict)):
                    # JSON-editor style for collections
                    current_config[k] = _json_text_input(label, v, key=key)

                elif _is_path_like(v):
                    # Treat paths as strings; show a text_input
                    sv = _coerce_path(v)
                    current_config[k] = st.text_input(label, sv, key=key, help=help_text)

                else:
                    # default: string
                    current_config[k] = st.text_input(k, str(v) if v is not None else "", key=key, help=help_text)

        # Submit is per-form (per tab in the wizard)
        submitted = st.form_submit_button("Save")
    return current_config, submitted


# ---------- “Wizard” selectors (Action → Env → Algo) ----------

def selector_action_env_algo(
    envs: Iterable[Dict[str, Any]],
    algos: Iterable[Dict[str, Any]],
    key_prefix: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Radio for action type, then env select (filtered), then algo select (filtered by env action).
    Returns (chosen_env, chosen_algo)
    """
    all_act_types = sorted({e.get("supported_action") for e in envs})
    chosen_type = st.radio(
        "Action type",
        options=all_act_types,
        index=None,
        key=f"{key_prefix}__acttype",
        horizontal=True,
    )

    # Filter envs
    env_options = [e for e in envs if e.get("supported_action") == chosen_type] if chosen_type else []
    chosen_env = st.selectbox(
        "Environment",
        options=env_options,
        format_func=lambda e: f"{e.get('env_name')} ({e.get('env_id')})",
        index=None,
        disabled=len(env_options) == 0,
        key=f"{key_prefix}__env",
    )

    # Filter algos by env's action type
    if chosen_env:
        ea = chosen_env.get("supported_action")
        algo_options = [a for a in algos if ea in (a.get("action_type") or [])]
    else:
        algo_options = []

    chosen_algo = st.selectbox(
        "Algorithm",
        options=algo_options,
        format_func=lambda a: f"{a.get('algo_name')}",
        index=None,
        disabled=len(algo_options) == 0,
        key=f"{key_prefix}__algo",
    )

    return chosen_env, chosen_algo
