# in frontend/streamlit_app/components/plot_card.py
import altair as alt
from typing import List, Optional
import pandas as pd
import streamlit as st

def plot_card(df: pd.DataFrame, metric: str, *, height: int = 180, smooth: int = 1) -> None:
    """
    Single-job wide DF: columns are metrics, index is step (classic).
    """
    if "step" not in df.columns or metric not in df.columns:
        st.info(f"No data for **{metric}**")
        return

    d = df[["step", metric]].dropna()
    if d.empty:
        st.info(f"No data for **{metric}**")
        return

    if smooth and smooth > 1:
        d = d.copy()
        d[metric] = d[metric].rolling(window=smooth, min_periods=1).mean()

    st.markdown(f"**{metric}**")
    st.line_chart(data=d.set_index("step"), height=height, width='stretch')


def render_plot_grid(
    df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    per_row: int = 3,
    height: int = 180,
    smooth: int = 1
) -> None:
    if df is None or df.empty:
        st.info("No results yet.")
        return

    if metrics is None:
        metrics = [c for c in df.columns if c != "step"]

    if not metrics:
        st.info("No plottable metrics found.")
        return

    for i in range(0, len(metrics), per_row):
        row_metrics = metrics[i : i + per_row]
        cols = st.columns(len(row_metrics))
        for col, m in zip(cols, row_metrics):
            with col:
                plot_card(df, m, height=height, smooth=smooth)


# ------- NEW: multi-job plotting from tidy DF (job_id, step, metric, value) -------

def plot_card_multi_jobs(df_long: pd.DataFrame, metric: str, *, height: int = 180, smooth: int = 1) -> None:
    """
    df_long columns: job_id, step, metric, value.
    Renders one chart with multiple lines (one per job_id) for a single metric.
    """
    needed = {"job_id", "step", "metric", "value"}
    if df_long is None or df_long.empty or not needed.issubset(df_long.columns):
        st.info(f"No data for **{metric}**")
        return

    d = df_long[df_long["metric"] == metric][["job_id", "step", "value"]].dropna()
    if d.empty:
        st.info(f"No data for **{metric}**")
        return

    # pivot to wide: index=step, columns=job_id, values=value
    wide = d.pivot(index="step", columns="job_id", values="value").sort_index()

    if smooth and smooth > 1:
        wide = wide.rolling(window=smooth, min_periods=1).mean()

    st.markdown(f"**{metric}**")
    st.line_chart(data=wide, height=height, width='stretch')


def _prepare_long_for_plot(
    df_long: pd.DataFrame,
    metric: str,
    smooth: int = 1,
) -> pd.DataFrame:
    """
    df_long columns: job_id, step, metric, value
    Returns long again but with aligned steps, step 0 = 0 if missing, and interpolated values.
    """
    sub = df_long[df_long["metric"] == metric].copy()
    if sub.empty:
        return sub

    # ensure correct types
    sub["job_id"] = sub["job_id"].astype(str)
    sub["step"] = sub["step"].astype(int)
    sub = sub.sort_values(["job_id", "step"])

    # Build per-metric step grid = union of existing steps + {0}
    step_grid = sorted(set(sub["step"].unique()).union({0}))

    # Wide: index=step, columns=job_id, values=value
    wide = sub.pivot_table(index="step", columns="job_id", values="value", aggfunc="last")

    # Reindex to the common grid
    wide = wide.reindex(step_grid)

    # Rule: step 0 missing -> fill with 0 for that job only
    if 0 in wide.index:
        wide.loc[0] = wide.loc[0].fillna(0.0)

    # Interpolate linearly per job (within range), then forward-fill to extend plateaus
    wide = wide.interpolate(method="linear", limit_direction="forward", axis=0).ffill()

    # Optional smoothing (rolling mean over steps)
    if smooth and smooth > 1:
        wide = wide.rolling(window=smooth, min_periods=1).mean()

    # Back to long
    long_interp = wide.reset_index().melt(
        id_vars="step", var_name="job_id", value_name="value"
    )
    long_interp["metric"] = metric
    return long_interp.dropna(subset=["value"])


def render_multi_plot_grid(
    df_long: pd.DataFrame,
    metrics: List[str],
    per_row: int = 3,
    height: int = 400,
    smooth: int = 1,
):
    """
    df_long must be tidy: (job_id, step, metric, value)
    Renders a grid of Altair line charts with one line per job_id for each metric.
    """
    if df_long is None or df_long.empty:
        st.info("No data to plot.")
        return

    # Keep only requested metrics that exist
    available = set(df_long["metric"].dropna().unique().tolist())
    metrics = [m for m in metrics if m in available]
    if not metrics:
        st.info("Selected metrics are not available.")
        return

    # Prepare all metrics (alignment + interpolation)
    parts = []
    for m in metrics:
        parts.append(_prepare_long_for_plot(df_long, m, smooth=smooth))
    ready = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if ready.empty:
        st.info("No data to plot after interpolation.")
        return

    # Build the charts in a grid
    rows = (len(metrics) + per_row - 1) // per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(per_row)
        for c in cols:
            if idx >= len(metrics):
                break
            metric = metrics[idx]
            plot_df = ready[ready["metric"] == metric]
            if plot_df.empty:
                c.info(f"No data for {metric}")
                idx += 1
                continue

            # Altair line chart (one line per job_id)
            chart = (
                alt.Chart(plot_df)
                .mark_line()
                .encode(
                    x=alt.X("step:Q", title="Step"),
                    y=alt.Y("value:Q", title=metric),
                    color=alt.Color("job_id:N", title="Job ID"),
                    tooltip=[
                        alt.Tooltip("job_id:N", title="Job"),
                        alt.Tooltip("step:Q", title="Step"),
                        alt.Tooltip("value:Q", title=metric, format=".4f"),
                    ],
                )
                .properties(height=height, title=metric,width="container",)
                .configure_axis(labelFontSize=11, titleFontSize=12)
                .configure_title(fontSize=13)
                .interactive()  # zoom/pan in Streamlit
            )
            c.altair_chart(chart, width='stretch')
            idx += 1
