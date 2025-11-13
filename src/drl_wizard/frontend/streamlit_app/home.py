import streamlit as st
from drl_wizard.frontend.streamlit_app.settings import BASE_URL


st.set_page_config(page_title="DRL Console", page_icon="🎛️", layout="wide")

st.title("🎛️ DRL Training Console")
st.caption(f"API: {BASE_URL}/docs")

st.markdown(
    """
    - Go to **Train** to start a run.
    - Check **Jobs** to see recent job IDs you’ve launched.
    """
)

st.page_link("pages/training.py", label="➡️ Train", icon="🏁")
st.page_link("pages/training_list.py", label="➡️ Jobs", icon="📋")
