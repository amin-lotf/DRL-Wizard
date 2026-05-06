import streamlit as st
from drl_wizard.frontend.streamlit_app.settings import BASE_URL


st.set_page_config(page_title="DRL Console", page_icon="🎛️", layout="wide")

st.title("🎛️ DRL Training Console")
st.caption(f"API: {BASE_URL}/docs")

st.markdown(
    """
    - Go to **Train** to start a run.
    - Check **Jobs** to see recent job IDs you’ve launched.
    - Use **Evaluate Model** to load a saved checkpoint and test it.
    """
)

st.page_link("pages/training.py", label="➡️ Train", icon="🏁")
st.page_link("pages/training_list.py", label="➡️ Jobs", icon="📋")
st.page_link("pages/evaluate_model.py", label="➡️ Evaluate Model", icon="🧪")
