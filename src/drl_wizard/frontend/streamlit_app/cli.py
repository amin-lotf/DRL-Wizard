import os
import sys
import pathlib
import runpy

def main():
    # Allow running a Streamlit app that lives inside the package.
    # You can keep a top-level 'home.py' in ui/ or a pages/ subfolder.
    pkg_dir = pathlib.Path(__file__).parent
    app_path = pkg_dir / "home.py"  # or "app.py"
    if not app_path.exists():
        # minimal inline app if you haven't created one yet
        (pkg_dir / "home.py").write_text(
            "import streamlit as st\nst.set_page_config(layout='wide')\n"
            "st.title('DRL-API UI')\nst.write('Hello from packaged Streamlit!')\n"
        )
        app_path = pkg_dir / "home.py"

    # Delegate to Streamlit
    args = ["streamlit", "run", str(app_path)]
    os.execvp(args[0], args)
