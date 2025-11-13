from pathlib import Path

import streamlit as st

from drl_wizard.common.types import ActionType, JobActions
from drl_wizard.common.utils import get_field
from drl_wizard.frontend.streamlit_app.components.job_card import job_general_card
from drl_wizard.frontend.streamlit_app.extras import handle_ui_action
from drl_wizard.frontend.streamlit_app.services.api import Api
from drl_wizard.frontend.streamlit_app.settings import BASE_URL

api = Api(BASE_URL)



if "ui_actions" not in st.session_state:
    st.session_state.ui_actions = []
if "__zip_paths__" not in st.session_state:
    st.session_state["__zip_paths__"] = {}  # { job_id: str(path) }
st.title("Training List!")

train_list = api.get_job_list()
download_slot = {}  # will hold the placeholder returned by the card
for job in train_list:
    job_id=get_field(job,'job_id')
    download_slot[job_id]=job_general_card(job, handle_ui_action, job_id)
need_rerun = False
while st.session_state.ui_actions:
    action = st.session_state.ui_actions.pop(0)
    try:
        match action['type']:
            case JobActions.STOP:
                ok = api.stop_job(action['job_id'])
                if ok:
                    st.toast(f"Job {action['job_id']} stopped successfully")
                else:
                    st.toast(f"Error stopping job {action['job_id']}")
                need_rerun = True
            case JobActions.DETAILS:
                st.session_state.job_id = action['job_id']
                st.switch_page("pages/_train_status.py")

            case JobActions.DELETE:
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

            case JobActions.DOWNLOAD:
                # 1) fetch to disk
                # need_rerun = True
                zip_path = api.download_job_zip(action["job_id"])
                st.session_state["__zip_paths__"][action["job_id"]] = str(zip_path)
                st.toast("Download prepared.")
                # 2) render the button *immediately* into the card's placeholder if available
                if download_slot[action["job_id"]] is not None:
                    with download_slot[action["job_id"]]:
                        p = Path(zip_path)
                        with open(p, "rb") as f:
                            st.download_button(
                                "Download ZIP",
                                data=f.read(),
                                file_name=p.name,
                                mime="application/zip",
                                key=f"dl_{action['job_id']}_{p.stat().st_mtime_ns}"
                            )

            case _:
                raise ValueError(f"Invalid action type {action['type']}")
    except RuntimeError as e:
        st.error(str(e))
        need_rerun = False
        break

if need_rerun:
    st.rerun()
