# frontend/streamlit_app/settings.py
import os

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")  # FastAPI root
POLL_MS = int(os.getenv("POLL_MS", "2000"))                    # Job detail refresh
DEFAULT_ALGOS = ["ppo"]                  # optional; or fetch from API later
DEFAULT_ENVS = ["CartPole-v1", "Ant-v5"]                       # optional; or fetch from API later
