import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")
import sys
import pathlib
import subprocess

def main():
    here = pathlib.Path(__file__).parent
    home_py = here / "frontend" / "streamlit_app" / "home.py"

    api_cmd = [
        sys.executable,
        "-m", "uvicorn",
        "drl_wizard.backend.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        # "--log-level", "warning",  # optional: less noisy logs
    ]

    ui_cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(home_py),
    ]

    print("[drl-wizard] Starting API on http://localhost:8000")
    api_proc = subprocess.Popen(api_cmd)

    print("[drl-wizard] Starting UI on http://localhost:8501")
    ui_proc = subprocess.Popen(ui_cmd)

    try:
        # Wait until UI exits (or Ctrl+C in this terminal)
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n[drl-wizard] Ctrl+C received, shutting down...")
    finally:
        # Ask both processes to terminate (SIGTERM)
        for proc in (ui_proc, api_proc):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()  # softer than SIGINT
                except Exception:
                    pass

        # Give them a few seconds to exit cleanly, then force kill if needed
        for proc in (ui_proc, api_proc):
            if proc and proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
