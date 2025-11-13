# training_worker.py
import os
import signal
import threading
import traceback




def train_entry(job_id: int, app_cfg, device_id: int | None, stop_evt):
    # mask GPU first
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import torch
    from drl_wizard.backend.services.training_service.job_runner import run_training
    # Make SIGTERM/SIGINT set the same event for graceful shutdown
    def _sig_handler(signum, frame):
        stop_evt.set()
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT,  _sig_handler)
    except Exception:
        # On some platforms (e.g., Windows spawn), signals behave differently; ignore if unsupported
        pass

    try:
        # pass the cross-process Event to your training
        run_training(job_id, app_cfg, stop_event=stop_evt)
    except Exception as e:
        print(f"train_entry crash for job {job_id}: {e}\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
