# backend/services/training_service/training_worker.py (or wherever run_training lives)
from __future__ import annotations

import threading
import traceback

import anyio  # anyio.run gives you a clean asyncio loop in this process

from drl_wizard.backend.services.logging.json_logger import SegmentedJsonlLogger
from drl_wizard.backend.services.storage.database import SessionLocal  # <-- async_sessionmaker
from drl_wizard.backend.services.training_service.repository import JobRepository
from drl_wizard.backend.services.training_service.service import TrainingService
from drl_wizard.common.types import JobStatus, AlgoType
from drl_wizard.configs.app_cfg import AppConfig


def run_training(job_id: int, app_cfg: AppConfig, stop_event=None) -> None:
    """
    Entry point executed in a spawned process.
    Spins up an event loop and uses AsyncSession/async service methods.
    Runners remain synchronous.
    """
    if stop_event is None:
        stop_event = threading.Event()

    async def _main() -> None:
        # Async DB session for the lifetime of this worker
        async with SessionLocal() as db:
            svc = TrainingService(JobRepository(db))
            logger = SegmentedJsonlLogger(svc=svc, app_cfg=app_cfg, save_dir=app_cfg.run_dir, job_id=job_id)

            try:
                # --- mark running
                await svc.mark_running(job_id)
                await logger.register_tracks()
                # --- pick runner (sync)
                if app_cfg.algo_cfg.algo_id == AlgoType.PPO:
                    from drl_wizard.algorithms.runners.gym_runners.ppo_runner import PPORunner
                    runner = PPORunner(app_cfg, logger=logger)
                elif app_cfg.algo_cfg.algo_id == AlgoType.SAC:
                    from drl_wizard.algorithms.runners.gym_runners.sac_runner import SACRunner
                    runner = SACRunner(app_cfg, logger=logger)
                elif app_cfg.algo_cfg.algo_id == AlgoType.DQN:
                    from drl_wizard.algorithms.runners.gym_runners.dqn_runner import DQNRunner
                    runner = DQNRunner(app_cfg, logger=logger)
                elif app_cfg.algo_cfg.algo_id == AlgoType.A2C:
                    from drl_wizard.algorithms.runners.gym_runners.a2c_runner import A2CRunner
                    runner = A2CRunner(app_cfg, logger=logger)
                elif app_cfg.algo_cfg.algo_id == AlgoType.TRPO:
                    from drl_wizard.algorithms.runners.gym_runners.trpo_runner import TRPORunner
                    runner = TRPORunner(app_cfg, logger=logger)
                else:
                    raise ValueError(f"Unsupported algo {app_cfg.algo_cfg.algo_id}")

                # --- run training (sync loop that periodically checks stop_event)
                runner.run(stop_event)
                # --- finalize status
                if stop_event.is_set():
                    await svc.mark_stopped(job_id)
                else:
                    await svc.mark_finished(job_id)

            except Exception as e:
                detail = f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"
                print(detail)
                await svc.mark_failed(job_id, detail)
            finally:
                status = await svc.get_job_status(job_id)
                if status not in (JobStatus.FINISHED, JobStatus.FAILED, JobStatus.STOPPED):
                    await svc.mark_stopped(job_id)
                # async with SessionLocal() will close the session automatically

    # Start an event loop in this process and run the async main
    anyio.run(_main)