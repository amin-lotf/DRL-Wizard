# backend/services/training_service/service.py
from __future__ import annotations

import json
import os
import shutil
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Optional, List, Iterable
from urllib.parse import urlparse

import anyio

from drl_wizard.backend.services.logging.log_manifest import Manifest
from drl_wizard.backend.services.storage.database import JobModel, JobResultsModel
from drl_wizard.backend.services.training_service import jobs as job_state
from drl_wizard.backend.services.training_service.algos import AlgoState
from drl_wizard.backend.services.training_service.environments import EnvironmentState
from drl_wizard.backend.services.training_service.job_results import create_job_result, JobResultState
from drl_wizard.backend.services.training_service.jobs import JobState
from drl_wizard.backend.services.training_service.repository import JobRepository
from drl_wizard.backend.services.training_service.training_worker import train_entry
from drl_wizard.backend.services.utils import pick_device_id
from drl_wizard.common.types import JobStatus, ResultType, AlgoType
from drl_wizard.configs.algo_cfg import BaseAlgoConfig
from drl_wizard.configs.app_cfg import AppConfig
from drl_wizard.configs.general_cfg import GeneralConfig
from drl_wizard.configs.log_cfg import LogConfig

mp_ctx = get_context("spawn")  # IMPORTANT for CUDA safety (avoid fork)
_active_processes: dict[int, "mp_ctx.Process"] = {}
_stop_events: dict[int, any] = {}  # job_id -> mp_ctx.Event

try:
    import zstandard as zstd
except ImportError:
    zstd = None  # compression optional

JobEventHandler = Callable[[JobState], None]


class TrainingService:
    """Service for managing training jobs and their lifecycle.

    This class provides functionality for starting, monitoring, and managing training jobs,
    including job state transitions, result handling, and configuration management.

    Args:
        repo (JobRepository): Repository for persisting job data
        event_handlers (Optional[List[JobEventHandler]]): Handlers for job state change events
    """

    def __init__(self, repo: JobRepository, event_handlers: Optional[List[JobEventHandler]] = None):
        self.repo = repo
        self.event_handlers:List[JobEventHandler] = event_handlers or []

    # --------------------------------------------------------------------- events
    def _emit(self, job: JobState) -> None:
        """Emit job state change event to all registered handlers.

        Args:
            job (JobState): Current job state to emit
        """
        for handler in self.event_handlers:
            try:
                handler(job)
            except Exception:
                # Add logging here
                pass

    # --------------------------------------------------------------------- jobs

    async def start_job(self, env_id: str, algo_id: AlgoType, app_cfg: AppConfig) -> JobState:
        """Start a new training job.

        Creates and starts a new training process for the specified environment and algorithm.

        Args:
            env_id (str): Environment identifier
            algo_id (AlgoType): Algorithm type to use
            app_cfg (AppConfig): Application configuration

        Returns:
            JobState: Initial state of the created job
        """
        job = job_state.create_job(env_id, algo_id)
        job = await self.repo.add_job(job)

        device_id = pick_device_id()
        stop_evt = mp_ctx.Event()
        _stop_events[job.job_id] = stop_evt

        p = mp_ctx.Process(
            target=train_entry,
            args=(job.job_id, app_cfg, device_id, stop_evt),
            daemon=False,
        )
        p.start()
        _active_processes[job.job_id] = p

        self._emit(job)
        return job

    async def mark_running(self, job_id: int) -> JobState:
        """Mark a job as running.

        Args:
            job_id (int): ID of the job to mark as running

        Returns:
            JobState: Updated job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        job_state.mark_running(job)
        await self.repo.update_job(job)
        self._emit(job)
        return job

    async def request_stop(self, job_id: int) -> JobState:
        """Request graceful stop of a running job.

        Attempts graceful shutdown first, then escalates to terminate and kill if needed.

        Args:
            job_id (int): ID of the job to stop

        Returns:
            JobState: Updated job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")

        job_state.mark_stop_requested(job)
        await self.repo.update_job(job)
        self._emit(job)

        # Signal the child process via shared Event
        evt = _stop_events.get(job_id)
        if evt is not None:
            evt.set()

        # Wait a short grace period then escalate; offload blocking joins to a worker thread
        p = _active_processes.get(job_id)

        def _graceful_stop(proc):
            if proc and proc.is_alive():
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=10)
                    if proc.is_alive():
                        try:
                            proc.kill()
                        except Exception:
                            pass

        if p:
            await anyio.to_thread.run_sync(_graceful_stop, p)

        return job

    async def mark_stopped(self, job_id: int) -> JobState:
        """Mark a job as stopped.

        Args:
            job_id (int): ID of the job to mark as stopped

        Returns:
            JobState: Updated job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        job_state.mark_stopped(job)
        await self.repo.update_job(job)
        self._emit(job)
        return job

    async def mark_finished(self, job_id: int) -> JobState:
        """Mark a job as successfully finished.

        Args:
            job_id (int): ID of the job to mark as finished

        Returns:
            JobState: Updated job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        job_state.mark_finished(job)
        await self.repo.update_job(job)
        self._emit(job)
        return job

    async def mark_failed(self, job_id: int, detail: Optional[str] = None) -> JobState:
        """Mark a job as failed.

        Args:
            job_id (int): ID of the job to mark as failed
            detail (Optional[str]): Optional failure details/message

        Returns:
            JobState: Updated job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        job_state.mark_failure(job, detail)
        await self.repo.update_job(job)
        self._emit(job)
        return job

    # --------------------------------------------------------------------- config getters

    def get_algo_config(self, algo_id: AlgoType) -> BaseAlgoConfig:
        """Get algorithm configuration.

        Args:
            algo_id (AlgoType): Algorithm identifier

        Returns:
            BaseAlgoConfig: Algorithm configuration
        """
        # pure in-memory
        return self.repo.get_algo_config(algo_id)

    def get_general_config(self, env_id: str) -> GeneralConfig:
        """Get general configuration for an environment.

        Args:
            env_id (str): Environment identifier

        Returns:
            GeneralConfig: General configuration
        """
        # pure in-memory
        return self.repo.get_general_config(env_id)

    def get_log_confi(self) -> LogConfig:
        """Get logging configuration.

        Returns:
            LogConfig: Logging configuration
        """
        # pure in-memory
        return self.repo.get_log_confi()

    def get_app_config(self, general_cfg: GeneralConfig, log_cfg: LogConfig, algo_cfg: BaseAlgoConfig) -> AppConfig:
        """Create application configuration from components.

        Args:
            general_cfg (GeneralConfig): General configuration
            log_cfg (LogConfig): Logging configuration
            algo_cfg (BaseAlgoConfig): Algorithm configuration

        Returns:
            AppConfig: Combined application configuration
        """
        # pure in-memory
        return AppConfig(**general_cfg.__dict__, log_cfg=log_cfg, algo_cfg=algo_cfg)

    # --------------------------------------------------------------------- catalogs

    def get_all_envs(self) -> List[EnvironmentState]:
        """Get all available environments.

        Returns:
            List[EnvironmentState]: List of environment states
        """
        return self.repo.get_all_envs()

    def get_env(self, env_id: str) -> EnvironmentState:
        """Get environment by ID.

        Args:
            env_id (str): Environment identifier

        Returns:
            EnvironmentState: Environment state

        Raises:
            KeyError: If environment with given ID is not found
        """
        env: EnvironmentState = self.repo.get_env(env_id)
        if not env:
            raise KeyError(f"Env with id {env_id} not found")
        return env

    def get_all_algos(self) -> List[AlgoState]:
        """Get all available algorithms.

        Returns:
            List[AlgoState]: List of algorithm states
        """
        return self.repo.get_all_algos()

    def get_algo(self, algo_id: AlgoType) -> AlgoState:
        """Get algorithm by ID.

        Args:
            algo_id (AlgoType): Algorithm identifier

        Returns:
            AlgoState: Algorithm state

        Raises:
            KeyError: If algorithm with given ID is not found
        """
        algo: AlgoState = self.repo.get_algo(algo_id)
        if not algo:
            raise KeyError(f"Algo with id {algo_id.value} not found")
        return algo

    def get_supported_algos(self, env_id: str) -> List[AlgoState]:
        """Get algorithms supported by a specific environment.

        Args:
            env_id (str): Environment identifier

        Returns:
            List[AlgoState]: List of supported algorithm states

        Raises:
            KeyError: If environment with given ID is not found
        """
        env = self.get_env(env_id)
        return [algo for algo in self.get_all_algos() if env.supported_action in algo.action_type]

    # --------------------------------------------------------------------- job queries

    async def get_job(self, job_id: int) -> JobState:
        """Get job by ID.

        Args:
            job_id (int): Job identifier

        Returns:
            JobState: Job state

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        return job

    async def delete_job(self, job_id: int) -> None:
        """Delete a job and its associated files.

        Removes the job from the repository and deletes any associated result files.

        Args:
            job_id (int): ID of the job to delete
        """
        # await self.request_stop(job_id)

        # Remove run dir (blocking I/O) if present
        try:
            manifest = await self.get_results_manifest(job_id, ResultType.EVALUATE)
        except Exception:
            manifest = None

        if manifest:
            run_dir = Path(manifest.path).parent

            def _rm_tree(p: Path):
                if p.exists():
                    shutil.rmtree(p)

            await anyio.to_thread.run_sync(_rm_tree, run_dir)

        await self.repo.delete_job(job_id)

    async def build_archive(self, job_id: int) -> Path:
        """Create a ZIP archive of job results.

        Args:
            job_id (int): Job identifier

        Returns:
            Path: Path to the created archive file

        Raises:
            KeyError: If job has no data
            FileNotFoundError: If results directory doesn't exist
        """
        manifest = await self.get_results_manifest(job_id, ResultType.EVALUATE)
        if not manifest:
            raise KeyError(f"Job {job_id} has no data")

        manifest_path = Path(manifest.path) if not isinstance(manifest.path, Path) else manifest.path
        run_dir = manifest_path.parent
        if not run_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {run_dir}")

        # Blocking archiving -> thread
        def _archive(dir_path: Path) -> str:
            tmpdir = tempfile.mkdtemp(prefix=f"job-{job_id}-")
            base_name = os.path.join(tmpdir, f"{job_id}-data")
            return shutil.make_archive(base_name=base_name, format="zip", root_dir=dir_path)

        archive_path_str = await anyio.to_thread.run_sync(_archive, run_dir)
        return Path(archive_path_str)

    async def get_all_jobs(self) -> List[JobState]:
        """Get all jobs.

        Returns:
            List[JobState]: List of all job states
        """
        return await self.repo.get_all_jobs()

    async def get_job_status(self, job_id: int) -> JobStatus:
        """Get status of a job.

        Args:
            job_id (int): Job identifier

        Returns:
            JobStatus: Current status of the job

        Raises:
            KeyError: If job with given ID is not found
        """
        job = await self.repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")
        return job.status

    async def add_job_results(
        self, job_id: int, result_type: ResultType, segment_steps: int, manifest_path: str
    ) -> JobResultState:
        """Add results for a job.

        Args:
            job_id (int): Job identifier
            result_type (ResultType): Type of results (TRAIN/EVALUATE)
            segment_steps (int): Number of steps in the segment
            manifest_path (str): Path to results manifest file

        Returns:
            JobResultState: Created job result state
        """
        job_results_state = create_job_result(job_id, result_type, segment_steps, manifest_path)
        return await self.repo.add_job_results(job_results_state)

    async def update_job_results(self, job_id: int, result_type: ResultType, latest_step: int) -> JobResultState:
        """Update results for a job.

        Args:
            job_id (int): Job identifier
            result_type (ResultType): Type of results (TRAIN/EVALUATE)
            latest_step (int): Latest step number

        Returns:
            JobResultState: Updated job result state
        """
        job_results_state = JobResultState(job_id=job_id, result_type=result_type, latest_step=latest_step)
        return await self.repo.update_results(job_results_state)

    async def get_results_manifest(self, job_id: int, result_type: ResultType) -> Manifest:
        """Get results manifest for a job.

        Args:
            job_id (int): Job identifier
            result_type (ResultType): Type of results to get

        Returns:
            Manifest: Results manifest

        Raises:
            KeyError: If job or requested results are not found
            ValueError: If result type is invalid
        """
        job: JobModel = await self.repo.get_job_with_results(job_id)
        if not job:
            raise KeyError(f"Job with id {job_id} not found")

        if result_type == ResultType.EVALUATE:
            job_results: JobResultsModel = job.evaluate_results
            if not job_results:
                raise KeyError(f"Job with id {job_id} has no evaluate results")
        elif result_type == ResultType.TRAIN:
            job_results: JobResultsModel = job.train_results
            if not job_results:
                raise KeyError(f"Job with id {job_id} has no train results")
        else:
            raise ValueError(f"Invalid result type {result_type}")

        # Manifest.load is blocking file I/O; offload
        def _load_manifest(uri: str) -> Manifest:
            manifest_uri = urlparse(uri)
            return Manifest.load(Path(manifest_uri.path))

        manifest = await anyio.to_thread.run_sync(_load_manifest, job_results.manifest_uri)
        return manifest

    # --------------------------------------------------------------------- streaming (sync generators by design)

    def stream_results(self, job_id: int, result_type: ResultType) -> Iterable[bytes]:
        """Stream results as NDJSON lines.

        Provides a synchronous generator for reading result data. The output should be
        wrapped with starlette.concurrency.iterate_in_threadpool(...) in routes.

        Args:
            job_id (int): Job identifier
            result_type (ResultType): Type of results to stream

        Returns:
            Iterable[bytes]: Generator yielding result data chunks
        """
        manifest: Manifest = anyio.run(lambda: self.get_results_manifest(job_id, result_type))  # safe shortcut here
        segs = [s for s in manifest.segments[result_type]]
        for s in segs:
            path = Path(urlparse(str(manifest.log_path)).path) / s.path
            if path.suffix == ".zst" and zstd is not None:
                dctx = zstd.ZstdDecompressor()
                with path.open("rb") as f, dctx.stream_reader(f) as r:
                    for line in r.read().splitlines(True):  # preserves newlines
                        yield line
            else:
                with path.open("rb") as f:
                    for chunk in f:
                        yield chunk

    def extract_results(self, manifest: Manifest, result_type: ResultType) -> Iterable[bytes]:
        """
        Synchronous generator used by multiplexed env streaming.
        """
        segs = [s for s in manifest.segments[result_type]]
        for s in segs:
            path = Path(urlparse(str(manifest.log_path)).path) / s.path
            if path.suffix == ".zst" and zstd is not None:
                dctx = zstd.ZstdDecompressor()
                with path.open("rb") as f, dctx.stream_reader(f) as r:
                    for line in r.read().splitlines(True):
                        yield line
            else:
                with path.open("rb") as f:
                    for chunk in f:
                        yield chunk

    async def get_env_jobs(self, env_id: str) -> List[JobState]:
        return await self.repo.get_env_jobs(env_id)

    def list_env_result_manifests(self, env_id: str, result_type: ResultType) -> list[JobResultState]:
        # DB call is async in repo; expose sync or async?
        # For route usage, we only read and then stream. Keep sync facade that routes will call via run_in_threadpool if needed.
        # But since repo method is async, provide an async wrapper as well:
        raise NotImplementedError("Use async_list_env_result_manifests instead")

    async def async_list_env_result_manifests(self, env_id: str, result_type: ResultType) -> list[JobResultState]:
        return await self.repo.get_env_job_results(env_id, result_type)

    def stream_env_results_multiplexed(self, env_id: str, result_type: ResultType) -> Iterable[bytes]:
        """
        Synchronous generator that multiplexes multiple NDJSON streams:
        Each line => {"job_id": <int>, "data": <original_json_object>}
        Routes must wrap with iterate_in_threadpool(...).
        """
        # 1) pick manifests for jobs in this env (blocking call to async repo is not allowed here).
        rows = anyio.run(lambda: self.repo.get_env_job_results(env_id, result_type))

        # 2) open iterators
        iters: list[tuple[int, Iterable[bytes]]] = []
        for r in rows:
            manifest = Manifest.load(Path(urlparse(r.manifest_uri).path))
            iters.append((r.job_id, self.extract_results(manifest, result_type)))

        # 3) simple round-robin interleave
        active = [(job_id, iter(stream)) for job_id, stream in iters]
        while active:
            next_active = []
            for job_id, it in active:
                try:
                    raw = next(it)
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except Exception:
                        obj = {"__opaque": raw.decode("utf-8", errors="replace")}
                    wrapped = json.dumps({"job_id": job_id, "data": obj}, ensure_ascii=False).encode("utf-8") + b"\n"
                    yield wrapped
                    next_active.append((job_id, it))
                except StopIteration:
                    pass
            active = next_active
