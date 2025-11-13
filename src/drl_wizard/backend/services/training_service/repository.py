# backend/services/training_service/repository.py
from __future__ import annotations
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from drl_wizard.algorithms.utils import discover_all_envs, discover_all_algos
from drl_wizard.backend.services.mappers import to_job_state, to_result_state
from drl_wizard.backend.services.storage.database import JobModel, JobResultsModel
from drl_wizard.backend.services.training_service.algos import AlgoState
from drl_wizard.backend.services.training_service.environments import EnvironmentState
from drl_wizard.backend.services.training_service.job_results import JobResultState
from drl_wizard.backend.services.training_service.jobs import JobState
from drl_wizard.common.types import AlgoType, ResultType
from drl_wizard.configs.algo_cfg import BaseAlgoConfig
from drl_wizard.configs.extras import ALGO_REGISTRY
from drl_wizard.configs.general_cfg import GeneralConfig
from drl_wizard.configs.log_cfg import LogConfig


class JobRepository:
    """Repository for managing training jobs, environments, algorithms and their results.

    This class provides methods for storing and retrieving training jobs, their results,
    available algorithms and environments. It uses both database storage for persistent
    data and in-memory caching for frequently accessed information.
    """

    def __init__(self, db: AsyncSession):
        """Initialize repository with database session.

        Args:
            db: AsyncSession instance for database operations
        """
        self._db = db
        self.buffered_envs: List[EnvironmentState] = []

    # -------------------- Environment/Algo discovery (sync, cached) --------------------

    @property
    def _environments(self) -> List[EnvironmentState]:
        """Get list of available environments, using cache if available.

        Returns:
            List of environment states
        """
        if self.buffered_envs:
            return self.buffered_envs
        envs: List[EnvironmentState] = discover_all_envs()
        self.buffered_envs = envs
        return envs

    # --------------------------------- Jobs -------------------------------------------

    async def add_job(self, job: JobState) -> JobState:
        """Add a new job to the database.

        Args:
            job: JobState instance containing job details

        Returns:
            Created JobState with updated database information
        """
        m = JobModel(
            job_id=job.job_id,
            status=job.status,
            algo_id=job.algo_id,
            env_id=job.env_id,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            detail=job.detail,
            stop_requested=job.stop_requested,
        )
        self._db.add(m)                 # no await (unit-of-work)
        await self._db.commit()
        await self._db.refresh(m)
        return to_job_state(m)

    async def get_job(self, job_id: int) -> Optional[JobState]:
        """Retrieve a job by its ID.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            JobState if found, None otherwise
        """
        stmt = select(JobModel).where(JobModel.job_id == job_id)
        m = (await self._db.scalars(stmt)).first()
        return to_job_state(m) if m else None

    async def delete_job(self, job_id: int) -> None:
        """Delete a job from the database.

        Args:
            job_id: ID of the job to delete

        Raises:
            KeyError: If job with given ID is not found
        """
        stmt = select(JobModel).where(JobModel.job_id == job_id)
        result = await self._db.scalars(stmt)
        m = result.first()
        if not m:
            raise KeyError(f"Job with id {job_id} not found")
        await self._db.delete(m)   # ✅ must be awaited!
        await self._db.commit()    # ✅ async commit

    async def get_job_with_results(self, job_id: int) -> JobModel:
        """Retrieve a job with its associated results.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            JobModel with loaded results

        Raises:
            KeyError: If job with given ID is not found
        """
        stmt = (
            select(JobModel)
            .where(JobModel.job_id == job_id)
            .options(selectinload(JobModel.results))
        )
        m = (await self._db.scalars(stmt)).first()
        if not m:
            raise KeyError(f"Job with id {job_id} not found")
        return m

    async def get_job_results(self, job_id: int, result_type: ResultType) -> JobResultState:
        """Retrieve specific type of results for a job.

        Args:
            job_id: ID of the job
            result_type: Type of results to retrieve

        Returns:
            JobResultState containing the results

        Raises:
            KeyError: If results are not found
        """
        stmt = (
            select(JobResultsModel)
            .where(JobResultsModel.job_id == job_id, JobResultsModel.result_type == result_type)
        )
        row = (await self._db.scalars(stmt)).first()
        if not row:
            raise KeyError(f"Result with id {job_id} not found")
        return to_result_state(row)

    async def get_all_jobs(self, limit: int | None = None, offset: int | None = None) -> list[JobState]:
        """Retrieve all jobs with pagination support.

        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of JobState objects
        """
        stmt = (
            select(JobModel)
            .order_by(JobModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        jobs = (await self._db.scalars(stmt)).all()
        return [to_job_state(m) for m in jobs]

    async def update_job(self, job: JobState) -> None:
        """Update existing job in database.

        Args:
            job: JobState with updated values

        Raises:
            KeyError: If job with given ID is not found
        """
        stmt = select(JobModel).where(JobModel.job_id == job.job_id)
        m = (await self._db.scalars(stmt)).first()
        if not m:
            raise KeyError(f"Job with id {job.job_id} not found")

        m.status = job.status
        m.started_at = job.started_at
        m.finished_at = job.finished_at
        m.detail = job.detail
        m.stop_requested = job.stop_requested

        await self._db.commit()

    # --------------------------------- Results ----------------------------------------

    async def add_job_results(self, result_state: JobResultState) -> JobResultState:
        """Add new results for a job.

        Args:
            result_state: JobResultState containing result information

        Returns:
            Created JobResultState with updated database information
        """
        m = JobResultsModel(
            job_id=result_state.job_id,
            result_type=result_state.result_type,
            manifest_uri=result_state.manifest_uri,
            segment_steps=result_state.segment_steps,
            latest_step=result_state.latest_step,
            created_at=result_state.created_at,
        )
        self._db.add(m)                 # no await
        await self._db.commit()
        await self._db.refresh(m)
        return to_result_state(m)

    async def update_results(self, result_state: JobResultState) -> JobResultState:
        """Update existing job results.

        Args:
            result_state: JobResultState with updated values

        Returns:
            Updated JobResultState

        Raises:
            KeyError: If results are not found
        """
        stmt = select(JobResultsModel).where(
            JobResultsModel.job_id == result_state.job_id,
            JobResultsModel.result_type == result_state.result_type,
        )
        m = (await self._db.scalars(stmt)).first()
        if not m:
            raise KeyError(f"Result with id {result_state.result_id} not found")
        m.latest_step = result_state.latest_step
        await self._db.commit()
        return to_result_state(m)

    # ----------------------------- Algo / Env (in-memory) -----------------------------

    def get_algo_config(self, algo_id: AlgoType) -> BaseAlgoConfig:
        """Get configuration for specified algorithm.

        Args:
            algo_id: Type of algorithm to get config for

        Returns:
            Algorithm configuration instance

        Raises:
            KeyError: If algorithm is not found in registry
        """
        ModelConfig = ALGO_REGISTRY.get(algo_id)
        if not ModelConfig:
            raise KeyError(f"Algo with id {algo_id.value} not found")
        return ModelConfig()

    def get_all_algos(self) -> List[AlgoState]:
        """Get list of all available algorithms.

        Returns:
            List of algorithm states
        """
        algos: List[AlgoState] = discover_all_algos()
        return algos

    def get_algo(self, algo_id: AlgoType) -> AlgoState:
        """Get state of specific algorithm.

        Args:
            algo_id: Type of algorithm to retrieve

        Returns:
            Algorithm state

        Raises:
            KeyError: If algorithm is not found
        """
        for algo in self.get_all_algos():
            if algo.algo_id == algo_id:
                return algo
        raise KeyError(f"Algo with id {algo_id.value} not found")

    def get_all_envs(self) -> List[EnvironmentState]:
        """Get list of all available environments.

        Returns:
            List of environment states
        """
        return self._environments

    def get_env(self, env_id: str) -> EnvironmentState:
        """Get state of specific environment.

        Args:
            env_id: ID of environment to retrieve

        Returns:
            Environment state

        Raises:
            KeyError: If environment is not found
        """
        for env in self.get_all_envs():
            if env.env_id == env_id:
                return env
        raise KeyError(f"Env with id {env_id} not found")

    async def get_env_jobs(self, env_id: str) -> list[JobState]:
        """Get all jobs for specific environment.

        Args:
            env_id: ID of environment to get jobs for

        Returns:
            List of job states for the environment
        """
        stmt = select(JobModel).where(JobModel.env_id == env_id)
        rows = (await self._db.scalars(stmt)).all()
        return [to_job_state(m) for m in rows]

    async def get_env_job_results(self, env_id: str, result_type: ResultType) -> list[JobResultState]:
        """Get all results of specific type for an environment's jobs.

        Args:
            env_id: ID of environment to get results for
            result_type: Type of results to retrieve

        Returns:
            List of job result states
        """
        stmt = (
            select(JobResultsModel)
            .join(JobModel, JobResultsModel.job_id == JobModel.job_id)
            .where(JobModel.env_id == env_id, JobResultsModel.result_type == result_type)
            .order_by(JobResultsModel.job_id.asc())
        )
        rows = (await self._db.scalars(stmt)).all()
        return [to_result_state(row) for row in rows]

    # ----------------------------- Defaults / Config ----------------------------------

    def get_general_config(self, env_id: str) -> GeneralConfig:
        """Get general configuration for environment.

        Args:
            env_id: ID of environment to get config for

        Returns:
            General configuration instance
        """
        return GeneralConfig(env_id=env_id)

    def get_log_confi(self) -> LogConfig:  # keeping original name for compatibility
        """Get logging configuration.

        Returns:
            Logging configuration instance
        """
        return LogConfig()
