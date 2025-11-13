# backend/services/training_service/training_route.py
from __future__ import annotations

import os
from typing import Annotated, List

from fastapi import APIRouter, status, HTTPException, Path, Depends, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import FileResponse

from sqlalchemy.ext.asyncio import AsyncSession

from drl_wizard.backend.schemas import EnvResponse, JobRequest, JobResponse
from drl_wizard.backend.schemas.algo_cfg_schema import AlgoConfigSchema
from drl_wizard.backend.schemas.algo_schema import AlgoResponse
from drl_wizard.backend.schemas.general_cfg_schema import GeneralConfigSchema
from drl_wizard.backend.schemas.log_cfg_schema import LogConfigSchema
from drl_wizard.backend.schemas.manifest_schema import ManifestSchema
from drl_wizard.backend.schemas.utils import build_meta
from drl_wizard.backend.schemas.wrapped_config import WrappedConfigSchema
from drl_wizard.backend.services.mappers import (
    to_algo_response,
    to_env_response,
    to_job_response,
    algo_domain_to_schema,
    manifest_domain_to_schema,
    general_domain_to_schema,
    log_domain_to_schema,
    algo_schema_to_domain,
    general_schema_to_domain,
    log_schema_to_domain,
)
from drl_wizard.backend.services.storage.database import get_db
from drl_wizard.backend.services.training_service.jobs import JobState
from drl_wizard.backend.services.training_service.repository import JobRepository
from drl_wizard.backend.services.training_service.service import TrainingService
from drl_wizard.common.types import ResultType, AlgoType
from drl_wizard.runtime.handlers import stop_event_handler


# --- dependencies -------------------------------------------------------------

db_dependency = Annotated[AsyncSession, Depends(get_db)]


def get_training_service(db: db_dependency) -> TrainingService:
    """Create and return a new TrainingService instance.

    Args:
        db: Database session dependency

    Returns:
        TrainingService: Configured training service instance with repository and event handlers
    """
    repo = JobRepository(db)
    svc = TrainingService(repo, event_handlers=[stop_event_handler])
    return svc

svc_dependency = Annotated[TrainingService, Depends(get_training_service)]

router = APIRouter(prefix="/training_service", tags=["training_service"])

# --- routes ------------------------------------------------------------------

@router.post("/train", status_code=status.HTTP_200_OK, response_model=JobResponse)
async def start_train(svc: svc_dependency, train_request: JobRequest) -> JobResponse:
    """Start a new training job.

    Args:
        svc: Training service dependency
        train_request: Training job configuration request

    Returns:
        JobResponse: Created job details

    Raises:
        HTTPException: If job creation fails
    """
    # schema -> domain (pure sync)
    model_config   = algo_schema_to_domain(train_request.algo_cfg)
    general_config = general_schema_to_domain(train_request.general_cfg)
    log_config     = log_schema_to_domain(train_request.log_cfg)

    app_cfg = svc.get_app_config(general_config, log_config, model_config)

    job_state = await svc.start_job(train_request.env_id, train_request.algo_id, app_cfg)

    # env/algo metadata are sync (in-memory discovery)
    job: JobResponse = to_job_response(
        state=job_state,
        env=svc.get_env(job_state.env_id),
        algo=svc.get_algo(job_state.algo_id),
    )
    return job


@router.patch("/{job_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_train(svc: svc_dependency, job_id: int = Path(..., gt=0)) -> Response:
    """Stop a running training job.

    Args:
        svc: Training service dependency
        job_id: ID of the job to stop

    Returns:
        Response: Empty response with 204 status code

    Raises:
        HTTPException: If job not found or stop request fails
    """
    await svc.request_stop(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/all", status_code=status.HTTP_200_OK, response_model=list[JobResponse])
async def train_list(svc: svc_dependency) -> list[JobResponse]:
    """Get list of all training jobs.

    Args:
        svc: Training service dependency

    Returns:
        list[JobResponse]: List of all jobs sorted by creation date
    """
    db_jobs: List[JobState] = await svc.get_all_jobs()
    # env/algo lookups are sync (in-memory)
    job_responses = [
        to_job_response(
            state=job_state,
            env=svc.get_env(job_state.env_id),
            algo=svc.get_algo(job_state.algo_id),
        )
        for job_state in db_jobs
    ]
    return sorted(job_responses, key=lambda job: job.created_at, reverse=True)


@router.get("/{job_id}/results/{result_type}/stream", response_class=StreamingResponse)
async def stream_results(
    svc: svc_dependency,
    job_id: int = Path(..., gt=0),
    result_type: ResultType = Path(...),
) -> StreamingResponse:
    """Stream results for a specific job and result type.

    Args:
        svc: Training service dependency
        job_id: ID of the job
        result_type: Type of results to stream

    Returns:
        StreamingResponse: Streamed result data

    Raises:
        HTTPException: If manifest not found or data not saved
    """
    # Validate manifest early
    try:
        await svc.get_results_manifest(job_id, result_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Manifest not found")
    except KeyError:
        raise HTTPException(status_code=404, detail="Unsaved data")

    # Wrap sync generator to avoid blocking
    generator = iterate_in_threadpool(svc.stream_results(job_id, result_type))
    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
        background=BackgroundTask(lambda: None),
    )


@router.get("/environments/{env_id:path}/supported_algorithms", response_model=list[AlgoResponse])
def get_algos_for_env(
    svc: svc_dependency,
    env_id: Annotated[
        str,
        Path(
            pattern=r"^[A-Za-z0-9._/\-]+$",
            description="Gym/Atari-style env id (may include /, e.g. ALE/Pong-v5)",
        ),
    ],
) -> list[AlgoResponse]:
    """Get list of algorithms supported by specified environment.

    Args:
        svc: Training service dependency
        env_id: Environment identifier

    Returns:
        list[AlgoResponse]: List of supported algorithms

    Raises:
        HTTPException: If environment not found
    """
    try:
        supported_algos = [to_algo_response(algo) for algo in svc.get_supported_algos(env_id)]
        return supported_algos
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")


@router.get("/environments", status_code=status.HTTP_200_OK, response_model=list[EnvResponse])
async def env_list(svc: svc_dependency) -> list[EnvResponse]:
    """Get list of all available environments.

    Args:
        svc: Training service dependency

    Returns:
        list[EnvResponse]: List of all environments
    """
    envs = [to_env_response(env) for env in svc.get_all_envs()]
    return envs


@router.get("/algorithms", status_code=status.HTTP_200_OK, response_model=list[AlgoResponse])
async def algo_list(svc: svc_dependency) -> list[AlgoResponse]:
    """Get list of all available algorithms.

    Args:
        svc: Training service dependency

    Returns:
        list[AlgoResponse]: List of all algorithms
    """
    algos = [to_algo_response(algo) for algo in svc.get_all_algos()]
    return algos


@router.get("/algorithms/{algo_id}", status_code=status.HTTP_200_OK, response_model=AlgoResponse)
async def algo_details(svc: svc_dependency, algo_id: AlgoType = Path(..., min_length=1)) -> AlgoResponse:
    """Get details for a specific algorithm.

    Args:
        svc: Training service dependency
        algo_id: Algorithm identifier

    Returns:
        AlgoResponse: Algorithm details

    Raises:
        HTTPException: If algorithm not found
    """
    try:
        return to_algo_response(svc.get_algo(algo_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Algo with id {algo_id.value} not found")


@router.get(
    "/algorithms/{algo_id}/config",
    status_code=status.HTTP_200_OK,
    response_model=WrappedConfigSchema[AlgoConfigSchema],
)
async def algo_config(svc: svc_dependency, algo_id: AlgoType) -> WrappedConfigSchema[AlgoConfigSchema]:
    """Get configuration schema for a specific algorithm.

    Args:
        svc: Training service dependency
        algo_id: Algorithm identifier

    Returns:
        WrappedConfigSchema[AlgoConfigSchema]: Algorithm configuration schema with metadata

    Raises:
        HTTPException: If algorithm not found
    """
    try:
        model_config = algo_domain_to_schema(svc.get_algo_config(algo_id))
        meta = build_meta(model_config.__class__)
        return {"config": model_config, "meta": meta}
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Algo with id {algo_id.value} not found")


@router.get(
    "/environments/{env_id:path}/general_config",
    status_code=status.HTTP_200_OK,
    response_model=WrappedConfigSchema[GeneralConfigSchema],
)
async def get_general_config(
    svc: svc_dependency,
    env_id: Annotated[
        str,
        Path(
            pattern=r"^[A-Za-z0-9._/\-]+$",
            description="Gym/Atari-style env id (may include /, e.g. ALE/Pong-v5)",
        ),
    ],
) -> WrappedConfigSchema[GeneralConfigSchema]:
    """Get general configuration for a specific environment.

    Args:
        svc: Training service dependency
        env_id: Environment identifier

    Returns:
        WrappedConfigSchema[GeneralConfigSchema]: Environment configuration with metadata

    Raises:
        HTTPException: If configuration not found
    """
    try:
        cfg = general_domain_to_schema(svc.get_general_config(env_id))
        meta = build_meta(GeneralConfigSchema)
        return {"config": cfg, "meta": meta}
    except KeyError:
        raise HTTPException(status_code=404, detail="General config not found")


@router.get("/logs/log_config", status_code=status.HTTP_200_OK, response_model=WrappedConfigSchema[LogConfigSchema])
def get_log_config(svc: svc_dependency) -> WrappedConfigSchema[LogConfigSchema]:
    """Get logging configuration schema.

    Args:
        svc: Training service dependency

    Returns:
        WrappedConfigSchema[LogConfigSchema]: Logging configuration with metadata

    Raises:
        HTTPException: If configuration not found
    """
    try:
        log_config = log_domain_to_schema(svc.get_log_confi())
        meta = build_meta(LogConfigSchema)
        return {"config": log_config, "meta": meta}
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log config not found")


@router.get(
    "/{job_id}/results/{result_type}/manifest",
    status_code=status.HTTP_200_OK,
    response_model=ManifestSchema,
)
async def get_results_manifest(
    svc: svc_dependency,
    result_type: ResultType,
    job_id: int = Path(..., gt=0),
) -> ManifestSchema:
    """Get results manifest for a specific job and result type.

    Args:
        svc: Training service dependency
        result_type: Type of results
        job_id: Job identifier

    Returns:
        ManifestSchema: Results manifest

    Raises:
        HTTPException: If job not found
    """
    try:
        manifest = await svc.get_results_manifest(job_id, result_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job with id {job_id} not found")
    return manifest_domain_to_schema(manifest)


@router.get("/{job_id}", status_code=status.HTTP_200_OK, response_model=JobResponse)
async def train_details(svc: svc_dependency, job_id: int = Path(..., gt=0)) -> JobResponse:
    """Get details for a specific training job.

    Args:
        svc: Training service dependency
        job_id: Job identifier

    Returns:
        JobResponse: Job details

    Raises:
        HTTPException: If job not found
    """
    try:
        job_state: JobState = await svc.get_job(job_id)
        job: JobResponse = to_job_response(
            state=job_state,
            env=svc.get_env(job_state.env_id),
            algo=svc.get_algo(job_state.algo_id),
        )
        return job
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job with id {job_id} not found")


@router.get(
    "/environments/{env_id:path}/results/{result_type}/stream",
    response_class=StreamingResponse,
)
async def stream_env_results(
    svc: svc_dependency,
    env_id: Annotated[
        str,
        Path(
            pattern=r"^[A-Za-z0-9._/\-]+$",
            description="Gym/Atari-style env id (may include /, e.g. ALE/Pong-v5)",
        ),
    ],
    result_type: ResultType,
) -> StreamingResponse:
    """Stream results for a specific environment and result type.

    Args:
        svc: Training service dependency
        env_id: Environment identifier
        result_type: Type of results to stream

    Returns:
        StreamingResponse: Streamed environment results

    Raises:
        HTTPException: If manifest not found
    """
    # quick existence check (async)
    try:
        _ = await svc.async_list_env_result_manifests(env_id, result_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Manifest not found for env/results")

    generator = iterate_in_threadpool(svc.stream_env_results_multiplexed(env_id, result_type))
    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
        background=BackgroundTask(lambda: None),
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(svc: svc_dependency, job_id: int = Path(..., gt=0)) -> Response:
    """Delete a specific training job.

    Args:
        svc: Training service dependency
        job_id: Job identifier

    Returns:
        Response: Empty response with 204 status code

    Raises:
        HTTPException: If job not found
    """
    try:
        await svc.delete_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job with id {job_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{job_id}/data/zip")
async def download_results_zip(
    svc: svc_dependency,
    job_id: int = Path(..., gt=0),
) -> FileResponse:
    """Download all results for a job as a ZIP archive.

    Args:
        svc: Training service dependency
        job_id: Job identifier

    Returns:
        FileResponse: ZIP file containing job results

    Raises:
        HTTPException: If job not found or results not available
    """
    try:
        zip_path = await svc.build_archive(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # clean up the temporary zip (and its temp dir) after sending
    def _cleanup(path: str):
        try:
            tmpdir = os.path.dirname(path)
            if os.path.exists(tmpdir):
                for name in os.listdir(tmpdir):
                    try:
                        os.remove(os.path.join(tmpdir, name))
                    except Exception:
                        pass
                os.rmdir(tmpdir)
        except Exception:
            pass

    filename = f"{job_id}-data.zip"
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=filename,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
        background=BackgroundTask(_cleanup, str(zip_path)),
    )
