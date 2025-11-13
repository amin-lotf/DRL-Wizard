import anyio
import pytest
from sqlalchemy import text

from backend.services.training_service.repository import JobRepository
from backend.services.storage.database import SessionLocal, init_db, engine

from backend.services.training_service.jobs import (
    create_job,
    mark_running,
    mark_finished,
    mark_stopped,
)
from backend.services.training_service.job_results import create_job_result
from common.types import AlgoType, ResultType, JobStatus


# ------------------------ DB bootstrap & cleanup ------------------------

@pytest.fixture(autouse=True, scope="function")
def _db_bootstrap_and_clean():
    """
    Ensure tables exist and clean state between tests.
    We avoid pytest-asyncio by using anyio.run in sync fixtures/tests.
    """
    anyio.run(init_db)

    async def _wipe():
        async with engine.begin() as conn:
            # FK order: clear children first
            await conn.execute(text("DELETE FROM job_train_results"))
            await conn.execute(text("DELETE FROM jobs"))

    anyio.run(_wipe)
    yield
    anyio.run(_wipe)


# ------------------------ Helper to run async in sync tests ------------------------

def _run(coro, *args, **kwargs):
    return anyio.run(lambda: coro(*args, **kwargs))


# ------------------------ Tests: Jobs CRUD ------------------------

def test_add_and_get_job_roundtrip():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            # Add a job
            j0 = create_job(env_id="CartPole-v1", algo_id=AlgoType.PPO)
            await repo.add_job(j0)

            # Fetch via get_all_jobs (ensures we have the DB-assigned id)
            all_jobs = await repo.get_all_jobs()
            assert len(all_jobs) == 1
            j = all_jobs[0]
            assert j.job_id is not None
            assert j.env_id == "CartPole-v1"
            assert j.algo_id == AlgoType.PPO
            assert j.status == JobStatus.QUEUED
            assert j.created_at is not None

    _run(_inner)


def test_update_job_status_and_fields():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            # Persist job, then re-read it to get the real id-backed state
            await repo.add_job(create_job(env_id="CartPole-v1", algo_id=AlgoType.SAC))
            j = (await repo.get_all_jobs())[0]

            # QUEUED -> RUNNING
            mark_running(j)
            await repo.update_job(j)

            j1 = (await repo.get_all_jobs())[0]
            assert j1.status == JobStatus.RUNNING
            assert j1.started_at is not None

            # RUNNING -> FINISHED
            mark_finished(j)
            await repo.update_job(j)

            j2 = (await repo.get_all_jobs())[0]
            assert j2.status == JobStatus.FINISHED
            assert j2.finished_at is not None

            # FINISHED -> STOPPED (exercise path)
            mark_stopped(j)
            await repo.update_job(j)

            j3 = (await repo.get_all_jobs())[0]
            assert j3.status == JobStatus.STOPPED

    _run(_inner)


def test_delete_job_and_missing_keyerror_on_second_delete():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            await repo.add_job(create_job(env_id="Ant-v5", algo_id=AlgoType.TRPO))
            j = (await repo.get_all_jobs())[0]

            # Delete once
            await repo.delete_job(j.job_id)
            assert await repo.get_job(j.job_id) is None

            # Delete again -> KeyError
            with pytest.raises(KeyError):
                await repo.delete_job(j.job_id)

    _run(_inner)


def test_get_all_jobs_order_limit_offset():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)
            await repo.add_job(create_job(env_id="A", algo_id=AlgoType.DQN))
            await repo.add_job(create_job(env_id="B", algo_id=AlgoType.A2C))
            await repo.add_job(create_job(env_id="C", algo_id=AlgoType.PPO))

            all_jobs = await repo.get_all_jobs()
            ids = [j.job_id for j in all_jobs]
            # created_at DESC => last added first
            assert ids == [all_jobs[0].job_id, all_jobs[1].job_id, all_jobs[2].job_id]

            top2 = await repo.get_all_jobs(limit=2)
            assert len(top2) == 2

            last1 = await repo.get_all_jobs(limit=2, offset=2)
            assert len(last1) == 1

    _run(_inner)


# ------------------------ Tests: Results CRUD ------------------------

def test_add_get_update_results_roundtrip():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            await repo.add_job(create_job(env_id="HalfCheetah-v5", algo_id=AlgoType.SAC))
            j = (await repo.get_all_jobs())[0]

            # IMPORTANT: use positional args (your create_job_result doesn't accept keywords)
            r_state = create_job_result(j.job_id, ResultType.TRAIN, 1000, "/tmp/train-manifest.json")
            r = await repo.add_job_results(r_state)
            assert r.job_id == j.job_id
            assert r.result_type == ResultType.TRAIN
            assert r.segment_steps == 1000
            assert r.latest_step == 0  # default at creation

            # Fetch by type
            got = await repo.get_job_results(j.job_id, ResultType.TRAIN)
            assert got.result_id == r.result_id
            assert got.manifest_uri.endswith("train-manifest.json")

            # Update latest_step
            got.latest_step = 4200
            u = await repo.update_results(got)
            assert u.latest_step == 4200

            # Missing type -> KeyError
            with pytest.raises(KeyError):
                await repo.get_job_results(j.job_id, ResultType.EVALUATE)

    _run(_inner)


def test_get_job_with_results_and_selectin_relationships():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            await repo.add_job(create_job(env_id="Walker2d-v5", algo_id=AlgoType.PPO))
            j = (await repo.get_all_jobs())[0]

            train = create_job_result(j.job_id, ResultType.TRAIN, 500, "/tmp/train.json")
            eval_ = create_job_result(j.job_id, ResultType.EVALUATE, 200, "/tmp/eval.json")
            await repo.add_job_results(train)
            await repo.add_job_results(eval_)

            m = await repo.get_job_with_results(j.job_id)
            # Ensure relationships are populated (selectin)
            assert m.train_results is not None
            assert m.evaluate_results is not None
            assert m.train_results.manifest_uri.endswith("train.json")
            assert m.evaluate_results.manifest_uri.endswith("eval.json")

    _run(_inner)


def test_get_env_job_results_join_filters_by_env_and_type():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            # Two jobs in different envs
            await repo.add_job(create_job(env_id="ALE/Pong-v5", algo_id=AlgoType.DQN))
            await repo.add_job(create_job(env_id="CartPole-v1", algo_id=AlgoType.A2C))
            all_jobs = await repo.get_all_jobs()
            # created_at DESC => index 0 is second job
            j2, j1 = all_jobs[0], all_jobs[1]  # j1: Pong, j2: CartPole

            await repo.add_job_results(create_job_result(j1.job_id, ResultType.TRAIN, 100, "/tmp/pong-train.json"))
            await repo.add_job_results(create_job_result(j2.job_id, ResultType.EVALUATE, 50, "/tmp/cp-eval.json"))

            pong_train = await repo.get_env_job_results("ALE/Pong-v5", ResultType.TRAIN)
            cp_eval = await repo.get_env_job_results("CartPole-v1", ResultType.EVALUATE)

            assert len(pong_train) == 1
            assert pong_train[0].job_id == j1.job_id
            assert pong_train[0].manifest_uri.endswith("pong-train.json")

            assert len(cp_eval) == 1
            assert cp_eval[0].job_id == j2.job_id
            assert cp_eval[0].manifest_uri.endswith("cp-eval.json")

            # No matches -> empty list
            none = await repo.get_env_job_results("ALE/Pong-v5", ResultType.EVALUATE)
            assert none == []

    _run(_inner)


def test_update_results_missing_raises_keyerror():
    async def _inner():
        async with SessionLocal() as db:
            repo = JobRepository(db)

            await repo.add_job(create_job(env_id="Ant-v5", algo_id=AlgoType.TRPO))
            j = (await repo.get_all_jobs())[0]

            # Create a result state but do NOT add it; then try to update -> KeyError
            missing = create_job_result(j.job_id, ResultType.TRAIN, 100, "/tmp/m.json")
            missing.latest_step = 10

            with pytest.raises(KeyError):
                await repo.update_results(missing)

    _run(_inner)
