import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, List

import anyio
import pytest

# Service under test
from backend.services.training_service import service as svc_mod
from backend.services.training_service.service import TrainingService

# Domain helpers / types
from backend.services.training_service.jobs import create_job, mark_running, mark_finished, mark_stopped
from backend.services.training_service.job_results import JobResultState, create_job_result
from common.types import AlgoType, ResultType, JobStatus


# ------------------------------- Test Utilities -------------------------------

def _run(coro, *args, **kwargs):
    """Drive async coroutines from sync tests."""
    return anyio.run(lambda: coro(*args, **kwargs))


class FakeJobModel:
    """Minimal ORM stand-in with relationships the service expects."""
    def __init__(self, job_id: int,
                 train_uri: Optional[str] = None,
                 eval_uri: Optional[str] = None):
        self.job_id = job_id
        # viewonly "selectin" relationships the service reads
        self.train_results = type("R", (), {})() if train_uri else None
        if self.train_results:
            self.train_results.job_id = job_id
            self.train_results.result_type = ResultType.TRAIN
            self.train_results.manifest_uri = train_uri

        self.evaluate_results = type("R", (), {})() if eval_uri else None
        if self.evaluate_results:
            self.evaluate_results.job_id = job_id
            self.evaluate_results.result_type = ResultType.EVALUATE
            self.evaluate_results.manifest_uri = eval_uri


class FakeRepo:
    """In-memory repository with just what TrainingService uses here."""
    def __init__(self):
        self.jobs: Dict[int, "JobState"] = {}
        self.results: Dict[tuple[int, ResultType], JobResultState] = {}
        self.models: Dict[int, FakeJobModel] = {}
        self._next_id = 1

    # ---- jobs ----
    async def add_job(self, job):
        # Ensure job_id is unique and stable
        job.job_id = self._next_id
        self._next_id += 1
        self.jobs[job.job_id] = job
        return job

    async def get_job(self, job_id: int):
        return self.jobs.get(job_id)

    async def update_job(self, job):
        if job.job_id not in self.jobs:
            raise KeyError(f"Job with id {job.job_id} not found")
        self.jobs[job.job_id] = job

    async def delete_job(self, job_id: int):
        if job_id not in self.jobs:
            raise KeyError(f"Job with id {job_id} not found")
        # Simulate cascade delete on results
        self.jobs.pop(job_id)
        for key in list(self.results.keys()):
            if key[0] == job_id:
                self.results.pop(key)
        self.models.pop(job_id, None)

    async def get_all_jobs(self):
        # return newest first by created_at semantics; our fake preserves insertion order
        return list(reversed(list(self.jobs.values())))

    async def get_job_with_results(self, job_id: int):
        m = self.models.get(job_id)
        if not m:
            raise KeyError(f"Job with id {job_id} not found")
        return m

    # ---- result rows ----
    async def add_job_results(self, result_state: JobResultState):
        self.results[(result_state.job_id, result_state.result_type)] = result_state
        # also surface a FakeJobModel for Manifest lookups
        jm = self.models.get(result_state.job_id) or FakeJobModel(result_state.job_id)
        if result_state.result_type == ResultType.TRAIN:
            jm.train_results = type("R", (), {})()
            jm.train_results.job_id = result_state.job_id
            jm.train_results.result_type = result_state.result_type
            jm.train_results.manifest_uri = result_state.manifest_uri
        else:
            jm.evaluate_results = type("R", (), {})()
            jm.evaluate_results.job_id = result_state.job_id
            jm.evaluate_results.result_type = result_state.result_type
            jm.evaluate_results.manifest_uri = result_state.manifest_uri
        self.models[result_state.job_id] = jm
        return result_state

    async def update_results(self, result_state: JobResultState):
        key = (result_state.job_id, result_state.result_type)
        if key not in self.results:
            raise KeyError("missing")
        self.results[key] = result_state
        return result_state

    async def get_env_job_results(self, env_id: str, result_type: ResultType) -> list[JobResultState]:
        # Not used in these tests; included for completeness
        return [r for (jid, rt), r in self.results.items() if rt == result_type]


# ------------------------------- Fakes for mp stop -------------------------------

class FakeEvent:
    def __init__(self):
        self._flag = False
    def set(self):
        self._flag = True
    def is_set(self):
        return self._flag


class FakeProcess:
    def __init__(self, alive=True):
        self._alive = alive
        self.join_calls: List[float] = []
        self.terminated = False
        self.killed = False
    def is_alive(self):
        return self._alive
    def join(self, timeout=None):
        self.join_calls.append(timeout)
        # join does not change alive state in our simulation
    def terminate(self):
        self.terminated = True
        self._alive = False
    def kill(self):
        self.killed = True
        self._alive = False


# ------------------------------- Manifest fakes -------------------------------

class FakeSegment:
    def __init__(self, path: str):
        self.path = path

class FakeManifest:
    """
    Only fields/methods the service touches:
      - path (Path-like)    -> for build_archive()
      - segments[ResultType] -> list[FakeSegment]
      - log_path            -> base dir to read segment files under stream_results()
      - load(Path) -> FakeManifest (classmethod/staticmethod patched in)
    """
    def __init__(self, path: Path, log_dir: Path, segments: Dict[ResultType, List[FakeSegment]]):
        self.path = path
        self.segments = segments
        self.log_path = str(log_dir)  # service uses urlparse(str(manifest.log_path)).path


# ------------------------------- Fixtures -------------------------------

@pytest.fixture()
def repo() -> FakeRepo:
    return FakeRepo()

@pytest.fixture()
def svc(repo: FakeRepo) -> TrainingService:
    return TrainingService(repo)


# ------------------------------- Tests: mark_* lifecycle -------------------------------

def test_mark_lifecycle_transitions(svc: TrainingService, repo: FakeRepo):
    async def _inner():
        j = await repo.add_job(create_job(env_id="CartPole-v1", algo_id=AlgoType.PPO))

        j1 = await svc.mark_running(j.job_id)
        assert j1.status == JobStatus.RUNNING and j1.started_at is not None

        j2 = await svc.mark_finished(j.job_id)
        assert j2.status == JobStatus.FINISHED and j2.finished_at is not None

        j3 = await svc.mark_stopped(j.job_id)
        assert j3.status == JobStatus.STOPPED

        j4 = await svc.mark_failed(j.job_id, "boom")
        assert j4.status == JobStatus.FAILED
        assert j4.detail and "boom" in j4.detail

    _run(_inner)


# ------------------------------- Tests: request_stop (graceful) -------------------------------

def test_request_stop_sets_event_and_terminates(monkeypatch, svc: TrainingService, repo: FakeRepo):
    async def _inner():
        # Create job
        j = await repo.add_job(create_job(env_id="Ant-v5", algo_id=AlgoType.TRPO))

        # Install fake event and process in the module-level maps the service uses
        evt = FakeEvent()
        proc = FakeProcess(alive=True)
        svc_mod._stop_events[j.job_id] = evt
        svc_mod._active_processes[j.job_id] = proc

        # Run
        await svc.request_stop(j.job_id)

        # Event set and process no longer alive (terminated by graceful path)
        assert evt.is_set()
        assert proc.terminated is True
        assert proc.killed is False    # terminate should suffice in our fake

        # Cleanup (what service would normally do later)
        svc_mod._stop_events.pop(j.job_id, None)
        svc_mod._active_processes.pop(j.job_id, None)

    _run(_inner)


# ------------------------------- Tests: build_archive -------------------------------

def test_build_archive_creates_zip(monkeypatch, svc: TrainingService, repo: FakeRepo, tmp_path: Path):
    async def _inner():
        # Setup a run directory with a manifest file inside it
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        manifest_file = run_dir / "manifest.json"
        manifest_file.write_text("{}", encoding="utf-8")

        # The DB row points to the manifest URI on disk
        j = await repo.add_job(create_job(env_id="Walker2d-v5", algo_id=AlgoType.SAC))
        # seed model row for get_job_with_results via add_job_results
        fake_result = create_job_result(j.job_id, ResultType.EVALUATE, 100, str(manifest_file))
        await repo.add_job_results(fake_result)

        # Patch Manifest.load to return a FakeManifest carrying `path=manifest_file`
        def fake_load(p: Path):
            return FakeManifest(path=p, log_dir=run_dir, segments={})
        monkeypatch.setattr(svc_mod.Manifest, "load", staticmethod(fake_load))

        # Exercise
        zip_path = await svc.build_archive(j.job_id)
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

        # Ensure it contains files from run_dir
        # (shutil.make_archive produced a zip named "<tmp>/<jobid>-data.zip")
        assert zip_path.stat().st_size > 0

    _run(_inner)


# ------------------------------- Tests: get_results_manifest selection -------------------------------

def test_get_results_manifest_chooses_correct_type(monkeypatch, svc: TrainingService, repo: FakeRepo, tmp_path: Path):
    async def _inner():
        run_dir = tmp_path / "run2"
        run_dir.mkdir(parents=True)
        train_manifest = run_dir / "train.json"
        eval_manifest = run_dir / "eval.json"
        train_manifest.write_text("{}", encoding="utf-8")
        eval_manifest.write_text("{}", encoding="utf-8")

        j = await repo.add_job(create_job(env_id="ALE/Pong-v5", algo_id=AlgoType.DQN))
        await repo.add_job_results(create_job_result(j.job_id, ResultType.TRAIN, 10, str(train_manifest)))
        await repo.add_job_results(create_job_result(j.job_id, ResultType.EVALUATE, 5, str(eval_manifest)))

        def fake_load(p: Path):
            # We return a minimal object with the same "path" the service passed in
            return FakeManifest(path=p, log_dir=run_dir, segments={})
        monkeypatch.setattr(svc_mod.Manifest, "load", staticmethod(fake_load))

        m_train = await svc.get_results_manifest(j.job_id, ResultType.TRAIN)
        m_eval = await svc.get_results_manifest(j.job_id, ResultType.EVALUATE)
        assert Path(m_train.path).name == "train.json"
        assert Path(m_eval.path).name == "eval.json"

        # Ask for invalid type -> ValueError
        with pytest.raises(ValueError):
            await svc.get_results_manifest(j.job_id, "weird")  # type: ignore

    _run(_inner)


# ------------------------------- Tests: stream_results (NDJSON) -------------------------------

# replace your existing test_stream_results_reads_segments with this sync version
def test_stream_results_reads_segments(monkeypatch, svc: TrainingService, repo: FakeRepo, tmp_path: Path):
    # Prepare files with NDJSON
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    seg1 = log_dir / "seg1.jsonl"
    seg2 = log_dir / "seg2.jsonl"
    seg1.write_bytes(b'{"a":1}\n{"b":2}\n')
    seg2.write_bytes(b'{"c":3}\n')

    # Create a dummy manifest file path (only used to derive run_dir in build_archive;
    # here we only need it to exist as a string)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}", encoding="utf-8")

    # Seed the fake repo *synchronously* (no awaits)
    job_id = 1
    # FakeJobModel lets the service find the manifest URI via get_job_with_results()
    repo.models[job_id] = FakeJobModel(
        job_id=job_id,
        train_uri=str(manifest_file),
        eval_uri=None,
    )

    # Monkeypatch Manifest.load to return a FakeManifest with our two segments
    def fake_load(p: Path):
        segments = {
            ResultType.TRAIN: [FakeSegment("seg1.jsonl"), FakeSegment("seg2.jsonl")],
            ResultType.EVALUATE: [],
        }
        return FakeManifest(path=p, log_dir=log_dir, segments=segments)

    monkeypatch.setattr(svc_mod.Manifest, "load", staticmethod(fake_load))

    # Call the sync generator under test
    chunks = list(svc.stream_results(job_id, ResultType.TRAIN))
    body = b"".join(chunks)
    assert body.splitlines() == [b'{"a":1}', b'{"b":2}', b'{"c":3}']



# ------------------------------- Tests: delete_job cleans results (no IO) -------------------------------

def test_delete_job_removes_rows(svc: TrainingService, repo: FakeRepo, tmp_path: Path):
    async def _inner():
        j = await repo.add_job(create_job(env_id="HalfCheetah-v5", algo_id=AlgoType.SAC))
        await repo.add_job_results(create_job_result(j.job_id, ResultType.TRAIN, 10, str(tmp_path / "m.json")))
        await repo.add_job_results(create_job_result(j.job_id, ResultType.EVALUATE, 10, str(tmp_path / "m2.json")))
        assert (j.job_id, ResultType.TRAIN) in repo.results
        assert (j.job_id, ResultType.EVALUATE) in repo.results

        await svc.delete_job(j.job_id)
        assert await repo.get_job(j.job_id) is None
        assert (j.job_id, ResultType.TRAIN) not in repo.results
        assert (j.job_id, ResultType.EVALUATE) not in repo.results

    _run(_inner)
