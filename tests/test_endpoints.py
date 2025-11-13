import asyncio
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import AsyncIterator, Iterable, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.routers.training_route import get_training_service, router
# ---- Import the real router & types from your app ----
# Adjust these imports only if your layout differs.

from common.types import ResultType, AlgoType


# ------------------------------ Fake Service ------------------------------

class FakeService:
    """
    Minimal stub to satisfy route calls.
    You can set attributes per-test to tune behavior.
    """

    # ----- toggles / holders you can override in tests -----
    # streaming
    stream_lines: List[bytes] = [b'{"x":1}\n', b'{"y":2}\n']
    # zip path
    zip_dir: Path | None = None
    zip_path: Path | None = None

    # list endpoints (return empty for schema-safe responses)
    def get_all_envs(self):
        return []  # env_list -> []

    def get_all_algos(self):
        return []  # algo_list -> []

    async def get_all_jobs(self):
        return []  # train_list -> []

    # start/stop/delete basics
    async def request_stop(self, job_id: int):
        self._last_stopped = job_id

    async def delete_job(self, job_id: int):
        if getattr(self, "raise_delete_404", False):
            raise KeyError(f"Job with id {job_id} not found")

    # manifest / streaming for a single job
    async def get_results_manifest(self, job_id: int, result_type: ResultType):
        if getattr(self, "raise_manifest_not_found", False):
            raise FileNotFoundError("no manifest")
        if getattr(self, "raise_manifest_unsaved", False):
            raise KeyError("unsaved")
        # Return a dummy object; routes only validate existence before streaming.
        class _Dummy:  # minimal stand-in
            pass
        return _Dummy()

    def stream_results(self, job_id: int, result_type: ResultType) -> Iterable[bytes]:
        # Return whatever lines the test configured
        for line in self.stream_lines:
            yield line

    # env-wide streaming
    async def async_list_env_result_manifests(self, env_id: str, result_type: ResultType):
        if getattr(self, "raise_env_manifest_not_found", False):
            raise FileNotFoundError("no env manifest")
        return [object()]  # only existence check matters in route

    def stream_env_results_multiplexed(self, env_id: str, result_type: ResultType) -> Iterable[bytes]:
        # Wrap per-line into NDJSON bytes (the worker already does it; we simulate final bytes)
        # Here we just pass through JSONL bytes like {"job_id": 1, "data": {...}}
        base = [{"job_id": 1, "data": {"k": 1}}, {"job_id": 2, "data": {"k": 2}}]
        for obj in base:
            yield (json.dumps(obj) + "\n").encode("utf-8")

    # zip download
    async def build_archive(self, job_id: int) -> Path:
        if getattr(self, "raise_zip_404_key", False):
            raise KeyError("no data")
        if getattr(self, "raise_zip_404_file", False):
            raise FileNotFoundError("missing")
        # Create a real temporary zip to let FileResponse stream it
        if self.zip_path and self.zip_path.exists():
            return self.zip_path
        self.zip_dir = Path(tempfile.mkdtemp(prefix=f"job-{job_id}-test-"))
        to_zip = self.zip_dir / "payload"
        to_zip.mkdir(parents=True, exist_ok=True)
        (to_zip / "a.txt").write_text("hello", encoding="utf-8")
        archive = shutil.make_archive(
            base_name=str(self.zip_dir / f"{job_id}-data"),
            format="zip",
            root_dir=str(to_zip),
        )
        self.zip_path = Path(archive)
        return self.zip_path

    # routes that we only hit on error paths in these tests
    def get_supported_algos(self, env_id: str):
        if getattr(self, "raise_env_404", False):
            raise KeyError("env not found")
        return []

    def get_algo(self, algo_id: AlgoType):
        if getattr(self, "raise_algo_404", False):
            raise KeyError("algo not found")
        class _A: pass
        return _A()

    def get_log_confi(self):
        if getattr(self, "raise_log_cfg_404", False):
            raise KeyError("log not found")
        class _L: pass
        return _L()

    def get_algo_config(self, algo_id: AlgoType):
        if getattr(self, "raise_algo_cfg_404", False):
            raise KeyError("algo cfg not found")
        class _C: pass
        return _C()

    def get_general_config(self, env_id: str):
        if getattr(self, "raise_general_cfg_404", False):
            raise KeyError("general not found")
        class _G: pass
        return _G()


# ------------------------------ Test App Fixture ------------------------------

@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    """
    Build a minimal FastAPI app with the real router and override the
    training service dependency to return our FakeService.
    """
    app = FastAPI()
    app.include_router(router)

    fake = FakeService()

    async def _override_svc():
        return fake

    app.dependency_overrides[get_training_service] = _override_svc
    # Expose fake so tests can toggle flags
    app.state.fake_svc = fake
    return app


# ------------------------------ Sync TestClient (good for 204/headers) ------------------------------

@pytest.fixture(scope="function")
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app)


# ------------------------------ Async client (good for streaming) ------------------------------

@pytest.fixture(scope="function")
async def aclient(test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


# ------------------------------ Tests: simple lists (OK with response_model=List[...]) ------------------------------

def test_env_list_empty(client: TestClient):
    r = client.get("/training_service/environments")
    assert r.status_code == 200
    assert r.json() == []

def test_algo_list_empty(client: TestClient):
    r = client.get("/training_service/algorithms")
    assert r.status_code == 200
    assert r.json() == []

def test_train_list_empty(client: TestClient):
    r = client.get("/training_service/all")
    assert r.status_code == 200
    assert r.json() == []


# ------------------------------ Tests: stop & delete ------------------------------

def test_stop_returns_204(client: TestClient, test_app: FastAPI):
    r = client.patch("/training_service/123/stop")
    assert r.status_code == 204
    assert test_app.state.fake_svc._last_stopped == 123

def test_delete_returns_204(client: TestClient):
    r = client.delete("/training_service/456")
    assert r.status_code == 204

def test_delete_404_maps_to_http_404(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_delete_404 = True
    r = client.delete("/training_service/999")
    assert r.status_code == 404
    assert r.json()["detail"] == "Job with id 999 not found"


# ------------------------------ Tests: streaming (single-job) ------------------------------

def test_stream_results_happy_path(client, test_app):
    test_app.state.fake_svc.raise_manifest_not_found = False
    test_app.state.fake_svc.raise_manifest_unsaved = False
    test_app.state.fake_svc.stream_lines = [b'{"a":1}\n', b'{"b":2}\n']

    url = "/training_service/10/results/train/stream"
    with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        body = b"".join(list(resp.iter_bytes()))
    assert body.splitlines() == [b'{"a":1}', b'{"b":2}']


def test_stream_results_404_manifest_not_found(client, test_app):
    test_app.state.fake_svc.raise_manifest_not_found = True
    r = client.get("/training_service/11/results/evaluate/stream")
    assert r.status_code == 404
    assert r.json()["detail"] == "Manifest not found"


def test_stream_results_404_unsaved(client, test_app):
    test_app.state.fake_svc.raise_manifest_unsaved = True
    r = client.get("/training_service/12/results/train/stream")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unsaved data"


def test_stream_env_results_happy_path(client, test_app):
    test_app.state.fake_svc.raise_env_manifest_not_found = False
    url = "/training_service/environments/ALE/Pong-v5/results/evaluate/stream"
    with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        body = b"".join(list(resp.iter_bytes()))
    lines = [json.loads(x) for x in body.decode("utf-8").strip().splitlines()]
    assert all("job_id" in x and "data" in x for x in lines)


def test_stream_env_results_404_when_no_manifest(client, test_app):
    test_app.state.fake_svc.raise_env_manifest_not_found = True

    url = "/training_service/environments/ALE/Breakout-v5/results/train/stream"
    resp = client.get(url)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Manifest not found for env/results"

# ------------------------------ Tests: streaming (env-multiplexed) ------------------------------





# ------------------------------ Tests: supported algos & details (error paths) ------------------------------

def test_supported_algos_404_on_unknown_env(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_env_404 = True
    r = client.get("/training_service/environments/ALE/Unknown-v5/supported_algorithms")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]

def test_algo_details_404(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_algo_404 = True
    r = client.get("/training_service/algorithms/PPO")  # value must match AlgoType parsing
    assert r.status_code == 404
    assert "Algo with id" in r.json()["detail"]


# ------------------------------ Tests: config getters (error paths) ------------------------------

def test_get_log_config_404(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_log_cfg_404 = True
    r = client.get("/training_service/logs/log_config")
    assert r.status_code == 404
    assert "Log config not found" in r.json()["detail"]

def test_get_algo_config_404(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_algo_cfg_404 = True
    r = client.get("/training_service/algorithms/PPO/config")  # ppo is fine as sample
    assert r.status_code == 404
    assert "Algo with id" in r.json()["detail"]

def test_get_general_config_404(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_general_cfg_404 = True
    r = client.get("/training_service/environments/ALE/Pong-v5/general_config")
    assert r.status_code == 404
    assert "General config not found" in r.json()["detail"]


# ------------------------------ Tests: zip download ------------------------------

def test_download_zip_success_headers_and_cleanup(client, test_app):
    r = client.get("/training_service/123/data/zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert 'attachment; filename="123-data.zip"' in r.headers["content-disposition"]
    assert len(r.content) > 0  # got actual bytes
    # Do NOT assert that the temp file/dir still exists; cleanup is expected.


def test_download_zip_404_on_keyerror(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_zip_404_key = True
    r = client.get("/training_service/77/data/zip")
    assert r.status_code == 404

def test_download_zip_404_on_filenotfound(client: TestClient, test_app: FastAPI):
    test_app.state.fake_svc.raise_zip_404_file = True
    r = client.get("/training_service/88/data/zip")
    assert r.status_code == 404
