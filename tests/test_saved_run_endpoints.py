from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from drl_wizard.backend.routers import training_route
from drl_wizard.backend.services.training_service.evaluation_service import (
    EvaluationSummary,
    RunDiscoveryResult,
    SavedRunDetails,
    SavedRunSummary,
)
from drl_wizard.common.types import AlgoType
from drl_wizard.configs.algo_cfg import PPOConfig
from drl_wizard.configs.app_cfg import AppConfig
from drl_wizard.configs.general_cfg import GeneralConfig
from drl_wizard.configs.log_cfg import LogConfig


def build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(training_route.router)
    return app


def build_saved_run(run_id: str = "7") -> SavedRunDetails:
    summary = SavedRunSummary(
        run_id=run_id,
        run_dir=Path(f"runs/{run_id}"),
        env_id="CartPole-v1",
        algo_id=AlgoType.PPO,
        checkpoint_label="latest",
        checkpoint_dir=Path(f"runs/{run_id}/checkpoints"),
        checkpoint_path=Path(f"runs/{run_id}/checkpoints/latest_actor.pt"),
    )
    app_config = AppConfig(
        **GeneralConfig(env_id="CartPole-v1").__dict__,
        log_cfg=LogConfig(),
        algo_cfg=PPOConfig(),
    )
    return SavedRunDetails(
        summary=summary,
        env_config={"env_id": "CartPole-v1", "n_envs": 1},
        algo_config={"algo_id": "PPO", "actor_lr": 0.0001},
        log_config={"segment_steps": 50000},
        raw_app_config={"env_id": "CartPole-v1", "algo_cfg": {"algo_id": "PPO"}},
        app_config=app_config,
    )


def test_saved_run_discovery_endpoint(monkeypatch):
    monkeypatch.setattr(
        training_route,
        "discover_runs_for_environment",
        lambda env_id: RunDiscoveryResult(
            runs=[
                build_saved_run("7").summary,
                build_saved_run("8").summary,
            ],
            warnings=["Skipped run 'broken': Missing config file"],
        ),
    )

    client = TestClient(build_app())
    response = client.get("/training_service/saved_runs", params={"env_id": "CartPole-v1"})

    assert response.status_code == 200
    payload = response.json()
    assert [run["run_id"] for run in payload["runs"]] == ["7", "8"]
    assert payload["runs"][0]["algo_id"] == "PPO"
    assert payload["warnings"] == ["Skipped run 'broken': Missing config file"]


def test_saved_run_details_endpoint(monkeypatch):
    monkeypatch.setattr(training_route, "load_saved_run", lambda run_dir: build_saved_run("9"))

    client = TestClient(build_app())
    response = client.get("/training_service/saved_runs/9")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["run_id"] == "9"
    assert payload["summary"]["checkpoint_label"] == "latest"
    assert payload["env_config"]["env_id"] == "CartPole-v1"
    assert payload["algo_config"]["algo_id"] == "PPO"
    assert payload["eval_episodes_default"] == GeneralConfig().eval_episodes


def test_saved_run_evaluation_endpoint_returns_base64_video(monkeypatch):
    monkeypatch.setattr(
        training_route,
        "evaluate_saved_run",
        lambda run_dir, env_id, episodes=None, render=False: EvaluationSummary(
            average_step_reward=1.75,
            average_episode_reward=12.5,
            rendered_video_bytes=b"video-bytes",
            rendered_video_mime_type="video/webm",
            render_warning=None,
        ),
    )

    client = TestClient(build_app())
    response = client.post(
        "/training_service/saved_runs/11/evaluate",
        json={"env_id": "CartPole-v1", "episodes": 3, "render": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["average_step_reward"] == 1.75
    assert payload["average_episode_reward"] == 12.5
    assert payload["rendered_video_mime_type"] == "video/webm"
    assert base64.b64decode(payload["rendered_video_base64"]) == b"video-bytes"
