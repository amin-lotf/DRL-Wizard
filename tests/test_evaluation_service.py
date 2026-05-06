import json
from pathlib import Path

from drl_wizard.backend.schemas.algo_cfg_schema import PPOConfigSchema
from drl_wizard.backend.schemas.general_cfg_schema import GeneralConfigSchema
from drl_wizard.backend.schemas.log_cfg_schema import LogConfigSchema
from drl_wizard.backend.services.training_service import evaluation_service as eval_svc
from drl_wizard.common.video import EncodedVideo


def _write_run(root: Path, run_id: str, env_id: str, with_checkpoint: bool = True) -> Path:
    run_dir = root / run_id
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    app_config = {
        **GeneralConfigSchema(env_id=env_id, run_dir=Path("./runs")).model_dump(mode="json"),
        "is_render": False,
        "algo_cfg": PPOConfigSchema().model_dump(mode="json"),
        "log_cfg": LogConfigSchema().model_dump(mode="json"),
    }
    (run_dir / "configs" / "app.json").write_text(
        json.dumps(app_config, indent=2),
        encoding="utf-8",
    )

    if with_checkpoint:
        (run_dir / "checkpoints" / "best_actor.pt").write_text("actor", encoding="utf-8")
        (run_dir / "checkpoints" / "best_critic.pt").write_text("critic", encoding="utf-8")

    return run_dir


def test_discover_runs_filters_environment_and_skips_broken_runs(tmp_path: Path):
    _write_run(tmp_path, "1", "CartPole-v1")
    _write_run(tmp_path, "2", "Ant-v5")
    (tmp_path / "3").mkdir()

    result = eval_svc.discover_runs_for_environment("CartPole-v1", runs_root=tmp_path)

    assert [run.run_id for run in result.runs] == ["1"]
    assert result.runs[0].checkpoint_label == "best"
    assert any("Skipped run '3'" in warning for warning in result.warnings)


def test_evaluate_saved_run_uses_loaded_config_and_selected_checkpoint(tmp_path: Path, monkeypatch):
    run_dir = _write_run(tmp_path, "7", "CartPole-v1")
    (run_dir / "checkpoints" / "latest_actor.pt").write_text("actor", encoding="utf-8")
    (run_dir / "checkpoints" / "latest_critic.pt").write_text("critic", encoding="utf-8")

    calls = {}

    class FakeRunner:
        def restore(self, is_best=False):
            calls["restore"] = is_best

        def evaluate_model(
            self,
            episodes: int,
            deterministic: bool = False,
            render_first_episode: bool = False,
        ):
            calls["episodes"] = episodes
            calls["deterministic"] = deterministic
            calls["render_first_episode"] = render_first_episode
            return {"average_step_reward": 1.25, "average_episode_reward": 9.5}

        def close(self):
            calls["closed"] = True

    def fake_build_gym_runner(app_cfg, logger=None, checkpoint_dir=None, allow_saving=True):
        calls["env_id"] = app_cfg.env_id
        calls["n_envs"] = app_cfg.n_envs
        calls["n_eval_envs"] = app_cfg.n_eval_envs
        calls["use_eval"] = app_cfg.use_eval
        calls["eval_episodes"] = app_cfg.eval_episodes
        calls["logger"] = logger
        calls["checkpoint_dir"] = checkpoint_dir
        calls["allow_saving"] = allow_saving
        return FakeRunner()

    monkeypatch.setattr(eval_svc, "build_gym_runner", fake_build_gym_runner)

    result = eval_svc.evaluate_saved_run(run_dir, env_id="CartPole-v1", episodes=6)

    assert result.average_step_reward == 1.25
    assert result.average_episode_reward == 9.5
    assert calls == {
        "env_id": "CartPole-v1",
        "n_envs": 1,
        "n_eval_envs": 1,
        "use_eval": False,
        "eval_episodes": 6,
        "logger": None,
        "checkpoint_dir": run_dir / "checkpoints",
        "allow_saving": False,
        "restore": False,
        "episodes": 6,
        "deterministic": True,
        "render_first_episode": False,
        "closed": True,
    }


def test_evaluate_saved_run_returns_rendered_video_when_requested(tmp_path: Path, monkeypatch):
    run_dir = _write_run(tmp_path, "8", "CartPole-v1")
    (run_dir / "checkpoints" / "latest_actor.pt").write_text("actor", encoding="utf-8")
    (run_dir / "checkpoints" / "latest_critic.pt").write_text("critic", encoding="utf-8")

    calls = {}

    class FakeRunner:
        def restore(self, is_best=False):
            calls["restore"] = is_best

        def evaluate_model(
            self,
            episodes: int,
            deterministic: bool = False,
            render_first_episode: bool = False,
        ):
            calls["episodes"] = episodes
            calls["deterministic"] = deterministic
            calls["render_first_episode"] = render_first_episode
            return {
                "average_step_reward": 2.5,
                "average_episode_reward": 11.0,
                "rendered_frames": ["frame-1", "frame-2"],
                "render_fps": 15,
            }

        def close(self):
            calls["closed"] = True

    def fake_build_gym_runner(app_cfg, logger=None, checkpoint_dir=None, allow_saving=True):
        calls["is_render"] = app_cfg.is_render
        return FakeRunner()

    monkeypatch.setattr(eval_svc, "build_gym_runner", fake_build_gym_runner)
    monkeypatch.setattr(
        eval_svc,
        "encode_video",
        lambda frames, fps: EncodedVideo(data=b"video-bytes", mime_type="video/webm"),
    )

    result = eval_svc.evaluate_saved_run(
        run_dir,
        env_id="CartPole-v1",
        episodes=4,
        render=True,
    )

    assert result.average_step_reward == 2.5
    assert result.average_episode_reward == 11.0
    assert result.rendered_video_bytes == b"video-bytes"
    assert result.rendered_video_mime_type == "video/webm"
    assert result.render_warning is None
    assert calls == {
        "is_render": True,
        "restore": False,
        "episodes": 4,
        "deterministic": True,
        "render_first_episode": True,
        "closed": True,
    }


def test_evaluate_saved_run_falls_back_to_non_render_eval_when_render_setup_fails(
    tmp_path: Path,
    monkeypatch,
):
    run_dir = _write_run(tmp_path, "9", "CartPole-v1")
    (run_dir / "checkpoints" / "latest_actor.pt").write_text("actor", encoding="utf-8")
    (run_dir / "checkpoints" / "latest_critic.pt").write_text("critic", encoding="utf-8")

    build_calls = []
    eval_calls = {}

    class FakeRunner:
        def restore(self, is_best=False):
            eval_calls["restore"] = is_best

        def evaluate_model(
            self,
            episodes: int,
            deterministic: bool = False,
            render_first_episode: bool = False,
        ):
            eval_calls["episodes"] = episodes
            eval_calls["deterministic"] = deterministic
            eval_calls["render_first_episode"] = render_first_episode
            return {"average_step_reward": 0.5, "average_episode_reward": 3.0}

        def close(self):
            eval_calls["closed"] = True

    def fake_build_gym_runner(app_cfg, logger=None, checkpoint_dir=None, allow_saving=True):
        build_calls.append(app_cfg.is_render)
        if app_cfg.is_render:
            raise RuntimeError("rgb_array rendering is not supported")
        return FakeRunner()

    monkeypatch.setattr(eval_svc, "build_gym_runner", fake_build_gym_runner)

    result = eval_svc.evaluate_saved_run(
        run_dir,
        env_id="CartPole-v1",
        episodes=2,
        render=True,
    )

    assert result.average_step_reward == 0.5
    assert result.average_episode_reward == 3.0
    assert result.rendered_video_bytes is None
    assert result.render_warning == "Rendering unavailable: rgb_array rendering is not supported"
    assert build_calls == [True, False]
    assert eval_calls == {
        "restore": False,
        "episodes": 2,
        "deterministic": True,
        "render_first_episode": False,
        "closed": True,
    }


def test_discover_prefers_latest_checkpoint_when_available(tmp_path: Path):
    run_dir = _write_run(tmp_path, "4", "CartPole-v1")
    (run_dir / "checkpoints" / "latest_actor.pt").write_text("actor", encoding="utf-8")
    (run_dir / "checkpoints" / "latest_critic.pt").write_text("critic", encoding="utf-8")

    result = eval_svc.discover_runs_for_environment("CartPole-v1", runs_root=tmp_path)

    assert len(result.runs) == 1
    assert result.runs[0].checkpoint_label == "latest"
