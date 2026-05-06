from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from drl_wizard.algorithms.runners.factory import build_gym_runner
from drl_wizard.backend.schemas.app_cfg_schema import AppConfigSchema
from drl_wizard.backend.services.mappers import app_schema_to_domain
from drl_wizard.common.video import DEFAULT_VIDEO_FPS, encode_video
from drl_wizard.common.types import AlgoType
from drl_wizard.configs.app_cfg import AppConfig
from drl_wizard.configs.general_cfg import GeneralConfig


DEFAULT_RUNS_DIR = Path(GeneralConfig().run_dir)

_CHECKPOINT_FILES: dict[AlgoType, tuple[str, ...]] = {
    AlgoType.PPO: ("actor.pt", "critic.pt"),
    AlgoType.A2C: ("actor.pt", "critic.pt"),
    AlgoType.TRPO: ("actor.pt", "critic.pt"),
    AlgoType.DQN: ("actor.pt",),
    AlgoType.SAC: ("actor.pt", "q1_net.pt", "q2_net.pt", "log_alpha.pt"),
}


@dataclass(frozen=True)
class SavedRunSummary:
    run_id: str
    run_dir: Path
    env_id: str
    algo_id: AlgoType
    checkpoint_label: str
    checkpoint_dir: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class SavedRunDetails:
    summary: SavedRunSummary
    env_config: dict[str, Any]
    algo_config: dict[str, Any]
    log_config: dict[str, Any]
    raw_app_config: dict[str, Any]
    app_config: AppConfig


@dataclass(frozen=True)
class RunDiscoveryResult:
    runs: list[SavedRunSummary]
    warnings: list[str]


@dataclass(frozen=True)
class EvaluationSummary:
    average_step_reward: float
    average_episode_reward: float
    rendered_video_bytes: bytes | None = None
    rendered_video_mime_type: str | None = None
    render_warning: str | None = None


def get_environment_key(saved_env_config: dict[str, Any]) -> str | None:
    env_id = saved_env_config.get("env_id")
    if env_id is None:
        return None
    env_key = str(env_id).strip()
    return env_key or None


def discover_runs_for_environment(env_id: str, runs_root: Path | None = None) -> RunDiscoveryResult:
    root = Path(runs_root or DEFAULT_RUNS_DIR)
    if not root.exists():
        return RunDiscoveryResult(runs=[], warnings=[])

    warnings: list[str] = []
    matches: list[SavedRunSummary] = []
    for run_dir in sorted(root.iterdir(), key=_run_sort_key):
        if not run_dir.is_dir() or run_dir.name == "models":
            continue
        try:
            details = load_saved_run(run_dir)
        except Exception as exc:
            if _looks_like_run_directory(run_dir):
                warnings.append(f"Skipped run '{run_dir.name}': {exc}")
            continue
        if details.summary.env_id == env_id:
            matches.append(details.summary)
    return RunDiscoveryResult(runs=matches, warnings=warnings)


def load_saved_run(run_dir: str | Path) -> SavedRunDetails:
    run_path = Path(run_dir)
    app_config_path = run_path / "configs" / "app.json"
    if not app_config_path.exists():
        raise FileNotFoundError(f"Missing config file: {app_config_path}")

    try:
        raw_app_config = json.loads(app_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unreadable config JSON: {exc}") from exc

    env_config = {
        key: value
        for key, value in raw_app_config.items()
        if key not in {"algo_cfg", "log_cfg"}
    }
    algo_config = raw_app_config.get("algo_cfg")
    log_config = raw_app_config.get("log_cfg")
    if not isinstance(algo_config, dict):
        raise ValueError("Missing or invalid algorithm config.")
    if not isinstance(log_config, dict):
        raise ValueError("Missing or invalid log config.")

    env_key = get_environment_key(env_config)
    if env_key is None:
        raise ValueError("Missing environment identifier in saved config.")

    try:
        app_schema = AppConfigSchema.model_validate(raw_app_config)
        app_config = app_schema_to_domain(app_schema)
    except Exception as exc:
        raise ValueError(f"Invalid saved app config: {exc}") from exc

    checkpoint_label, checkpoint_dir, checkpoint_path = _resolve_checkpoint(
        run_path / "checkpoints",
        app_config.algo_cfg.algo_id,
    )

    return SavedRunDetails(
        summary=SavedRunSummary(
            run_id=run_path.name,
            run_dir=run_path,
            env_id=env_key,
            algo_id=app_config.algo_cfg.algo_id,
            checkpoint_label=checkpoint_label,
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
        ),
        env_config=env_config,
        algo_config=algo_config,
        log_config=log_config,
        raw_app_config=raw_app_config,
        app_config=app_config,
    )


def evaluate_saved_run(
    run_dir: str | Path,
    env_id: str,
    episodes: int | None = None,
    render: bool = False,
) -> EvaluationSummary:
    details = load_saved_run(run_dir)
    if details.summary.env_id != env_id:
        raise ValueError(
            f"Run '{details.summary.run_id}' was trained on '{details.summary.env_id}', not '{env_id}'."
        )

    eval_episodes = episodes or details.app_config.eval_episodes
    if eval_episodes <= 0:
        raise ValueError("Evaluation episodes must be greater than zero.")

    render_warning: str | None = None
    render_enabled = bool(render)

    try:
        runner = _build_evaluation_runner(
            details=details,
            env_id=env_id,
            eval_episodes=eval_episodes,
            render=render_enabled,
        )
    except Exception as exc:
        if not render_enabled:
            raise
        render_enabled = False
        render_warning = f"Rendering unavailable: {exc}"
        runner = _build_evaluation_runner(
            details=details,
            env_id=env_id,
            eval_episodes=eval_episodes,
            render=False,
        )

    try:
        runner.restore(is_best=details.summary.checkpoint_label == "best")
        metrics = runner.evaluate_model(
            eval_episodes,
            deterministic=True,
            render_first_episode=render_enabled,
        )
    finally:
        runner.close()

    encoded_video = None
    render_warning = _merge_render_warning(render_warning, metrics.get("render_warning"))
    rendered_frames = metrics.get("rendered_frames") or []
    render_fps = metrics.get("render_fps") or DEFAULT_VIDEO_FPS

    if render_enabled and rendered_frames:
        try:
            encoded_video = encode_video(rendered_frames, fps=int(render_fps))
        except Exception as exc:
            render_warning = _merge_render_warning(render_warning, f"Video encoding failed: {exc}")
    elif render and not render_warning:
        render_warning = "Rendering was requested, but no video was generated."

    return EvaluationSummary(
        average_step_reward=float(metrics["average_step_reward"]),
        average_episode_reward=float(metrics["average_episode_reward"]),
        rendered_video_bytes=encoded_video.data if encoded_video is not None else None,
        rendered_video_mime_type=encoded_video.mime_type if encoded_video is not None else None,
        render_warning=render_warning,
    )


def _build_evaluation_runner(
    details: SavedRunDetails,
    env_id: str,
    eval_episodes: int,
    render: bool,
):
    eval_config = replace(
        details.app_config,
        env_id=env_id,
        n_envs=1,
        n_eval_envs=1,
        use_eval=False,
        is_render=render,
        eval_episodes=eval_episodes,
    )
    return build_gym_runner(
        eval_config,
        logger=None,
        checkpoint_dir=details.summary.checkpoint_dir,
        allow_saving=False,
    )


def _merge_render_warning(current: str | None, extra: Any) -> str | None:
    extra_text = str(extra).strip() if extra is not None else ""
    if not extra_text:
        return current
    if current:
        return f"{current} {extra_text}"
    return extra_text


def _resolve_checkpoint(checkpoint_dir: Path, algo_id: AlgoType) -> tuple[str, Path, Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoint_dir}")

    expected_files = _CHECKPOINT_FILES.get(algo_id)
    if not expected_files:
        raise ValueError(f"Unsupported algorithm for checkpoint discovery: {algo_id}")

    for checkpoint_label in ("latest", "best"):
        checkpoint_files = [checkpoint_dir / f"{checkpoint_label}_{name}" for name in expected_files]
        if all(path.exists() for path in checkpoint_files):
            return checkpoint_label, checkpoint_dir, checkpoint_files[0]

    raise FileNotFoundError(
        f"Missing complete checkpoint set for {algo_id.value} in {checkpoint_dir}"
    )


def _looks_like_run_directory(run_dir: Path) -> bool:
    return run_dir.name.isdigit() or (run_dir / "configs").exists() or (run_dir / "checkpoints").exists()


def _run_sort_key(run_dir: Path):
    if run_dir.name.isdigit():
        return 0, -int(run_dir.name)
    return 1, run_dir.name
