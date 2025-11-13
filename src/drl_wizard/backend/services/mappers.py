from typing import Optional, cast

from drl_wizard.backend.schemas import JobResponse, EnvResponse
from drl_wizard.backend.schemas.algo_cfg_schema import AlgoConfigSchema, SACConfigSchema, PPOConfigSchema, DQNConfigSchema, \
    A2CConfigSchema, TRPOConfigSchema
from drl_wizard.backend.schemas.algo_schema import AlgoResponse
from drl_wizard.backend.schemas.app_cfg_schema import AppConfigSchema
from drl_wizard.backend.schemas.general_cfg_schema import GeneralConfigSchema
from drl_wizard.backend.schemas.log_cfg_schema import LogConfigSchema
from drl_wizard.backend.schemas.manifest_schema import ManifestSchema, SegmentSchema
from drl_wizard.backend.services.logging.log_manifest import Manifest, Segment
from drl_wizard.backend.services.storage.database import JobResultsModel, JobModel
from drl_wizard.backend.services.training_service.algos import AlgoState
from drl_wizard.backend.services.training_service.environments import EnvironmentState
from drl_wizard.backend.services.training_service.job_results import JobResultState
from drl_wizard.backend.services.training_service.jobs import JobState
from drl_wizard.backend.services.utils import ensure_aware_utc
from drl_wizard.configs.algo_cfg import BaseAlgoConfig, PPOConfig, SACConfig, DQNConfig, A2CConfig, TRPOConfig
from drl_wizard.configs.app_cfg import AppConfig
from drl_wizard.configs.general_cfg import GeneralConfig
from drl_wizard.configs.log_cfg import LogConfig


def to_env_response(env: EnvironmentState) -> EnvResponse:
    return EnvResponse(
        env_id=env.env_id,
        env_name=env.env_name,
        origin=env.origin,
        supported_action=env.supported_action
    )


def to_algo_response(algo: AlgoState) -> AlgoResponse:
    return AlgoResponse(
        algo_id=algo.algo_id,
        algo_name=algo.algo_name,
        action_type=algo.action_type
    )


def to_job_response(state: JobState, env: Optional[EnvironmentState] = None,
                    algo: Optional[AlgoState] = None) -> JobResponse:
    return JobResponse(
        job_id=state.job_id,
        status=state.status,
        env=to_env_response(env) if env else None,
        algo=to_algo_response(algo) if algo else None,
        created_at=ensure_aware_utc(state.created_at),
        started_at=ensure_aware_utc(state.started_at),
        finished_at=ensure_aware_utc(state.finished_at),
        detail=state.detail,
    )


def to_job_state(job_model: JobModel) -> JobState:
    job=JobState(
        job_id=job_model.job_id,
        status=job_model.status,
        algo_id=job_model.algo_id,
        env_id=job_model.env_id,
        created_at=ensure_aware_utc(job_model.created_at),
        started_at=ensure_aware_utc(job_model.started_at),
        finished_at=ensure_aware_utc(job_model.finished_at),
        detail=job_model.detail,
        stop_requested=bool(job_model.stop_requested)
    )
    return job


def to_result_state(job_result: JobResultsModel) -> JobResultState:
    return JobResultState(
        result_id=job_result.id,
        job_id=job_result.job_id,
        created_at=ensure_aware_utc(job_result.created_at),
        result_type=job_result.result_type,
        manifest_uri=job_result.manifest_uri,
        segment_steps=job_result.segment_steps,
        latest_step=job_result.latest_step
    )


def algo_schema_to_domain(s: AlgoConfigSchema) -> BaseAlgoConfig:
    if isinstance(s, PPOConfigSchema):
        return PPOConfig(
            # algo_id fixed in domain, no need to pass
            num_agents=s.num_agents,
            actor_lr=s.actor_lr,
            critic_lr=s.critic_lr,
            lr_gamma=s.lr_gamma,
            opti_eps=s.opti_eps,
            weight_decay=s.weight_decay,
            gamma=s.gamma,
            use_gae=s.use_gae,
            gae_lambda=s.gae_lambda,
            clip_ratio=s.clip_ratio,
            ent_coef=s.ent_coef,
            vf_coef=s.vf_coef,
            episode_length=s.episode_length,
            minibatch_size=s.minibatch_size,
            use_relu=s.use_relu,
            use_orthogonal=s.use_orthogonal,
            cnn_hidden=s.cnn_hidden,
            cnn_kernel_size=s.cnn_kernel_size,
            cnn_stride=s.cnn_stride,
            fc_hidden=s.fc_hidden,
            fc_num_hidden=s.fc_num_hidden,
            use_feature_normalization=s.use_feature_normalization,
            use_clipped_value_loss=s.use_clipped_value_loss,
            use_max_grad_norm=s.use_max_grad_norm,
            use_lr_decay=s.use_lr_decay,
            max_grad_norm=s.max_grad_norm,
            gain=s.gain,
            num_epochs=s.num_epochs,
        )

    elif isinstance(s, TRPOConfigSchema):
        s = cast(TRPOConfigSchema, s)
        return TRPOConfig(
            # algo_id fixed in domain, no need to pass
            num_agents=s.num_agents,
            critic_lr=s.critic_lr,
            lr_gamma=s.lr_gamma,
            opti_eps=s.opti_eps,
            weight_decay=s.weight_decay,
            gamma=s.gamma,
            use_gae=s.use_gae,
            gae_lambda=s.gae_lambda,
            # clip_ratio present for compatibility across UI/paths; domain can ignore if unused
            clip_ratio=s.clip_ratio,
            ent_coef=s.ent_coef,
            vf_coef=s.vf_coef,
            episode_length=s.episode_length,
            minibatch_size=s.minibatch_size,
            use_relu=s.use_relu,
            use_orthogonal=s.use_orthogonal,
            cnn_hidden=s.cnn_hidden,
            cnn_kernel_size=s.cnn_kernel_size,
            cnn_stride=s.cnn_stride,
            fc_hidden=s.fc_hidden,
            fc_num_hidden=s.fc_num_hidden,
            use_feature_normalization=s.use_feature_normalization,
            use_clipped_value_loss=s.use_clipped_value_loss,
            use_max_grad_norm=s.use_max_grad_norm,
            use_lr_decay=s.use_lr_decay,
            max_grad_norm=s.max_grad_norm,
            gain=s.gain,
            num_epochs=s.num_epochs,
            # TRPO-specific
            max_kl=s.max_kl,
            cg_iters=s.cg_iters,
            cg_residual_tol=s.cg_residual_tol,
            damping=s.damping,
            backtrack_coeff=s.backtrack_coeff,
            backtrack_steps=s.backtrack_steps,
            vf_train_iters=s.vf_train_iters,
        )
    elif isinstance(s, SACConfigSchema):
        return SACConfig(
        num_agents = s.num_agents,
        buffer_size = s.buffer_size,
        batch_size = s.batch_size,
        minibatch_size = s.minibatch_size,
        use_relu = s.use_relu,
        use_orthogonal = s.use_orthogonal,
        cnn_hidden = s.cnn_hidden,
        cnn_kernel_size = s.cnn_kernel_size,
        cnn_stride = s.cnn_stride,
        fc_hidden = s.fc_hidden,
        fc_num_hidden = s.fc_num_hidden,
        use_feature_normalization = s.use_feature_normalization,
        use_max_grad_norm = s.use_max_grad_norm,
        use_lr_decay = s.use_lr_decay,
        max_grad_norm = s.max_grad_norm,
        gain = s.gain,
        num_epochs = s.num_epochs,
        actor_lr = s.actor_lr,
        q_lr = s.q_lr,
        alpha_lr = s.alpha_lr,
        lr_gamma = s.lr_gamma,
        opti_eps = s.opti_eps,
        weight_decay = s.weight_decay,
        gamma = s.gamma,
        alpha_init = s.alpha_init,
        target_entropy_scale = s.target_entropy_scale,
        tau = s.tau,
        update_interval = s.update_interval,
        warmup_steps = s.warmup_steps
    )
    if isinstance(s,DQNConfigSchema):
        return DQNConfig(
            num_agents=s.num_agents,
            buffer_size=s.buffer_size,
            batch_size=s.batch_size,
            minibatch_size=s.minibatch_size,
            use_relu=s.use_relu,
            use_orthogonal=s.use_orthogonal,
            cnn_hidden=s.cnn_hidden,
            cnn_kernel_size=s.cnn_kernel_size,
            cnn_stride=s.cnn_stride,
            fc_hidden=s.fc_hidden,
            fc_num_hidden=s.fc_num_hidden,
            use_feature_normalization=s.use_feature_normalization,
            use_max_grad_norm=s.use_max_grad_norm,
            use_lr_decay=s.use_lr_decay,
            max_grad_norm=s.max_grad_norm,
            gain=s.gain,
            num_epochs=s.num_epochs,
            actor_lr=s.actor_lr,
            lr_gamma=s.lr_gamma,
            opti_eps=s.opti_eps,
            weight_decay=s.weight_decay,
            gamma=s.gamma,
            dqn_epsilon_start=s.dqn_epsilon_start,
            dqn_epsilon_end=s.dqn_epsilon_end,
            dqn_epsilon_decay_last_episode=s.dqn_epsilon_decay_last_episode,
            use_dueling=s.use_dueling,
            use_double_dqn=s.use_double_dqn,
            tau=s.tau
        )
    elif isinstance(s, A2CConfigSchema):
        return A2CConfig(
            num_agents=s.num_agents,
            actor_lr=s.actor_lr,
            critic_lr=s.critic_lr,
            lr_gamma=s.lr_gamma,
            opti_eps=s.opti_eps,
            weight_decay=s.weight_decay,
            gamma=s.gamma,
            use_gae=s.use_gae,
            gae_lambda=s.gae_lambda,
            ent_coef=s.ent_coef,
            vf_coef=s.vf_coef,
            episode_length=s.episode_length,
            minibatch_size=s.minibatch_size,
            use_relu=s.use_relu,
            use_orthogonal=s.use_orthogonal,
            cnn_hidden=s.cnn_hidden,
            cnn_kernel_size=s.cnn_kernel_size,
            cnn_stride=s.cnn_stride,
            fc_hidden=s.fc_hidden,
            fc_num_hidden=s.fc_num_hidden,
            use_feature_normalization=s.use_feature_normalization,
            use_max_grad_norm=s.use_max_grad_norm,
            use_lr_decay=s.use_lr_decay,
            max_grad_norm=s.max_grad_norm,
            gain=s.gain,
            num_epochs=s.num_epochs
        )

    raise NotImplementedError(f"Unknown algo schema type: {type(s)}")





def general_schema_to_domain(g: GeneralConfigSchema) -> GeneralConfig:
    return GeneralConfig(
        device=g.device,
        env_id=g.env_id,
        seed=g.seed,
        n_envs=g.n_envs,
        n_eval_envs=g.n_eval_envs,
        total_steps=g.total_steps,
        run_dir=g.run_dir,
        save_interval=g.save_interval,
        log_interval=g.log_interval,
        rescale_frames=g.rescale_frames,
        use_eval=g.use_eval,
        eval_interval=g.eval_interval,
        eval_episodes=g.eval_episodes,
    )


def log_schema_to_domain(l: LogConfigSchema) -> LogConfig:
    return LogConfig(
        segment_steps=l.segment_steps,
        buffer_rows=l.buffer_rows,
        compress=l.compress,
        tb_writer=l.tb_writer
    )


def algo_domain_to_schema(d: BaseAlgoConfig) -> AlgoConfigSchema:
    if isinstance(d, PPOConfig):
        assert isinstance(d, PPOConfig)
        d = cast(PPOConfig, d)
        return PPOConfigSchema(
            num_agents=d.num_agents,
            actor_lr=d.actor_lr,
            critic_lr=d.critic_lr,
            lr_gamma=d.lr_gamma,
            opti_eps=d.opti_eps,
            weight_decay=d.weight_decay,
            gamma=d.gamma,
            use_gae=d.use_gae,
            gae_lambda=d.gae_lambda,
            clip_ratio=d.clip_ratio,
            ent_coef=d.ent_coef,
            vf_coef=d.vf_coef,
            episode_length=d.episode_length,
            minibatch_size=d.minibatch_size,
            use_relu=d.use_relu,
            use_orthogonal=d.use_orthogonal,
            cnn_hidden=d.cnn_hidden,
            cnn_kernel_size=d.cnn_kernel_size,
            cnn_stride=d.cnn_stride,
            fc_hidden=d.fc_hidden,
            fc_num_hidden=d.fc_num_hidden,
            use_feature_normalization=d.use_feature_normalization,
            use_clipped_value_loss=d.use_clipped_value_loss,
            use_max_grad_norm=d.use_max_grad_norm,
            use_lr_decay=d.use_lr_decay,
            max_grad_norm=d.max_grad_norm,
            gain=d.gain,
            num_epochs=d.num_epochs,
        )
    elif isinstance(d, TRPOConfig):
        d = cast(TRPOConfig, d)
        return TRPOConfigSchema(
            num_agents=d.num_agents,
            critic_lr=d.critic_lr,
            lr_gamma=d.lr_gamma,
            opti_eps=d.opti_eps,
            weight_decay=d.weight_decay,
            gamma=d.gamma,
            use_gae=d.use_gae,
            gae_lambda=d.gae_lambda,
            clip_ratio=d.clip_ratio,
            ent_coef=d.ent_coef,
            vf_coef=d.vf_coef,
            episode_length=d.episode_length,
            minibatch_size=d.minibatch_size,
            use_relu=d.use_relu,
            use_orthogonal=d.use_orthogonal,
            cnn_hidden=d.cnn_hidden,
            cnn_kernel_size=d.cnn_kernel_size,
            cnn_stride=d.cnn_stride,
            fc_hidden=d.fc_hidden,
            fc_num_hidden=d.fc_num_hidden,
            use_feature_normalization=d.use_feature_normalization,
            use_clipped_value_loss=d.use_clipped_value_loss,
            use_max_grad_norm=d.use_max_grad_norm,
            use_lr_decay=d.use_lr_decay,
            max_grad_norm=d.max_grad_norm,
            gain=d.gain,
            num_epochs=d.num_epochs,
            max_kl=d.max_kl,
            cg_iters=d.cg_iters,
            cg_residual_tol=d.cg_residual_tol,
            damping=d.damping,
            backtrack_coeff=d.backtrack_coeff,
            backtrack_steps=d.backtrack_steps,
            vf_train_iters=d.vf_train_iters,
        )
    elif isinstance(d, SACConfig):
        assert isinstance(d, SACConfig)
        d = cast(SACConfig, d)
        return SACConfigSchema(
            num_agents=d.num_agents,
            buffer_size=d.buffer_size,
            batch_size=d.batch_size,
            minibatch_size=d.minibatch_size,
            use_relu=d.use_relu,
            use_orthogonal=d.use_orthogonal,
            cnn_hidden=d.cnn_hidden,
            cnn_kernel_size=d.cnn_kernel_size,
            cnn_stride=d.cnn_stride,
            fc_hidden=d.fc_hidden,
            fc_num_hidden=d.fc_num_hidden,
            use_feature_normalization=d.use_feature_normalization,
            use_max_grad_norm=d.use_max_grad_norm,
            use_lr_decay=d.use_lr_decay,
            max_grad_norm=d.max_grad_norm,
            gain=d.gain,
            num_epochs=d.num_epochs,
            actor_lr=d.actor_lr,
            q_lr=d.q_lr,
            alpha_lr=d.alpha_lr,
            lr_gamma=d.lr_gamma,
            opti_eps=d.opti_eps,
            weight_decay=d.weight_decay,
            gamma=d.gamma,
            alpha_init=d.alpha_init,
            target_entropy_scale=d.target_entropy_scale,
            tau=d.tau,
            update_interval=d.update_interval,
            warmup_steps=d.warmup_steps
        )
    elif isinstance(d,DQNConfig):
        assert isinstance(d, DQNConfig)
        d = cast(DQNConfig, d)
        return DQNConfigSchema(
            num_agents=d.num_agents,
            buffer_size=d.buffer_size,
            batch_size=d.batch_size,
            minibatch_size=d.minibatch_size,
            use_relu=d.use_relu,
            use_orthogonal=d.use_orthogonal,
            cnn_hidden=d.cnn_hidden,
            cnn_kernel_size=d.cnn_kernel_size,
            cnn_stride=d.cnn_stride,
            fc_hidden=d.fc_hidden,
            fc_num_hidden=d.fc_num_hidden,
            use_feature_normalization=d.use_feature_normalization,
            use_max_grad_norm=d.use_max_grad_norm,
            use_lr_decay=d.use_lr_decay,
            max_grad_norm=d.max_grad_norm,
            gain=d.gain,
            num_epochs=d.num_epochs,
            actor_lr=d.actor_lr,
            lr_gamma=d.lr_gamma,
            opti_eps=d.opti_eps,
            weight_decay=d.weight_decay,
            gamma=d.gamma,
            dqn_epsilon_start=d.dqn_epsilon_start,
            dqn_epsilon_end=d.dqn_epsilon_end,
            dqn_epsilon_decay_last_episode=d.dqn_epsilon_decay_last_episode,
            use_dueling=d.use_dueling,
            use_double_dqn=d.use_double_dqn,
            tau=d.tau
        )
    elif isinstance(d, A2CConfig):
        assert isinstance(d, A2CConfig)
        d = cast(A2CConfig, d)
        return A2CConfigSchema(
            num_agents=d.num_agents,
            actor_lr=d.actor_lr,
            critic_lr=d.critic_lr,
            lr_gamma=d.lr_gamma,
            opti_eps=d.opti_eps,
            weight_decay=d.weight_decay,
            gamma=d.gamma,
            use_gae=d.use_gae,
            gae_lambda=d.gae_lambda,
            ent_coef=d.ent_coef,
            vf_coef=d.vf_coef,
            episode_length=d.episode_length,
            minibatch_size=d.minibatch_size,
            use_relu=d.use_relu,
            use_orthogonal=d.use_orthogonal,
            cnn_hidden=d.cnn_hidden,
            cnn_kernel_size=d.cnn_kernel_size,
            cnn_stride=d.cnn_stride,
            fc_hidden=d.fc_hidden,
            fc_num_hidden=d.fc_num_hidden,
            use_feature_normalization=d.use_feature_normalization,
            use_max_grad_norm=d.use_max_grad_norm,
            use_lr_decay=d.use_lr_decay,
            max_grad_norm=d.max_grad_norm,
            gain=d.gain,
            num_epochs=d.num_epochs
        )

    raise NotImplementedError(f"Unknown algo domain type: {type(d)}")


def general_domain_to_schema(d: GeneralConfig) -> GeneralConfigSchema:
    return GeneralConfigSchema(
        device=d.device,
        env_id=d.env_id,
        seed=d.seed,
        n_envs=d.n_envs,
        n_eval_envs=d.n_eval_envs,
        total_steps=d.total_steps,
        rescale_frames=d.rescale_frames,
        run_dir=d.run_dir,
        save_interval=d.save_interval,
        log_interval=d.log_interval,
        use_eval=d.use_eval,
        eval_interval=d.eval_interval,
        eval_episodes=d.eval_episodes,
    )


def log_domain_to_schema(d: LogConfig) -> LogConfigSchema:
    return LogConfigSchema(
        segment_steps=d.segment_steps,
        buffer_rows=d.buffer_rows,
        compress=d.compress,
        tb_writer=d.tb_writer
    )


def app_schema_to_domain(s: AppConfigSchema) -> AppConfig:
    return AppConfig(
        # AppDomain inherits GeneralDomain fields
        **general_schema_to_domain(s).__dict__,
        algo_cfg=algo_schema_to_domain(s.algo_cfg),
        log_cfg=log_schema_to_domain(s.log_cfg),
    )



def manifest_domain_to_schema(m: Manifest) -> ManifestSchema:
    return ManifestSchema(
        job_id=m.job_id,
        path=m.path,
        log_path=m.log_path,
        configs_path=m.configs_path,
        checkpoints_path=m.checkpoints_path,
        created_at=ensure_aware_utc(m.created_at),
        schema_version=m.schema_version,
        segments={k: [SegmentSchema(**s.__dict__) for s in v] for k, v in m.segments.items()},
    )

def manifest_schema_to_domain(s: ManifestSchema) -> Manifest:
    return Manifest(
        job_id=s.job_id,
        path=s.path,
        log_path=s.log_path,
        configs_path=s.configs_path,
        checkpoints_path=s.checkpoints_path,
        created_at=ensure_aware_utc(s.created_at),
        schema_version=s.schema_version,
        segments={k: [Segment(**seg.model_dump()) for seg in v] for k, v in s.segments.items()},
    )
