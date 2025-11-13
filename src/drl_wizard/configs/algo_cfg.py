from pydantic.dataclasses import dataclass
from dataclasses import field
from drl_wizard.common.types import AlgoType
from drl_wizard.configs.extras import register_algo


# ---------- Base ----------
@dataclass
class BaseAlgoConfig:
    # no Literal here (keep domain free of transport concerns)
    algo_id: AlgoType

# ---------- PPO (domain) ----------
@register_algo
@dataclass
class PPOConfig(BaseAlgoConfig):
    # class-level discriminator value (used by registry)
    algo_id: AlgoType = field(default=AlgoType.PPO, init=False)
    num_agents: int = field(default=1)
    actor_lr: float = field(default=3e-4)
    critic_lr: float = field(default=4e-4)
    lr_gamma: float = field(default=0.99)
    opti_eps: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    gamma: float = field(default=0.99)
    use_gae: bool = field(default=True)
    gae_lambda: float = field(default=0.95)
    clip_ratio: float = field(default=0.2)
    ent_coef: float = field(default=0.01)
    vf_coef: float = field(default=0.5)
    episode_length: int = field(default=256)
    minibatch_size: int = field(default=128)
    use_relu: bool = field(default=True)
    use_orthogonal: bool = field(default=True)
    cnn_hidden: int = field(default=256)
    cnn_kernel_size: int = field(default=3)
    cnn_stride: int = field(default=1)
    fc_hidden: int = field(default=64)
    fc_num_hidden: int = field(default=2)
    use_feature_normalization: bool = field(default=True)
    use_clipped_value_loss: bool = field(default=True)
    use_max_grad_norm: bool = field(default=True)
    use_lr_decay: bool = field(default=True)
    max_grad_norm: float = field(default=0.5)
    gain: float = field(default=0.01)
    num_epochs: int = field(default=5)

    def __post_init__(self):
        if self.episode_length <= 0 or self.minibatch_size <= 0:
            raise ValueError("episode_length and minibatch_size must be > 0")
        if self.episode_length % self.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")
        # add more range checks as you like



# ---------- SAC (domain) ----------
@register_algo
@dataclass
class SACConfig(BaseAlgoConfig):
    # class-level discriminator value (used by registry)
    algo_id: AlgoType = field(default=AlgoType.SAC, init=False)
    num_agents: int = field(default=1)
    buffer_size:int = field(default=1000000)
    batch_size: int = field(default=256)
    minibatch_size : int = field(default=256)
    use_relu: bool = field(default=True)
    use_orthogonal: bool = field(default=True)
    cnn_hidden: int = field(default=256)
    cnn_kernel_size: int = field(default=3)
    cnn_stride: int = field(default=1)
    fc_hidden: int = field(default=256)
    fc_num_hidden: int = field(default=2)
    use_feature_normalization: bool = field(default=False)
    use_max_grad_norm: bool = field(default=False)
    use_lr_decay: bool = field(default=True)
    max_grad_norm: float = field(default=10)
    gain: float = field(default=0.01)
    num_epochs: int = field(default=1)
    actor_lr: float = field(default=3e-4)
    q_lr: float = field(default=3e-4)
    alpha_lr: float = field(default=3e-4)
    lr_gamma: float = field(default=0.99)
    opti_eps: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    gamma: float = field(default=0.99)
    alpha_init:float = field(default=0.2)
    target_entropy_scale: float = field(default=-1.0)
    tau: float = field(default=0.005)
    update_interval:int = field(default=20)
    warmup_steps:int = field(default=1000)



# ---------- DQN (domain) ----------
@register_algo
@dataclass
class DQNConfig(BaseAlgoConfig):
    # class-level discriminator value (used by registry)
    algo_id: AlgoType = field(default=AlgoType.DQN, init=False)
    num_agents: int = field(default=1)
    use_dueling: bool = field(default=True)
    use_double_dqn: bool = field(default=True)
    dqn_epsilon_start: float = field(default=1.0)
    dqn_epsilon_end: float = field(default=0.01)
    dqn_epsilon_decay_last_episode: float = field(default=50000)
    buffer_size:int = field(default=30000)
    batch_size: int = field(default=128)
    minibatch_size : int = field(default=128)
    lr_gamma: float = field(default=0.99)
    use_relu: bool = field(default=True)
    use_orthogonal: bool = field(default=True)
    tau: float = field(default=0.005)
    cnn_hidden: int = field(default=256)
    cnn_kernel_size: int = field(default=3)
    cnn_stride: int = field(default=1)
    fc_hidden: int = field(default=64)
    fc_num_hidden: int = field(default=1)
    use_feature_normalization: bool = field(default=False)
    use_max_grad_norm: bool = field(default=False)
    use_lr_decay: bool = field(default=True)
    max_grad_norm: float = field(default=10)
    gain: float = field(default=0.01)
    num_epochs: int = field(default=1)
    actor_lr: float = field(default=1e-3)
    opti_eps: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    gamma: float = field(default=0.99)
    update_interval:int = field(default=1)
    warmup_steps:int = field(default=1000)




# ---------- A2C (domain) ----------
@register_algo
@dataclass
class A2CConfig(BaseAlgoConfig):
    # class-level discriminator value (used by registry)
    algo_id: AlgoType = field(default=AlgoType.A2C, init=False)
    num_agents: int = field(default=1)
    actor_lr: float = field(default=1e-4)
    critic_lr: float = field(default=2e-4)
    lr_gamma: float = field(default=0.99)
    opti_eps: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    gamma: float = field(default=0.99)
    use_gae: bool = field(default=True)
    gae_lambda: float = field(default=0.95)
    ent_coef: float = field(default=0.01)
    clip_ratio: float = field(default=0.2)
    vf_coef: float = field(default=0.5)
    episode_length: int = field(default=128)
    minibatch_size: int = field(default=64)
    use_relu: bool = field(default=True)
    use_orthogonal: bool = field(default=True)
    cnn_hidden: int = field(default=256)
    cnn_kernel_size: int = field(default=3)
    cnn_stride: int = field(default=1)
    fc_hidden: int = field(default=64)
    fc_num_hidden: int = field(default=1)
    use_feature_normalization: bool = field(default=True)
    use_clipped_value_loss: bool = field(default=True)
    use_max_grad_norm: bool = field(default=True)
    use_lr_decay: bool = field(default=True)
    max_grad_norm: float = field(default=0.5)
    gain: float = field(default=0.01)
    num_epochs: int = field(default=5)

    def __post_init__(self):
        if self.episode_length <= 0 or self.minibatch_size <= 0:
            raise ValueError("episode_length and minibatch_size must be > 0")
        if self.episode_length % self.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")
        # add more range checks as you like




# ---------- TRPO (domain) ----------
@register_algo
@dataclass
class TRPOConfig(BaseAlgoConfig):
    # class-level discriminator value (used by registry)
    algo_id: AlgoType = field(default=AlgoType.TRPO, init=False)
    num_agents: int = field(default=1)
    critic_lr: float = field(default=4e-4)
    lr_gamma: float = field(default=0.99)
    opti_eps: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    gamma: float = field(default=0.99)
    use_gae: bool = field(default=True)
    gae_lambda: float = field(default=0.95)
    clip_ratio: float = field(default=0.2)
    ent_coef: float = field(default=0.01)
    vf_coef: float = field(default=0.5)
    episode_length: int = field(default=256)
    minibatch_size: int = field(default=128)
    use_relu: bool = field(default=True)
    use_orthogonal: bool = field(default=True)
    cnn_hidden: int = field(default=256)
    cnn_kernel_size: int = field(default=3)
    cnn_stride: int = field(default=1)
    fc_hidden: int = field(default=64)
    fc_num_hidden: int = field(default=2)
    use_feature_normalization: bool = field(default=True)
    use_clipped_value_loss: bool = field(default=True)
    use_max_grad_norm: bool = field(default=True)
    use_lr_decay: bool = field(default=True)
    max_grad_norm: float = field(default=0.5)
    gain: float = field(default=0.01)
    num_epochs: int = field(default=5)
    max_kl: float = 1e-2  # δ (typical 0.5e-2 ... 2e-2)
    cg_iters: int = 10
    cg_residual_tol: float = 1e-10
    damping: float = 0.1  # Fisher damping to stabilize HVP
    backtrack_coeff: float = 0.8
    backtrack_steps: int = 10
    vf_train_iters: int = 1  # value regression iters per minibatch (you can keep 1 as in PPO)

    def __post_init__(self):
        if self.episode_length <= 0 or self.minibatch_size <= 0:
            raise ValueError("episode_length and minibatch_size must be > 0")
        if self.episode_length % self.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")
        # add more range checks as you like
