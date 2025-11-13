# api/schemas/algo.py  (transport)
from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field
from drl_wizard.common.types import AlgoType

class BaseAlgoConfigSchema(BaseModel):
    algo_id: AlgoType

class PPOConfigSchema(BaseAlgoConfigSchema):
    algo_id: Literal[AlgoType.PPO] = AlgoType.PPO

    num_agents: int = Field(
        default=1, gt=0,
        title="Number of Agents",
        description="Number of agents (for multi-agent or parallel actor setups)."
    )
    actor_lr: float = Field(
        default=1e-4, gt=0.0, le=1.0,
        title="Actor Learning Rate",
        description="Learning rate for the actor (policy) network."
    )
    critic_lr: float = Field(
        default=2e-4, gt=0.0, le=1.0,
        title="Critic Learning Rate",
        description="Learning rate for the critic (value) network."
    )
    lr_gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="LR Decay Gamma",
        description="Decay factor for learning rate scheduling."
    )
    opti_eps: float = Field(
        default=1e-5, ge=0.0,
        title="Optimizer Epsilon",
        description="Small epsilon to improve optimizer numerical stability."
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        title="Weight Decay",
        description="L2 regularization coefficient."
    )
    gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="Discount Factor (Gamma)",
        description="Discount factor for future rewards."
    )
    use_gae: bool = Field(
        default=True,
        title="Use GAE",
        description="Enable Generalized Advantage Estimation (GAE)."
    )
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0,
        title="GAE Lambda",
        description="Lambda parameter for GAE advantage calculation."
    )
    clip_ratio: float = Field(
        default=0.2, gt=0.0,
        title="Clip Ratio",
        description="Clipping range for the PPO objective (policy ratio)."
    )
    ent_coef: float = Field(
        default=0.01, ge=0.0,
        title="Entropy Coefficient",
        description="Weight for entropy term to encourage exploration."
    )
    vf_coef: float = Field(
        default=0.5, ge=0.0,
        title="Value Function Coefficient",
        description="Weight for critic (value) loss term."
    )
    episode_length: int = Field(
        default=128, gt=0,
        title="Episode Length",
        description="Number of steps per rollout before optimization."
    )
    minibatch_size: int = Field(
        default=64, gt=0,
        title="Mini-batch Size",
        description="Number of samples per PPO update batch."
    )
    use_relu: bool = Field(
        default=True,
        title="Use ReLU Activations",
        description="Use ReLU activation instead of Tanh."
    )
    use_orthogonal: bool = Field(
        default=True,
        title="Use Orthogonal Initialization",
        description="Apply orthogonal initialization to network layers."
    )
    cnn_hidden: int = Field(
        default=256, gt=0,
        title="CNN Hidden Channels",
        description="Number of output channels for CNN feature extractor."
    )
    cnn_kernel_size: int = Field(
        default=3, gt=0,
        title="CNN Kernel Size",
        description="Kernel size for CNN convolutions."
    )
    cnn_stride: int = Field(
        default=1, gt=0,
        title="CNN Stride",
        description="Stride for CNN convolutions."
    )
    fc_hidden: int = Field(
        default=64, gt=0,
        title="Fully Connected Hidden Size",
        description="Number of hidden units per fully connected layer."
    )
    fc_num_hidden: int = Field(
        default=1, gt=0,
        title="Number of FC Hidden Layers",
        description="How many hidden layers in the fully connected network."
    )
    use_feature_normalization: bool = Field(
        default=True,
        title="Feature Normalization",
        description="Normalize extracted features before feeding to the network."
    )
    use_clipped_value_loss: bool = Field(
        default=True,
        title="Use Clipped Value Loss",
        description="Clip the value loss term to stabilize training."
    )
    use_max_grad_norm: bool = Field(
        default=True,
        title="Use Gradient Clipping",
        description="Enable max gradient norm clipping for stability."
    )
    use_lr_decay: bool = Field(
        default=True,
        title="Use LR Decay",
        description="Enable linear learning rate decay over time."
    )
    max_grad_norm: float = Field(
        default=0.5, gt=0.0,
        title="Max Gradient Norm",
        description="Maximum gradient norm (for clipping)."
    )
    gain: float = Field(
        default=0.01, gt=0.0,
        title="Gain Initialization Scale",
        description="Gain used for layer initialization."
    )
    num_epochs: int = Field(
        default=5, gt=0,
        title="Number of Epochs",
        description="Number of PPO update epochs per rollout."
    )

    # Cross-field validation (same as before)
    @classmethod
    def model_validate(cls, obj, *a, **kw):
        m = super().model_validate(obj, *a, **kw)
        if m.episode_length % m.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")
        return m


class TRPOConfigSchema(BaseAlgoConfigSchema):
    algo_id: Literal[AlgoType.TRPO] = AlgoType.TRPO

    # --- Shared/regular RL knobs (mirroring PPO style where applicable) ---
    num_agents: int = Field(
        default=1, gt=0,
        title="Number of Agents",
        description="Number of agents (for multi-agent or parallel actor setups)."
    )
    critic_lr: float = Field(
        default=4e-4, gt=0.0, le=1.0,
        title="Critic Learning Rate",
        description="Learning rate for the critic (value) network."
    )
    lr_gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="LR Decay Gamma",
        description="Decay factor for learning rate scheduling."
    )
    opti_eps: float = Field(
        default=1e-5, ge=0.0,
        title="Optimizer Epsilon",
        description="Small epsilon to improve optimizer numerical stability."
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        title="Weight Decay",
        description="L2 regularization coefficient."
    )
    gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="Discount Factor (Gamma)",
        description="Discount factor for future rewards."
    )
    use_gae: bool = Field(
        default=True,
        title="Use GAE",
        description="Enable Generalized Advantage Estimation (GAE)."
    )
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0,
        title="GAE Lambda",
        description="Lambda parameter for GAE advantage calculation."
    )
    # Note: TRPO doesn't use PPO ratio clipping, but keeping this if your codebase expects it.
    clip_ratio: float = Field(
        default=0.2, gt=0.0,
        title="Clip Ratio (compat)",
        description="Compatibility field kept for shared code paths; not used by core TRPO update."
    )
    ent_coef: float = Field(
        default=0.01, ge=0.0,
        title="Entropy Coefficient",
        description="Weight for entropy term to encourage exploration."
    )
    vf_coef: float = Field(
        default=0.5, ge=0.0,
        title="Value Function Coefficient",
        description="Weight for critic (value) loss term."
    )
    episode_length: int = Field(
        default=256, gt=0,
        title="Episode Length",
        description="Number of steps per rollout before optimization."
    )
    minibatch_size: int = Field(
        default=128, gt=0,
        title="Mini-batch Size",
        description="Number of samples per update batch."
    )
    use_relu: bool = Field(
        default=True,
        title="Use ReLU Activations",
        description="Use ReLU activation instead of Tanh."
    )
    use_orthogonal: bool = Field(
        default=True,
        title="Use Orthogonal Initialization",
        description="Apply orthogonal initialization to network layers."
    )
    cnn_hidden: int = Field(
        default=256, gt=0,
        title="CNN Hidden Channels",
        description="Number of output channels for CNN feature extractor."
    )
    cnn_kernel_size: int = Field(
        default=3, gt=0,
        title="CNN Kernel Size",
        description="Kernel size for CNN convolutions."
    )
    cnn_stride: int = Field(
        default=1, gt=0,
        title="CNN Stride",
        description="Stride for CNN convolutions."
    )
    fc_hidden: int = Field(
        default=64, gt=0,
        title="Fully Connected Hidden Size",
        description="Number of hidden units per fully connected layer."
    )
    fc_num_hidden: int = Field(
        default=2, gt=0,
        title="Number of FC Hidden Layers",
        description="How many hidden layers in the fully connected network."
    )
    use_feature_normalization: bool = Field(
        default=True,
        title="Feature Normalization",
        description="Normalize extracted features before feeding to the network."
    )
    use_clipped_value_loss: bool = Field(
        default=True,
        title="Use Clipped Value Loss",
        description="Clip the value loss term to stabilize training."
    )
    use_max_grad_norm: bool = Field(
        default=True,
        title="Use Gradient Clipping",
        description="Enable max gradient norm clipping for stability."
    )
    use_lr_decay: bool = Field(
        default=True,
        title="Use LR Decay",
        description="Enable linear learning rate decay over time."
    )
    max_grad_norm: float = Field(
        default=0.5, gt=0.0,
        title="Max Gradient Norm",
        description="Maximum gradient norm (for clipping)."
    )
    gain: float = Field(
        default=0.01, gt=0.0,
        title="Gain Initialization Scale",
        description="Gain used for layer initialization."
    )
    num_epochs: int = Field(
        default=5, gt=0,
        title="Number of Epochs",
        description="Number of update epochs per rollout (kept for compatibility; often 1 for TRPO)."
    )

    # --- TRPO-specific knobs ---
    max_kl: float = Field(
        default=1e-2, gt=0.0, le=1.0,
        title="Max KL Divergence",
        description="Trust-region step size constraint (typ. 5e-3 to 2e-2)."
    )
    cg_iters: int = Field(
        default=10, gt=0,
        title="Conjugate Gradient Iterations",
        description="Number of CG iterations to approximate the natural gradient."
    )
    cg_residual_tol: float = Field(
        default=1e-10, gt=0.0,
        title="CG Residual Tolerance",
        description="Stopping tolerance for the CG solver residual."
    )
    damping: float = Field(
        default=0.1, ge=0.0,
        title="Fisher Damping",
        description="Damping added to the Fisher-vector product for stability."
    )
    backtrack_coeff: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        title="Backtracking Coefficient",
        description="Multiplier for step-size during line search backtracking."
    )
    backtrack_steps: int = Field(
        default=10, ge=1,
        title="Backtracking Steps",
        description="Max number of backtracking reductions in line search."
    )
    vf_train_iters: int = Field(
        default=1, ge=1,
        title="Value Function Train Iters",
        description="Number of value regression iterations per minibatch."
    )

    # Cross-field validation
    @classmethod
    def model_validate(cls, obj, *a, **kw):
        m = super().model_validate(obj, *a, **kw)
        if m.episode_length % m.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")
        # (Optional) Additional sanity checks that reflect TRPO practice:
        if not (0.0 < m.backtrack_coeff < 1.0):
            raise ValueError("backtrack_coeff must be in (0, 1).")
        return m


class SACConfigSchema(BaseAlgoConfigSchema):
    algo_id: Literal[AlgoType.SAC] = AlgoType.SAC

    # --- High-level training/layout ---
    num_agents: int = Field(
        default=1, gt=0,
        title="Number of Agents",
        description="Number of agents (for multi-agent or parallel actor setups)."
    )
    buffer_size: int = Field(
        default=1_000_000, gt=0,
        title="Replay Buffer Size",
        description="Maximum number of transitions stored in the replay buffer."
    )
    batch_size: int = Field(
        default=256, gt=0,
        title="Batch Size",
        description="Number of transitions sampled from the replay buffer per gradient update."
    )
    minibatch_size: int = Field(
        default=256, gt=0,
        title="Mini-batch Size",
        description="Size of each mini-batch if splitting one update into multiple mini-batches. "
                    "Must divide batch_size."
    )
    num_epochs: int = Field(
        default=1, gt=0,
        title="Gradient Passes per Update",
        description="How many passes/epochs over the sampled batch per update call."
    )
    update_interval: int = Field(
        default=20, gt=0,
        title="Update Interval (env steps)",
        description="Number of environment steps between training updates."
    )
    warmup_steps: int = Field(
        default=1000, gt=0,
        title="Warmup Steps",
        description="Number of initial environment steps collected before starting updates."
    )

    # --- Networks / architecture ---
    use_relu: bool = Field(
        default=True,
        title="Use ReLU Activations",
        description="Use ReLU activation (otherwise Tanh) in MLP/CNN blocks."
    )
    use_orthogonal: bool = Field(
        default=True,
        title="Use Orthogonal Initialization",
        description="Apply orthogonal initialization to network layers."
    )
    cnn_hidden: int = Field(
        default=256, gt=0,
        title="CNN Hidden Channels",
        description="Number of output channels for CNN feature extractor."
    )
    cnn_kernel_size: int = Field(
        default=3, gt=0,
        title="CNN Kernel Size",
        description="Kernel size for CNN convolutions."
    )
    cnn_stride: int = Field(
        default=1, gt=0,
        title="CNN Stride",
        description="Stride for CNN convolutions."
    )
    fc_hidden: int = Field(
        default=256, gt=0,
        title="Fully Connected Hidden Size",
        description="Number of hidden units per fully connected layer."
    )
    fc_num_hidden: int = Field(
        default=2, gt=0,
        title="Number of FC Hidden Layers",
        description="How many hidden layers in the fully connected network."
    )
    use_feature_normalization: bool = Field(
        default=False,
        title="Feature Normalization",
        description="Normalize extracted features before feeding to the network."
    )

    # --- Optimization ---
    actor_lr: float = Field(
        default=3e-4, gt=0.0, le=1.0,
        title="Actor Learning Rate",
        description="Learning rate for the policy network."
    )
    q_lr: float = Field(
        default=3e-4, gt=0.0, le=1.0,
        title="Q-Network Learning Rate",
        description="Learning rate for the Q/critic networks."
    )
    alpha_lr: float = Field(
        default=3e-4, gt=0.0, le=1.0,
        title="Entropy Coef Learning Rate",
        description="Learning rate for the temperature (alpha) in entropy tuning."
    )
    lr_gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="LR Decay Gamma",
        description="Decay factor for learning rate scheduling (if enabled)."
    )
    opti_eps: float = Field(
        default=1e-5, ge=0.0,
        title="Optimizer Epsilon",
        description="Small epsilon to improve optimizer numerical stability."
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        title="Weight Decay",
        description="L2 regularization coefficient."
    )
    use_max_grad_norm: bool = Field(
        default=False,
        title="Use Gradient Clipping",
        description="Enable max gradient norm clipping for stability."
    )
    max_grad_norm: float = Field(
        default=10.0, gt=0.0,
        title="Max Gradient Norm",
        description="Maximum gradient norm when clipping is enabled."
    )
    use_lr_decay: bool = Field(
        default=True,
        title="Use LR Decay",
        description="Enable linear/exponential learning rate decay over time."
    )
    gain: float = Field(
        default=0.01, gt=0.0,
        title="Gain Initialization Scale",
        description="Gain used for layer initialization."
    )

    # --- SAC specifics ---
    gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="Discount Factor (Gamma)",
        description="Discount factor for future returns."
    )
    alpha_init: float = Field(
        default=0.2, gt=0.0,
        title="Initial Entropy Coefficient (alpha)",
        description="Initial temperature; used when starting automatic entropy tuning."
    )
    target_entropy_scale: float = Field(
        default=-1.0,
        title="Target Entropy Scale",
        description="Multiplier for target entropy: target_entropy = scale × action_dim. "
                    "Use -1.0 for the common heuristic of -|A|."
    )
    tau: float = Field(
        default=0.005, gt=0.0, le=1.0,
        title="Soft Update Coefficient (tau)",
        description="Polyak averaging coefficient for target network updates."
    )

    # --- Cross-field validation (mirrors the PPO style) ---
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        m = super().model_validate(obj, *args, **kwargs)

        # batch_size must be divisible by minibatch_size (if you split updates)
        if m.batch_size % m.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")

        # warmup must be less than buffer capacity
        if m.warmup_steps >= m.buffer_size:
            raise ValueError("warmup_steps must be < buffer_size")

        # if lr decay is enabled, gamma < 1.0 makes sense (else it's no-op or growth)
        if m.use_lr_decay and not (0.0 <= m.lr_gamma < 1.0):
            raise ValueError("With use_lr_decay=True, lr_gamma should be in [0.0, 1.0)")

        # gradient clipping requirements
        if m.use_max_grad_norm and m.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be > 0.0 when use_max_grad_norm=True")

        # tau must be in (0,1]
        if not (0.0 < m.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")

        return m


class DQNConfigSchema(BaseAlgoConfigSchema):
    algo_id: Literal[AlgoType.DQN] = AlgoType.DQN

    # --- High-level / rollout ---
    num_agents: int = Field(
        default=1, gt=0,
        title="Number of Agents",
        description="Number of agents (for multi-agent or parallel actor setups)."
    )
    buffer_size: int = Field(
        default=1_000_000, gt=0,
        title="Replay Buffer Size",
        description="Maximum number of transitions stored in the replay buffer."
    )
    batch_size: int = Field(
        default=256, gt=0,
        title="Batch Size",
        description="Number of transitions sampled from the replay buffer per update."
    )
    minibatch_size: int = Field(
        default=256, gt=0,
        title="Mini-batch Size",
        description="Size of each mini-batch if you split an update into multiple mini-batches. "
                    "Must divide batch_size."
    )
    num_epochs: int = Field(
        default=1, gt=0,
        title="Gradient Passes per Update",
        description="How many passes/epochs over the sampled batch per update call."
    )
    update_interval: int = Field(
        default=1, gt=0,
        title="Update Interval (env steps)",
        description="Number of environment steps between training updates."
    )
    warmup_steps: int = Field(
        default=1000, gt=0,
        title="Warmup Steps",
        description="Number of initial environment steps collected before starting updates."
    )

    # --- Exploration (ε-greedy) ---
    dqn_epsilon_start: float = Field(
        default=1.0, ge=0.0, le=1.0,
        title="Epsilon Start",
        description="Initial ε for ε-greedy exploration."
    )
    dqn_epsilon_end: float = Field(
        default=0.01, ge=0.0, le=1.0,
        title="Epsilon End",
        description="Final ε after decay completes."
    )
    dqn_epsilon_decay_last_episode: float = Field(
        default=10_000, gt=0.0,
        title="Epsilon Decay Horizon",
        description="Number of steps/episodes over which ε decays from start to end."
    )

    # --- DQN variants ---
    use_dueling: bool = Field(
        default=False,
        title="Use Dueling DQN",
        description="Use dueling architecture (separate value and advantage streams)."
    )
    use_double_dqn: bool = Field(
        default=False,
        title="Use Double DQN",
        description="Use Double DQN target (reduces overestimation bias)."
    )

    # --- Networks / architecture ---
    use_relu: bool = Field(
        default=True,
        title="Use ReLU Activations",
        description="Use ReLU activation (otherwise Tanh) in MLP/CNN blocks."
    )
    use_orthogonal: bool = Field(
        default=True,
        title="Use Orthogonal Initialization",
        description="Apply orthogonal initialization to network layers."
    )
    cnn_hidden: int = Field(
        default=256, gt=0,
        title="CNN Hidden Channels",
        description="Number of output channels for CNN feature extractor."
    )
    cnn_kernel_size: int = Field(
        default=3, gt=0,
        title="CNN Kernel Size",
        description="Kernel size for CNN convolutions."
    )
    cnn_stride: int = Field(
        default=1, gt=0,
        title="CNN Stride",
        description="Stride for CNN convolutions."
    )
    fc_hidden: int = Field(
        default=64, gt=0,
        title="Fully Connected Hidden Size",
        description="Number of hidden units per fully connected layer."
    )
    fc_num_hidden: int = Field(
        default=1, gt=0,
        title="Number of FC Hidden Layers",
        description="How many hidden layers in the fully connected network."
    )
    use_feature_normalization: bool = Field(
        default=False,
        title="Feature Normalization",
        description="Normalize extracted features before feeding to the network."
    )

    # --- Optimization ---
    actor_lr: float = Field(
        default=1e-3, gt=0.0, le=1.0,
        title="Q-Network Learning Rate",
        description="Learning rate for the Q-network (named 'actor_lr' for parity with other algos)."
    )
    lr_gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="LR Decay Gamma",
        description="Decay factor for learning rate scheduling (if enabled)."
    )
    opti_eps: float = Field(
        default=1e-5, ge=0.0,
        title="Optimizer Epsilon",
        description="Small epsilon to improve optimizer numerical stability."
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        title="Weight Decay",
        description="L2 regularization coefficient."
    )
    use_max_grad_norm: bool = Field(
        default=False,
        title="Use Gradient Clipping",
        description="Enable max gradient norm clipping for stability."
    )
    max_grad_norm: float = Field(
        default=10.0, gt=0.0,
        title="Max Gradient Norm",
        description="Maximum gradient norm when clipping is enabled."
    )
    use_lr_decay: bool = Field(
        default=True,
        title="Use LR Decay",
        description="Enable linear/exponential learning rate decay over time."
    )
    gain: float = Field(
        default=0.01, gt=0.0,
        title="Gain Initialization Scale",
        description="Gain used for layer initialization."
    )

    # --- RL specifics ---
    gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="Discount Factor (Gamma)",
        description="Discount factor for future returns."
    )
    tau: float = Field(
        default=0.005, gt=0.0, le=1.0,
        title="Soft Update Coefficient (tau)",
        description="Polyak averaging coefficient for target network updates. "
                    "Set near 1.0 for hard/instant updates if your implementation interprets τ=1 as hard copy."
    )

    # --- Cross-field validation (parity with PPO/SAC style) ---
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        m = super().model_validate(obj, *args, **kwargs)

        # minibatch should divide batch_size
        if m.batch_size % m.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")

        # warmup less than buffer capacity
        if m.warmup_steps >= m.buffer_size:
            raise ValueError("warmup_steps must be < buffer_size")

        # lr decay gamma sensible when enabled
        if m.use_lr_decay and not (0.0 <= m.lr_gamma < 1.0):
            raise ValueError("With use_lr_decay=True, lr_gamma should be in [0.0, 1.0)")

        # gradient clipping requirements
        if m.use_max_grad_norm and m.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be > 0.0 when use_max_grad_norm=True")

        # epsilon schedule sanity
        if m.dqn_epsilon_start < m.dqn_epsilon_end:
            raise ValueError("dqn_epsilon_start must be >= dqn_epsilon_end")
        if not (0.0 <= m.dqn_epsilon_start <= 1.0) or not (0.0 <= m.dqn_epsilon_end <= 1.0):
            raise ValueError("epsilon values must be in [0, 1]")

        # tau in (0,1]
        if not (0.0 < m.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")

        return m


class A2CConfigSchema(BaseAlgoConfigSchema):
    algo_id: Literal[AlgoType.A2C] = AlgoType.A2C

    # --- High-level / rollout ---
    num_agents: int = Field(
        default=1, gt=0,
        title="Number of Agents",
        description="Number of agents (for multi-agent or parallel actor setups)."
    )
    episode_length: int = Field(
        default=128, gt=0,
        title="Rollout Length",
        description="Number of environment steps per rollout before updates."
    )
    minibatch_size: int = Field(
        default=64, gt=0,
        title="Mini-batch Size",
        description="Samples per update mini-batch. Must divide episode_length."
    )
    num_epochs: int = Field(
        default=5, gt=0,
        title="Optimization Epochs",
        description="Number of passes over collected rollout data per update."
    )

    # --- Optimization ---
    actor_lr: float = Field(
        default=1e-4, gt=0.0, le=1.0,
        title="Actor Learning Rate",
        description="Learning rate for the policy (actor) network."
    )
    critic_lr: float = Field(
        default=2e-4, gt=0.0, le=1.0,
        title="Critic Learning Rate",
        description="Learning rate for the value (critic) network."
    )
    lr_gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="LR Decay Gamma",
        description="Decay factor for learning rate scheduling (if enabled)."
    )
    opti_eps: float = Field(
        default=1e-5, ge=0.0,
        title="Optimizer Epsilon",
        description="Small epsilon to improve optimizer numerical stability."
    )
    weight_decay: float = Field(
        default=0.0, ge=0.0,
        title="Weight Decay",
        description="L2 regularization coefficient."
    )
    use_max_grad_norm: bool = Field(
        default=True,
        title="Use Gradient Clipping",
        description="Enable max gradient norm clipping for stability."
    )
    max_grad_norm: float = Field(
        default=0.5, gt=0.0,
        title="Max Gradient Norm",
        description="Maximum gradient norm when clipping is enabled."
    )
    use_lr_decay: bool = Field(
        default=True,
        title="Use LR Decay",
        description="Enable linear/exponential learning rate decay over time."
    )
    gain: float = Field(
        default=0.01, gt=0.0,
        title="Gain Initialization Scale",
        description="Gain used for layer initialization."
    )

    # --- Loss / advantages ---
    gamma: float = Field(
        default=0.99, ge=0.0, le=1.0,
        title="Discount Factor (Gamma)",
        description="Discount factor for future rewards."
    )
    use_gae: bool = Field(
        default=True,
        title="Use GAE",
        description="Enable Generalized Advantage Estimation."
    )
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0,
        title="GAE Lambda",
        description="Lambda parameter for GAE advantage calculation."
    )
    ent_coef: float = Field(
        default=0.01, ge=0.0,
        title="Entropy Coefficient",
        description="Weight for entropy term to encourage exploration."
    )
    vf_coef: float = Field(
        default=0.5, ge=0.0,
        title="Value Function Coefficient",
        description="Weight for critic (value) loss term."
    )
    clip_ratio: float = Field(
        default=0.2, gt=0.0,
        title="(Optional) Advantage Clip Ratio",
        description="If your implementation supports advantage or value clipping, "
                    "this ratio controls clipping range. Retained for parity with PPO-style UIs."
    )
    use_clipped_value_loss: bool = Field(
        default=True,
        title="Use Clipped Value Loss",
        description="Clip the value loss term to stabilize training (if supported)."
    )

    # --- Networks / architecture ---
    use_relu: bool = Field(
        default=True,
        title="Use ReLU Activations",
        description="Use ReLU activation (otherwise Tanh) in MLP/CNN blocks."
    )
    use_orthogonal: bool = Field(
        default=True,
        title="Use Orthogonal Initialization",
        description="Apply orthogonal initialization to network layers."
    )
    use_feature_normalization: bool = Field(
        default=True,
        title="Feature Normalization",
        description="Normalize extracted features before feeding to the network."
    )
    cnn_hidden: int = Field(
        default=256, gt=0,
        title="CNN Hidden Channels",
        description="Number of output channels for CNN feature extractor."
    )
    cnn_kernel_size: int = Field(
        default=3, gt=0,
        title="CNN Kernel Size",
        description="Kernel size for CNN convolutions."
    )
    cnn_stride: int = Field(
        default=1, gt=0,
        title="CNN Stride",
        description="Stride for CNN convolutions."
    )
    fc_hidden: int = Field(
        default=64, gt=0,
        title="Fully Connected Hidden Size",
        description="Number of hidden units per fully connected layer."
    )
    fc_num_hidden: int = Field(
        default=1, gt=0,
        title="Number of FC Hidden Layers",
        description="How many hidden layers in the fully connected network."
    )


    # --- Cross-field validation (parity with PPO style) ---
    @classmethod
    def model_validate(cls, obj, *a, **kw):
        m = super().model_validate(obj, *a, **kw)

        # rollout divisibility
        if m.episode_length % m.minibatch_size != 0:
            raise ValueError("episode_length must be divisible by minibatch_size")

        # lr decay gamma sensible when enabled
        if m.use_lr_decay and not (0.0 <= m.lr_gamma < 1.0):
            raise ValueError("With use_lr_decay=True, lr_gamma should be in [0.0, 1.0)")

        # gradient clipping requirement
        if m.use_max_grad_norm and m.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be > 0.0 when use_max_grad_norm=True")

        # if GAE disabled, gae_lambda is irrelevant but we maintain bounds already via Field
        return m


AlgoConfigSchema = Annotated[
    Union[PPOConfigSchema, SACConfigSchema, DQNConfigSchema, A2CConfigSchema,TRPOConfigSchema],
    Field(discriminator="algo_id"),
]

