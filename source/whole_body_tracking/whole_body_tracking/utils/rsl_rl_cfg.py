from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg
)


@configclass
class RslRlPpoActorCriticWithRefVelSkipCfg(RslRlPpoActorCriticCfg):
    """
    Actor-Critic configuration with ref_vel skip connection support.

    When enabled, estimated ref_vel skips the first layer of the policy network
    and connects directly to the second layer.

    Architecture:
        policy_obs → layer1 → layer1_out
        ref_vel ─────────────────────┘
                                      ↓
        [layer1_out, ref_vel] → layer2 → ... → output
    """
    ref_vel_skip_first_layer: bool = False
    """Enable ref_vel skip connection (default: False)."""
    ref_vel_dim: int = 3
    """Dimension of estimated ref_vel (default: 3)."""


@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTransformer"
    seq_len: int = 1
    d_model: int = 512
    nhead: int = 4
    num_layers: int = 2
    activation_transformer: str = "gelu"


@configclass
class RslRlPpoActorCriticFSQCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFSQ"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    activation_fsq: str = "elu"
    latent_dim: int = 8
    num_levels: int = 5

@configclass
class RslRlPpoActorCriticVQCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticVQ"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    encoder_output_dim: int = 256
    activation_vq: str = "elu"
    num_embeddings: int = 512
    embedding_dim: int = 32
    commitment_weight: float = 0.25
    vq_loss_coef: float = 0.1

@configclass
class RslRlPpoActorCriticAttentionCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticAttention"
    num_actor_proprio: int = 1
    encoder_hidden_dims: list[int] = [1024, 1024]
    activation_attn: str = "elu"
    attention_dim: int = 256
    nhead: int = 4

@configclass
class RslRlDistillationCfg(RslRlPpoActorCriticCfg):
    class_name: str = "StudentTeacher"
    student_hidden_dims: list[int] = [256, 256, 256]
    teacher_hidden_dims: list[int] = [256, 256, 256]





@configclass
class RslRlResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Residual Actor-Critic configuration for ResMimic-style residual learning.

    Architecture:
    - GMT policy (frozen): Loaded from checkpoint, provides base actions
    - Residual network (trainable): Learns corrections Δa
    - Final action: a_final = a_gmt + Δa_residual
    """
    class_name: str = "ResidualActorCritic"

    # Residual network configuration
    residual_hidden_dims: list[int] = [512, 256, 128]
    """Hidden dimensions for residual network."""
    residual_last_layer_gain: float = 0.01
    """Xavier initialization gain for last layer (small value for near-zero initial output)."""

    # GMT configuration
    gmt_checkpoint_path: str = MISSING
    """Path to GMT policy checkpoint (.pt file). Required."""
    gmt_policy_cfg: dict | None = None
    """Optional GMT policy architecture config. If None, auto-inferred from checkpoint."""
    init_critic_from_gmt: bool = False
    """Initialize residual critic weights from the GMT checkpoint if dimensions match."""

    # Ref vel estimator configuration
    num_ref_vel_estimator_obs: int | None = None
    """Dimension of ref_vel_estimator observations (e.g., 305). If None, estimator is not used."""
    ref_vel_estimator_checkpoint_path: str | None = None
    """Path to ref_vel estimator checkpoint (.pt file). If None, zero padding is used."""
    ref_vel_estimator_type: str = "mlp"
    """Type of estimator: 'mlp' or 'transformer'."""


@configclass
class RslRlMOSAICAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    MOSAIC algorithm configuration.

    MOSAIC is a plugin-style extension of PPO that adds hybrid learning:
    1. PPO: Standard reinforcement learning (optional)
    2. Offline expert BC: Learn from pre-collected expert trajectories (.npy file)
    3. Online teacher BC: Learn from teacher policy with privileged observations

    This configuration supports all modes:
    - PPO only: use_ppo=True, expert_trajectory_path=None, lambda_teacher_init=0.0
    - PPO + Expert BC: use_ppo=True, expert_trajectory_path=set, lambda_teacher_init=0.0
    - PPO + Teacher BC: use_ppo=True, expert_trajectory_path=None, lambda_teacher_init>0.0
    - Pure Teacher BC: use_ppo=False, expert_trajectory_path=None, lambda_teacher_init>0.0
    - Full MOSAIC: use_ppo=True, expert_trajectory_path=set, lambda_teacher_init>0.0
    """
    class_name: str = "MOSAIC"

    # Mode selection
    hybrid: bool = True
    """True = hybrid mode (random mini-batches, per-batch updates), False = pure BC mode (sequential data, gradient accumulation)."""

    # PPO switch
    use_ppo: bool = True
    """Enable PPO reinforcement learning. Set to False for pure BC mode."""

    # Offline Expert BC parameters
    expert_trajectory_path: str | None = None
    """Path to expert trajectory .npy file for offline BC. Set to None to disable."""
    lambda_off_policy: float = 0.3
    """Initial weight for offline expert BC loss."""
    lambda_off_policy_decay: float = 0.995
    """Decay rate for offline BC weight (1.0 = no decay, 0.995 = slow decay)."""
    lambda_off_policy_min: float = 0.01
    """Minimum offline BC weight after decay."""
    off_policy_batch_size: int = 256
    """Batch size for sampling expert trajectories."""
    expert_allow_repeat_sampling: bool = False
    """Allow sampling with replacement if batch_size > dataset_size."""
    expert_loss_type: str = "mse"
    """Loss function for expert BC: 'kl' (KL divergence) or 'mse' (MSE on action means)."""
    expert_normalize_obs: bool = True
    """Whether to normalize expert observations with student's normalizer."""
    expert_update_normalizer: bool = False
    """Whether expert observations should update normalizer statistics (False=recommended)."""

    # Online Teacher BC parameters
    teacher_checkpoint_path: str | dict[str, str] | None = None
    """Path to teacher checkpoint .pt file. Supports single teacher (str) or multi-teacher (dict: group_name -> path). Required if lambda_teacher_init > 0.0."""
    teacher_obs_source_mapping: dict[str, str] | None = None
    """Maps teacher group names to observation sources for multi-teacher mode. Options: 'policy', 'teacher', 'critic'. Example: {'lafan': 'teacher', 'fld': 'policy'}"""
    teacher_critic_checkpoint_path: str | None = None
    """Path to separate teacher critic checkpoint .pt file. If provided, loads critic weights from this checkpoint."""
    teacher_critic_frozen: bool = True
    """Whether to freeze teacher critic (True=frozen, False=allow fine-tuning). Only applies when teacher_critic_checkpoint_path is provided."""
    train_critic_during_distillation: bool = False
    """Whether to train critic during distillation (use_ppo=False). If True, critic is trained via value loss even when PPO is disabled."""
    lambda_teacher_init: float = 1.0
    """Initial weight for online teacher BC loss. Set to 0.0 to disable."""
    lambda_teacher_decay: float = 0.995
    """Decay rate for teacher BC weight (0.995 = slow decay to encourage early learning)."""
    lambda_teacher_min: float = 0.1
    """Minimum teacher BC weight after decay."""
    teacher_loss_type: str = "mse"
    """Loss function for teacher BC: 'kl' (KL divergence) or 'mse' (MSE on action means)."""

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    """Number of mini-batches to accumulate gradients before optimizer step. 1 = no accumulation."""

    # Reference Velocity Estimator
    use_estimate_ref_vel: bool = False
    """Whether to use learned reference velocity estimator."""
    ref_vel_estimator_checkpoint_path: str | None = None
    """Path to reference velocity estimator checkpoint (.pt file). Required if use_estimate_ref_vel=True."""
    ref_vel_estimator_type: str = "mlp"
    """Type of velocity estimator: 'mlp' or 'transformer'."""


@configclass
class RslRlKLDistillationAlgorithmCfg:
    """
    KL Distillation algorithm configuration.

    Improved distillation using KL divergence instead of MSE loss.
    This matches MOSAIC's teacher BC approach for better distribution matching.
    """
    class_name: str = "KLDistillation"

    num_learning_epochs: int = 5
    """Number of passes through the dataset per update."""
    gradient_length: int = 15
    """Number of steps to accumulate gradients before optimizer step."""
    learning_rate: float = 1.0e-3
    """Learning rate for student optimizer."""
    loss_type: str = "kl"
    """Loss function type: 'kl' (recommended), 'mse', or 'huber'."""
    kl_reduction: str = "mean"
    """How to reduce KL loss: 'mean' or 'sum'."""

# ====== FrontRES Stage 1: Supervised Learning ======

@configclass # policy
class RslRlSuperviseJointPosCfg(RslRlPpoActorCriticCfg):
    """
    FrontRES Policy Training: Loss Computer & Gradient Update
    """
    class_name: str = "SuperviseLearning"
    student_hidden_dims: list[int] = [256, 256, 256]
    gmt_path: str = ""

@configclass # algorithm
class RslRlSuperviseAlgorithmCfg:
    """FrontRES Stage 1 Supervised Training"""
    class_name: str = "SuperviseTrainer"
    num_learning_epochs: int = 5
    learning_rate: float = 1.0e-3
    gradient_length: int = 15
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

# ====== FrontRES Stage 1: Supervised Learning ======

# ========== FrontRES Stage 2: RL Finetune ==========

@configclass # policy
class RslRlFrontResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Front-End Residual Actor-Critic configuration.
    Stage 2 RL Finetuning: FrontRES outputs Δq, which modifies q_ref before entering GMT.
    """
    class_name: str = "FrontRESActorCritic"

    # FrontRES network configuration
    residual_hidden_dims: list[int] = [1024, 1024, 512, 256]
    residual_last_layer_gain: float = 0.01

    # Observation index configuration
    q_ref_start_idx: int = MISSING
    """The starting index of q_ref in the flattened observation vector."""

    # GMT configuration
    gmt_checkpoint_path: str = MISSING
    gmt_policy_cfg: dict | None = None
    init_critic_from_gmt: bool = False

    # Ref vel estimator configuration (for GMT input)
    num_ref_vel_estimator_obs: int | None = None
    ref_vel_estimator_checkpoint_path: str | None = None
    ref_vel_estimator_type: str = "mlp"

    # Action noise type: "scalar" (shared std per action) or "log" (log-parameterized std)
    noise_std_type: str = "scalar"


@configclass  # algorithm
class RslRlPpoFrontRESAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    FrontRES Stage-2 RL Fine-tuning 专用 PPO 配置。

    在标准 PPO 基础上新增：
    - 正则化项 L_reg = λ_reg * ||Δq_mean||^2：
        拉近 FrontRES 输出与零（等价于 ||q' - q_ref||^2），
        防止策略为规避奖惩而输出过大的修正量，保持动作序列接近原始运动参考。
    - PCGrad（use_pcgrad=True 时生效）：
        当 PPO 梯度与正则化梯度方向冲突时，将各自梯度投影到对方的法平面，
        消除相互抑制，实现自适应权重协调。
    """
    class_name: str = "PPO"

    # 正则化权重
    lambda_reg_init:  float = 0.01
    """正则化初始权重。0.0 表示禁用。"""
    lambda_reg_decay: float = 1.0
    """每次 update() 后乘以此系数衰减。1.0 = 不衰减（推荐）。"""
    lambda_reg_min:   float = 0.0
    """正则化权重下限。"""

    # 梯度冲突协调
    use_pcgrad: bool = False
    """True = 启用 PCGrad 在 PPO loss 与正则化 loss 之间协调梯度；
    False = 直接加权相加（更快但可能互相抑制）。"""

