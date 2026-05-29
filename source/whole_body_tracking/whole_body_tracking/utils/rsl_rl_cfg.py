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

    # Unified Stage 1+2: supervised auxiliary loss (FrontRES task-space mode)
    lambda_supervised: float = 0.0
    """Initial weight for supervised ΔSE3 auxiliary loss. 0 = disabled (legacy behaviour)."""
    lambda_supervised_min: float = 0.05
    """Floor for lambda_supervised after decay. Acts as regulariser preventing drift."""
    lambda_supervised_decay: float = 0.997
    """Per-iteration decay multiplier once the trigger threshold is crossed."""
    supervised_trigger_cosine_sim: float = 0.85
    """EMA cosine-similarity threshold that starts the decay (FrontRES has learned direction)."""
    supervised_rpy_loss_weight: float = 1.0
    """Weight for Δrpy component relative to Δpos in the supervised loss."""
    supervised_conf_loss_weight: float = 0.05
    """Small BCE weight for task-space confidence heads; keeps gates learnable before PPO signal is strong."""


@configclass
class RslRlFrontRESUnifiedAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """FrontRES-only unified training config.

    This intentionally does not inherit from MOSAIC: FrontRES uses PPO plus a
    supervised ΔSE3 auxiliary loss, without teacher or off-policy BC branches.
    """

    class_name: str = "FrontRESUnified"

    hybrid: bool = True
    """Kept for runner/config compatibility; FrontRESUnified requires True."""
    use_ppo: bool = True
    """Kept for runner/config compatibility; FrontRESUnified requires True."""
    gradient_accumulation_steps: int = 1
    """Kept for config compatibility; FrontRESUnified requires 1."""

    use_estimate_ref_vel: bool = False
    """Whether to use learned reference velocity estimator."""
    ref_vel_estimator_checkpoint_path: str | None = None
    """Path to reference velocity estimator checkpoint (.pt file)."""
    ref_vel_estimator_type: str = "mlp"
    """Type of velocity estimator: 'mlp' or 'transformer'."""

    lambda_supervised: float = 0.0
    """Initial weight for FrontRES supervised auxiliary loss."""
    lambda_supervised_min: float = 0.05
    """Minimum supervised loss weight after decay."""
    lambda_supervised_decay: float = 0.997
    """Per-update multiplicative decay after cosine trigger."""
    supervised_trigger_cosine_sim: float = 0.85
    """EMA cosine-similarity threshold that starts supervised weight decay."""
    supervised_rpy_loss_weight: float = 1.0
    """Weight for Δrpy component relative to Δpos in the supervised loss."""
    supervised_conf_loss_weight: float = 0.05
    """Small BCE weight for task-space confidence heads."""
    supervised_direction_loss_weight: float = 0.1
    """Cosine-direction loss weight on non-zero supervised targets."""
    supervised_valid_loss_weight: float = 4.0
    """Extra sample weight for non-zero supervised targets before normalization."""
    supervised_magnitude_loss_weight: float = 0.0
    """Weight for matching correction magnitude to the clean restoration target."""
    supervised_over_loss_weight: float = 0.0
    """Weight for penalizing corrections whose norm exceeds the clean target norm."""
    supervised_smooth_loss_weight: float = 0.0
    """Weight for matching temporal first differences of corrections to the target sequence."""
    supervised_coeff_sparse_weight: float = 0.0
    """L1 weight on inactive per-axis repair coefficients in basis_restore."""
    supervised_coeff_miss_weight: float = 0.0
    """Penalty for closing coefficients on active target axes in basis_restore."""
    supervised_coeff_smooth_weight: float = 0.0
    """Temporal smoothness weight for per-axis repair coefficients in basis_restore."""
    supervised_harm_loss_weight: float = 1.0
    """Explicit no-op penalty weight on rollout samples where FEMR is worse than noisy."""
    frontres_hsl_rollout_label_enabled: bool = False
    """Use Clean/Noisy/FEMR rollout states to build continuous HSL supervised targets."""
    frontres_hsl_rollout_eta: float = 1.0
    """Step size for converting FEMR-vs-clean rollout residual into a correction target."""
    frontres_hsl_rot_error_scale: float = 0.25
    """Scale rotation error when computing continuous HSL sample difficulty."""
    frontres_hsl_safe_threshold: float = 0.03
    """Samples below this rollout error are treated as mostly safe/no-op."""
    frontres_hsl_broken_threshold: float = 0.35
    """Samples above this rollout error are treated as mostly broken/no-op."""
    frontres_hsl_safe_temperature: float = 0.01
    """Temperature for the safe side of the double-sigmoid HSL gate."""
    frontres_hsl_broken_temperature: float = 0.05
    """Temperature for the broken side of the double-sigmoid HSL gate."""
    frontres_hsl_harm_temperature: float = 0.02
    """Temperature for the harmful-repair sigmoid gate."""
    frontres_hsl_safe_noop_weight: float = 1.0
    """No-op supervision weight for safe samples."""
    frontres_hsl_broken_noop_weight: float = 1.0
    """No-op supervision weight for broken samples."""
    frontres_hsl_harm_noop_weight: float = 2.0
    """No-op supervision weight when FEMR rollout is worse than noisy rollout."""
    frontres_hsl_max_sample_weight: float = 4.0
    """Upper bound on the combined HSL supervised sample weight."""
    frontres_supervised_lr_schedule: str = "fixed"
    """Supervised-only LR schedule: fixed or cosine_anneal."""
    frontres_supervised_lr_start: float | None = None
    """Initial LR for supervised cosine warmup."""
    frontres_supervised_lr_peak: float | None = None
    """Peak LR for supervised cosine schedule."""
    frontres_supervised_lr_min: float | None = None
    """Final LR floor for supervised cosine schedule."""
    frontres_supervised_lr_warmup_iters: int = 0
    """Linear warmup iterations before cosine decay."""
    frontres_supervised_lr_cosine_iters: int = 1000
    """Iterations used for cosine decay after warmup."""
    frontres_restore_debug_print_interval: int = 100
    """Iteration interval for low-frequency FrontRES restore consistency prints. <=0 disables prints."""
    ppo_actor_warmup_iterations: int = 0
    """Number of PPO iterations with actor surrogate disabled; critic and supervised loss still train."""
    ppo_actor_ramp_iterations: int = 0
    """Number of PPO iterations used to linearly ramp actor surrogate from 0 to 1."""
    ppo_advantage_focal_power: float = 0.0
    """Optional |advantage| focal exponent for actor surrogate. 0.0 gives standard PPO."""
    frontres_training_objective: str = "ppo_hrl"
    """FrontRES update objective: 'ppo_hrl' keeps PPO+supervised, 'supervised_restore' uses only supervised restoration."""
    frontres_active_task_dims: list[int] | None = None
    """Optional supervised-loss mask for task-space FrontRES correction dims."""
    diagnose_gradient_conflict: bool = True
    """Log cosine/norm diagnostics between PPO actor and supervised actor gradients."""


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
    # Number of auxiliary z-correction outputs appended after the Δq outputs.
    # Total FrontRES output = num_actions (Δq) + num_z_outputs (Δz).
    # Set to 1 for the standard [Δq (29), Δz (1)] architecture.
    num_z_outputs: int = 0
    # Task-space mode: when >0, FrontRES outputs [Δpos(3), Δrpy(3)] instead of Δq+Δz.
    # Set to 6 to enable full SE(3) correction mode.
    num_task_corrections: int = 0

@configclass # algorithm
class RslRlSuperviseAlgorithmCfg:
    """FrontRES Stage 1 Supervised Training"""
    class_name: str = "SuperviseTrainer"
    num_learning_epochs: int = 5
    learning_rate: float = 1.0e-3
    gradient_length: int = 15
    max_grad_norm: float = 1.0
    loss_type: str = "huber"
    # Static lower-limb joint weighting: set joint indices for hip/knee/ankle
    lower_limb_indices: list | None = None
    lower_limb_weight: float = 2.0
    # Temporal gate: joints whose Δq_gt changes by more than this (rad) in one
    # step are excluded from the loss (pre-fall tracking failure detection)
    jump_threshold: float = 0.2
    # Split point: first num_joint_outputs dims are Δq (get temporal gate + cascade mask),
    # remaining dims are auxiliary outputs (Δz etc., get only terminal mask).
    # Inactive when num_task_corrections > 0 (task-space mode).
    num_joint_outputs: int = 29
    # Weight for auxiliary (Δz) loss relative to Δq loss.
    z_loss_weight: float = 0.5
    # Task-space mode: weight for Δrpy loss relative to Δpos loss.
    # Active only when num_task_corrections > 0.
    rpy_loss_weight: float = 1.0
    # Task-space mode: number of task-space outputs (0 = disabled, 6 = [Δpos(3)+Δrpy(3)]).
    num_task_corrections: int = 0

# ====== FrontRES Stage 1: Supervised Learning ======

# ========== FrontRES Stage 2: RL Finetune ==========

@configclass # policy
class RslRlFrontResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Front-End Residual Actor-Critic configuration.
    Supports legacy joint-space [Δq, Δz] correction and the current task-space
    FrontRES path, where ΔSE(3) is applied by the runner/env command term.
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

    # Δz output configuration (must match Stage 1 setting; inactive when num_task_corrections > 0)
    num_z_outputs: int = 0
    """Number of root z-correction outputs after Δq. 1 → total output = num_actions + 1."""
    max_delta_z: float = 0.3
    """tanh clip for Δz output (metres). 0.3 m covers typical float/sink artifacts."""

    # FrontRES-specific observation: number of leading dims of the full policy obs
    # that are reference-frame only (command + motion_anchor_ori_b).
    # When set, FrontRES processes only this subset; GMT continues to use the full obs.
    # 320 = (58 command + 6 anchor_ori) × 5 history frames.
    # Default 0 = use full policy obs (backward compatible).
    num_frontres_obs: int = 0
    """FrontRES observation subset dims. 0 = use full policy_obs (legacy). 320 = ref-only."""

    # Task-space correction mode: replaces Δq+Δz with [Δpos(3), Δrpy(3)]
    num_task_corrections: int = 0
    """When >0, FrontRES outputs SE(3) anchor corrections instead of joint Δq. Set to 6."""
    task_conf_dim: int = 2
    """Confidence/coefficient dims: 2 legacy c_pos/c_rpy, 6 per-axis repair coefficients."""
    max_delta_pos: float = 0.3
    """tanh clip for position correction (metres). 0.3 m covers float/sink/slip artifacts."""
    max_delta_rpy: float = 0.3
    """tanh clip for orientation correction (radians). 0.3 rad ≈ 17° covers tilt artifacts."""

    # Ref vel estimator configuration (for GMT input)
    num_ref_vel_estimator_obs: int | None = None
    ref_vel_estimator_checkpoint_path: str | None = None
    ref_vel_estimator_type: str = "mlp"

    # Action noise type: "scalar" (shared std per action) or "log" (log-parameterized std)
    noise_std_type: str = "scalar"

    # Output clipping: Δq ∈ (-max_delta_q, max_delta_q) per joint via tanh.
    # Prevents action explosion during RL fine-tuning.  Default 0.5 rad (≈ ±28.6°).
    max_delta_q: float = 0.5


@configclass  # algorithm
class RslRlPpoFrontRESAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    FrontRES Stage-2 RL Fine-tuning PPO Configs

    Adding at Standard PPO:
    - Regularization terms L_reg = λ_reg * ||Δq_mean||^2:
        Pulling FrontRES's output and 0 (equal to ||q' - q_ref||^2),
        In case policy output enormous Δq to avoid punishment, 
        Keeping q' close to q_ref
    - PCGrad (use_pcgrad=True):
        when the gradients of PPO and regularization are conflict, 
        project gradients into others' normal plane to eliminate mutual inhibition
    """
    class_name: str = "PPO"

    # ── 监督损失：锚定 FrontRES 方向 ───────────────────────────────────────
    # λ_sup ∈ [0, 1]：supervised_loss = λ_sup × HuberLoss(pred, target)
    # 1.0 = 纯监督方向，0.0 = 纯 PPO。建议从 1.0 衰减到 0.1
    lambda_supervised: float = 1.0
    lambda_supervised_min: float = 0.1
    lambda_supervised_decay: float = 0.999  # 每 iter 乘以此系数

    # 正则化权重
    lambda_reg_init:  float = 0.01

    # 正则化初始权重, 0.0表示禁用
    lambda_reg_decay: float = 1.0

    # 每次 update()后乘以此系数衰减, 1.0=不衰减 (推荐)
    lambda_reg_min:   float = 0.0

    # 梯度冲突协调, True启用PCGrad, False直接加权相加
    use_pcgrad: bool = False

# ========== FrontRES Stage 2: RL Finetune ==========
