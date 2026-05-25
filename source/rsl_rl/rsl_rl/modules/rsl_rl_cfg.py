from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
class RslRlSuperviseJointPosCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SuperviseLearning"
    student_hidden_dims: list[int] = [256, 256, 256]


@configclass
class RslRlFrontEndResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """
    Front-End Residual Actor-Critic configuration.
    Legacy joint-space Stage 2 path: FrontRES outputs Δq, which modifies q_ref
    before entering GMT. The current task-space FrontRES config lives in the
    FrontRESActorCritic/unified-runner path.
    """
    class_name: str = "FrontEndResidualActorCritic"

    # FrontRES network configuration (this is our trainable actor)
    residual_hidden_dims: list[int] = [1024, 1024, 512, 256]

    # Observation index configuration
    q_ref_start_idx: int = MISSING
    """The starting index of q_ref in the flattened observation vector."""

    # GMT configuration (frozen)
    gmt_checkpoint_path: str = MISSING
    """Path to the pre-trained GMT policy checkpoint (.pt file)."""

    # Critic configuration
    critic_hidden_dims: list[int] = [512, 256, 128]


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
class RslRlFrontRESUnifiedAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """FrontRES-only unified training configuration.

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
    frontres_state_supervised_controller_enabled: bool = True
    """Use FrontRES reward diagnostics to permanently decay supervised loss into a weak anchor."""
    frontres_supervised_anchor_weight: float = 0.02
    """Final weak supervised anchor weight once PPO has a learnable executable signal."""
    frontres_supervised_decay_good: float = 0.985
    """Multiplicative supervised decay when PPO signal is learnable and safe."""
    frontres_supervised_decay_conflict: float = 0.97
    """Faster supervised decay when PPO and supervised actor gradients are nearly orthogonal/conflicting."""
    frontres_supervised_positive_gain_trigger: float = 0.52
    """Positive-gain fraction required before supervised weight is allowed to decay."""
    frontres_supervised_harm_limit: float = 0.06
    """Harm-rate ceiling for supervised decay."""
    frontres_supervised_grad_cos_low: float = 0.03
    """Gradient cosine below which supervised is treated as directionally unhelpful."""
    frontres_supervised_min_hold_iters: int = 5
    """Number of diagnostic iterations before the state controller can decay supervised loss."""
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
    ppo_actor_warmup_iterations: int = 0
    """Number of PPO iterations with actor surrogate disabled; critic and supervised loss still train."""
    ppo_actor_ramp_iterations: int = 0
    """Number of PPO iterations used to linearly ramp actor surrogate from 0 to 1."""
    ppo_advantage_focal_power: float = 0.0
    """Optional |advantage| focal exponent for actor surrogate. 0.0 gives standard PPO."""
    frontres_training_objective: str = "ppo_hrl"
    """FrontRES update objective: 'ppo_hrl' keeps PPO+supervised, 'supervised_restore' uses only supervised restoration."""
    frontres_exec_reward_signal: str = "gain"
    """Executable reward signal: gain, ratio, or family_preference."""
    frontres_selective_reward_enabled: bool = True
    """Use safe/repairable/broken gates so FrontRES learns when to abstain and when to repair."""
    frontres_min_effective_gain: float = 0.006
    """Minimum per-step executable gain treated as an effective repair inside the repairable band."""
    frontres_effective_gain_bonus_weight: float = 0.5
    """Extra reward weight for repair gains above frontres_min_effective_gain."""
    frontres_safe_cost_weight: float = 1.0
    """Intervention-cost weight for safe samples where FrontRES should abstain."""
    frontres_repair_cost_weight: float = 0.15
    """Intervention-cost weight inside the repairable band; kept weak so useful repairs can grow."""
    frontres_broken_cost_weight: float = 1.0
    """Intervention-cost weight for deeply broken samples where FrontRES should abstain."""
    frontres_broken_harm_weight: float = 1.0
    """Harm penalty gate for deeply broken samples."""
    frontres_family_preference_scale: float = 0.02
    """Per-step scale for bounded pairwise preference reward."""
    frontres_family_preference_tau: float = 1.0
    """Temperature for tanh(gain / family_std / tau)."""
    frontres_family_preference_alpha: float = 0.7
    """Blend weight for bounded preference vs. bounded repair ratio."""
    frontres_family_gain_ema_alpha: float = 0.05
    """EMA update rate for per-perturb-family gain statistics."""
    frontres_family_gain_initial_std: float = 0.01
    """Initial per-family gain std used before enough statistics are observed."""
    frontres_family_gain_min_std: float = 0.002
    """Lower bound for per-family gain std normalization."""
    frontres_per_mode_supervised_mask: bool = True
    """Mask supervised ΔSE3 targets by the perturbation family assigned to each env."""
    frontres_adaptive_perturb_curriculum_enabled: bool = True
    """Choose perturbation-family complexity from repairability diagnostics instead of fixed iteration progress."""
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
