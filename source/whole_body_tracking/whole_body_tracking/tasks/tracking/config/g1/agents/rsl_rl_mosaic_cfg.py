import os
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlMOSAICAlgorithmCfg,
    RslRlFrontRESUnifiedAlgorithmCfg,
    RslRlResidualActorCriticCfg,
    RslRlFrontResidualActorCriticCfg,
    RslRlPpoActorCriticWithRefVelSkipCfg,
)

@configclass
class G1FlatMOSAICRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "g1_flat_mosaic"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=False, 
        ref_vel_dim=3, )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True, 

        # PPO parameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # MOSAIC specific parameters
        use_ppo=True,
        expert_trajectory_path=None, 
        lambda_off_policy=0.3,
        lambda_off_policy_decay=0.995,
        lambda_off_policy_min=0.01,
        off_policy_batch_size=256,
        expert_allow_repeat_sampling=False,
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,)


@configclass
class G1FlatMOSAICHybridRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Hybrid Mode Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=False,
        ref_vel_dim=3,)

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,  # Hybrid mode

        # ========== PPO Configuration ==========
        use_ppo=True,

        # PPO hyperparameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path=None, 
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.9995,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse", 

        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False, 

        # ========== Pure BC only ==========
        gradient_accumulation_steps=15,  # For Pure BC mode

        # ========== Expert BC Configuration (Offline Data Distillation) ==========
        expert_trajectory_path=None,
        lambda_off_policy=0.1,
        lambda_off_policy_decay=0.99,
        lambda_off_policy_min=0.1,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",  
        expert_normalize_obs=True,  
        expert_update_normalizer=False, 
        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False, 
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer", )


@configclass
class G1FlatMOSAICPureDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Pure Distillation Configuration.
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=0.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=True, 
        ref_vel_dim=3,)

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,

        # ========== PPO Configuration ==========
        use_ppo=False,

        # PPO hyperparameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path="path/to/teacher_checkpoint.pt",  
        lambda_teacher_init=1.0,
        lambda_teacher_decay=1.0,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse",

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,
        train_critic_during_distillation=True, 

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1,

        # ========== Expert BC Configuration ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0, 
        lambda_off_policy_decay=1.0,
        lambda_off_policy_min=1.0,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=True, 
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",  # Optional checkpoint for pre-trained estimator
        ref_vel_estimator_type="mlp",)  # "mlp" or "transformer"


@configclass
class G1FlatMOSAICRLContinueRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC RL Continue Configuration.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid" 
    empirical_normalization = True

    # ========== Student Checkpoint Configuration ==========
    student_checkpoint_path="path/to/distilled_checkpoint.pt",

    # ========== Noise Std Reset Configuration ==========
    reset_noise_std_on_resume = True

    # ========== Normalizer Freeze Configuration ==========
    freeze_normalizer_on_resume = False

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=0.5, 
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,  # Hybrid mode

        # ========== PPO Configuration (ENABLED) ==========
        use_ppo=True,  # Enable PPO for RL training

        # PPO hyperparameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path=None,
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,
        teacher_loss_type="mse",

        # ========== Expert BC Configuration ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0,  
        lambda_off_policy_decay=0.995,  
        lambda_off_policy_min=1.0, 
        off_policy_batch_size=100000, 
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1,

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False, 

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False,
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer",)  # "mlp" or "transformer"


@configclass
class G1FlatMOSAICRLContinueResidualRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC RL Continue with Residual Learning Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True 

    policy = RslRlResidualActorCriticCfg(
        # Residual network configuration
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,

        # GMT configuration
        gmt_checkpoint_path="path/to/gmt_checkpoint.pt",
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt=True,

        # Standard ActorCritic parameters
        init_noise_std=0.8,
        noise_std_type="scalar",
        activation="elu",)

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,
        use_ppo=True,

        # ========== PPO Configuration ==========
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # ========== Teacher BC Configuration (DISABLED) ======
        teacher_checkpoint_path="path/to/teacher_checkpoint.pt",
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,
        teacher_loss_type="mse",

        # ========== Expert BC Configuration (Regularization) ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0,
        lambda_off_policy_decay=0.99,
        lambda_off_policy_min=1.0,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1, 

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False, 
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer",)  # "mlp" or "transformer"


@configclass
class G1FlatMOSAICMultiTeacherResidualRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Multi-Teacher Residual BC Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlResidualActorCriticCfg(
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,
        gmt_checkpoint_path="path/to/gmt_checkpoint.pt",
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        activation="elu",
        num_ref_vel_estimator_obs=305,
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",
        ref_vel_estimator_type="mlp",)
    

    algorithm = RslRlMOSAICAlgorithmCfg(
        hybrid=True,
        use_ppo=False,  # Pure BC mode
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Multi-teacher BC configuration
        teacher_checkpoint_path={
            "motions": "path/to/gmt_checkpoint.pt", # same as gmt_checkpoint_path="path/to/gmt_checkpoint.pt when using one stage rl training for gmt",
            "teleop_motions": "path/to/finetuned_gmt_checkpoint.pt",},
        
        # Teacher observation source mapping: which observations each teacher uses
        teacher_obs_source_mapping={
            "motions": "teacher",  # Use teacher observations
            "teleop_motions": "policy",},     # Use policy observations
        
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse",

        use_estimate_ref_vel=False,  # Enable estimator to track velocity error
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",
        ref_vel_estimator_type="mlp",

        expert_trajectory_path=None,
        lambda_off_policy=1.0,
        lambda_off_policy_decay=0.995,
        lambda_off_policy_min=0.01,
        off_policy_batch_size=100000,

        gradient_accumulation_steps=1,
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,
        train_critic_during_distillation=False,)


# ========== FrontRES Unified Training (Stage 1 + Stage 2 merged) ==========

@configclass
class G1FlatFrontRESUnifiedRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Unified FrontRES training: supervised ΔSE3 warmup + B1 delta-reward RL in one run.

    Architecture:
      FrontRES (trainable) → ΔSE3 [Δpos(3), Δrpy(3)]
      → patch anchor root pose → frozen GMT → robot actions

    Loss:
      L = L_PPO(B1 delta-reward) + λ_sup(t) * L_supervised(ΔSE3)

    λ_sup decays from lambda_supervised → lambda_supervised_min once
    cosine_similarity_ema(pred, target) ≥ supervised_trigger_cosine_sim.

    Mandatory fields to fill before training:
      policy.gmt_checkpoint_path        — path to frozen GMT .pt checkpoint
      policy.q_ref_start_idx            — index of q_ref in the flattened policy obs

    Recommended MotionPerturbationCfg (all dims covered):
      float_prob=0.3,        float_ratio=0.15,      # Δpos[2]+
      sink_prob=0.3,         sink_ratio=0.15,       # Δpos[2]-
      foot_slip_prob=0.2,    foot_slip_ratio=0.05,  # Δpos[0]
      lateral_drift_prob=0.3, lateral_drift_std=0.05, # Δpos[1]
      root_tilt_prob=0.3,    root_tilt_max_rad=0.08, # Δroll, Δpitch
    """
    num_steps_per_env = 24
    # Short default run for reward/DR debugging.  Use --max_iterations to
    # launch a long formal run once the diagnostics look healthy.
    max_iterations    = 1500
    save_interval     = 100
    experiment_name   = "g1_flat_frontres_unified"
    empirical_normalization = True
    # Resume semantics:
    # True  = full checkpoint resume (actor + critic + optimizer + iteration).
    # False = checkpoint as initialization (residual actor only; critic/optimizer/iteration reset).
    # Specialist rp-z finetune should start from a broad FrontRES actor but
    # relearn critic/optimizer state for the narrowed vertical-contact reward.
    is_full_resume = False

    # ── Curriculum Oracle ─────────────────────────────────────────────────────
    # Mixes oracle (-OU ground truth) with FrontRES output to bootstrap training.
    # ── Oracle mixup DISABLED ─────────────────────────────────────────────────
    # Now handled by λ_supervised in the PPO loss space (anchors μ directly),
    # which is cleaner — no credit-assignment distortion in the action space.
    oracle_curriculum              = False
    oracle_mix_cos_low             = 0.3    # below this: pure oracle
    oracle_mix_cos_high            = 0.85   # above this: pure FrontRES

    # ── Fix 2: Low-pass filter on anchor corrections ──────────────────────────
    correction_smooth_alpha        = 0.4

    # ── FrontRES rp demo specialist ───────────────────────────────────────────
    # Task-space action layout:
    #   [dx, dy, dz, droll, dpitch, dyaw, c_pos, c_rpy]
    # Isolate roll/pitch first so we can measure GMT's angular robustness limit
    # without coupling the signal to root-z contact discontinuities.
    frontres_specialist_mode       = "rp"
    frontres_active_task_dims      = [3, 4, 7]
    frontres_perturbation_channels = "rp"

    # "More executable" reward:
    #   damage_gap  = R_feasible_oracle - R_perturbed
    #   repair_gain = R_frontres - R_perturbed
    #   repair_ratio = repair_gain / max(damage_gap, gap_floor)
    # A double-sigmoid window maps executable damage to one repairability
    # weight mu.  Safe and deeply broken samples are suppressed; only the
    # middle repairable band receives the executable-gain reward.
    # Demo-oriented FEMR: clean restoration is the primary objective.  The
    # executable score remains a safety diagnostic/constraint, not the behavior
    # target that PPO is allowed to hack.
    frontres_exec_reward_weight    = 0.0
    frontres_exec_reward_temp      = 1.0
    # Train on direct executable gain by default.  The normalized ratio is still
    # logged as a difficulty diagnostic, but using it as reward divided by small
    # gaps and over-amplified tiny per-step noise during Actor takeover.
    frontres_exec_reward_signal    = "gain"  # "gain" or "ratio"
    frontres_repair_reward_scale   = 1.0
    # FrontRES-specific executability score.  This deliberately excludes
    # teleop rewards and actuator penalties from RewardsExpertCfg.
    #
    # Full-output test: planar and vertical executability are both active.  Task
    # velocity consistency remains a weak anti-cheat prior so the policy cannot
    # simply make the reference easier by erasing motion.
    frontres_exec_planar_weight    = 1.0
    frontres_exec_vertical_weight  = 1.0
    frontres_exec_task_weight      = 0.0
    # Cone-aware scalarization used by reward/gap/gain after executable
    # diagnostics are computed.  Each perturbation family reads its matching
    # component: planar->xy, yaw->yaw, global_z->z, local_rp->rp.
    frontres_exec_cone_planar_weight = 1.0
    frontres_exec_cone_yaw_weight = 1.0
    frontres_exec_cone_vertical_weight = 1.0
    frontres_exec_cone_rp_weight = 1.0
    frontres_exec_cone_task_weight = 0.0
    frontres_exec_anchor_xy_threshold = 0.35
    frontres_exec_anchor_yaw_threshold = 0.45
    frontres_exec_anchor_xy_vel_std = 1.0
    frontres_exec_anchor_yaw_rate_std = 1.0
    frontres_exec_anchor_xy_weight = 1.0
    frontres_exec_anchor_yaw_weight = 1.0
    frontres_exec_anchor_xy_vel_weight = 0.5
    frontres_exec_anchor_yaw_rate_weight = 0.5
    frontres_exec_foot_phase_weight = 0.5
    frontres_exec_foot_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    frontres_exec_foot_phase_z_threshold = 0.12
    frontres_exec_foot_phase_gate_temp = 0.03
    frontres_exec_foot_phase_xy_threshold = 0.25
    frontres_exec_anchor_z_threshold = 0.20
    frontres_exec_anchor_ori_threshold = 0.25
    frontres_exec_ee_z_threshold   = 0.18
    # Vertical score is mostly stability margin.  Keep z/EE tracking weak so
    # sink artifacts are not rewarded by simply lifting the whole reference.
    frontres_exec_anchor_z_weight  = 0.25
    frontres_exec_anchor_ori_weight = 1.0
    frontres_exec_ee_z_weight      = 0.25
    frontres_exec_body_lin_vel_std = 1.0
    frontres_exec_body_ang_vel_std = 3.14
    frontres_exec_anchor_lin_vel_std = 1.0
    frontres_exec_ee_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    frontres_exec_velocity_body_names = None
    frontres_gap_floor_per_step    = 0.005
    frontres_safe_gap_per_step     = 0.003
    frontres_broken_gap_per_step   = 0.10
    frontres_gap_gate_temp         = 0.005
    frontres_side_actor_gate_weight = 0.05
    frontres_harm_epsilon          = 0.001
    frontres_harm_penalty_weight   = 1.0
    frontres_executable_harm_weight = 1.0
    frontres_reward_scale_dr_reference = 1.25
    frontres_reward_progress_min = 0.0
    frontres_constraint_progress_exponent = 2.0
    frontres_side_harm_weight      = 0.0
    frontres_harm_action_cost_floor = 0.001
    frontres_harm_action_cost_ref   = 0.01
    frontres_state_supervised_controller_enabled = True
    # Keep PPO near the supervised anti-perturbation cone.  A low anchor lets
    # late-stage PPO drift into self-induced harmful corrections.
    frontres_supervised_anchor_weight = 0.20
    frontres_supervised_decay_good = 0.985
    frontres_supervised_decay_conflict = 0.97
    frontres_supervised_positive_gain_trigger = 0.52
    frontres_supervised_harm_limit = 0.06
    frontres_supervised_grad_cos_low = 0.03
    frontres_supervised_min_hold_iters = 5
    frontres_exec_reward_signal = "gain"
    frontres_selective_reward_enabled = True
    frontres_min_effective_gain = 0.008
    frontres_effective_gain_bonus_weight = 0.0
    frontres_safe_cost_weight = 1.0
    frontres_repair_cost_weight = 0.0
    frontres_broken_cost_weight = 1.0
    frontres_broken_harm_weight = 1.0
    frontres_family_preference_scale = 0.02
    frontres_family_preference_tau = 1.0
    frontres_family_preference_alpha = 0.7
    frontres_family_gain_ema_alpha = 0.05
    frontres_family_gain_initial_std = 0.01
    frontres_family_gain_min_std = 0.002
    frontres_per_mode_supervised_mask = True
    frontres_adaptive_perturb_curriculum_enabled = True
    frontres_warmup_energy_loss_weight = 1.0
    # Demo restoration reward:
    #   r_restore = ||noisy - clean|| - ||corrected - clean||
    # It is computed from the anchor error against the clean reference.  PPO now
    # optimizes visual/reference restoration directly; executable harm is a
    # constraint that prevents dynamically damaging corrections.
    frontres_restore_z_weight = 0.0
    frontres_restore_xy_weight = 0.0
    frontres_restore_rp_weight = 1.0
    frontres_restore_yaw_weight = 0.0
    frontres_geometry_reward_weight = 1.0
    frontres_rescue_reward_weight  = 0.0
    # Clean-bounded action regularization.  The legacy magnitude cost is kept
    # off by default because it suppresses necessary repairs.  The active cost
    # penalizes only corrections past the Clean target or away from its direction.
    frontres_intervention_cost_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    frontres_overcorrection_margin = 0.10
    frontres_overcorrection_weight = 0.5
    frontres_clean_bound_side_weight = 0.1
    frontres_min_restore_ratio = 0.6
    frontres_under_repair_weight = 0.2
    frontres_oracle_clean_gap_tau = 0.05
    frontres_oracle_clean_gap_threshold = 1.0e9
    frontres_restore_eval_min_error = 0.01

    # ── Fast debug mode: shortens the feedback loop for reward/DR tuning ─────
    # Formal training leaves this False.  Enable with:
    #   --frontres_debug_training
    # or override the debug_* fields below from Hydra.
    frontres_debug_training        = False
    debug_supervised_warmup_iterations = 200
    debug_supervised_warmup_diag_interval = 40
    debug_critic_warmup_iterations = 0
    debug_ppo_actor_warmup_iterations = 0
    debug_ppo_actor_ramp_iterations = 100
    debug_dr_scale_init            = 1.0
    debug_dr_min_scale             = 0.3
    debug_dr_ema_alpha             = 0.90
    debug_dr_p_gain                = 0.20
    debug_dr_i_gain                = 0.03
    debug_dr_start_ppo_actor_weight = 1.0
    debug_frontres_safe_gap_per_step = 0.003
    debug_frontres_broken_gap_per_step = 0.08
    debug_frontres_gap_gate_temp   = 0.005

    # ── Joint warmup before PPO loop ───────────────────────────────────────
    # Same rollout batch trains both halves of the concept:
    #   Actor:  Δ ≈ -noise
    #   Critic: E(s_noisy) ≈ max(R_feasible_oracle - R_noisy, 0)
    # PPO still keeps an online supervised anchor afterwards, so the transition
    # is gradual rather than a hard switch.
    supervised_warmup_iterations   = 400
    supervised_warmup_steps_per_iter = 8
    supervised_warmup_max_envs_per_step = 4096
    supervised_warmup_dr_scale_start = 0.35  # curriculum start: easy enough for stable direction learning
    supervised_warmup_dr_scale      = 1.25   # curriculum end: expose rp beyond the easy GMT regime
    supervised_warmup_lr           = 1e-4
    supervised_warmup_epochs       = 3
    supervised_warmup_diag_interval = 40
    # Each warmup rollout uses one active perturbation family, balanced across
    # planar, yaw, global_z, and local_rp inside the warmup update. This keeps
    # supervised labels clean without serially fitting only one repair direction
    # for many consecutive updates. Composite perturbations are postponed to PPO.
    supervised_warmup_perturbation_schedule = "balanced_single"

    # ── Adaptive DR: repairable-boundary controller ────────────────────────
    dr_scale_init                  = 1.0    # fixed during Actor takeover; calibrated near repairable GMT damage
    dr_adapt_speed                 = 0.001  # per-iteration step size
    dr_max_scale                   = 3.0    # upper limit; rp base 0.08 rad reaches 0.24 rad
    dr_min_scale                   = 0.30   # do not collapse below the xy/yaw debug signal floor
    dr_ema_alpha                   = 0.95   # r_delta EMA smoothing
    dr_start_ppo_actor_weight      = 1.0    # marks the actor-takeover phase for DR scheduling
    frontres_boundary_dr_enabled   = True
    frontres_boundary_dr_during_actor_takeover = True
    frontres_boundary_dr_ema_alpha = 0.90
    frontres_boundary_dr_step      = 0.03
    frontres_boundary_safe_high    = 0.45
    frontres_boundary_repair_low   = 0.45
    frontres_boundary_repair_high  = 0.70
    frontres_boundary_broken_target = 0.25
    frontres_boundary_broken_high  = 0.35
    frontres_boundary_positive_gain_low = 0.45
    frontres_boundary_positive_gain_high = 0.55

    # ── Perturbation curriculum ────────────────────────────────────────────
    # Per-env rollout mode curriculum.  Early training keeps each sample
    # single-family, but every PPO rollout mixes families across env triplets;
    # later training adds pair/three/full combinations. Boundary DR controls
    # magnitude inside each sampled mode.
    frontres_perturbation_curriculum_enabled = True
    frontres_curriculum_total_iterations = 1500
    frontres_curriculum_single_until = 0.30
    frontres_curriculum_two_until = 0.70
    frontres_curriculum_two_mid_prob = 0.35
    frontres_curriculum_two_late_prob = 0.40
    frontres_curriculum_three_prob = 0.10
    frontres_curriculum_full_prob = 0.05

    # ── Task-space correction ramp ────────────────────────────────────────────
    # Alpha must be 1.0 from the start so task-space corrections reach the
    # command term.  Stability is handled by conservative projection, confidence,
    # supervised warmup, and low initial DR scale.
    # ── IID step-jump perturbation probabilities (per-axis, per-step) ──────
    # Full-output test: use both local root artifacts (clear planar/yaw signal)
    # and vertical/tilt artifacts (clear z/rp signal).  IID channels are kept as
    # short shocks; OU-like float/sink/root tilt maintain persistent damage.
    iid_prob_z                     = 0.3
    iid_prob_xy                    = 0.1
    iid_prob_rp                    = 0.4
    iid_prob_ya                    = 0.1
    iid_std_z                      = 0.06   # Z jump std (m), scaled by dr_scale
    iid_std_xy                     = 0.15
    iid_std_rp                     = 0.08   # RP jump std (rad)
    iid_std_ya                     = 0.15
    local_root_artifact_prob       = 0.3
    local_root_artifact_min_steps  = 6
    local_root_artifact_max_steps  = 12
    local_root_artifact_xy_std     = 0.18   # metres at dr_scale=1; local contact/heading inconsistency
    local_root_artifact_yaw_std    = 0.24   # radians at dr_scale=1
    foot_slip_prob                 = 0.2
    foot_slip_ratio                = 0.008
    lateral_drift_prob             = 0.3
    lateral_drift_std              = 0.02
    joint_noise_prob               = 0.0
    joint_noise_std                = 0.0
    float_prob                     = 0.3
    float_ratio                    = 0.05
    # Sink artifacts are kept in the DR distribution, but the executable-gap
    # oracle only credits the part that is feasible under the active action cone
    # (mostly roll/pitch repair and jump-time penetration removal).  Root-level
    # upward Δz remains blocked during action application.
    sink_prob                      = 0.3
    sink_ratio                     = 0.04
    root_tilt_prob                 = 0.5
    root_tilt_max_rad              = 0.08

    # ── Legacy Critic warmup ───────────────────────────────────────────────
    # Disabled because Critic now learns executable damage during joint warmup.
    critic_warmup_iterations       = 0

    # ── DR PI controller, velocity form (Phase 3) ────────────────────────────
    # Δu = Kp×(e−e_prev) + Ki×e  →  dr_scale += Δu
    # Stays constant at error=0; no double-integration risk.
    dr_target_r_delta              = 0.01   # target r_delta/step; PI tracks this level
    dr_p_gain                      = 0.10   # P: reacts to error change (damping)
    dr_i_gain                      = 0.01   # I: reacts to error level  (steady-state)

    # 两台服务器上的 MOSAIC 根目录（不含实验子目录）
    candidate_gmt_paths = [
        "/home/yuxuancheng/MOSAIC/model/model_27000.pt", # SUST_Main_1
        "/hdd1/cyx/MOSAIC/model/model_27000.pt", # SUST_Main_2
    ]

    # 自动选择第一个真实存在的路径
    gmt_checkpoint_path_ = None
    for path in candidate_gmt_paths:
        if os.path.exists(path):
            gmt_checkpoint_path_ = path
            break

    policy = RslRlFrontResidualActorCriticCfg(
        class_name             = "FrontRESActorCritic",
        # ── Network ──────────────────────────────────────────────────────────
        residual_hidden_dims   = [512, 256, 128],
        residual_last_layer_gain = 0.01,
        critic_hidden_dims     = [1024, 1024, 512, 256],
        activation             = "elu",
        init_noise_std         = 0.01,     # fixed: 3mm noise floor, sufficient for PPO IS ratio
                                           # pos noise = 1.5cm, rpy noise = 1.5° — well below
                                           # DR perturbation (5-20cm) and GMT tracking error.
                                           # ratio ∈ [0.6, 1.6]: meaningful PPO dynamic range.
                                           # Tune range: 0.03-0.08 (effect is narrow).
        noise_std_type         = "scalar",
        # ── Task-space SE(3) correction mode ─────────────────────────────────
        num_task_corrections   = 6,        # bounded correction head = [Δpos(3), Δrpy(3)]; policy appends c_pos/c_rpy
        max_delta_pos          = 0.3,      # tanh clip (metres)
        max_delta_rpy          = 0.1,      # tanh clip (radians ≈ 5.7°); keep FrontRES rotation conservative
        # ── GMT (frozen) ─────────────────────────────────────────────────────
        gmt_checkpoint_path    = gmt_checkpoint_path_,
        init_critic_from_gmt   = False,
        # ── Observation layout ───────────────────────────────────────────────
        q_ref_start_idx        = 232,      # q_ref offset in 800-dim policy obs
        num_frontres_obs       = 0,        # 0 = FrontRES sees full obs
        # ── Δq / Δz unused in task-space mode ────────────────────────────────
        num_z_outputs          = 0,
        max_delta_q            = 0.5,
    )

    algorithm = RslRlFrontRESUnifiedAlgorithmCfg(
        # ── Mode ─────────────────────────────────────────────────────────────
        hybrid  = True,
        use_ppo = True,

        # ── PPO ──────────────────────────────────────────────────────────────
        value_loss_coef      = 1.0,
        use_clipped_value_loss = True,
        clip_param           = 0.2,
        entropy_coef         = 0.0,        # 0: FrontRES is a corrector, not an explorer.
                                           # Any entropy>0 unconditionally pushes σ up,
                                           # causing tanh saturation → random corrections
                                           # → anchor_ori terminations → σ explosion.
        num_learning_epochs  = 5,
        num_mini_batches     = 4,
        learning_rate        = 3.0e-5,
        schedule             = "fixed",    # fixed: adaptive KL deadlocks with FrontRES 8-DoF output
        gamma                = 0.99,
        lam                  = 0.95,
        desired_kl           = 0.01,       # kept as reference (not used in fixed mode)
        max_grad_norm        = 0.5,

        # ── Supervised auxiliary loss (λ_sup schedule) ────────────────────────
        frontres_training_objective  = "supervised_restore",
        lambda_supervised             = 1.0,   # initial weight
        lambda_supervised_min         = 1.0,   # supervised_restore keeps the target as the main objective
        lambda_supervised_decay       = 1.0,   # no decay in the supervised branch
        supervised_trigger_cosine_sim = 0.85,  # EMA threshold to start decay
        supervised_rpy_loss_weight    = 1.0,
        supervised_conf_loss_weight   = 0.0,   # BCE drives c→1 always (OU≠0); let PPO learn gating
        supervised_direction_loss_weight = 0.1,
        supervised_valid_loss_weight     = 4.0,
        # Joint warmup already initializes the Critic's executable-energy
        # estimate, but Actor takeover still changes the corrected-reference
        # distribution.  Ramp slowly so PPO does not push the warmup solution
        # out of its executable neighborhood in the first few hundred iters.
        ppo_actor_warmup_iterations   = 0,
        ppo_actor_ramp_iterations     = 400,
        ppo_advantage_focal_power     = 0.0,
        frontres_active_task_dims      = [3, 4, 7],
        diagnose_gradient_conflict    = True,

        # ── Misc ─────────────────────────────────────────────────────────────
        gradient_accumulation_steps    = 1,
        use_estimate_ref_vel           = False,
        ref_vel_estimator_checkpoint_path = None,
        ref_vel_estimator_type         = "mlp",
    )
