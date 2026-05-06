from pathlib import Path

from isaaclab.utils import configclass

# IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/
# algorithm: RslRlPpoAlgorithmCfg, RslRlDistillationAlgorithmCfg
# policy: RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationAlgorithmCfg
)

# Unified training (Stage 1+2 merged) → see rsl_rl_mosaic_cfg.py :: G1FlatFrontRESUnifiedRunnerCfg
# The imports below were only needed for the deprecated two-stage approach:
# from whole_body_tracking.utils.supervise import (
#     SuperviseTrainer                          # Stage 1 supervised trainer
# )

from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlPpoActorCriticTransformerCfg,
    RslRlPpoActorCriticFSQCfg,
    RslRlPpoActorCriticVQCfg,
    RslRlPpoActorCriticAttentionCfg,
    RslRlDistillationCfg,
    # Deprecated two-stage imports (kept for reference, not used):
    # RslRlSuperviseJointPosCfg,           # Stage 1 policy  → replaced by lambda_supervised
    # RslRlSuperviseAlgorithmCfg,          # Stage 1 algorithm → replaced by lambda_supervised
    # RslRlFrontResidualActorCriticCfg,    # Stage 2 policy  → now in rsl_rl_mosaic_cfg.py
    # RslRlPpoFrontRESAlgorithmCfg,        # Stage 2 algorithm → now in rsl_rl_mosaic_cfg.py
)


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "g1_flat_mosaic_hybrid"
    # experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

    algorithm = RslRlPpoAlgorithmCfg(
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
        max_grad_norm=1.0,)


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)

@configclass
class G1FlatDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length = 15)


@configclass
class G1FlatKLDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Configuration for KL-based distillation (improved version).

    This uses KL divergence loss instead of MSE, matching MOSAIC's approach.
    Expected to provide better imitation performance than standard distillation.
    """
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_kl_distillation"
    empirical_normalization = True

    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

# ======================================================================
# DEPRECATED: Two-stage (Stage 1 supervised + Stage 2 RL) approach.
# Both classes below are superseded by G1FlatFrontRESUnifiedRunnerCfg
# in rsl_rl_mosaic_cfg.py, which merges both stages into a single run
# via lambda_supervised + B1 delta-reward PPO.
# ======================================================================

# -- Shared GMT path (referenced by deprecated classes, kept for record) --
# _GMT_P1 = Path("/home/yuxuancheng/MOSAIC/model/model_27000.pt")  # SUST_Main
# _GMT_P2 = Path("/home/chengyuxuan/MOSAIC/model/model_27000.pt")  # Wujie_4090
# _GMT_PATH = _GMT_P1 if _GMT_P1.exists() else (_GMT_P2 if _GMT_P2.exists() else None)

# -- Stage 1: Supervised pre-training (replaced by lambda_supervised in MOSAIC) --
# @configclass
# class G1FlatSupervisedRunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 25000
#     save_interval = 500
#     experiment_name = "g1_flat_supervised"
#     empirical_normalization = True
#     policy = RslRlSuperviseJointPosCfg(
#         class_name="SuperviseLearning",
#         init_noise_std=1.0,
#         student_hidden_dims=[1024, 1024, 512, 256],
#         activation="elu",
#         gmt_path=_GMT_PATH,
#         num_task_corrections=6,
#     )
#     algorithm = RslRlSuperviseAlgorithmCfg(
#         loss_type="huber",
#         num_task_corrections=6,
#         rpy_loss_weight=1.0,
#         lower_limb_indices=list(range(12)),
#         lower_limb_weight=2.0,
#         jump_threshold=0.2,
#         num_joint_outputs=29,
#     )

# -- Stage 2 path variables (only used by G1FlatFrontRESFinetuneRunnerCfg) --
# _S1_CKPT_ITER       = 10000
# _STAGE2_EXTRA_ITERS = 40000
# _S1_CKPT_P1 = Path("/home/yuxuancheng/MOSAIC/stage1/model_10000.pt")
# _S1_CKPT_P2 = Path("/home/chengyuxuan/MOSAIC/stage1/model_10000.pt")

# -- Stage 2: RL fine-tuning (replaced by unified runner in rsl_rl_mosaic_cfg.py) --
# Key parameters preserved here for reference when configuring the unified runner:
#   q_ref_start_idx  = 232   (current-frame q_ref_pos offset in 800-dim obs)
#   num_frontres_obs = 320   (command[58×5] + anchor_ori[6×5] = ref-only subset)
#   num_task_corrections = 6  ([Δpos(3), Δrpy(3)] task-space SE3 output)
#   max_delta_pos = 0.3      (tanh clip, metres)
#   max_delta_rpy = 0.3      (tanh clip, radians ≈ 17°)
#   lambda_reg_init = 0.005  (regularisation weight, matched to r_delta magnitude)
#   learning_rate   = 3e-5   (fixed schedule, avoids adaptive lr deadlock)
#
# @configclass
# class G1FlatFrontRESFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
#     ...  (full body omitted — see git history if needed)
