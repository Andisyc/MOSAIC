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

# PAMR: Stage 1 Training
from whole_body_tracking.utils.supervise import (
    SuperviseTrainer
)

# policy: RslRlDistillationCfg
from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlPpoActorCriticTransformerCfg,
    RslRlPpoActorCriticFSQCfg,
    RslRlPpoActorCriticVQCfg,
    RslRlPpoActorCriticAttentionCfg,
    RslRlDistillationCfg,
    RslRlSuperviseJointPosCfg,           # FrontRES Stage 1 policy
    RslRlSuperviseAlgorithmCfg,          # FrontRES Stage 1 algorithm
    RslRlFrontResidualActorCriticCfg,    # FrontRES Stage 2 policy
    RslRlPpoFrontRESAlgorithmCfg,        # FrontRES Stage 2 algorithm (with reg + PCGrad)
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

# ====== FrontRES Stage 1: Supervised Learning ======

# Stage 1 训练总迭代数，Stage 2 的 max_iterations 必须在此基础上累加，
# 否则 runner 加载 Stage 1 checkpoint 后发现 current_iter >= max_iterations 会直接退出。
_STAGE1_MAX_ITERATIONS = 200000
_STAGE2_EXTRA_ITERATIONS = 50000

@configclass
class G1FlatSupervisedRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = _STAGE1_MAX_ITERATIONS
    save_interval = 500
    experiment_name = "g1_flat_supervised"
    empirical_normalization = True

    # GMT ONNX 路径（SuperviseLearning 用 onnxruntime 加载，支持双服务器路径选择）
    _p1 = Path("/home/yuxuancheng/MOSAIC/model/exported/policy.onnx")
    _p2 = Path("/home/chengyuxuan/MOSAIC/model/exported/policy.onnx")
    _gmt_onnx_path = _p1 if _p1.exists() else (_p2 if _p2.exists() else None)

    policy = RslRlSuperviseJointPosCfg(
        class_name="SuperviseLearning",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        gmt_path=_gmt_onnx_path,)

    algorithm = RslRlSuperviseAlgorithmCfg()

# ====== FrontRES Stage 1: Supervised Learning ======

# ========== FrontRES Stage 2: RL Finetune ==========

@configclass
class G1FlatFrontRESFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Stage 2: RL Finetuning of FrontRES."""
    num_steps_per_env = 24
    max_iterations = _STAGE1_MAX_ITERATIONS + _STAGE2_EXTRA_ITERATIONS  # 250000
    save_interval = 500
    experiment_name = "g1_flat_frontres_finetune"
    empirical_normalization = True
    resume = True  # 从 Stage 1 checkpoint 加载权重

    # Stage 1 → Stage 2 过渡时需要重置探索噪声 std，
    # 否则 runner 会沿用 checkpoint 里的旧值，忽略 init_noise_std=0.1 的设定。
    reset_noise_std_on_resume = True

    # Critic 预热：训练开始后冻结 Actor（FrontRES residual_actor）前 N 个 iteration，
    # 让 Critic 先在正确的奖励函数下收敛，再解冻 Actor 进行联合训练。
    # 设为 0 则禁用预热，直接联合训练。
    critic_warmup_iterations = 1000

    # GMT .pt 路径（FrontRESActorCritic 用 torch.load() 加载，必须是 .pt 而非 .onnx，支持双服务器路径选择）
    _p1 = Path("/home/yuxuancheng/MOSAIC/logs/rsl_rl/g1_flat/exported/policy.pt")
    _p2 = Path("/home/chengyuxuan/MOSAIC/logs/rsl_rl/g1_flat/exported/policy.pt")
    _gmt_pt_path = _p1 if _p1.exists() else (_p2 if _p2.exists() else None)

    policy = RslRlFrontResidualActorCriticCfg(
        class_name="FrontRESActorCritic",

        residual_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",

        init_noise_std=0.1,                # 微调使用较小初始探索噪声，防止动作崩坏
        gmt_checkpoint_path=_gmt_pt_path,  # 冻结的 GMT 权重（.pt 格式，torch.load 加载）

        # q_ref 在策略观测向量中的起始索引。
        # ObservationsCfg.PolicyCfg 的第一项是 command = [joint_pos(29), joint_vel(29)]，
        # 因此 q_ref (joint_pos) 从 index 0 开始。
        # 如果启用了 history_length=5，需要确认 MultiMotionCommand 在哪一帧切片。
        # 建议先以 q_ref_start_idx=0 运行一步 debug，打印 obs[0, 0:29] 与 q_ref 比对。
        q_ref_start_idx=0,
    )

    algorithm = RslRlPpoFrontRESAlgorithmCfg(
        # --- 基础运行参数 ---
        num_learning_epochs=5,       # 每次拿 buffer 里的数据训练几次
        num_mini_batches=4,          # 切分 batch
        gamma=0.99,                  # 折扣因子
        lam=0.95,                    # GAE 参数

        # --- 微调优化参数 ---
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,              # 限制策略更新幅度
        entropy_coef=0.001,          # 微调阶段探索熵系数可以很小
        learning_rate=5.0e-4,        # 微调必须用比从零训练更小的学习率
        schedule="adaptive",
        desired_kl=0.008,            # 要求策略更新比标准 PPO 更平滑
        max_grad_norm=1.0,

        # --- FrontRES 正则化：防止修正量过大 ---
        # L_reg = λ_reg * ||Δq_mean||^2 = ||q' - q_ref||^2 的等价形式
        lambda_reg_init=0.01,        # 初始权重，约为 PPO loss 量级的 1/10
        lambda_reg_decay=1.0,        # 不衰减（正则化是持续的安全约束）
        lambda_reg_min=0.0,

        # --- PCGrad：自适应协调 PPO 梯度与正则化梯度 ---
        # True = 启用投影；False = 简单加权（先用 False 调稳定后再开）
        use_pcgrad=False,
    )

# ========== FrontRES Stage 2: RL Finetune ==========
