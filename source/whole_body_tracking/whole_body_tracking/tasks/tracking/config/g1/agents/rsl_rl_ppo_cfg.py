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

# ── 公共路径：GMT backbone（Stage 1 和 Stage 2 共用，始终需要）──────────────
_GMT_P1 = Path("/home/yuxuancheng/MOSAIC/model/model_27000.pt")  # SUST_Main
_GMT_P2 = Path("/home/chengyuxuan/MOSAIC/model/model_27000.pt")  # Wujie_4090
_GMT_PATH = _GMT_P1 if _GMT_P1.exists() else (_GMT_P2 if _GMT_P2.exists() else None)

@configclass
class G1FlatSupervisedRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 25000
    save_interval = 500
    experiment_name = "g1_flat_supervised"
    empirical_normalization = True

    policy = RslRlSuperviseJointPosCfg(
        class_name="SuperviseLearning",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        gmt_path=_GMT_PATH,
        # Task-space mode: FrontRES outputs [Δx, Δy, Δz, Δroll, Δpitch, Δyaw] (6 dims).
        # obs layout (history_length=5, motion_horizon=1): 800 dims, no future frames
        #   [0:290]    command  q_ref_pos(29)+q_ref_vel(29)  × 5 history frames
        #   [290:320]  motion_anchor_ori_b  6 dims           × 5 history frames
        #   [320:335]  base_ang_vel         3 dims           × 5 history frames
        #   [335:480]  joint_pos_rel        29 dims          × 5 history frames
        #   [480:625]  joint_vel_rel        29 dims          × 5 history frames
        #   [625:770]  actions              29 dims          × 5 history frames
        #   [770:785]  anchor_root_pos_error_w  3 dims       × 5 history frames
        #   [785:800]  anchor_root_rpy_error_w  3 dims       × 5 history frames
        num_task_corrections=6,
    )

    algorithm = RslRlSuperviseAlgorithmCfg(
        loss_type="huber",
        num_task_corrections=6,
        rpy_loss_weight=1.0,
        # Δq-mode params (inactive in task-space mode, retained for reference):
        lower_limb_indices=list(range(12)),
        lower_limb_weight=2.0,
        jump_threshold=0.2,
        num_joint_outputs=29,
    )

# ====== FrontRES Stage 1: Supervised Learning ======

# ========== FrontRES Stage 2: RL Finetune ==========

# ── 迭代数说明 ─────────────────────────────────────────────────────────────
# max_iterations 是绝对上限（非增量）。runner 加载 checkpoint 后以
#   remaining = max_iterations - current_iter  作为实际运行轮次。
# Stage 2 RL 微调规划（40000 轮）：
#   Phase 1 (DR ramp)   5000 轮 — DR 从 0 线性爬坡到全强度
#   Phase 2 (Full PPO) 35000 轮 — 标准 PPO 全强度训练
# _S1_CKPT_ITER: 所选 Stage 1 checkpoint 的迭代数（runner 从该数继续计数）。
_S1_CKPT_ITER        = 10000   # ← 更换 Stage 1 checkpoint 时同步修改此处
_STAGE2_EXTRA_ITERS  = 40000
# max_iterations = _S1_CKPT_ITER + _STAGE2_EXTRA_ITERS = 50000

# ── Checkpoint 路径 ────────────────────────────────────────────────────────
# 从 Stage 1 冷启动（首次）：填 Stage 1 的 model_XXXXX.pt，is_full_resume=False
# 从 Stage 2 断点续训：填 Stage 2 最新的 model_XXXXX.pt，is_full_resume=True
_S1_CKPT_P1 = Path("/home/yuxuancheng/MOSAIC/stage1/model_10000.pt")
_S1_CKPT_P2 = Path("/home/chengyuxuan/MOSAIC/stage1/model_10000.pt")
# 续训时将下方路径填入 student_checkpoint_path（替换 _S1_CKPT_P*）：
# /home/yuxuancheng/MOSAIC/resume/model_XXXXX.pt
# /home/chengyuxuan/MOSAIC/resume/model_XXXXX.pt

@configclass
class G1FlatFrontRESFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Stage 2: RL Finetuning of FrontRES."""
    num_steps_per_env = 24
    max_iterations = _S1_CKPT_ITER + _STAGE2_EXTRA_ITERS  # 50000
    save_interval = 500
    experiment_name = "g1_flat_finetune"
    empirical_normalization = True
    resume = True

    # ══════════════════════════════════════════════════════════════════════
    #  模式切换（每次启动前确认以下两项）
    # ══════════════════════════════════════════════════════════════════════
    #
    #  Stage 1 → Stage 2  冷启动（首次 RL 微调）
    #    student_checkpoint_path = _S1_CKPT_P1 if ... else _S1_CKPT_P2 ...
    #    is_full_resume = False
    #      效果：仅迁移 FrontRES 网络权重，优化器/噪声 std 重置为初始值
    #
    #  Stage 2 → Stage 2  断点续训
    #    student_checkpoint_path = _S2_CKPT_P1 if ... else _S2_CKPT_P2 ...
    #    is_full_resume = True
    #      效果：恢复优化器 Adam moments、学习率、噪声 std，无缝续训
    #
    # ══════════════════════════════════════════════════════════════════════
    student_checkpoint_path = (
        _S1_CKPT_P1 if _S1_CKPT_P1.exists() else
        _S1_CKPT_P2 if _S1_CKPT_P2.exists() else None
    )
    is_full_resume: bool = False  # Stage 1→2 冷启动；续训改为 True

    reset_lr_on_resume: bool = True   # fixed schedule 下确保 lr 从配置值 3e-5 开始

    # ── 自适应 DR：目标存活率 PI 控制器 ───────────────────────────────────────
    # 原理：以训练环境的逐步存活率（per-step survival）作为反馈，
    #        连续调节 DR 倍率 dr_scale ∈ [0, dr_max_scale]，使存活率维持在目标附近。
    # 优势：无需手动指定台阶，DR 随 FrontRES 能力自动升降，训练不稳定时可自动退回。
    #
    # dr_target_episode_length: 目标平均回合长度（步数）。
    #   对应 per-step 存活率目标 = 1 - 1/target_ep_len。
    #   60 步 ≈ 半倍 GMT 基线（125步），此时机器人有足够的失败率提供 r_delta 信号。
    dr_target_episode_length: int   = 60     # 目标回合长度（步）
    #
    # dr_adapt_speed: 每轮迭代 dr_scale 最大变化量。
    #   0.0005/iter → 从 0 到 1.0（base 值全强度）最快需 2000 轮；到 4.0 最快 8000 轮。
    #   实际因存活率降低而减速，总爬升约 15000–25000 轮。
    dr_adapt_speed:  float = 0.0005
    #
    # dr_max_scale: dr_scale 上限（base 值的倍率）。
    #   4.0 × base_values（base 已设为原始值的 1/4）= 恢复到原始最大扰动强度。
    dr_max_scale:    float = 4.0
    #
    # dr_ema_alpha: 存活率 EMA 平滑系数，防止单步噪声触发过度调整。
    dr_ema_alpha:    float = 0.95
    #
    # dr_deadband: 死区（per-step 存活率），避免在目标附近频繁调整。
    #   0.005 ≈ ±5 步回合长度差（在 60 步基准下）。
    dr_deadband:     float = 0.005
    #
    # dr_init_scale: Stage1→Stage2 冷启动时的初始 dr_scale（仅在 is_full_resume=False 时生效）。
    #   默认 1.0：从 Stage1 训练所用扰动强度（Level 1）出发，确保 Stage1 修正策略
    #   在 Stage2 初期仍然适用，避免干净参考上修正有害导致的即时崩溃。
    #   断点续训（is_full_resume=True）时该值被忽略，直接从 checkpoint 恢复。
    dr_init_scale:   float = 1.0
    #
    # dr_min_scale: dr_scale 下限（PI 控制器不会将 dr_scale 降至此值以下）。
    #   0.3 保证始终存在 Level 0.3 扰动，防止 FrontRES 滑入"零修正"无效果捷径，
    #   同时不给 GMT 施加过大的负担使 episode 崩溃。
    #   若生存率在 dr_min 处仍低于目标，说明 GMT 无法跟踪该运动，需从根本修复。
    dr_min_scale:    float = 0.3

    # Critic 预热：禁用。actor-frozen warmup 会制造 V 估计分布错配，PPO 自然处理 OOD 冷启动。
    critic_warmup_iterations = 0

    # FrontRES 监督预热：训练的前 N 轮使用监督损失（Huber）而非 PPO。
    # 目标 = -(perturbed - original)，来源 = command term 的 anchor_dr_delta_*。
    # 0 = 禁用（纯 PPO）；>0 = 先监督预热再切 PPO。
    # 例如 2000：前 2000 轮用 Huber 学习扰动检测，之后切换 PPO 微调。
    supervised_warmup_iterations: int = 0
    supervised_warmup_lr:        float = 1e-4  # warmup 阶段的学习率
    supervised_warmup_epochs:    int   = 5     # 每次 rollout 后监督学习的 epoch 数

    # Δq alpha 课程：禁用。alpha=0.1 会使 Critic 学到 V≈0，与后续 alpha=1.0 严重错配。
    delta_q_alpha_init         = 1.0
    delta_q_alpha_ramp_iterations = 0

    policy = RslRlFrontResidualActorCriticCfg(
        class_name="FrontRESActorCritic",

        residual_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",

        init_noise_std=0.1,                # 微调使用较小初始探索噪声，防止动作崩坏
        gmt_checkpoint_path=_GMT_PATH,  # 冻结的 GMT 权重（.pt 格式，torch.load 加载）

        # q_ref 在策略观测向量中的起始索引。
        #
        # IsaacLab 以 per-group 方式拼接历史：先将所有时间步的 command 拼在一起，
        # 再拼 motion_anchor_ori_b，以此类推（而非 per-timestep 交错）。
        # 历史顺序为 oldest→newest：[t-4, t-3, t-2, t-1, t]
        #
        # obs 布局（history_length=5，obs_dim=800）：
        #   [0:290]    command (q_ref_pos(29)+q_ref_vel(29)) × 5 帧
        #     [0:29]   q_ref_pos at t-4  ← index 0（最旧帧，已过去，错误目标）
        #     [232:261] q_ref_pos at t   ← 当前帧，FrontRES 修正目标
        #   [290:320]  motion_anchor_ori_b (6 dim) × 5 帧
        #   [320:335]  base_ang_vel (3 dim) × 5 帧
        #   [335:480]  joint_pos_rel (29 dim) × 5 帧
        #   [480:625]  joint_vel_rel (29 dim) × 5 帧
        #   [625:770]  actions (29 dim) × 5 帧
        #   [770:785]  anchor_root_pos_error_w (3 dim) × 5 帧  ← 任务空间位置误差
        #   [785:800]  anchor_root_rpy_error_w (3 dim) × 5 帧  ← 任务空间姿态误差
        #
        # q_ref_start_idx 不受新增 term 影响（新 term 追加在末尾）。
        # 计算：当前帧偏移 = (history_length - 1) × command_per_frame
        #                   = (5 - 1) × 58 = 232
        # q_ref_pos 在当前帧的起始索引 = 232（q_ref_vel 在 261:290）
        q_ref_start_idx=232,

        # Δq 输出截断：tanh(raw) * max_delta_q（在 task-space 模式下不使用 Δq）。
        max_delta_q=0.5,

        # 任务空间模式：FrontRES 输出 [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]（6 维）。
        # max_delta_pos=0.3m 覆盖典型 float/sink/slip 幅度。
        # max_delta_rpy=0.3rad ≈ 17° 覆盖典型倾斜幅度。
        num_task_corrections=6,
        max_delta_pos=0.3,
        max_delta_rpy=0.3,

        # FrontRES 只处理参考帧信息（前 320 维 = command + motion_anchor_ori_b），
        # 移除 proprioception（base_ang_vel, joint_pos/vel, actions, anchor errors）。
        # GMT 继续使用完整 800 维观测。
        num_frontres_obs=320,
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
        entropy_coef=0.01,           # OOD 初始阶段需要足够探索：0.001 太小（梯度~0.001/σ），0.01 有效
        learning_rate=3e-5,          # 从 lr=1e-5 温和升至 3e-5（3×），避免破坏已有行为
        schedule="fixed",            # 彻底绕开 adaptive schedule 死锁：
                                     #   adaptive 在此系统的结构性最小 KL≈3 >> desired_kl=0.23，
                                     #   闭环只降不升，lr 从第一天起永久压在 floor(1e-5)。
                                     #   fixed 模式下 desired_kl 不生效，lr 由配置直接控制。
        desired_kl=0.23,             # 保留作记录（fixed 模式下不生效）
        max_grad_norm=1.0,

        # --- FrontRES 正则化：防止修正量过大 ---
        # L_reg = λ_reg * ||Δq_mean||^2 = ||q' - q_ref||^2 的等价形式
        # lambda_reg_init=0.1 时 reg_penalty_per_step ≈ 0.028，是 r_delta 信号(≈0.001)的28倍，
        # 导致 mean_r_delta 被掩盖在 -3→0 的范围内，严重阻碍学习。
        # 0.005 时 reg_penalty_per_step ≈ 0.0014，与 r_delta 信号量级匹配。
        lambda_reg_init=0.005,       # 降低至 0.005（原为 0.1，28倍过强）
        lambda_reg_decay=1.0,        # 不衰减（正则化是持续的安全约束）
        lambda_reg_min=0.0,

        # --- PCGrad：自适应协调 PPO 梯度与正则化梯度 ---
        # True = 启用投影；False = 简单加权（先用 False 调稳定后再开）
        use_pcgrad=False,
    )

# ========== FrontRES Stage 2: RL Finetune ==========
