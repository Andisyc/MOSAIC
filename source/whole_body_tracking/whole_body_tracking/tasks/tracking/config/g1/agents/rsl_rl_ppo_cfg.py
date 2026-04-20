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

# Stage 1 实际训练停止的迭代数（由 checkpoint 决定）。
# Stage 2 的 max_iterations 必须在此基础上累加，否则 runner 加载 Stage 1
# checkpoint 后发现 current_iter >= max_iterations 会直接退出。
# IsaacLab 的工作流将 max_iterations - current_iter 作为 learn() 的实际轮次，
# 所以 max_iterations 是绝对迭代数上限，而非相对增量。
#
# Stage 1 已在第 25000 轮收敛（valid_ratio>0.99，MAE≈0，cosine_sim≈1），
# 尖刺为数据集中的困难动作片段（temporal gate 触发），属于监督学习天花板。
_STAGE1_MAX_ITERATIONS = 25000
# Stage 2 RL 微调额外轮次：
#   Phase 0 (Critic warmup)  1000 轮  — Actor 冻结，α = alpha_init = 0.1
#   Phase 1 (Ramp)           9000 轮  — α: 0.1→1.0，DR: min→max
#   Phase 2 (Full PPO)      30000 轮  — 标准 PPO，处理困难样本
#   合计                    40000 轮
_STAGE2_EXTRA_ITERATIONS = 40000

@configclass
class G1FlatSupervisedRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = _STAGE1_MAX_ITERATIONS
    save_interval = 500
    experiment_name = "g1_flat_supervised"
    empirical_normalization = True

    # GMT .pt checkpoint 路径（与 Stage 2 共用同一个 checkpoint，支持双服务器路径选择）
    _p1 = Path("/home/yuxuancheng/MOSAIC/model/model_27000.pt") # SUST_Main
    _p2 = Path("/home/chengyuxuan/MOSAIC/model/model_27000.pt") # Wujie_4090
    _gmt_pt_path = _p1 if _p1.exists() else (_p2 if _p2.exists() else None)

    policy = RslRlSuperviseJointPosCfg(
        class_name="SuperviseLearning",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        gmt_path=_gmt_pt_path,
    )

    algorithm = RslRlSuperviseAlgorithmCfg(
        loss_type="huber",
        # G1 29-DOF lower-limb joint indices (hip × 6, knee × 2, ankle × 4 = 12 joints).
        # Adjust if your joint ordering differs — cross-check with the obs joint_pos slice.
        lower_limb_indices=list(range(12)),
        lower_limb_weight=2.0,
        jump_threshold=0.2,
    )

# ====== FrontRES Stage 1: Supervised Learning ======

# ========== FrontRES Stage 2: RL Finetune ==========

@configclass
class G1FlatFrontRESFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Stage 2: RL Finetuning of FrontRES."""
    num_steps_per_env = 24
    max_iterations = _STAGE1_MAX_ITERATIONS + _STAGE2_EXTRA_ITERATIONS  # 65000
    save_interval = 500
    experiment_name = "g1_flat_frontres_finetune"
    empirical_normalization = True
    resume = True  # 从 Stage 1 checkpoint 加载权重

    # Checkpoint 路径（绝对路径，直接传给 runner.load()，绕过 log_root_path 拼接）
    # train.py 检测到 student_checkpoint_path 存在时优先使用，否则回退到 load_run/load_checkpoint 机制
    # is_full_resume=False → Stage 1 权重迁移（冷启动）; is_full_resume=True → Stage 2 断点续训
    _s1 = Path("/home/yuxuancheng/MOSAIC/stage2/model_58500.pt")  # SUST_Main
    _s2 = Path("/home/chengyuxuan/MOSAIC/stage2/model_58500.pt")  # Wujie_4090
    student_checkpoint_path = _s1 if _s1.exists() else (_s2 if _s2.exists() else None)

    # ── 断点续训模式控制 ──────────────────────────────────────────────────────
    # True  = Stage 2 → Stage 2 断点续训：
    #           恢复优化器矩估计（Adam moments）、学习率、噪声 std
    #           适用于：因参数调整暂停后继续训练
    # False = Stage 1 → Stage 2 权重迁移（冷启动）：
    #           仅加载 residual_actor/critic 权重，重置优化器和 std
    #           适用于：首次从 Stage 1 checkpoint 启动 Stage 2
    is_full_resume: bool = True

    # lr 重置：schedule="fixed" 时 lr 由配置直接控制，无需 reset（固定值覆盖 checkpoint）。
    # schedule="adaptive" 且 checkpoint lr 因 KL bug 被压至下限时才需要 True。
    reset_lr_on_resume: bool = True  # fixed 模式下同样生效：确保 lr 从配置的 3e-5 开始

    # DR 课程：Stage 2 开始时 MotionPerturber 强度线性从 0 增长到 motion_perturbations 配置值。
    # 防止 FrontRES 在 Stage 2 冷启动时面对完全 OOD 的 q_ref，导致全负 r_delta → Δq=0 捷径陷阱。
    # 设为 0 禁用（直接使用全强度 DR，仅在 FrontRES 已对 Stage 2 DR 鲁棒时使用）。
    dr_curriculum_iterations: int = 5000
    # Stage 2 起始绝对迭代数，用于 DR 课程进度计算，支持断点续训时正确恢复 DR 强度。
    stage2_start_iteration: int = _STAGE1_MAX_ITERATIONS

    # 阶梯 DR 课程：初始课程完成后，自动按台阶递增扰动幅度直至 FrontRES 崩溃或训练结束。
    # 工作原理：
    #   监控 r_delta_per_step 的 EMA 斜率；当斜率趋近于零（平台期）且停留时间 ≥
    #   dr_staircase_min_plateau_iters 时，自动以 dr_staircase_ramp_iters 线性爬坡
    #   到下一台阶倍率，然后继续监控。
    # 倍率列表（相对 motion_perturbations 基础值，仅缩放幅度 ratio，不改变概率 prob）：
    #   例：float_ratio=0.05 × 1.6 → 0.08 m；× 2.4 → 0.12 m；× 3.6 → 0.18 m；× 5.0 → 0.25 m
    # 设为空字符串 "" 禁用阶梯课程（只用初始 0→full 的单次课程）。
    dr_staircase_multipliers: str = "1.6,2.4,3.6,5.0"
    # 每次台阶切换时线性爬坡到新幅度的迭代数（过渡期，防止突变触发 Δq=0 捷径陷阱）
    dr_staircase_ramp_iters: int = 2000
    # 每台阶最短停留迭代数（短于此时间不触发晋级，避免噪声误判为平台期）
    dr_staircase_min_plateau_iters: int = 3000
    # EMA 斜率绝对值低于此阈值即判定为平台期（单位：r_delta/iter）
    dr_staircase_plateau_threshold: float = 2e-7
    # 断点续训时从第几个台阶开始（0 = 基础台阶；1 = 第一次晋级后，以此类推）
    # 首次从旧 checkpoint (无 staircase 状态) 续训时设为 1：
    #   跳过已确认饱和的 level 0 (float_ratio=0.05)，直接从 1.6× 爬坡开始。
    # 后续续训由 checkpoint 自动恢复正确 level，此字段不再生效。
    dr_staircase_start_level: int = 1

    # Critic 预热：禁用（=0）。
    #
    # 禁用原因：actor-frozen warmup 会在 Critic 与训练阶段之间制造人为的分布错配。
    #   - Warmup 期间 Critic 学 V(s | π_frozen, α_init)
    #   - 训练开始后策略变为 π_trainable, α=1.0 → 完全不同的分布
    #   - Critic 的 V 估计立刻过时 → 优势方差爆炸 → std 崩塌
    #
    # OOD 担忧（Stage 1 → Stage 2 DR 分布偏移）不是 Critic warmup 要解决的问题：
    #   - OOD 初始表现差 → r_delta 为负 → Critic 正确学到 V(s) < 0 → 有效梯度信号
    #   - tanh × max_delta_q 限制 Δq 幅度，防止灾难性摔倒
    #   - 这就是标准 RL 的"初始策略不佳"问题，PPO 自然处理
    critic_warmup_iterations = 0

    # Δq alpha 课程：禁用（直接 alpha=1.0 全量训练）。
    #
    # 禁用原因：alpha=0.1 导致 r_delta≈0，Critic 学到 V≈0，
    # 与后续 alpha=1.0 时的真实价值函数完全不匹配，训练不稳定。
    # alpha 直接设为 1.0 意味着 Critic 全程跟踪实际策略分布，收敛更稳定。
    delta_q_alpha_init = 1.0          # 全程 alpha=1.0，无课程
    delta_q_alpha_ramp_iterations = 0  # 无 ramp

    # GMT .pt 路径（FrontRESActorCritic 用 torch.load() 加载，必须是 .pt 而非 .onnx，支持双服务器路径选择）
    _p1 = Path("/home/yuxuancheng/MOSAIC/model/model_27000.pt")
    _p2 = Path("/home/chengyuxuan/MOSAIC/model/model_27000.pt")
    _gmt_pt_path = _p1 if _p1.exists() else (_p2 if _p2.exists() else None)

    policy = RslRlFrontResidualActorCriticCfg(
        class_name="FrontRESActorCritic",

        residual_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",

        init_noise_std=0.1,                # 微调使用较小初始探索噪声，防止动作崩坏
        gmt_checkpoint_path=_gmt_pt_path,  # 冻结的 GMT 权重（.pt 格式，torch.load 加载）

        # q_ref 在策略观测向量中的起始索引。
        #
        # IsaacLab 以 per-group 方式拼接历史：先将所有时间步的 command 拼在一起，
        # 再拼 motion_anchor_ori_b，以此类推（而非 per-timestep 交错）。
        # 历史顺序为 oldest→newest：[t-4, t-3, t-2, t-1, t]
        #
        # obs 布局（history_length=5，obs_dim=770）：
        #   [0:290]    command (q_ref_pos(29)+q_ref_vel(29)) × 5 帧
        #     [0:29]   q_ref_pos at t-4  ← index 0（最旧帧，已过去，错误目标）
        #     [232:261] q_ref_pos at t   ← 当前帧，FrontRES 应修正的目标
        #   [290:320]  motion_anchor_ori_b × 5 帧
        #   [320:335]  base_ang_vel × 5 帧
        #   [335:480]  joint_pos_rel × 5 帧
        #   [480:625]  joint_vel_rel × 5 帧
        #   [625:770]  actions × 5 帧
        #
        # 计算：当前帧偏移 = (history_length - 1) × command_per_frame
        #                   = (5 - 1) × 58 = 232
        # q_ref_pos 在当前帧的起始索引 = 232（q_ref_vel 在 261:290）
        q_ref_start_idx=232,

        # Δq 输出截断：tanh(raw) * max_delta_q，防止 action explosion。
        # 0.5 rad ≈ ±28.6°/关节，覆盖典型 sim-to-real 修正范围。
        max_delta_q=0.5,
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
