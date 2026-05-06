# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation, MOSAIC
from whole_body_tracking.utils.supervise import SuperviseTrainer
from rsl_rl.modules.supervise_learning import SuperviseLearning
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticFSQ,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
    ActorCriticTransformer,
    ActorCriticVQ,
    ActorCriticAttention,
    ResidualActorCritic,
    FrontRESActorCritic, # 引入第二阶段模型
)
from rsl_rl.utils import store_code_state
from isaaclab.utils.math import quat_from_euler_xyz


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm 训练算法
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "MOSAIC":
            self.training_type = "mosaic"  # MOSAIC has its own training type with teacher action storage
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        elif self.alg_cfg["class_name"] == "SuperviseTrainer":
            self.training_type = "supervise"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations 观测量维度
        obs, extras = self.env.get_observations()
        obs_dict = extras.get("observations", {})
        if "policy" in obs_dict:
            self.policy_obs_type = "policy"
            obs = obs_dict["policy"]
        else:
            self.policy_obs_type = None
        if "teacher" in obs_dict:
            self.teacher_obs_type = "teacher"
        else:
            self.teacher_obs_type = None
        num_obs = obs.shape[1]

        # resolve type of privileged observations 特权信息
        if self.training_type == "rl":
            if "critic" in obs_dict:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learning, e.g., PPO
            else:
                self.privileged_obs_type = None
        elif self.training_type == "mosaic":
            # MOSAIC uses critic observations for value function when available.
            # Teacher observations are handled separately for teacher BC.
            has_teacher_obs = "teacher" in obs_dict
            has_critic_obs = "critic" in obs_dict
            if has_critic_obs:
                self.privileged_obs_type = "critic"
                print(f"[MOSAIC] Using 'critic' observations for value estimation.")
            elif has_teacher_obs:
                self.privileged_obs_type = "teacher"
                print(f"[MOSAIC] Using 'teacher' observations for value estimation (no critic obs available).")
            else:
                self.privileged_obs_type = None
        elif self.training_type == "distillation":
            if "teacher" in obs_dict:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None
        elif self.training_type == "supervise":
            if "target" in obs_dict:
                self.privileged_obs_type = "target"
            else:
                self.privileged_obs_type = None

        # resolve type of ref_vel_estimator observations (for MOSAIC with velocity estimator) 速度估计器
        if "ref_vel_estimator" in obs_dict:
            self.ref_vel_estimator_obs_type = "ref_vel_estimator"
            num_ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].shape[1]
            print(f"[Runner] Found 'ref_vel_estimator' observations for velocity estimation (dim={num_ref_vel_estimator_obs}).")
        else:
            self.ref_vel_estimator_obs_type = None

        # resolve dimensions of privileged observations 特权信息维度
        if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
            num_privileged_obs = obs_dict[self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
            num_teacher_obs = obs_dict[self.teacher_obs_type].shape[1]
        else:
            num_teacher_obs = None

        # Adjust actor input dimension if using velocity estimator (MOSAIC with estimated ref vel)
        # The actor will receive obs_augmented = [obs, estimated_ref_vel] where estimated_ref_vel is 3D
        # IMPORTANT: Keep num_obs unchanged for normalizer initialization!
        # IMPORTANT: For ResidualActorCritic, do NOT adjust num_actor_obs (it handles estimator internally)
        num_actor_obs = num_obs  # Start with policy obs dimension 动作维度

        # evaluate the policy class (非常危险的做法)
        # eval会将字符串直接作为python代码执行, class_name="ResidualActorCritic"
        # eval会直接将字符串"ResidualActorCritic"变为ResidualActorCritic类的实例
        policy_class = eval(self.policy_cfg.pop("class_name"))

        # Check if using ResidualActorCritic (special handling for estimator dimension)
        is_residual_policy = policy_class in [ResidualActorCritic, FrontRESActorCritic]

        if self.training_type == "mosaic" and self.alg_cfg.get("use_estimate_ref_vel", False):
            if not is_residual_policy:
                # For normal ActorCritic: adjust input dimension to include estimated ref_vel
                num_actor_obs += 3  # Add 3 dimensions for estimated reference velocity (x, y, z)
                print(f"[Runner] Velocity estimator enabled: actor input dimension adjusted to {num_actor_obs} (policy obs {num_obs} + 3D velocity)")
            else:
                # For ResidualActorCritic: keep num_actor_obs unchanged (770)
                # ResidualActorCritic handles estimator internally:
                # - residual_actor uses num_actor_obs (770)
                # - GMT policy uses num_actor_obs + 3 (773)
                print(f"[Runner] Velocity estimator enabled for ResidualActorCritic: residual_actor uses {num_actor_obs} dims, GMT uses {num_actor_obs + 3} dims")
        
        # 选择网络架构 (Actor-Critic是网络架构, PPO是更新算法, AMP是Loss) (Actor-Critic与Teacher-Student可叠加)
        # 无记忆Actor-Critic, 有记忆的Actor-Critic, 无记忆Teacher-Student, 有记忆Teacher-Student
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_actor_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm 实例化训练方式
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = eval(alg_class_name)
        self.alg: PPO | Distillation | MOSAIC = alg_class(
            policy,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        # Track whether task-space FrontRES needs partial obs normalization.
        # When set, first _frontres_gmt_obs_dim dims are GMT-normalized;
        # remaining dims (anchor error terms) use Stage-1 empirical stats.
        self._frontres_gmt_obs_dim: int | None = None
        self._frontres_extra_mean: torch.Tensor | None = None  # (1, K) Stage-1 mean for extra dims
        self._frontres_extra_std:  torch.Tensor | None = None  # (1, K) Stage-1 std  for extra dims

        # Check if using ResidualActorCritic (special handling for GMT normalizer)
        if isinstance(policy, (ResidualActorCritic, FrontRESActorCritic)):
            # Use GMT's frozen normalizer for observations
            if policy.gmt_normalizer is not None:
                self.obs_normalizer = policy.gmt_normalizer
                print("[Runner] Using GMT's frozen normalizer for ResidualActorCritic")
                # Task-space mode: student obs may have extra anchor-error dims beyond
                # what the GMT normalizer expects.  Detect and store the split point.
                if (isinstance(policy, FrontRESActorCritic)
                        and getattr(policy, 'num_task_corrections', 0) > 0):
                    _gmt_mean = getattr(policy.gmt_normalizer, '_mean', None)
                    gmt_norm_dim = _gmt_mean.shape[-1] if _gmt_mean is not None else num_obs
                    if num_obs > gmt_norm_dim:
                        self._frontres_gmt_obs_dim = gmt_norm_dim
                        print(f"[Runner] FrontRES task-space: GMT normalizes first "
                              f"{gmt_norm_dim} obs dims; last "
                              f"{num_obs - gmt_norm_dim} anchor-error dims pass-through")
            else:
                print("[Runner] WARNING: ResidualActorCritic has no GMT normalizer, using Identity")
                self.obs_normalizer = torch.nn.Identity().to(self.device)

            # Create privileged obs normalizer (for critic)
            if self.empirical_normalization:
                self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], 
                                                                        until=1.0e8).to(self.device)
            else:
                self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

            # Teacher obs normalizer (not used for residual learning)
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        elif self.training_type == "supervise":
            # Student obs: empirical normalization is fine for MLP inputs
            if self.empirical_normalization:
                self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            else:
                self.obs_normalizer = torch.nn.Identity().to(self.device)
            # Target Δq must NOT be normalized: it is in physical units (radians) and will be
            # added directly to q_ref in Stage 2. Normalizing would corrupt the scale.
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        elif self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], 
                                                                    until=1.0e8).to(self.device)
            if num_teacher_obs is not None:
                self.teacher_obs_normalizer = EmpiricalNormalization(shape=[num_teacher_obs], 
                                                                     until=1.0e8).to(self.device)
            else:
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # For MOSAIC, use teacher normalizer from checkpoint and freeze it. 教师观测量归一器
        # IMPORTANT: In multi-teacher mode, skip runner-level normalization
        # because each teacher will use its own normalizer in MOSAIC.update()
        if (alg_class_name == "MOSAIC" and self.teacher_obs_type == "teacher"):
            # Check for multi-teacher mode
            if hasattr(self.alg, "teacher_normalizers") and self.alg.teacher_normalizers is not None:
                # Multi-teacher: skip runner-level normalization
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
                print("[Runner] Multi-teacher mode: skipping runner-level teacher_obs normalization (each teacher uses its own normalizer)")
            elif hasattr(self.alg, "teacher_normalizer") and self.alg.teacher_normalizer is not None:
                # Single teacher: use teacher's normalizer
                self.teacher_obs_normalizer = self.alg.teacher_normalizer
                self.teacher_obs_normalizer.eval()  # Freeze teacher normalizer
                print("[Runner] Using teacher observation normalizer from checkpoint (frozen)")
            # else: keep the EmpiricalNormalization created above

        # For MOSAIC, pass obs_normalizer and privileged_obs_normalizer for teacher BC 学生观测量&特权信息归一器
        if alg_class_name == "MOSAIC":
            self.alg.obs_normalizer = self.obs_normalizer
            self.alg.privileged_obs_normalizer = self.privileged_obs_normalizer
            print("[Runner] Passed obs_normalizer and privileged_obs_normalizer to MOSAIC for teacher BC")

            # Pass environment's group mapping to MOSAIC for multi-teacher consistency
            env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                motion_command = env.command_manager._terms['motion']
                if hasattr(motion_command, 'group_name_to_idx'):
                    self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                    print(f"[Runner] Passed environment's group mapping to MOSAIC: {self.alg.env_group_name_to_idx}")

            # If MOSAIC loaded a teacher critic normalizer, use it for privileged obs
            if hasattr(self.alg, "teacher_critic_normalizer") and self.alg.teacher_critic_normalizer is not None:
                self.privileged_obs_normalizer = self.alg.teacher_critic_normalizer
                print("[Runner] Using teacher critic normalizer for privileged observations")

        # init storage and model
        if self.training_type == "mosaic":
            # For FrontRESActorCritic in task-space mode, the "action" stored in the
            # rollout buffer is 6-dim SE(3) correction [Δpos, Δrpy], NOT 29-dim robot joints.
            _mosaic_action_dim = getattr(policy, 'total_output_dim', None) or self.env.num_actions
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [_mosaic_action_dim],
                teacher_obs_shape=[num_teacher_obs] if num_teacher_obs is not None else None,
                ref_vel_estimator_obs_shape=[num_ref_vel_estimator_obs] if self.ref_vel_estimator_obs_type is not None else None,)
        elif self.training_type == "supervise":
            # action_shape must match the supervision target dim (num_privileged_obs),
            # NOT self.env.num_actions (robot DOFs). The target is [Δq(29), Δz(1)] = 30 dims.
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [num_privileged_obs],)
        else:
            # For FrontRESActorCritic in task-space mode, the "policy action" stored in the
            # rollout buffer is the 6-dim SE(3) correction [Δpos, Δrpy], NOT the 29-dim
            # robot joint targets produced by GMT. Use total_output_dim when available.
            _policy_action_dim = getattr(policy, 'total_output_dim', None) or self.env.num_actions
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [_policy_action_dim],)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        print("[Runner] learn() entered — initializing logger...", flush=True)
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        print("[Runner] Logger initialized — starting training setup...", flush=True)
        # Pass writer and log_interval to algorithm for logging (needed by MOSAIC)
        if hasattr(self, 'writer') and self.writer is not None:
            self.alg.writer = self.writer

        self.alg.log_interval = 1 # Default log interval

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # For MOSAIC multi-teacher: ensure env_group_name_to_idx is set before training 整理动作序号
        if self.training_type == "mosaic" and hasattr(self.alg, 'use_multi_teacher') and self.alg.use_multi_teacher:
            if self.alg.env_group_name_to_idx is None:
                print("[Runner] Retrieving environment's group mapping...")
                env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                    motion_command = env.command_manager._terms['motion']
                    if hasattr(motion_command, 'group_name_to_idx'):
                        self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                        print(f"[Runner] Successfully retrieved environment's group mapping: {self.alg.env_group_name_to_idx}")
                    else:
                        raise RuntimeError(
                            "[Runner] FATAL: motion_command does not have 'group_name_to_idx' attribute!\n"
                            "Multi-teacher training cannot proceed without environment's group mapping.")
                else:
                    raise RuntimeError(
                        "[Runner] FATAL: Cannot retrieve environment's group mapping!\n"
                        f"Environment type: {type(env)}\n"
                        f"Has command_manager: {hasattr(env, 'command_manager')}\n"
                        "Multi-teacher training cannot proceed.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # start learning
        obs, extras = self.env.get_observations() # obs.shape=[num_env, 770], 770=[t, t-1, t-2, t-3, t-4]
        obs_dict = extras.get("observations", {})
        if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
            obs = obs_dict[self.policy_obs_type]

        # 获取特权信息 & 教师观测
        privileged_obs = obs_dict.get(self.privileged_obs_type, obs) 
        teacher_obs = obs_dict.get(self.teacher_obs_type) # 
        obs = obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device)
        if teacher_obs is not None:
            teacher_obs = teacher_obs.to(self.device)
        else:
            teacher_obs = privileged_obs
        
        # Initialize ref_vel_estimator observations (NO normalization!) 速度估计器
        ref_vel_estimator_obs = obs_dict.get(self.ref_vel_estimator_obs_type)
        if ref_vel_estimator_obs is not None:
            ref_vel_estimator_obs = ref_vel_estimator_obs.to(self.device)

        # For Stage 1: save raw obs BEFORE obs_normalizer for GMT ONNX input.
        # The exported ONNX includes the normalizer in the computation graph, so
        # get_gmt_action() must receive raw (unnormalized) observations.
        if self.training_type == "supervise":
            obs_raw_for_gmt = obs.clone()

        # Normalize initial observations (same as in training loop) 观测归一器
        obs = self._apply_obs_normalizer(obs) # 三种观测量分别使用不同观测归一器

        # 使用观测量归一化器对观测量进行处理
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        self.train_mode() # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)  # FrontRES envs: r_delta per episode; others: raw reward
        lenbuffer = deque(maxlen=100)  # FrontRES training envs episode lengths
        # B1: separate GMT baseline reward buffer (only populated when _is_frontres)
        rewbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: raw GMT reward per episode
        lenbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: episode lengths (key diagnostic)

        # self.env.num_envs: 仿真中同时运行的机器人数量
        # cur_reward_sum & cur_episode_length: 每个机器人的总得分与存活时间
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Velocity estimator error tracking
        vel_est_error_buffer = deque(maxlen=100)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Critic warmup: freeze Actor for the first N iterations so the Critic
        # can converge before Actor weights (pretrained from Stage 1) are updated.
        # Only applied to FrontRESActorCritic; other policy types are unaffected.
        critic_warmup_iters = self.cfg.get("critic_warmup_iterations", 0)
        _warmup_actor_frozen = False  # internal state flag

        # Δq curriculum scheduler — only for FrontRESActorCritic
        # alpha_init: fixed α during critic warmup (non-zero so critic learns correct V(s))
        # alpha_ramp_iterations: after warmup, linearly ramp alpha_init → 1.0
        _is_frontres = isinstance(self.alg.policy, FrontRESActorCritic)
        _is_task_space_mode = _is_frontres and getattr(self.alg.policy, 'num_task_corrections', 0) > 0
        _alpha_init  = float(self.cfg.get("delta_q_alpha_init", 1.0))
        _alpha_ramp  = int(self.cfg.get("delta_q_alpha_ramp_iterations", 0))
        if _is_frontres and (_alpha_ramp > 0 or _alpha_init < 1.0):
            print(f"[Runner] Δq curriculum enabled: alpha_init={_alpha_init}, "
                  f"ramp_iterations={_alpha_ramp} (starts after warmup={critic_warmup_iters})")

        # B1 split-env delta-reward: first N_train envs run FrontRES, last N_base envs run GMT-only.
        # R_baseline = mean reward of GMT-only envs each step (exact, from real simulation).
        # r_delta = r_total[:N_train] − r_baseline  →  credit assignment to FrontRES contribution only.
        # GMT envs' advantage → 0 in steady state (V converges to GMT reward level) → gradient vanishes.
        if _is_frontres:
            N_train = self.env.num_envs // 2
            N_base  = self.env.num_envs - N_train
            print(f"[Runner] FrontRES B1 delta-reward: "
                  f"{N_train} training envs (FrontRES) + {N_base} baseline envs (GMT-only)")
            # Separate cumulative reward tracker for GMT envs (raw GMT reward, for logging only).
            # We zero GMT rewards in the PPO storage so V(s) only learns from FrontRES r_delta.
            # GMT raw rewards must be tracked separately before they are zeroed.
            cur_reward_sum_gmt = torch.zeros(N_base, dtype=torch.float, device=self.device)

        # ── Adaptive DR: target-survival-rate PI controller ───────────────────
        # Maintains training_survival_rate ≈ dr_target_survival by continuously
        # adjusting dr_scale ∈ [0, dr_max_scale].  Replaces the old exp-ramp +
        # staircase approach with a single self-correcting feedback loop:
        #   survival > target + deadband  →  dr_scale += adapt_speed  (harder)
        #   survival < target - deadband  →  dr_scale -= adapt_speed  (easier)
        #
        # Survival target is expressed as target episode length (steps), converted
        # internally: target_per_step = 1 - 1/dr_target_episode_length.
        _dr_target_ep   = int(self.cfg.get("dr_target_episode_length", 60))
        _dr_target_surv = 1.0 - 1.0 / max(1, _dr_target_ep)  # per-step survival target
        _dr_speed       = float(self.cfg.get("dr_adapt_speed",  0.0005))
        _dr_max         = float(self.cfg.get("dr_max_scale",    4.0))
        _dr_min         = float(self.cfg.get("dr_min_scale",    0.0))
        _dr_ema_alpha   = float(self.cfg.get("dr_ema_alpha",    0.95))
        _dr_deadband    = float(self.cfg.get("dr_deadband",     0.005))

        # Restore dr_scale from checkpoint (set by load()); start at 0 for fresh runs.
        _dr_scale     = float(getattr(self, '_dr_scale', 0.0))
        _dr_scale     = max(_dr_scale, _dr_min)  # enforce floor on resume
        _survival_ema = 1.0  # optimistic initial estimate; warms up quickly via EMA

        # Read base perturbation values from env config (scaled by dr_scale each iteration).
        # Only ratio/magnitude fields are scaled; prob fields remain constant.
        _perturb_target = None
        if _is_frontres:
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(_env_raw, 'cfg') and hasattr(_env_raw.cfg, 'motion_perturbations'):
                from types import SimpleNamespace as _NS
                _pt = _env_raw.cfg.motion_perturbations
                _perturb_target = _NS(
                    float_prob        = float(_pt.float_prob),
                    float_ratio       = float(_pt.float_ratio),
                    sink_prob         = float(_pt.sink_prob),
                    sink_ratio        = float(_pt.sink_ratio),
                    foot_slip_prob    = float(_pt.foot_slip_prob),
                    foot_slip_ratio   = float(_pt.foot_slip_ratio),
                    root_tilt_prob    = float(getattr(_pt, 'root_tilt_prob',    0.0)),
                    root_tilt_max_rad = float(getattr(_pt, 'root_tilt_max_rad', 0.0)),
                    joint_noise_prob  = float(getattr(_pt, 'joint_noise_prob',  0.0)),
                    joint_noise_std   = float(getattr(_pt, 'joint_noise_std',   0.0)),
                )
                print(
                    f"[Runner] Adaptive DR (PI controller): "
                    f"target_ep_len={_dr_target_ep} steps "
                    f"(per-step target={_dr_target_surv:.4f}), "
                    f"speed={_dr_speed}, max_scale={_dr_max}, "
                    f"base float_ratio={_perturb_target.float_ratio:.3f}m, "
                    f"base root_tilt={_perturb_target.root_tilt_max_rad:.3f}rad, "
                    f"resume dr_scale={_dr_scale:.3f}"
                )
            else:
                print("[Runner] WARNING: FrontRES DR enabled but env.cfg.motion_perturbations not found")

        # ── FrontRES supervised warmup (Stage 1 → Stage 2 merge) ──────────────────
        # Runs BEFORE the PPO loop. Teaches FrontRES to detect perturbations via
        # supervised learning on the anti-DR target:  target = -(perturbed - original).
        # After warmup, PPO fine-tunes with r_delta reward (same architecture, same obs).
        _warmup_iters = int(self.cfg.get("supervised_warmup_iterations", 0))
        if _is_frontres and _warmup_iters > 0:
            _warmup_lr     = float(self.cfg.get("supervised_warmup_lr", 1e-4))
            _warmup_epochs = int(self.cfg.get("supervised_warmup_epochs", 5))
            _warmup_opt = torch.optim.Adam(
                self.alg.policy.residual_actor.parameters(), lr=_warmup_lr)
            _warmup_loss_fn = torch.nn.HuberLoss(delta=1.0)

            # Import once to avoid per-step overhead
            from whole_body_tracking.tasks.tracking.mdp.observations import \
                get_supervision_target_task_space as _get_warmup_target

            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            _nfo = self.alg.policy.num_frontres_obs

            print(f"[Runner] === Supervised warmup: {_warmup_iters} iters "
                  f"(lr={_warmup_lr}, epochs={_warmup_epochs}, "
                  f"frontres_input={_nfo} dims) ===")

            for _wu in range(_warmup_iters):
                _wo_list: list[torch.Tensor] = []
                _wt_list: list[torch.Tensor] = []

                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        obs, extras = self.env.get_observations()
                        obs_dict = extras.get("observations", {})
                        _p_obs = obs_dict.get(self.policy_obs_type, obs).to(self.device)

                        # GMT-only actions (FrontRES correction = zero during warmup)
                        env_actions = self.alg.policy.get_env_action(
                            _p_obs,
                            torch.zeros(_p_obs.shape[0], self.alg.policy.total_output_dim,
                                        device=self.device))

                        obs, _, dones, extras = self.env.step(env_actions.to(self.env.device))
                        obs_dict = extras.get("observations", {})
                        _p_obs = obs_dict.get(self.policy_obs_type, obs).to(self.device)

                        # Both obs and target reflect the perturbation applied this step
                        _wo_list.append(_p_obs[:, :_nfo])
                        _wt_list.append(_get_warmup_target(_env_raw, "motion"))

                # Supervised SGD over collected rollout data
                _all_obs = torch.cat(_wo_list, dim=0)      # (S*E, _nfo)
                _all_tgt = torch.cat(_wt_list, dim=0)      # (S*E, 6)
                _N = _all_obs.shape[0]

                for epoch in range(_warmup_epochs):
                    perm = torch.randperm(_N, device=self.device)
                    for i in range(0, _N, 4096):
                        idx = perm[i:i + 4096]
                        pred = self.alg.policy.residual_actor(_all_obs[idx])
                        loss = _warmup_loss_fn(pred, _all_tgt[idx])
                        _warmup_opt.zero_grad()
                        loss.backward()
                        _warmup_opt.step()

                if (_wu + 1) % max(1, _warmup_iters // 5) == 0:
                    print(f"[Runner]   warmup {_wu + 1}/{_warmup_iters}: "
                          f"loss={loss.item():.6f}")

            print(f"[Runner] === Supervised warmup complete (final loss={loss.item():.6f}) ===")
        # ── END supervised warmup ─────────────────────────────────────────────────

        for it in range(start_iter, tot_iter):
            start = time.time()

            # --- Critic warmup management ---
            if isinstance(self.alg.policy, FrontRESActorCritic) and critic_warmup_iters > 0:
                warmup_active = (it - start_iter) < critic_warmup_iters
                if warmup_active and not _warmup_actor_frozen:
                    for param in self.alg.policy.residual_actor.parameters():
                        param.requires_grad = False
                    _warmup_actor_frozen = True
                    print(f"[Runner] Critic warmup started: Actor frozen for {critic_warmup_iters} iterations")
                elif not warmup_active and _warmup_actor_frozen:
                    for param in self.alg.policy.residual_actor.parameters():
                        param.requires_grad = True
                    _warmup_actor_frozen = False
                    print(f"[Runner] Critic warmup complete at iteration {it}: Actor unfrozen")

            # --- Δq alpha curriculum update ---
            if _is_frontres:
                local_it = it - start_iter
                if local_it < critic_warmup_iters:
                    # Phase 0: critic warmup — fixed alpha_init
                    new_alpha = _alpha_init
                elif _alpha_ramp > 0:
                    # Phase 1: linear ramp from alpha_init to 1.0
                    ramp_progress = min(1.0, (local_it - critic_warmup_iters) / _alpha_ramp)
                    new_alpha = _alpha_init + ramp_progress * (1.0 - _alpha_init)
                else:
                    # No ramp configured: jump to 1.0 immediately after warmup
                    new_alpha = 1.0
                self.alg.policy.delta_q_alpha = new_alpha

            # --- DR PI controller: update survival EMA and adjust dr_scale ---
            # Runs BEFORE the rollout so the perturber is set for this iteration.
            # survival EMA is updated with last iteration's value (initialized to 1.0).
            if _is_frontres and _perturb_target is not None:
                # Update survival EMA (first iteration uses the initial 1.0 → dr_scale increases)
                _survival_ema = (_dr_ema_alpha * _survival_ema
                                 + (1.0 - _dr_ema_alpha) * getattr(self, '_last_survival_rate', 1.0))
                # PI control: adjust dr_scale based on deviation from target
                if _survival_ema > _dr_target_surv + _dr_deadband:
                    _dr_scale = min(_dr_scale + _dr_speed, _dr_max)
                elif _survival_ema < _dr_target_surv - _dr_deadband:
                    _dr_scale = max(_dr_scale - _dr_speed, _dr_min)
                # Persist for resume
                self._dr_scale = _dr_scale
                # Apply dr_scale to perturber (prob fields unchanged; only ratio/magnitude fields)
                _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                if hasattr(_env_raw, 'command_manager') and 'motion' in _env_raw.command_manager._terms:
                    _mcmd = _env_raw.command_manager._terms['motion']
                    if hasattr(_mcmd, 'perturber'):
                        _mcmd.perturber.cfg.float_prob        = _perturb_target.float_prob
                        _mcmd.perturber.cfg.float_ratio       = _perturb_target.float_ratio       * _dr_scale
                        _mcmd.perturber.cfg.sink_prob         = _perturb_target.sink_prob
                        _mcmd.perturber.cfg.sink_ratio        = _perturb_target.sink_ratio        * _dr_scale
                        _mcmd.perturber.cfg.foot_slip_prob    = _perturb_target.foot_slip_prob
                        _mcmd.perturber.cfg.foot_slip_ratio   = _perturb_target.foot_slip_ratio   * _dr_scale
                        _mcmd.perturber.cfg.root_tilt_prob    = _perturb_target.root_tilt_prob
                        _mcmd.perturber.cfg.root_tilt_max_rad = _perturb_target.root_tilt_max_rad * _dr_scale
                        _mcmd.perturber.cfg.joint_noise_prob  = _perturb_target.joint_noise_prob
                        _mcmd.perturber.cfg.joint_noise_std   = _perturb_target.joint_noise_std   * _dr_scale

            # FrontRES reward-shaping state: reset at the start of each rollout.
            _frontres_prev_delta_q: torch.Tensor | None = None  # [N, A] for smoothness penalty
            # Accumulators for wandb logging (per iteration, divided by shaping steps)
            _frontres_rdelta_sum:         float = 0.0
            _frontres_baseline_sum:       float = 0.0
            _frontres_smooth_penalty_sum: float = 0.0
            _frontres_reg_penalty_sum:    float = 0.0
            _frontres_delta_z_abs_sum:    float = 0.0   # mean |task correction| per step
            _frontres_jump_degree_sum:    float = 0.0   # mean jump_degree (gate activation monitor)
            _frontres_shaping_steps:      int   = 0
            # Termination tracking for training envs (used to compute survival_rate this rollout)
            _frontres_term_count: int = 0
            _frontres_step_count: int = 0
            # reg_penalty activates once dr_scale ≥ 1.0 (base values fully applied).
            # Before that, reg pushing corrections→0 reinforces the no-op shortcut trap.
            _lambda_reg = getattr(self.alg, 'lambda_reg_current', 0.0) if _is_frontres else 0.0
            _dr_done    = _is_frontres and (_dr_scale >= 1.0)

            # Rollout: 训练首先需要积攒数据, 等数据攒够才能调用self.alg.update()更新权重
            with torch.inference_mode(): # 关闭计算图的梯度追踪, 只进行推理
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    if self.training_type == "mosaic":
                        # Extract motion groups for multi-teacher support
                        motion_groups = None

                        # CRITICAL FIX: Use unwrapped env to access command_manager
                        # 对仿真环境进行unwrapped, 直接调用底层指令管理器得到正在运行的动作标签
                        # 因为仿真环境知道正在运行的动作序列, 但算法不知道, 需要把动作序号传递给算法
                        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                            motion_command = env.command_manager._terms['motion']
                            if hasattr(motion_command, 'env_motion_groups'):
                                motion_groups = motion_command.env_motion_groups.clone()

                        # 前向传播获得动作
                        actions = self.alg.act(obs,
                                               privileged_obs,
                                               teacher_obs=teacher_obs,
                                               ref_vel_estimator_obs=ref_vel_estimator_obs,
                                               motion_groups=motion_groups)

                        # Track velocity estimator error if available 仿真器有速度真值, 但学生模型只能瞎猜
                        if hasattr(self.alg, 'last_estimated_ref_vel') and self.alg.last_estimated_ref_vel is not None:
                            # Get ground truth ref_anchor_lin_vel_b from environment using existing mdp function
                            # This uses anchor body coordinate system to match offline training
                            from whole_body_tracking.tasks.tracking.mdp import observations as mdp
                            gt_ref_vel_b = mdp.ref_base_lin_vel_b(self.env.unwrapped, "motion")  # [N, 3]

                            # Compute MAE (same metric as training validation) 计算两者均方差
                            # MAE per environment (averaging across 3 velocity dimensions)
                            vel_error = (self.alg.last_estimated_ref_vel - gt_ref_vel_b).abs().mean(dim=-1)  # [N]
                            vel_est_error_buffer.extend(vel_error.cpu().numpy().tolist()) # 送入buffer

                        # PAMR action mapping (RL action → env action)
                        if hasattr(self.alg.policy, 'get_env_action'):
                            env_actions = self.alg.policy.get_env_action(obs, actions)
                        else:
                            env_actions = actions

                        # B1 split-env: override stored actions/log_probs for GMT baseline envs
                        # and set frontres_mask so the critic update is masked correctly.
                        # This mirrors the else-branch B1 logic but skips env_actions split
                        # (task-space mode: env_actions are pure GMT for all envs anyway;
                        # the FrontRES corrections reach the env via _frontres_pos/quat_correction).
                        if _is_frontres:
                            _zeros_gmt = torch.zeros(N_base, self.alg.transition.actions.shape[-1],
                                                     device=self.device)
                            self.alg.transition.actions[N_train:] = _zeros_gmt
                            _mean_gmt = self.alg.policy.action_mean[N_train:].clone()
                            _std_gmt  = self.alg.policy.action_std[N_train:]
                            _logp_zeros = torch.distributions.Normal(_mean_gmt, _std_gmt) \
                                              .log_prob(_zeros_gmt).sum(dim=-1)
                            self.alg.transition.actions_log_prob[N_train:] = _logp_zeros
                            _frontres_mask = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_mask[:N_train] = 1.0
                            self.alg.transition.frontres_mask = _frontres_mask

                    elif self.training_type == "supervise":
                        # Stage 1 Supervised Learning rollout:
                        #   - GMT drives the environment (produces joint position targets)
                        #   - Student only records (obs, target_delta_q) for offline training
                        #
                        # GMT ONNX includes the normalizer in the computation graph, so it
                        # expects RAW (unnormalized) observations.  `obs_raw_for_gmt` holds
                        # the un-normalized policy obs from the previous step.
                        env_actions = self.alg.policy.get_gmt_action(obs_raw_for_gmt)

                        # Record current (obs_t, target_delta_q_t) transition for training.
                        # privileged_obs == obs_dict["target"] == Δq_gt (Identity normalizer, raw units).
                        # The return value (student's Δq_pred) is discarded here; prediction is
                        # recomputed with gradients inside SuperviseTrainer.update().
                        _ = self.alg.act(obs, privileged_obs)

                    else:
                        actions = self.alg.act(obs, privileged_obs)

                        if _is_frontres:
                            # B1 split: FrontRES envs [0:N_train] use policy delta_q,
                            #           GMT envs [N_train:] receive zero delta_q (pure GMT).
                            # Override the transition's stored action/log_prob for GMT envs
                            # BEFORE process_env_step() calls add_transitions() — this ensures
                            # the IS ratio used in PPO is correct (zero was the actual env action).
                            _zeros_gmt = torch.zeros_like(actions[N_train:])  # [N_base, A]
                            self.alg.transition.actions[N_train:] = _zeros_gmt
                            _mean_gmt   = self.alg.policy.action_mean[N_train:].clone()
                            _std_gmt    = self.alg.policy.action_std[N_train:]  # [N_base, A]
                            _logp_zeros = torch.distributions.Normal(_mean_gmt, _std_gmt) \
                                              .log_prob(_zeros_gmt).sum(dim=-1)
                            self.alg.transition.actions_log_prob[N_train:] = _logp_zeros
                            # Mark which envs are FrontRES (1) vs GMT baseline (0) for critic masking.
                            _frontres_mask = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_mask[:N_train] = 1.0
                            self.alg.transition.frontres_mask = _frontres_mask
                            # Build split env_actions (one physics step covers both groups)
                            env_actions_fr  = self.alg.policy.get_env_action(obs[:N_train], actions[:N_train])
                            env_actions_gmt = self.alg.policy.get_env_action(obs[N_train:], _zeros_gmt)
                            env_actions = torch.cat([env_actions_fr, env_actions_gmt], dim=0)
                        elif hasattr(self.alg.policy, 'get_env_action'):
                            env_actions = self.alg.policy.get_env_action(obs, actions)
                        else:
                            env_actions = actions

                    # Apply task-space anchor corrections for FrontRES (task-space mode).
                    # Must happen BEFORE env.step() so rewards and next obs see the corrected anchor.
                    if _is_task_space_mode:
                        _task_corr = getattr(self.alg.policy, 'last_task_correction', None)
                        if _task_corr is not None:
                            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                            for _cmd_term in _env_raw.command_manager._terms.values():
                                if hasattr(_cmd_term, '_frontres_pos_correction'):
                                    # ── Jump-degree soft gate ──────────────────────────────
                                    # Gate Δpos by (1 - jump_degree): suppresses position
                                    # correction during free flight (a_z ≈ -g).
                                    # Δrpy (orientation) is NOT gated: tilt correction is
                                    # always valid and does not interfere with jump physics.
                                    _pos_corr = _task_corr[:N_train, :3].clone()
                                    if hasattr(_cmd_term, 'jump_degree'):
                                        _jd   = _cmd_term.jump_degree[:N_train].to(_task_corr.device)
                                        _gate = (1.0 - _jd).clamp(0.0, 1.0).unsqueeze(-1)
                                        _pos_corr = _pos_corr * _gate
                                    # ── end gate ──────────────────────────────────────────
                                    _cmd_term._frontres_pos_correction[:N_train].copy_(_pos_corr)
                                    # Gate Δrpy with the same jump_degree.
                                    # During free flight (jump_degree≈1), orientation correction
                                    # is suppressed so FrontRES does not disturb GMT's natural
                                    # jump arc.  Small-angle approximation (max 0.3 rad ≈ 17°):
                                    # scaling Euler angles ≈ SLERP to within O(θ²) ≈ 5% error.
                                    _rpy = _task_corr[:N_train, 3:]
                                    if hasattr(_cmd_term, 'jump_degree'):
                                        _rpy = _rpy * _gate  # _gate already (N_train, 1)
                                    _quat_corr = quat_from_euler_xyz(
                                        _rpy[:, 0], _rpy[:, 1], _rpy[:, 2])
                                    _cmd_term._frontres_quat_correction[:N_train].copy_(_quat_corr)
                                    # Baseline envs: identity (no correction)
                                    _cmd_term._frontres_pos_correction[N_train:].zero_()
                                    _cmd_term._frontres_quat_correction[N_train:].zero_()
                                    _cmd_term._frontres_quat_correction[N_train:, 0] = 1.0

                    # Read supervised target BEFORE env.step: the command term's cache holds
                    # the perturbation that generated the CURRENT obs (used by FrontRES this step).
                    # After env.step, _update_command() overwrites the cache for the next step.
                    if _is_task_space_mode and getattr(self.alg, 'lambda_supervised', 0.0) > 0:
                        # Use a local env reference to avoid relying on _env_raw which is only
                        # assigned inside the _task_corr is not None branch above.
                        _env_for_sup = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        for _cmd_sup in _env_for_sup.command_manager._terms.values():
                            if hasattr(_cmd_sup, 'supervised_target'):
                                self.alg.transition.supervised_target = \
                                    _cmd_sup.supervised_target.clone().to(self.device)
                                break

                    # Step the environment 仿真环境更新观测量/动作评分/序列结束与否/监控数据
                    # NOTE: This is where the environment computes the *next* observation internally.
                    # The result is returned here and then used in the next loop iteration.
                    obs, rewards, dones, infos = self.env.step(env_actions.to(self.env.device))

                    # Move to device
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    # ── FrontRES B1 delta-reward ────────────────────────────────────────
                    # GMT baseline envs [N_train:] ran with delta_q=0 → rewards ≈ GMT-only.
                    # r_delta = r_total[:N_train] − r_baseline isolates FrontRES contribution.
                    # GMT env rewards are zeroed → returns ≈ 0 → advantage ≈ 0 → no policy gradient.
                    if _is_frontres:
                        r_raw_gmt = rewards[N_train:].view(-1).clone()  # [N_base] save before zeroing
                        r_total   = rewards[:N_train].view(-1)          # [N_train]
                        # Per-env paired baseline: env i vs env i+N_train on the same motion context.
                        # Falls back to scalar mean only when N_train != N_base (odd num_envs).
                        if N_train == N_base:
                            r_base = r_raw_gmt                          # [N_base] element-wise pairing
                        else:
                            r_base = r_raw_gmt.mean()                   # scalar fallback
                        r_delta   = r_total - r_base                    # [N_train]

                        rewards_mod = rewards.clone()
                        if rewards_mod.dim() == 2:
                            rewards_mod[:N_train] = r_delta.unsqueeze(-1)
                            rewards_mod[N_train:] = 0.0  # zero GMT → V(s) target = 0 → advantage = 0
                        else:
                            rewards_mod[:N_train] = r_delta
                            rewards_mod[N_train:] = 0.0
                        rewards = rewards_mod

                        # Optional smoothness penalty on top of delta reward
                        _lambda_smooth = getattr(self.alg, 'lambda_smooth', 0.0)
                        if _lambda_smooth > 0.0 and _frontres_prev_delta_q is not None:
                            _diff = actions[:N_train] - _frontres_prev_delta_q[:N_train]  # [N_train, A]
                            _smooth_penalty = -_lambda_smooth * _diff.pow(2).mean(dim=-1)  # [N_train]
                            if rewards.dim() == 2:
                                rewards[:N_train] = rewards[:N_train] + _smooth_penalty.unsqueeze(-1)
                            else:
                                rewards[:N_train] = rewards[:N_train] + _smooth_penalty
                            _frontres_smooth_penalty_sum += _smooth_penalty.mean().item()

                        # lambda_reg reward shaping: penalize ||Δq||² to discourage unbounded corrections.
                        # Gated by _dr_done: do NOT apply during DR curriculum ramp-up, so Actor can
                        # freely explore non-zero Δq without the penalty reinforcing the Δq=0 shortcut.
                        # Uses sampled actions (unbiased estimator of mean penalty); mean over joints
                        # so scale is per-joint and consistent with per-step reward magnitude.
                        if _lambda_reg > 0.0 and _dr_done:
                            _reg_penalty = -_lambda_reg * actions[:N_train].pow(2).mean(dim=-1)  # [N_train]
                            if rewards.dim() == 2:
                                rewards[:N_train] = rewards[:N_train] + _reg_penalty.unsqueeze(-1)
                            else:
                                rewards[:N_train] = rewards[:N_train] + _reg_penalty
                            _frontres_reg_penalty_sum += _reg_penalty.mean().item()

                        # Logging accumulators
                        _frontres_rdelta_sum   += r_delta.mean().item()
                        _frontres_baseline_sum += r_base.mean().item()
                        # Task-space mode: log mean correction magnitude; else log Δz
                        if _is_task_space_mode:
                            _tc = getattr(self.alg.policy, 'last_task_correction', None)
                            if _tc is not None:
                                _frontres_delta_z_abs_sum += _tc.abs().mean().item()
                            # Accumulate jump_degree for gate activation monitoring
                            for _cmd_term in (self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env).command_manager._terms.values():
                                if hasattr(_cmd_term, 'jump_degree'):
                                    _frontres_jump_degree_sum += _cmd_term.jump_degree[:N_train].mean().item()
                                    break
                        else:
                            _dz = getattr(self.alg.policy, 'last_delta_z', None)
                            if _dz is not None:
                                _frontres_delta_z_abs_sum += _dz.abs().mean().item()
                        _frontres_shaping_steps += 1
                        # Adaptive DR: count terminations in training envs
                        _frontres_term_count += int((dones[:N_train] > 0).sum().item())
                        _frontres_step_count += N_train

                        # Update prev_delta_q for smoothness tracking (all N envs)
                        _done_mask = dones.bool().view(-1)
                        if _frontres_prev_delta_q is None:
                            _frontres_prev_delta_q = actions.clone()
                        else:
                            _frontres_prev_delta_q = actions.clone()
                            _frontres_prev_delta_q[_done_mask] = 0.0
                    # ── END FrontRES B1 delta-reward ─────────────────────────────────────

                    obs_dict = infos.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                        obs = obs_dict[self.policy_obs_type].to(self.device)
                    else:
                        obs = obs.to(self.device)

                    # For Stage 1: update raw obs BEFORE normalization for GMT ONNX next step
                    if self.training_type == "supervise":
                        obs_raw_for_gmt = obs.clone()

                    # perform normalization 对本次循环的观测量进行归一化, 用于计算下步动作
                    obs = self._apply_obs_normalizer(obs)
                    if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                        privileged_obs = self.privileged_obs_normalizer(
                            obs_dict[self.privileged_obs_type].to(self.device))
                    else:
                        privileged_obs = obs
                    if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
                        teacher_obs = self.teacher_obs_normalizer(
                            obs_dict[self.teacher_obs_type].to(self.device))
                    else:
                        teacher_obs = privileged_obs
                    
                    # Extract ref_vel_estimator observations (NO normalization - must match offline training!)
                    if self.ref_vel_estimator_obs_type is not None and self.ref_vel_estimator_obs_type in obs_dict:
                        ref_vel_estimator_obs = obs_dict[self.ref_vel_estimator_obs_type].to(self.device)
                    else: # 提取速度估计器的速度观测量
                        ref_vel_estimator_obs = None

                    # process the step 更新回放池的数据 (奖励值, 完成布尔值, 额外信息)
                    self.alg.process_env_step(rewards, dones, infos) # 这里存入的 actions 依然是纯粹的 delta_q

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if hasattr(self.alg, "rnd") and self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        
                        # Update rewards
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards  # GMT envs contribute 0 (zeroed above)
                        # GMT raw reward tracking (separate, uses pre-zeroed values saved earlier)
                        if _is_frontres:
                            cur_reward_sum_gmt += r_raw_gmt  # [N_base] raw GMT per-step reward
                        
                        # Update episode length
                        cur_episode_length += 1

                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if _is_frontres and len(new_ids) > 0:
                            # B1: split done envs into FrontRES (r_delta) and GMT (raw reward + length)
                            _env_idx  = new_ids[:, 0]
                            _fr_done  = new_ids[_env_idx < N_train]
                            _gmt_done = new_ids[_env_idx >= N_train]
                            if len(_fr_done) > 0:
                                rewbuffer.extend(cur_reward_sum[_fr_done][:, 0].cpu().numpy().tolist())
                                lenbuffer.extend(cur_episode_length[_fr_done][:, 0].cpu().numpy().tolist())
                            if len(_gmt_done) > 0:
                                _gmt_local = _gmt_done[:, 0] - N_train
                                rewbuffer_gmt.extend(cur_reward_sum_gmt[_gmt_local].cpu().numpy().tolist())
                                lenbuffer_gmt.extend(cur_episode_length[_gmt_done][:, 0].cpu().numpy().tolist())
                                cur_reward_sum_gmt[_gmt_local] = 0
                        elif len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # -- intrinsic and extrinsic rewards
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns 计算广义优势值
                if self.training_type in ["rl", "mosaic"]:
                    self.alg.compute_returns(privileged_obs)

            # update policy Rollout结束, 开始使用buffer计算Loss更新权重
            # Pass current iteration to algorithm for logging (needed by MOSAIC)
            self.alg.current_learning_iteration = it
            loss_dict = self.alg.update() # 调用mosaic.py中的update()函数进行权重更新

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # expose curriculum state and B1 delta-reward metrics to log() via locals()
            frontres_alpha         = self.alg.policy.delta_q_alpha if _is_frontres else None
            frontres_warmup_active = int(_warmup_actor_frozen) if _is_frontres else None
            frontres_rdelta_mean   = (_frontres_rdelta_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_baseline_mean = (_frontres_baseline_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_smooth_penalty_mean = (_frontres_smooth_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_reg_penalty_mean    = (_frontres_reg_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_survival_rate       = (1.0 - _frontres_term_count / _frontres_step_count
                                            if _is_frontres and _frontres_step_count > 0 else None)
            frontres_delta_z_abs_mean    = (_frontres_delta_z_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_jump_degree_mean    = (_frontres_jump_degree_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)

            # Store survival rate for next iteration's PI controller update.
            if frontres_survival_rate is not None:
                self._last_survival_rate = frontres_survival_rate

            # DR scale for logging: current value (set by PI controller at top of iteration)
            frontres_dr_scale = _dr_scale if _is_frontres else None

            # Removed: staircase advancement logic (replaced by PI controller above)
            _staircase_level_for_log = None
            _staircase_mult_for_log  = None

            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear() # 清空记录机器人人得分和存活长度的字典

            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files 防呆设计, 自动扫描本地改动并保存在log_dir
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)

                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size

        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)

                if self.training_type == "supervise":
                    # Stage 1: only log termination-related keys under GMT/ prefix.
                    # Reward and other RL metrics are meaningless here.
                    key_lower = key.lower().replace("/", "_")
                    if any(r in key_lower for r in ["rew", "reward"]):
                        continue  # skip reward metrics entirely
                    # Everything else (e.g. termination reasons) → GMT/ namespace
                    log_key = key if "/" in key else f"GMT/{key}"
                    self.writer.add_scalar(log_key, value, locs["it"])
                    ep_string += f"""{f'GMT {key}:':>{pad}} {value:.4f}\n"""
                else:
                    # log to logger and terminal
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs["it"])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, locs["it"])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # -- Stage 1: GMT episode-length as action-completion proxy
        if self.training_type == "supervise" and len(locs["lenbuffer"]) > 0:
            gmt_ep_len = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("GMT/mean_episode_length", gmt_ep_len, locs["it"])

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses (reclassify FrontRES diagnostics out of Loss/ into FrontRES/)
        _frontres_diag_keys = {"delta_q_norm_mean", "delta_q_norm_std", "smooth_metric", "lambda_reg"}
        _stage1_dz_keys     = {"loss_dz", "dz_pred_abs", "dz_gt_abs", "dz_mae"}
        for key, value in locs["loss_dict"].items():
            if self.training_type == "supervise" and key in _stage1_dz_keys:
                self.writer.add_scalar(f"Stage1/DeltaZ/{key}", value, locs["it"])
            elif isinstance(self.alg.policy, FrontRESActorCritic) and key in _frontres_diag_keys:
                self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy (not meaningful during supervised learning)
        if self.training_type != "supervise":
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- FrontRES Δq curriculum + B1 delta-reward diagnostics (wandb)
        if isinstance(self.alg.policy, FrontRESActorCritic):
            self.writer.add_scalar("Curriculum/delta_q_alpha",
                                   locs.get("frontres_alpha", self.alg.policy.delta_q_alpha), locs["it"])
            if locs.get("frontres_warmup_active") is not None:
                self.writer.add_scalar("Curriculum/critic_warmup_active",
                                       locs["frontres_warmup_active"], locs["it"])
            # -- B1 delta-reward: r_delta and GMT baseline per step
            if locs.get("frontres_rdelta_mean") is not None:
                self.writer.add_scalar("FrontRES/r_delta_per_step",
                                       locs["frontres_rdelta_mean"], locs["it"])
            if locs.get("frontres_baseline_mean") is not None:
                self.writer.add_scalar("FrontRES/baseline_per_step",
                                       locs["frontres_baseline_mean"], locs["it"])
            if locs.get("frontres_smooth_penalty_mean") is not None:
                self.writer.add_scalar("FrontRES/smooth_penalty_per_step",
                                       locs["frontres_smooth_penalty_mean"], locs["it"])
            if locs.get("frontres_reg_penalty_mean") is not None:
                self.writer.add_scalar("FrontRES/reg_penalty_per_step",
                                       locs["frontres_reg_penalty_mean"], locs["it"])
            if locs.get("frontres_dr_scale") is not None:
                self.writer.add_scalar("Curriculum/dr_scale",
                                       locs["frontres_dr_scale"], locs["it"])
            if locs.get("frontres_delta_z_abs_mean") is not None:
                self.writer.add_scalar("FrontRES/delta_z_abs_mean",
                                       locs["frontres_delta_z_abs_mean"], locs["it"])
            if locs.get("frontres_survival_rate") is not None:
                self.writer.add_scalar("Curriculum/training_survival_rate",
                                       locs["frontres_survival_rate"], locs["it"])
            # Log PI controller state
            if isinstance(self.alg.policy, FrontRESActorCritic):
                self.writer.add_scalar("Curriculum/survival_ema",
                                       locs.get("_survival_ema", 1.0), locs["it"])
                self.writer.add_scalar("Curriculum/dr_target_survival",
                                       locs.get("_dr_target_surv", 0.983), locs["it"])
                if locs.get("frontres_jump_degree_mean") is not None:
                    self.writer.add_scalar("FrontRES/jump_degree_mean",
                                           locs["frontres_jump_degree_mean"], locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training (RL metrics: skip during supervised learning to avoid confusing oscillation)
        if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

            # everything else
            if isinstance(self.alg.policy, FrontRESActorCritic):
                # B1: rewbuffer = FrontRES r_delta per episode; rewbuffer_gmt = GMT raw reward
                self.writer.add_scalar("Train/mean_r_delta",   statistics.mean(locs["rewbuffer"]),     locs["it"])
                if len(locs.get("rewbuffer_gmt", [])) > 0:
                    self.writer.add_scalar("Train/mean_reward_gmt", statistics.mean(locs["rewbuffer_gmt"]), locs["it"])
                if len(locs.get("lenbuffer_gmt", [])) > 0:
                    self.writer.add_scalar("Train/mean_episode_length_gmt",
                                           statistics.mean(locs["lenbuffer_gmt"]), locs["it"])
            else:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n""")

            if isinstance(self.alg.policy, FrontRESActorCritic):
                log_string += f"""{'Mean r_delta (FrontRES):':>{pad}} {statistics.mean(locs['rewbuffer']):.4f}\n"""
                if len(locs.get("rewbuffer_gmt", [])) > 0:
                    log_string += f"""{'Mean reward_GMT (baseline):':>{pad}} {statistics.mean(locs['rewbuffer_gmt']):.4f}\n"""
                if len(locs.get("lenbuffer_gmt", [])) > 0:
                    log_string += f"""{'Mean ep_len_GMT (baseline):':>{pad}} {statistics.mean(locs['lenbuffer_gmt']):.1f}\n"""
            else:
                log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- Velocity estimator error (if available)
            if 'vel_est_error_buffer' in locs and len(locs['vel_est_error_buffer']) > 0:
                log_string += f"""{'Mean vel_estimator error:':>{pad}} {statistics.mean(locs['vel_est_error_buffer']):.4f}\n"""
            # -- FrontRES curriculum state
            if locs.get("frontres_alpha") is not None:
                warmup_str = " [WARMUP]" if locs.get("frontres_warmup_active") else ""
                log_string += f"""{'Δq alpha:':>{pad}} {locs['frontres_alpha']:.4f}{warmup_str}\n"""
            if locs.get("frontres_dr_scale") is not None:
                log_string += f"""{'DR curriculum scale:':>{pad}} {locs['frontres_dr_scale']:.4f}\n"""
            if locs.get("frontres_survival_rate") is not None:
                log_string += f"""{'Training survival rate:':>{pad}} {locs['frontres_survival_rate']:.3f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")

            if self.training_type == "supervise":
                # Stage 1 console summary: SL losses + GMT episode length
                for key, value in locs["loss_dict"].items():
                    log_string += f"""{f'SL {key}:':>{pad}} {value:.4f}\n"""
                if len(locs["lenbuffer"]) > 0:
                    log_string += f"""{'GMT mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
            else:
                log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                for key, value in locs["loss_dict"].items():
                    log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n""")
        
        print(log_string)

    def save(self, path: str, infos=None):
        # Check if using ResidualActorCritic (special handling)
        if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
            # Save only residual network + critic (GMT is frozen, no need to save)
            model_state_dict = {
                'residual_actor': self.alg.policy.residual_actor.state_dict(),
                'critic': self.alg.policy.critic.state_dict(),}
            
            # Save noise std parameter
            if hasattr(self.alg.policy, 'std'):
                model_state_dict['std'] = self.alg.policy.std
            elif hasattr(self.alg.policy, 'log_std'):
                model_state_dict['log_std'] = self.alg.policy.log_std
        else:
            # Standard save: entire policy
            model_state_dict = self.alg.policy.state_dict()

        # -- Save model
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,}

        # Persist adaptive DR state so resume picks up at the correct scale.
        if hasattr(self, '_dr_scale'):
            saved_dict["dr_scale"] = self._dr_scale
        
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
            # Save teacher normalizer for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    saved_dict["teacher_obs_norm_state_dict"] = self.teacher_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
        loaded_dict = torch.load(path, weights_only=False)

        # ── 断点续训模式控制 ────────────────────────────────────────────────────────
        # is_full_resume=True  (Stage2→Stage2 断点续训): 恢复优化器矩估计+学习率, 保留 std
        # is_full_resume=False (Stage1→Stage2 权重迁移): 仅权重, 重置优化器和 std
        # load_optimizer 参数仍可从外部显式覆盖（例如强制跳过优化器加载）。
        is_full_resume: bool = self.cfg.get('is_full_resume', True)
        if not is_full_resume:
            load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
        print(f"[Runner] is_full_resume={is_full_resume} → "
              f"load_optimizer={load_optimizer}, reset_noise_std={not is_full_resume}")

        # Check if using ResidualActorCritic (special handling)
        if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
            # 智能映射：尝试从阶段一 (SuperviseLearning) 提取 student 权重
            if isinstance(self.alg.policy, FrontRESActorCritic) and "student.0.weight" in loaded_dict["model_state_dict"]:
                mapped_dict = {k.replace("student.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("student.")}
                self.alg.policy.residual_actor.load_state_dict(mapped_dict, strict=True)
                print("[Runner] Success: Auto-mapped Stage 1 'student' weights to Stage 2 'residual_actor'!")
            else:
                self.alg.policy.residual_actor.load_state_dict(loaded_dict["model_state_dict"]["residual_actor"])

            if load_critic:
                if "critic" in loaded_dict["model_state_dict"]:
                    self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
                else:
                    print("[Runner] No critic weights found. Critic will be initialized from scratch.")
            # Load noise std parameter
            if "std" in loaded_dict["model_state_dict"]:
                self.alg.policy.std.data = loaded_dict["model_state_dict"]["std"].data
            elif "log_std" in loaded_dict["model_state_dict"]:
                self.alg.policy.log_std.data = loaded_dict["model_state_dict"]["log_std"].data
            if load_critic:
                print("[Runner] Loaded residual network + critic from checkpoint (GMT remains frozen)")
            else:
                print("[Runner] Loaded residual network only (skipping critic from checkpoint)")
            resumed_training = True
        else:
            if load_critic:
                # Standard load: entire policy
                resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            else:
                actor_only_state_dict = {
                    key: value
                    for key, value in loaded_dict["model_state_dict"].items()
                    if not key.startswith("critic.")}
                
                resumed_training = self.alg.policy.load_state_dict(actor_only_state_dict, strict=False)

        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # Load observation normalizers if used
        if self.empirical_normalization:
            if resumed_training:
                # Resuming training: load student obs normalizer
                # For ResidualActorCritic / FrontRESActorCritic, obs_normalizer IS GMT's frozen
                # normalizer — never overwrite it with a checkpoint's normalizer statistics.
                if not isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
                    self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                elif (isinstance(self.alg.policy, FrontRESActorCritic)
                        and self._frontres_gmt_obs_dim is not None
                        and "obs_norm_state_dict" in loaded_dict):
                    # Task-space FrontRES: anchor-error dims [gmt_dim:] are not covered by
                    # the GMT normalizer.  Restore Stage-1's empirical stats for those dims
                    # so Stage 2 sees the same normalized scale that Stage 1 trained on.
                    _s1_sd   = loaded_dict["obs_norm_state_dict"]
                    _s1_mean = _s1_sd.get("_mean", None)  # shape (1, 800)
                    _s1_std  = _s1_sd.get("_std",  None)  # shape (1, 800)
                    if _s1_mean is not None and _s1_std is not None:
                        gmt_dim = self._frontres_gmt_obs_dim
                        self._frontres_extra_mean = _s1_mean[:, gmt_dim:].to(self.device)
                        self._frontres_extra_std  = _s1_std[:,  gmt_dim:].to(self.device)
                        print(f"[Runner] Loaded Stage-1 anchor-error normalizer stats "
                              f"(dims {gmt_dim}–{_s1_mean.shape[-1]}) for FrontRES task-space.")

                if self.training_type == "mosaic":
                    # For MOSAIC: determine whether to load privileged_obs_normalizer from checkpoint
                    # Only skip loading if teacher_critic was loaded from a separate checkpoint AND is frozen
                    load_privileged_normalizer = load_critic
                    if hasattr(self.alg, 'teacher_critic_checkpoint_path') and self.alg.teacher_critic_checkpoint_path is not None:
                        if hasattr(self.alg, 'teacher_critic_frozen') and self.alg.teacher_critic_frozen:
                            load_privileged_normalizer = False
                            print("[Runner] Keeping privileged_obs_normalizer from teacher_critic_checkpoint (frozen).")

                    if load_privileged_normalizer:
                        # Load critic normalizer from student checkpoint
                        if "privileged_obs_norm_state_dict" in loaded_dict:
                            self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                            print("[Runner] Loaded privileged_obs_normalizer from checkpoint.")
                        else:
                            print("[Runner] WARNING: No privileged_obs_norm_state_dict in checkpoint!")

                    # Load teacher obs normalizer if available (for teacher BC)
                    if "teacher_obs_norm_state_dict" in loaded_dict:
                        self.teacher_obs_normalizer.load_state_dict(loaded_dict["teacher_obs_norm_state_dict"])
                        print("[Runner] Loaded teacher_obs_normalizer from checkpoint.")
                else:
                    # For PPO and Distillation: load both normalizers
                    if load_critic:
                        priv_sd = loaded_dict.get("privileged_obs_norm_state_dict", {})
                        if priv_sd and "_mean" in priv_sd:
                            self.privileged_obs_normalizer.load_state_dict(priv_sd)
                        else:
                            # Stage 1 (SuperviseLearning) checkpoint has no valid
                            # privileged_obs_norm_state_dict — critic normalizer starts fresh.
                            print("[Runner] WARNING: privileged_obs_norm_state_dict missing or invalid — "
                                  "privileged_obs_normalizer starts fresh (expected for Stage 1 → Stage 2 transfer).")
            else:
                # Not resuming (e.g., Distillation after RL): load teacher normalizer
                # For Distillation: the checkpoint's obs_norm is the teacher's normalizer
                if load_critic:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            if not load_critic:
                print("[Runner] Skipping optimizer load because load_critic=False.")
            else:
                try:
                    # -- algorithm optimizer
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                    print("[Runner] Loaded optimizer state from checkpoint.")
                    # ── 学习率同步 ─────────────────────────────────────────────────────
                    # PPO.update() 每次 epoch 都用 self.alg.learning_rate 覆盖
                    # optimizer.param_groups["lr"]。load_state_dict 已将 param_groups["lr"]
                    # 恢复为 checkpoint 时的值，但 self.alg.learning_rate 仍是配置初始值。
                    # 此处同步，避免第一次 update() 将已恢复的学习率覆盖为初始值。
                    if is_full_resume and hasattr(self.alg, 'learning_rate'):
                        restored_lr = self.alg.optimizer.param_groups[0]['lr']
                        reset_lr = bool(self.cfg.get('reset_lr_on_resume', False))
                        if reset_lr:
                            # lr 被 adaptive schedule 压至下限时（如因 desired_kl 配置错误），
                            # 直接重置为算法配置的初始学习率，避免续训起点过低。
                            config_lr = float(self.alg_cfg.get('learning_rate', 5e-4))
                            self.alg.learning_rate = config_lr
                            for pg in self.alg.optimizer.param_groups:
                                pg['lr'] = config_lr
                            print(f"[Runner] Reset learning_rate → {config_lr:.2e} "
                                  f"(reset_lr_on_resume=True; checkpoint had {restored_lr:.2e})")
                        else:
                            self.alg.learning_rate = restored_lr
                            print(f"[Runner] Synced learning_rate = {restored_lr:.2e} (from optimizer checkpoint)")
                except (ValueError, KeyError) as e:
                    # Optimizer state mismatch (e.g., different parameter groups between stages)
                    # This can happen when:
                    # - Stage 1 had frozen critic (optimizer only has actor params)
                    # - Stage 2 unfreezes critic (optimizer has actor + critic params)
                    print(f"[Runner] WARNING: Could not load optimizer state: {e}")
                    print("[Runner] Optimizer will be initialized from scratch (learning rate, momentum, etc. reset)")
                    print("[Runner] This is expected when transitioning between training stages with different frozen parameters.")

                # -- RND optimizer if used
                if hasattr(self.alg, "rnd") and self.alg.rnd:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]

        # ── 噪声 std 控制 ──────────────────────────────────────────────────────────
        # is_full_resume=True:  保留 checkpoint 中已自然适应的 std（断点续训）
        # is_full_resume=False: 重置为 init_noise_std（Stage1→Stage2 冷启动）
        # 向后兼容：若 cfg 中显式设置了 reset_noise_std_on_resume，以其为准。
        reset_noise: bool
        if 'reset_noise_std_on_resume' in self.cfg:
            reset_noise = bool(self.cfg.get('reset_noise_std_on_resume'))
            print(f"[Runner] reset_noise_std_on_resume = {reset_noise} (explicit config override)")
        else:
            reset_noise = not is_full_resume   # is_full_resume=True → 不重置; False → 重置
            print(f"[Runner] reset_noise_std = {reset_noise} (derived from is_full_resume={is_full_resume})")

        if reset_noise and (hasattr(self.alg.policy, 'std') or hasattr(self.alg.policy, 'log_std')):
            init_noise_std = self.policy_cfg.get("init_noise_std", 1.0)
            noise_std_type = self.policy_cfg.get("noise_std_type", "scalar")
            num_actions = (self.alg.policy.std.shape[0] if hasattr(self.alg.policy, 'std')
                           else self.alg.policy.log_std.shape[0])
            if noise_std_type == "scalar":
                self.alg.policy.std.data = torch.ones(num_actions, device=self.device) * init_noise_std
                print(f"[Runner] Reset noise std → {init_noise_std}")
            elif noise_std_type == "log":
                self.alg.policy.log_std.data = torch.log(
                    torch.ones(num_actions, device=self.device) * init_noise_std)
                print(f"[Runner] Reset log_std → log({init_noise_std})")
        else:
            if hasattr(self.alg.policy, 'std'):
                print(f"[Runner] Kept noise std from checkpoint = {self.alg.policy.std.mean().item():.4f}")

        # -- Freeze normalizer if specified in config (for stage transitions)
        # This prevents normalizer statistics from drifting when resuming from distillation
        freeze_normalizer = self.cfg.get("freeze_normalizer_on_resume", False)
        print(f"[Runner] freeze_normalizer_on_resume = {freeze_normalizer}")
        if freeze_normalizer and self.empirical_normalization:
            # Freeze obs normalizer
            self.obs_normalizer.eval()
            if hasattr(self.obs_normalizer, 'until'):
                self.obs_normalizer.until = self.obs_normalizer.count  # Stop updating
            print(f"[Runner] Froze obs_normalizer (count={self.obs_normalizer.count})")

            # Freeze privileged obs normalizer
            self.privileged_obs_normalizer.eval()
            if hasattr(self.privileged_obs_normalizer, 'until'):
                self.privileged_obs_normalizer.until = self.privileged_obs_normalizer.count
            print(f"[Runner] Froze privileged_obs_normalizer (count={self.privileged_obs_normalizer.count})")

        # Restore adaptive DR scale so resume continues from the correct DR level.
        # is_full_resume=True  (Stage2断点续训): 恢复 checkpoint 中的 dr_scale
        # is_full_resume=False (Stage1→Stage2冷启动): 忽略 checkpoint dr_scale，
        #   改用 cfg 中的 dr_init_scale（默认 1.0），确保 Stage2 从 Stage1 训练强度出发，
        #   避免 dr_scale=0 时 Stage1 修正策略作用于干净参考导致的即时崩溃。
        if is_full_resume:
            self._dr_scale = loaded_dict.get("dr_scale", 0.0)
            print(f"[Runner] Adaptive DR scale restored from checkpoint: {self._dr_scale:.4f}")
        else:
            _dr_init = float(self.cfg.get("dr_init_scale", 1.0))
            self._dr_scale = _dr_init
            print(f"[Runner] Stage1→Stage2 cold-start: dr_scale initialised to "
                  f"dr_init_scale={_dr_init:.4f} (ignoring checkpoint value "
                  f"{loaded_dict.get('dr_scale', 0.0):.4f})")

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def _apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply obs_normalizer, with partial pass-through for FrontRES task-space mode.

        In task-space FrontRES:
          - First _frontres_gmt_obs_dim dims → GMT normalizer (frozen, matches GMT training)
          - Remaining anchor-error dims       → Stage-1 empirical stats (loaded from checkpoint)
            If those stats are not available, pass the anchor-error dims through unchanged.
        """
        if self._frontres_gmt_obs_dim is not None and obs.shape[-1] > self._frontres_gmt_obs_dim:
            gmt_dim = self._frontres_gmt_obs_dim
            gmt_part   = self.obs_normalizer(obs[:, :gmt_dim])
            extra      = obs[:, gmt_dim:]
            _s1_mean = getattr(self, '_frontres_extra_mean', None)
            _s1_std  = getattr(self, '_frontres_extra_std',  None)
            if _s1_mean is not None and _s1_std is not None:
                extra = (extra - _s1_mean) / (_s1_std + 1e-8)
            return torch.cat([gmt_part, extra], dim=-1)
        return self.obs_normalizer(obs)

    def _move_normalizer_to_device(self, device):
        if hasattr(self, 'obs_normalizer') and self.obs_normalizer is not None:
            for param in self.obs_normalizer.parameters():
                param.data = param.data.to(device)
            if hasattr(self.obs_normalizer, '_mean') and self.obs_normalizer._mean is not None:
                self.obs_normalizer._mean = self.obs_normalizer._mean.to(device)
            if hasattr(self.obs_normalizer, '_std') and self.obs_normalizer._std is not None:
                self.obs_normalizer._std = self.obs_normalizer._std.to(device)

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,}  # total number of processes

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
