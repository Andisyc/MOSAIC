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
from rsl_rl.algorithms import PPO, Distillation, MOSAIC, FrontRESUnified
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
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz


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
        elif self.alg_cfg["class_name"] == "FrontRESUnified":
            self.training_type = "frontres"
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
        elif self.training_type in ("mosaic", "frontres"):
            # MOSAIC uses critic observations for value function when available.
            # Teacher observations are handled separately for teacher BC.
            has_teacher_obs = "teacher" in obs_dict
            has_critic_obs = "critic" in obs_dict
            if has_critic_obs:
                self.privileged_obs_type = "critic"
                print(f"[{self.alg_cfg['class_name']}] Using 'critic' observations for value estimation.")
            elif has_teacher_obs:
                self.privileged_obs_type = "teacher"
                print(f"[{self.alg_cfg['class_name']}] Using 'teacher' observations for value estimation (no critic obs available).")
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

        if self.training_type in ("mosaic", "frontres") and self.alg_cfg.get("use_estimate_ref_vel", False):
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
        self.alg: PPO | Distillation | MOSAIC | FrontRESUnified = alg_class(
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
                print(f"[Runner] Using GMT's frozen normalizer for {type(policy).__name__}")
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

        # MOSAIC needs these for teacher BC; FrontRESUnified uses them for its
        # own supervised auxiliary loss and checkpoint resume path.
        if alg_class_name in ("MOSAIC", "FrontRESUnified"):
            self.alg.obs_normalizer = self.obs_normalizer
            self.alg.privileged_obs_normalizer = self.privileged_obs_normalizer
            print(f"[Runner] Passed obs_normalizer and privileged_obs_normalizer to {alg_class_name}")

            # Pass environment's group mapping for multi-teacher consistency in MOSAIC
            # and paired motion bookkeeping in FrontRESUnified.
            env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                motion_command = env.command_manager._terms['motion']
                if hasattr(motion_command, 'group_name_to_idx'):
                    self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                    print(f"[Runner] Passed environment's group mapping to {alg_class_name}: {self.alg.env_group_name_to_idx}")

            # If MOSAIC loaded a teacher critic normalizer, use it for privileged obs
            if hasattr(self.alg, "teacher_critic_normalizer") and self.alg.teacher_critic_normalizer is not None:
                self.privileged_obs_normalizer = self.alg.teacher_critic_normalizer
                print("[Runner] Using teacher critic normalizer for privileged observations")

        # init storage and model
        if self.training_type in ("mosaic", "frontres"):
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
                teacher_obs_shape=(
                    [num_teacher_obs]
                    if self.training_type == "mosaic" and num_teacher_obs is not None
                    else None
                ),
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
        print("[Runner] Getting initial observations...", flush=True)
        obs, extras = self.env.get_observations() # obs.shape=[num_env, 770], 770=[t, t-1, t-2, t-3, t-4]
        print("[Runner] Observations received.", flush=True)
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
        print(
            f"[Runner] Initial obs moved to device: obs={tuple(obs.shape)} {obs.device}, "
            f"priv={tuple(privileged_obs.shape)} {privileged_obs.device}, "
            f"teacher={tuple(teacher_obs.shape)} {teacher_obs.device}",
            flush=True,
        )
        
        # Initialize ref_vel_estimator observations (NO normalization!) 速度估计器
        ref_vel_estimator_obs = obs_dict.get(self.ref_vel_estimator_obs_type)
        if ref_vel_estimator_obs is not None:
            ref_vel_estimator_obs = ref_vel_estimator_obs.to(self.device)
            print(
                f"[Runner] ref_vel_estimator_obs moved: "
                f"{tuple(ref_vel_estimator_obs.shape)} {ref_vel_estimator_obs.device}",
                flush=True,
            )

        # For Stage 1: save raw obs BEFORE obs_normalizer for GMT ONNX input.
        # The exported ONNX includes the normalizer in the computation graph, so
        # get_gmt_action() must receive raw (unnormalized) observations.
        if self.training_type == "supervise":
            obs_raw_for_gmt = obs.clone()

        # Normalize initial observations (same as in training loop) 观测归一器
        print("[Runner] Applying obs normalizer...", flush=True)
        obs = self._apply_obs_normalizer(obs) # 三种观测量分别使用不同观测归一器
        print("[Runner] Policy obs normalized.", flush=True)

        # 使用观测量归一化器对观测量进行处理
        print("[Runner] Applying privileged/teacher normalizers...", flush=True)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        print("[Runner] Privileged/teacher obs normalized.", flush=True)

        print("[Runner] Switching modules to train mode...", flush=True)
        self.train_mode() # switch to train mode (for dropout for example)
        print("[Runner] Train mode set.", flush=True)

        # Book keeping
        print("[Runner] Initializing bookkeeping buffers...", flush=True)
        ep_infos = []
        rewbuffer = deque(maxlen=100)  # FrontRES envs: r_delta per episode; others: raw reward
        lenbuffer = deque(maxlen=100)  # FrontRES training envs episode lengths
        # B1: separate GMT baseline reward buffer (only populated when _is_frontres)
        rewbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: raw GMT reward per episode
        lenbuffer_gmt    = deque(maxlen=100)  # GMT-only envs: episode lengths (key diagnostic)

        # self.env.num_envs: 仿真中同时运行的机器人数量
        # cur_reward_sum & cur_episode_length: 每个机器人的总得分与存活时间
        print("[Runner] Allocating episode reward/length tensors...", flush=True)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        print("[Runner] Episode tensors allocated.", flush=True)

        # create buffers for logging extrinsic and intrinsic rewards
        print("[Runner] Checking RND buffers...", flush=True)
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            print("[Runner] Allocating RND reward buffers...", flush=True)
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            print("[Runner] RND buffers allocated.", flush=True)

        # Velocity estimator error tracking
        vel_est_error_buffer = deque(maxlen=100)
        print("[Runner] Bookkeeping buffers ready.", flush=True)

        # Ensure all parameters are in-synced
        print(f"[Runner] Distributed enabled: {self.is_distributed}", flush=True)
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            print(f"[Runner] Parameter synchronization complete for rank {self.gpu_global_rank}.", flush=True)
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        print("[Runner] Preparing iteration counters...", flush=True)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        print(f"[Runner] Iteration counters ready: start={start_iter}, total={tot_iter}", flush=True)

        _is_frontres = isinstance(self.alg.policy, FrontRESActorCritic)

        # Critic warmup: freeze Actor for the first N iterations so the Critic
        # can converge before Actor weights (pretrained from Stage 1) are updated.
        # Only applied to FrontRESActorCritic; other policy types are unaffected.
        if _is_frontres and bool(self.cfg.get("frontres_debug_training", False)):
            def _cfg_set(key, value):
                self.cfg[key] = value

            def _debug_value(debug_key, default):
                # In debug mode we want a genuinely shortened feedback loop.
                # Some Hydra/ConfigClass paths may drop debug_* class defaults
                # from agent_cfg.to_dict(); falling back to formal-training
                # values would silently turn debug into the slow schedule.
                return self.cfg.get(debug_key, default)

            _debug_overrides = {
                "supervised_warmup_iterations": int(_debug_value("debug_supervised_warmup_iterations", 200)),
                "supervised_warmup_diag_interval": int(_debug_value("debug_supervised_warmup_diag_interval", 40)),
                "critic_warmup_iterations": int(_debug_value("debug_critic_warmup_iterations", 50)),
                "dr_scale_init": float(_debug_value("debug_dr_scale_init", 0.5)),
                "dr_min_scale": float(_debug_value("debug_dr_min_scale", 0.3)),
                "dr_ema_alpha": float(_debug_value("debug_dr_ema_alpha", 0.90)),
                "dr_p_gain": float(_debug_value("debug_dr_p_gain", 0.20)),
                "dr_i_gain": float(_debug_value("debug_dr_i_gain", 0.03)),
                "dr_start_ppo_actor_weight": float(_debug_value("debug_dr_start_ppo_actor_weight", 1.0)),
                "frontres_safe_gap_per_step": float(_debug_value("debug_frontres_safe_gap_per_step", 0.003)),
                "frontres_broken_gap_per_step": float(_debug_value("debug_frontres_broken_gap_per_step", 0.08)),
                "frontres_gap_gate_temp": float(_debug_value("debug_frontres_gap_gate_temp", 0.005)),
            }
            for _key, _value in _debug_overrides.items():
                _cfg_set(_key, _value)

            _actor_warmup_debug = int(_debug_value(
                "debug_ppo_actor_warmup_iterations",
                50,
            ))
            _actor_ramp_debug = int(_debug_value(
                "debug_ppo_actor_ramp_iterations",
                200,
            ))
            self.cfg["ppo_actor_warmup_iterations"] = _actor_warmup_debug
            self.cfg["ppo_actor_ramp_iterations"] = _actor_ramp_debug
            self.alg_cfg["ppo_actor_warmup_iterations"] = _actor_warmup_debug
            self.alg_cfg["ppo_actor_ramp_iterations"] = _actor_ramp_debug

            print(
                "[Runner] === FrontRES DEBUG TRAINING enabled ===\n"
                f"[Runner]   supervised_warmup_iterations={self.cfg['supervised_warmup_iterations']}, "
                f"critic_warmup_iterations={self.cfg['critic_warmup_iterations']}\n"
                f"[Runner]   ppo_actor_warmup_iterations={_actor_warmup_debug}, "
                f"ppo_actor_ramp_iterations={_actor_ramp_debug}\n"
                f"[Runner]   dr_scale_init={self.cfg['dr_scale_init']}, "
                f"dr_min_scale={self.cfg['dr_min_scale']}, "
                f"dr_p_gain={self.cfg['dr_p_gain']}, dr_i_gain={self.cfg['dr_i_gain']}, "
                f"dr_start_ppo_actor_weight={self.cfg['dr_start_ppo_actor_weight']}\n"
                f"[Runner]   frontres_safe_gap_per_step={self.cfg['frontres_safe_gap_per_step']}, "
                f"frontres_broken_gap_per_step={self.cfg['frontres_broken_gap_per_step']}, "
                f"frontres_gap_gate_temp={self.cfg['frontres_gap_gate_temp']}",
                flush=True,
            )

        critic_warmup_iters = self.cfg.get("critic_warmup_iterations", 0)

        print("[Runner] Checking FrontRES mode...", flush=True)
        _is_task_space_mode = _is_frontres and getattr(self.alg.policy, 'num_task_corrections', 0) > 0
        print(
            f"[Runner] FrontRES mode: is_frontres={_is_frontres}, "
            f"task_space={_is_task_space_mode}",
            flush=True,
        )
        if _is_frontres and critic_warmup_iters > 0:
            print(f"[Runner] Critic warmup (fixed DR scale, Actor active): {critic_warmup_iters} iters")

        def _frontres_ppo_actor_weight_for_iter(iteration: int) -> float:
            """Linear supervised-to-PPO takeover schedule for the current iteration."""
            if not (_is_frontres and hasattr(self.alg, "ppo_actor_weight")):
                return 1.0
            _actor_warmup = int(self.alg_cfg.get(
                "ppo_actor_warmup_iterations",
                self.cfg.get("ppo_actor_warmup_iterations", 0)))
            _actor_ramp = int(self.alg_cfg.get(
                "ppo_actor_ramp_iterations",
                self.cfg.get("ppo_actor_ramp_iterations", 0)))
            _phase_iter = max(0, iteration - start_iter)
            if _phase_iter < _actor_warmup:
                return 0.0
            if _actor_ramp > 0 and _phase_iter < _actor_warmup + _actor_ramp:
                _weight = (_phase_iter - _actor_warmup + 1) / float(_actor_ramp)
                return max(0.0, min(1.0, _weight))
            return 1.0

        # B1 triplet delta-reward:
        #   [0:N_train)                  FrontRES on noisy reference
        #   [N_train:N_train+N_base)      GMT on the same noisy reference
        #   [N_train+N_base:...)          GMT on the clean reference
        # This exposes the executable gap:
        #   R_clean - R_noisy
        # and the repair gain:
        #   R_frontres - R_noisy
        if _is_frontres:
            N_pair = self.env.num_envs // 3
            N_train = N_pair
            N_base  = N_pair
            N_clean = self.env.num_envs - N_train - N_base
            print(f"[Runner] FrontRES B1 triplet reward: "
                  f"{N_train} FrontRES envs + {N_base} noisy-GMT envs + {N_clean} clean-GMT envs",
                  flush=True)
            _env_pair = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(_env_pair, 'command_manager') and 'motion' in _env_pair.command_manager._terms:
                _mcmd_pair = _env_pair.command_manager._terms['motion']
                if hasattr(_mcmd_pair, 'set_frontres_triplet_baseline'):
                    _mcmd_pair.set_frontres_triplet_baseline(N_train, N_base, N_clean)
                    print("[Runner] FrontRES B1 triplet baseline enabled: "
                          "FrontRES/noisy-GMT/clean-GMT share motion/frame; clean-GMT has zero perturbation",
                          flush=True)
                elif hasattr(_mcmd_pair, 'set_frontres_paired_baseline'):
                    _mcmd_pair.set_frontres_paired_baseline(N_train)
                    print("[Runner] FrontRES B1 paired baseline enabled (legacy two-way fallback): "
                          "env i and env i+N_train share motion/frame/perturbation",
                          flush=True)
            # Separate cumulative reward tracker for GMT envs (raw GMT reward, for logging only).
            # We zero GMT rewards in the PPO storage so V(s) only learns from FrontRES r_delta.
            # GMT raw rewards must be tracked separately before they are zeroed.
            cur_reward_sum_gmt = torch.zeros(N_base + N_clean, dtype=torch.float, device=self.device)

            # REMOVED: set_baseline_envs() was originally added to fix ep_len_gmt≈10
            # caused by large OU noise at high dr_scale hitting termination thresholds.
            # However, zeroing perturbations for baseline envs makes them track a CLEAN
            # (unperturbed) reference, so r_delta = reward(perturbed+correction) -
            # reward(clean) ≤ 0 always — a mathematically impossible training signal.
            #
            # Correct baseline: GMT with the SAME perturbation but NO correction.
            # r_delta = reward(perturbed+correction) - reward(perturbed)
            # This is positive when FrontRES correction actually helps, providing the
            # correct learning signal for PPO to improve FrontRES.
            #
            # The ep_len_gmt≈10 root cause was the obs ordering bug (anchor errors at
            # [0:30] instead of [770:800]), which has been fixed separately.

        _frontres_task_action_mask = None
        if _is_task_space_mode:
            _active_dims = self.cfg.get("frontres_active_task_dims", None)
            if _active_dims is not None:
                _frontres_task_action_mask = torch.zeros(8, device=self.device)
                for _idx in _active_dims:
                    _idx = int(_idx)
                    if not 0 <= _idx < 8:
                        raise ValueError(
                            "frontres_active_task_dims must contain indices in [0, 7] "
                            "for [dx,dy,dz,droll,dpitch,dyaw,c_pos,c_rpy]."
                        )
                    _frontres_task_action_mask[_idx] = 1.0
                print(
                    "[Runner] FrontRES task-space action mask enabled: "
                    "[dx,dy,dz,droll,dpitch,dyaw,c_pos,c_rpy] mask="
                    f"{_frontres_task_action_mask.detach().cpu().tolist()}",
                    flush=True,
                )

        def _mask_frontres_task_actions(_actions: torch.Tensor) -> torch.Tensor:
            if _frontres_task_action_mask is None:
                return _actions
            return _actions * _frontres_task_action_mask.view(1, -1)

        # ── Adaptive DR: PI controller ────────────────────────────────────────
        # Tracks dr_target_r_delta performance level.
        #   error = r_delta_ema - dr_target_r_delta
        #   error > 0: FrontRES above target → increase DR (harder)
        #   error < 0: FrontRES below target → decrease DR (easier)
        _dr_max         = float(self.cfg.get("dr_max_scale",    4.0))
        _dr_min         = float(self.cfg.get("dr_min_scale",    0.0))
        _dr_ema_alpha   = float(self.cfg.get("dr_ema_alpha",    0.95))

        # Restore dr_scale from checkpoint (set by load()); start at 0 for fresh runs.
        _dr_scale_init = float(self.cfg.get("dr_scale_init", 0.3))
        _dr_scale     = float(getattr(self, '_dr_scale', _dr_scale_init))
        _dr_scale     = max(_dr_scale, _dr_scale_init)  # enforce floor on resume
        _r_delta_ema  = 0.1  # optimistic: push DR up from dr_init=1.0

        # Read base perturbation values from env config (scaled by dr_scale each iteration).
        # Only ratio/magnitude fields are scaled; prob fields remain constant.
        _perturb_target = None
        if _is_frontres:
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(_env_raw, 'cfg') and hasattr(_env_raw.cfg, 'motion_perturbations'):
                from types import SimpleNamespace as _NS
                _pt = _env_raw.cfg.motion_perturbations
                _perturb_target = _NS(
                    float_prob           = float(_pt.float_prob),
                    float_ratio          = float(_pt.float_ratio),
                    sink_prob            = float(_pt.sink_prob),
                    sink_ratio           = float(_pt.sink_ratio),
                    foot_slip_prob       = float(_pt.foot_slip_prob),
                    foot_slip_ratio      = float(_pt.foot_slip_ratio),
                    lateral_drift_prob   = float(getattr(_pt, 'lateral_drift_prob',   0.0)),
                    lateral_drift_std    = float(getattr(_pt, 'lateral_drift_std',    0.0)),
                    root_tilt_prob       = float(getattr(_pt, 'root_tilt_prob',       0.0)),
                    root_tilt_max_rad    = float(getattr(_pt, 'root_tilt_max_rad',    0.0)),
                    joint_noise_prob     = float(getattr(_pt, 'joint_noise_prob',     0.0)),
                    joint_noise_std      = float(getattr(_pt, 'joint_noise_std',      0.0)),
                    # IID step-jump base values
                    iid_prob_z           = float(getattr(_pt, 'iid_prob_z',           0.0)),
                    iid_std_z            = float(getattr(_pt, 'iid_std_z',            0.0)),
                    iid_prob_xy          = float(getattr(_pt, 'iid_prob_xy',          0.0)),
                    iid_std_xy           = float(getattr(_pt, 'iid_std_xy',           0.0)),
                    iid_prob_rp          = float(getattr(_pt, 'iid_prob_rp',          0.0)),
                    iid_std_rp           = float(getattr(_pt, 'iid_std_rp',           0.0)),
                    iid_prob_ya          = float(getattr(_pt, 'iid_prob_ya',          0.0)),
                    iid_std_ya           = float(getattr(_pt, 'iid_std_ya',           0.0)),
                    local_root_artifact_prob = float(getattr(_pt, 'local_root_artifact_prob', 0.0)),
                    local_root_artifact_xy_std = float(getattr(_pt, 'local_root_artifact_xy_std', 0.0)),
                    local_root_artifact_yaw_std = float(getattr(_pt, 'local_root_artifact_yaw_std', 0.0)),
                )
                print(
                    f"[Runner] Adaptive DR (PI controller): "
                    f"Kp={self.cfg.get('dr_p_gain', 0.10)}, "
                    f"Ki={self.cfg.get('dr_i_gain', 0.01)}, "
                    f"target_r_delta={self.cfg.get('dr_target_r_delta', 0.01)}, "
                    f"max_scale={_dr_max}, ema_alpha={_dr_ema_alpha}, "
                    f"start_actor_weight={self.cfg.get('dr_start_ppo_actor_weight', 1.0)}, "
                    f"resume dr_scale={_dr_scale:.3f}"
                )
            else:
                print("[Runner] WARNING: FrontRES DR enabled but env.cfg.motion_perturbations not found")

        def _apply_frontres_dr_scale(scale: float) -> None:
            if not (_is_frontres and _perturb_target is not None):
                return
            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if not (hasattr(_env_raw, 'command_manager') and 'motion' in _env_raw.command_manager._terms):
                return
            _mcmd = _env_raw.command_manager._terms['motion']
            if not hasattr(_mcmd, 'perturber'):
                return

            def _pt(attr, default=0.0):
                return getattr(_perturb_target, attr, default)

            _mcmd.perturber.cfg.float_prob          = _pt('float_prob')
            _mcmd.perturber.cfg.float_ratio         = _pt('float_ratio')         * scale
            _mcmd.perturber.cfg.sink_prob           = _pt('sink_prob')
            _mcmd.perturber.cfg.sink_ratio          = _pt('sink_ratio')          * scale
            _mcmd.perturber.cfg.foot_slip_prob      = _pt('foot_slip_prob')
            _mcmd.perturber.cfg.foot_slip_ratio     = _pt('foot_slip_ratio')     * scale
            _mcmd.perturber.cfg.lateral_drift_prob  = _pt('lateral_drift_prob')
            _mcmd.perturber.cfg.lateral_drift_std   = _pt('lateral_drift_std')   * scale
            _mcmd.perturber.cfg.root_tilt_prob      = _pt('root_tilt_prob')
            _mcmd.perturber.cfg.root_tilt_max_rad   = _pt('root_tilt_max_rad')   * scale
            _mcmd.perturber.cfg.joint_noise_prob    = _pt('joint_noise_prob')
            _mcmd.perturber.cfg.joint_noise_std     = _pt('joint_noise_std')     * scale
            _mcmd.perturber.cfg.iid_prob_z          = _pt('iid_prob_z')
            _mcmd.perturber.cfg.iid_std_z           = _pt('iid_std_z')           * scale
            _mcmd.perturber.cfg.iid_prob_xy         = _pt('iid_prob_xy')
            _mcmd.perturber.cfg.iid_std_xy          = _pt('iid_std_xy')          * scale
            _mcmd.perturber.cfg.iid_prob_rp         = _pt('iid_prob_rp')
            _mcmd.perturber.cfg.iid_std_rp          = _pt('iid_std_rp')          * scale
            _mcmd.perturber.cfg.iid_prob_ya         = _pt('iid_prob_ya')
            _mcmd.perturber.cfg.iid_std_ya          = _pt('iid_std_ya')          * scale
            _mcmd.perturber.cfg.local_root_artifact_prob = _pt('local_root_artifact_prob')
            # Local artifact magnitudes are multiplied by perturber._dr_scale
            # at burst sampling time, so keep cfg as the unscaled base value.
            _mcmd.perturber.cfg.local_root_artifact_xy_std = _pt('local_root_artifact_xy_std')
            _mcmd.perturber.cfg.local_root_artifact_yaw_std = _pt('local_root_artifact_yaw_std')
            _mcmd.perturber._dr_scale = float(scale)

        # ── FrontRES supervised warmup (Stage 1 → Stage 2 merge) ──────────────────
        # Runs BEFORE the PPO loop. Teaches FrontRES to detect perturbations via
        # supervised learning on the anti-DR target:  target = -(perturbed - original).
        # After warmup, PPO fine-tunes with r_delta reward (same architecture, same obs).
        _warmup_iters = int(self.cfg.get("supervised_warmup_iterations", 0))
        if start_iter > 0 and _warmup_iters > 0:
            print(f"[Runner] Resuming from iter {start_iter} — skipping supervised warmup", flush=True)
            _warmup_iters = 0
        if getattr(self, "_frontres_warmup_complete", False) and _warmup_iters > 0:
            print("[Runner] Loaded a completed FrontRES warmup checkpoint — skipping supervised warmup",
                  flush=True)
            _warmup_iters = 0
        if _is_frontres and _warmup_iters > 0:
            _warmup_dr_scale_end = float(self.cfg.get("supervised_warmup_dr_scale", _dr_scale_init))
            _warmup_dr_scale_start = float(self.cfg.get(
                "supervised_warmup_dr_scale_start",
                self.cfg.get("supervised_warmup_dr_scale_min", _warmup_dr_scale_end),
            ))
            _warmup_dr_scale_start = max(0.0, _warmup_dr_scale_start)
            _warmup_dr_scale_end = max(0.0, _warmup_dr_scale_end)
            _warmup_lr     = float(self.cfg.get("supervised_warmup_lr", 1e-4))
            _warmup_epochs = int(self.cfg.get("supervised_warmup_epochs", 5))
            _warmup_steps  = int(self.cfg.get("supervised_warmup_steps_per_iter", self.num_steps_per_env))
            _warmup_steps  = max(1, min(_warmup_steps, self.num_steps_per_env))
            _warmup_max_envs = int(self.cfg.get("supervised_warmup_max_envs_per_step", self.env.num_envs))
            _warmup_max_envs = max(1, min(_warmup_max_envs, self.env.num_envs))
            _warmup_valid_w = float(getattr(self.alg, "supervised_valid_loss_weight", 4.0))
            _warmup_dir_w = float(getattr(self.alg, "supervised_direction_loss_weight", 0.1))
            _warmup_energy_w = float(self.cfg.get("frontres_warmup_energy_loss_weight", 1.0))
            _warmup_diag_interval = int(self.cfg.get(
                "supervised_warmup_diag_interval", max(1, _warmup_iters // 5)))
            _warmup_diag_interval = max(1, _warmup_diag_interval)
            _warmup_opt = torch.optim.Adam(
                list(self.alg.policy.residual_actor.parameters())
                + list(self.alg.policy.critic.parameters()),
                lr=_warmup_lr,
            )

            # Import once to avoid per-step overhead
            from whole_body_tracking.tasks.tracking.mdp.observations import \
                get_supervision_target_task_space as _get_warmup_target

            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            _nfo = self.alg.policy.num_frontres_obs
            if _nfo <= 0:
                _nfo = self.alg.policy.num_actor_obs  # use full obs when no subset configured

            _warmup_dr_desc = (
                f"{_warmup_dr_scale_start}->{_warmup_dr_scale_end}"
                if abs(_warmup_dr_scale_end - _warmup_dr_scale_start) > 1e-8
                else f"{_warmup_dr_scale_end}"
            )
            print(f"[Runner] === Joint warmup: {_warmup_iters} iters "
                  f"(dr_scale={_warmup_dr_desc}, lr={_warmup_lr}, epochs={_warmup_epochs}, "
                  f"steps_per_iter={_warmup_steps}, "
                  f"max_envs_per_step={_warmup_max_envs}, "
                  f"frontres_input={_nfo} dims, energy_w={_warmup_energy_w}) ===",
                  flush=True)

            for _wu in range(_warmup_iters):
                if _warmup_iters > 1:
                    _warmup_frac = _wu / float(_warmup_iters - 1)
                else:
                    _warmup_frac = 1.0
                # Smooth curriculum: spend early warmup on learnable direction,
                # then expose the critic/actor to near-boundary perturbations.
                _warmup_frac = _warmup_frac * _warmup_frac * (3.0 - 2.0 * _warmup_frac)
                _warmup_dr_scale = (
                    _warmup_dr_scale_start
                    + (_warmup_dr_scale_end - _warmup_dr_scale_start) * _warmup_frac
                )
                _apply_frontres_dr_scale(_warmup_dr_scale)

                _wo_list: list[torch.Tensor] = []
                _wt_list: list[torch.Tensor] = []
                _wc_list: list[torch.Tensor] = []
                _we_list: list[torch.Tensor] = []

                # Use no_grad rather than inference_mode: warmup samples are
                # later fed back through trainable actor/critic networks, and
                # some PyTorch versions reject inference tensors in backward.
                with torch.no_grad():
                    for _ in range(_warmup_steps):
                        obs, extras = self.env.get_observations()
                        obs_dict = extras.get("observations", {})
                        _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                        _p_obs = self._apply_obs_normalizer(_p_obs_raw)

                        # GMT-only actions (FrontRES correction = zero during warmup)
                        env_actions = self.alg.policy.get_env_action(
                            _p_obs,
                            torch.zeros(_p_obs.shape[0], self.alg.policy.total_output_dim,
                                        device=self.device))

                        obs, rewards_wu, dones, extras = self.env.step(env_actions.to(self.env.device))
                        obs_dict = extras.get("observations", {})
                        _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                        _p_obs = self._apply_obs_normalizer(_p_obs_raw)
                        if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                            _c_obs = self.privileged_obs_normalizer(
                                obs_dict[self.privileged_obs_type].to(self.device)
                            )
                        else:
                            _c_obs = _p_obs
                        _target = _get_warmup_target(_env_raw, "motion").to(self.device)
                        if "N_train" in locals() and N_train > 0 and N_base > 0 and N_clean > 0:
                            _n_energy = min(N_train, N_base, N_clean)
                            _r_noisy_wu = rewards_wu[N_train:N_train + _n_energy].view(-1).to(self.device)
                            _clean_start_wu = N_train + N_base
                            _r_clean_wu = rewards_wu[
                                _clean_start_wu:_clean_start_wu + _n_energy
                            ].view(-1).to(self.device)
                            _energy_target = (_r_clean_wu - _r_noisy_wu).clamp(min=0.0).unsqueeze(-1)
                            _p_obs = _p_obs[:_n_energy]
                            _c_obs = _c_obs[:_n_energy]
                            _target = _target[:_n_energy]
                        else:
                            _energy_target = torch.zeros(_p_obs.shape[0], 1, device=self.device)

                        if _warmup_max_envs < _p_obs.shape[0]:
                            _sample_ids = torch.randperm(
                                _p_obs.shape[0], device=self.device)[:_warmup_max_envs]
                            _p_obs = _p_obs[_sample_ids]
                            _c_obs = _c_obs[_sample_ids]
                            _target = _target[_sample_ids]
                            _energy_target = _energy_target[_sample_ids]

                        # Both obs and target reflect the perturbation applied this step
                        _wo_list.append(_p_obs[:, :_nfo])
                        _wt_list.append(_target)
                        _wc_list.append(_c_obs)
                        _we_list.append(_energy_target)

                # Joint SGD over collected rollout data:
                #   Actor:  Δ ≈ -noise
                #   Critic: E(s_noisy) ≈ max(R_clean - R_noisy, 0)
                _all_obs = torch.cat(_wo_list, dim=0)      # (S*E, _nfo)
                _all_tgt = torch.cat(_wt_list, dim=0)      # (S*E, 6)
                _all_critic_obs = torch.cat(_wc_list, dim=0)
                _all_energy = torch.cat(_we_list, dim=0)
                _N = _all_obs.shape[0]
                _last_actor_loss = torch.tensor(0.0, device=self.device)
                _last_energy_loss = torch.tensor(0.0, device=self.device)

                for epoch in range(_warmup_epochs):
                    perm = torch.randperm(_N, device=self.device)
                    for i in range(0, _N, 4096):
                        idx = perm[i:i + 4096]
                        pred = self.alg.policy.residual_actor(_all_obs[idx])
                        # pred: [N, 8] (Δpos_raw, Δrpy_raw, conf_raw)
                        # tgt:  [N, 6] (Δpos, Δrpy) — compare pos/rpy only
                        if getattr(self.alg.policy, 'num_task_corrections', 0) > 0:
                            pred_sup = torch.cat([
                                torch.tanh(pred[:, :3]) * self.alg.policy.max_delta_pos,
                                torch.tanh(pred[:, 3:6]) * self.alg.policy.max_delta_rpy,
                            ], dim=-1)
                            target_sup = torch.cat([
                                _all_tgt[idx, :3].clamp(
                                    -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                                _all_tgt[idx, 3:].clamp(
                                    -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                            ], dim=-1)
                        else:
                            pred_sup = pred[:, :_all_tgt.shape[-1]]
                            target_sup = _all_tgt[idx]

                        target_norm = target_sup.norm(dim=-1)
                        valid = target_norm > 1e-4
                        sample_weight = torch.ones_like(target_norm)
                        if valid.any():
                            sample_weight[valid] = _warmup_valid_w
                        sample_weight = sample_weight / sample_weight.mean().clamp(min=1e-6)

                        pos_err = torch.nn.functional.huber_loss(
                            pred_sup[:, :3], target_sup[:, :3].detach(), reduction="none").mean(dim=-1)
                        rpy_err = torch.nn.functional.huber_loss(
                            pred_sup[:, 3:], target_sup[:, 3:].detach(), reduction="none").mean(dim=-1)
                        _rpy_w = float(getattr(self.alg, 'supervised_rpy_loss_weight', 1.0))
                        loss = (pos_err * sample_weight).mean() + _rpy_w * (rpy_err * sample_weight).mean()
                        if valid.any() and _warmup_dir_w > 0.0:
                            direction_loss = 1.0 - torch.nn.functional.cosine_similarity(
                                pred_sup[valid], target_sup[valid].detach(), dim=-1).mean()
                            loss = loss + _warmup_dir_w * direction_loss
                        _conf_w = float(getattr(self.alg, 'supervised_conf_loss_weight', 0.0))
                        if getattr(self.alg.policy, 'num_task_corrections', 0) > 0 and pred.shape[-1] >= 8 and _conf_w > 0:
                            target_conf = valid.view(-1, 1).to(pred.dtype)
                            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                pred[:, 6:8], target_conf.expand(-1, 2))
                            loss = loss + _conf_w * conf_loss
                        actor_loss = loss
                        value_pred = self.alg.policy.evaluate(_all_critic_obs[idx])
                        energy_loss = torch.nn.functional.huber_loss(
                            value_pred, _all_energy[idx].detach(), reduction="mean"
                        )
                        loss = actor_loss + _warmup_energy_w * energy_loss
                        _warmup_opt.zero_grad()
                        loss.backward()
                        _warmup_opt.step()
                        _last_actor_loss = actor_loss.detach()
                        _last_energy_loss = energy_loss.detach()

                if (_wu + 1) % _warmup_diag_interval == 0 or (_wu + 1) == _warmup_iters:
                    with torch.inference_mode():
                        _valid_all = _all_tgt.norm(dim=-1) > 1e-4
                        _pred_all_raw = self.alg.policy.residual_actor(_all_obs[:, :_nfo])
                        if getattr(self.alg.policy, 'num_task_corrections', 0) > 0:
                            _pred_all = torch.cat([
                                torch.tanh(_pred_all_raw[:, :3]) * self.alg.policy.max_delta_pos,
                                torch.tanh(_pred_all_raw[:, 3:6]) * self.alg.policy.max_delta_rpy,
                            ], dim=-1)
                            _target_all = torch.cat([
                                _all_tgt[:, :3].clamp(
                                    -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                                _all_tgt[:, 3:].clamp(
                                    -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                            ], dim=-1)
                        else:
                            _pred_all = _pred_all_raw[:, :_all_tgt.shape[-1]]
                            _target_all = _all_tgt

                        _valid_all = _target_all.norm(dim=-1) > 1e-4
                        _valid_pos = _target_all[:, :3].norm(dim=-1) > 1e-4
                        _valid_rpy = _target_all[:, 3:].norm(dim=-1) > 1e-4

                        def _masked_cos(a, b, mask):
                            if mask.any():
                                return torch.nn.functional.cosine_similarity(
                                    a[mask], b[mask], dim=-1).mean().item()
                            return 0.0

                        def _masked_mae(a, b, mask):
                            if mask.any():
                                return (a[mask] - b[mask]).abs().mean().item()
                            return 0.0

                        def _masked_norm(a, mask):
                            if mask.any():
                                return a[mask].norm(dim=-1).mean().item()
                            return 0.0

                        if _valid_all.any():
                            _warmup_cos = torch.nn.functional.cosine_similarity(
                                _pred_all[_valid_all], _target_all[_valid_all], dim=-1).mean().item()
                        else:
                            _warmup_cos = 0.0
                        _valid_frac = _valid_all.float().mean().item()
                        _valid_pos_frac = _valid_pos.float().mean().item()
                        _valid_rpy_frac = _valid_rpy.float().mean().item()
                        _cos_pos = _masked_cos(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                        _cos_rpy = _masked_cos(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                        _mae_pos = _masked_mae(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                        _mae_rpy = _masked_mae(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                        _pred_pos_norm = _masked_norm(_pred_all[:, :3], _valid_pos)
                        _tgt_pos_norm = _masked_norm(_target_all[:, :3], _valid_pos)
                        _pred_rpy_norm = _masked_norm(_pred_all[:, 3:], _valid_rpy)
                        _tgt_rpy_norm = _masked_norm(_target_all[:, 3:], _valid_rpy)
                        _energy_pred_all = self.alg.policy.evaluate(_all_critic_obs)
                        _energy_loss_all = torch.nn.functional.huber_loss(
                            _energy_pred_all, _all_energy, reduction="mean").item()
                        _energy_mae = (_energy_pred_all - _all_energy).abs().mean().item()
                        _energy_target_mean = _all_energy.mean().item()
                        _energy_pred_mean = _energy_pred_all.mean().item()
                        _energy_target_std = _all_energy.std(unbiased=False).item()
                        _energy_pred_std = _energy_pred_all.std(unbiased=False).item()
                        _energy_cov = (
                            (_energy_pred_all - _energy_pred_all.mean())
                            * (_all_energy - _all_energy.mean())
                        ).mean()
                        _energy_corr = (
                            _energy_cov
                            / (_energy_pred_all.std(unbiased=False) * _all_energy.std(unbiased=False)).clamp(min=1e-6)
                        ).item()
                        _safe_gap_diag = float(self.cfg.get("frontres_safe_gap_per_step", 0.003))
                        _broken_gap_diag = float(self.cfg.get("frontres_broken_gap_per_step", 0.08))
                        _energy_damage_frac = (_all_energy.view(-1) > _safe_gap_diag).float().mean().item()
                        _energy_broken_frac = (_all_energy.view(-1) > _broken_gap_diag).float().mean().item()
                    print(f"[Runner]   warmup {_wu + 1}/{_warmup_iters}: "
                          f"dr_scale={_warmup_dr_scale:.3f}, "
                          f"loss={loss.item():.6f}, actor={_last_actor_loss.item():.6f}, "
                          f"energy={_last_energy_loss.item():.6f}, cos={_warmup_cos:.4f}, "
                          f"valid={_valid_frac:.3f}",
                          flush=True)
                    print(f"[Runner]      diag: "
                          f"cos_pos={_cos_pos:+.4f}, cos_rpy={_cos_rpy:+.4f}, "
                          f"valid_pos={_valid_pos_frac:.3f}, valid_rpy={_valid_rpy_frac:.3f}",
                          flush=True)
                    print(f"[Runner]      diag: "
                          f"mae_pos={_mae_pos:.5f}m, mae_rpy={_mae_rpy:.5f}rad, "
                          f"|pred_pos|/|tgt_pos|={_pred_pos_norm:.5f}/{_tgt_pos_norm:.5f}, "
                          f"|pred_rpy|/|tgt_rpy|={_pred_rpy_norm:.5f}/{_tgt_rpy_norm:.5f}",
                          flush=True)
                    print(f"[Runner]      energy: "
                          f"loss={_energy_loss_all:.6f}, mae={_energy_mae:.6f}, "
                          f"pred/target={_energy_pred_mean:.6f}/{_energy_target_mean:.6f}",
                          flush=True)
                    print(f"[Runner]      energy: "
                          f"corr={_energy_corr:+.4f}, std_pred/target={_energy_pred_std:.6f}/{_energy_target_std:.6f}, "
                          f"damage_frac={_energy_damage_frac:.3f}, broken_frac={_energy_broken_frac:.3f}",
                          flush=True)

            print(f"[Runner] === Joint warmup complete (final loss={loss.item():.6f}) ===",
                  flush=True)
            # Save warmup checkpoint so subsequent runs can skip this phase via --resume
            if self.log_dir is not None:
                self._frontres_warmup_complete = True
                self._dr_scale = _dr_scale
                warmup_path = os.path.join(self.log_dir, f"model_warmup.pt")
                self.save(warmup_path)
                print(f"[Runner] Warmup checkpoint saved to {warmup_path}", flush=True)
            _apply_frontres_dr_scale(_dr_scale)
        # ── END supervised warmup ─────────────────────────────────────────────────

        print(
            f"[Runner] Entering PPO loop: start_iter={start_iter}, "
            f"tot_iter={tot_iter}, steps_per_env={self.num_steps_per_env}",
            flush=True,
        )
        for it in range(start_iter, tot_iter):
            start = time.time()
            _ppo_actor_weight_current = _frontres_ppo_actor_weight_for_iter(it)
            if _is_frontres and hasattr(self.alg, "ppo_actor_weight"):
                # Set before collection as well as before update so diagnostics,
                # curriculum gates, and PPO loss all see the same phase.
                self.alg.ppo_actor_weight = _ppo_actor_weight_current

            # --- Critic warmup: fixed low DR scale ─────────────────────────
            # Critic learns V(s) under real low perturbations.  PPO actor may
            # be frozen separately by ppo_actor_warmup_iterations; supervised
            # loss remains active throughout.
            _critic_warmup = (isinstance(self.alg.policy, FrontRESActorCritic)
                              and critic_warmup_iters > 0
                              and (it - start_iter) < critic_warmup_iters)

            # --- DR PI controller: r_delta → dr_scale ---------------------------
            # Tracks a target r_delta performance level.
            # error = r_delta_ema - dr_target_r_delta
            #   error > 0: FrontRES above target → task too easy → increase DR
            #   error < 0: FrontRES below target → task too hard → decrease DR
            # P term: proportional response (reacts to magnitude, not just sign)
            # I term: removes steady-state offset; anti-windup prevents saturation
            _actor_takeover_active = False
            if _is_frontres and _perturb_target is not None:
                _dr_start_actor_weight = float(self.cfg.get("dr_start_ppo_actor_weight", 1.0))
                _actor_takeover_active = (
                    (not _critic_warmup)
                    and _ppo_actor_weight_current < _dr_start_actor_weight
                )
                # Update r_delta EMA from last iteration's mean
                _r_delta_ema = (_dr_ema_alpha * _r_delta_ema
                                + (1.0 - _dr_ema_alpha) * getattr(self, '_last_r_delta_mean', 0.0))
                if _critic_warmup or _actor_takeover_active:
                    # Hold at dr_scale_init during Critic warmup.
                    # Also hold during Actor takeover.  Otherwise p(s_noisy) and
                    # p(s_noisy + Δ_actor) drift at the same time, which makes
                    # Critic targets hard to interpret and PPO updates unstable.
                    _dr_scale = _dr_scale_init
                    self._dr_hold_just_ended = True
                else:
                    if getattr(self, '_dr_hold_just_ended', False):
                        _dr_scale = _dr_scale_init
                        # The hold phase also runs at dr_scale_init, so the EMA is
                        # real signal.  Keep it; only reset the PI previous error
                        # so the first adaptive step is not a stale jump.
                        if hasattr(self, "_dr_prev_error"):
                            delattr(self, "_dr_prev_error")
                        self._dr_hold_just_ended = False

                    _kp        = float(self.cfg.get("dr_p_gain",          0.10))
                    _ki        = float(self.cfg.get("dr_i_gain",          0.01))
                    _dr_target = float(self.cfg.get("dr_target_r_delta",  0.01))

                    error      = _r_delta_ema - _dr_target
                    _prev_err  = getattr(self, '_dr_prev_error', error)
                    # Velocity-form PI: Δu = Kp×Δe + Ki×e
                    # dr_scale += Δu  (stays constant when error = 0)
                    # No double-integration: P reacts to error *change*, I to error *level*.
                    _delta = _kp * (error - _prev_err) + _ki * error
                    self._dr_prev_error = error

                    _dr_scale = max(_dr_min, min(_dr_max, _dr_scale + _delta))

                # Persist for resume
                self._dr_scale = _dr_scale
                # Apply dr_scale to perturber (prob fields unchanged; only ratio/magnitude fields)
                _apply_frontres_dr_scale(_dr_scale)

            # FrontRES reward-shaping state: reset at the start of each rollout.
            _frontres_prev_delta_q: torch.Tensor | None = None  # [N, A] for smoothness penalty
            # Accumulators for wandb logging (per iteration, divided by shaping steps)
            _frontres_rdelta_sum:         float = 0.0
            _frontres_baseline_sum:       float = 0.0
            _frontres_r_z_sum:            float = 0.0
            _frontres_r_xy_sum:           float = 0.0
            _frontres_r_rp_sum:           float = 0.0
            _frontres_r_yaw_sum:          float = 0.0
            _frontres_r_rescue_sum:       float = 0.0
            _frontres_r_exec_sum:         float = 0.0
            _frontres_r_geom_sum:         float = 0.0
            _frontres_intervention_cost_sum: float = 0.0
            _frontres_reward_frontres_sum: float = 0.0
            _frontres_damage_gap_sum:     float = 0.0
            _frontres_repair_gain_sum:    float = 0.0
            _frontres_repair_ratio_sum:   float = 0.0
            _frontres_safe_gate_sum:      float = 0.0
            _frontres_fragile_gate_sum:   float = 0.0
            _frontres_damage_gate_sum:    float = 0.0
            _frontres_broken_gate_sum:    float = 0.0
            _frontres_actor_gate_sum:     float = 0.0
            _frontres_exec_gate_sum:      float = 0.0
            _frontres_cost_gate_sum:      float = 0.0
            _frontres_dr_z_abs_sum:       float = 0.0
            _frontres_dr_xy_abs_sum:      float = 0.0
            _frontres_dr_rp_abs_sum:      float = 0.0
            _frontres_dr_yaw_abs_sum:     float = 0.0
            _frontres_corr_z_abs_sum:     float = 0.0
            _frontres_corr_xy_abs_sum:    float = 0.0
            _frontres_corr_rp_abs_sum:    float = 0.0
            _frontres_corr_yaw_abs_sum:   float = 0.0
            _frontres_smooth_penalty_sum: float = 0.0
            _frontres_reg_penalty_sum:    float = 0.0
            _frontres_delta_pos_abs_sum:  float = 0.0   # mean |Δpos[:3]| per step (task-space mode)
            _frontres_delta_rpy_abs_sum:  float = 0.0   # mean |Δrpy[3:]| per step (task-space mode)
            _frontres_delta_z_abs_sum:    float = 0.0   # mean |Δz| per step (joint-space mode only)
            _frontres_jump_degree_sum:    float = 0.0   # mean jump_degree (gate activation monitor)
            _frontres_shaping_steps:      int   = 0
            _frontres_reward_diag_steps:  int   = 0
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
                    if self.training_type in ("mosaic", "frontres"):
                        # Extract motion groups for MOSAIC multi-teacher support.
                        # FrontRESUnified ignores this argument; paired-baseline bookkeeping
                        # is handled directly by the motion command.
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
                                               teacher_obs=teacher_obs if self.training_type == "mosaic" else None,
                                               ref_vel_estimator_obs=ref_vel_estimator_obs,
                                               motion_groups=motion_groups)
                        if _is_task_space_mode:
                            actions = _mask_frontres_task_actions(actions)
                            self.alg.transition.actions = actions.detach()
                            self.alg.transition.actions_log_prob = \
                                self.alg.policy.get_actions_log_prob(actions).detach()

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
                            _zeros_gmt = torch.zeros(N_base + N_clean, self.alg.transition.actions.shape[-1],
                                                     device=self.device)
                            self.alg.transition.actions[N_train:] = _zeros_gmt
                            _mean_gmt = self.alg.policy.action_mean[N_train:].clone()
                            _std_gmt  = self.alg.policy.action_std[N_train:]
                            _logp_zeros = torch.distributions.Normal(_mean_gmt, _std_gmt) \
                                              .log_prob(_zeros_gmt).sum(dim=-1)
                            # TanhNormal Jacobian correction for zero action:
                            # bounded=0 → raw=atanh(0)=0, Jacobian = Σlog(max_d)
                            # so TanhNormal.log_prob(0) = Normal.log_prob(0) - Σlog(max_d)
                            # This ensures IS ratio = 1 for GMT envs (zero correction taken).
                            if _is_task_space_mode:
                                _logp_zeros = _logp_zeros - (
                                    3 * math.log(self.alg.policy.max_delta_pos)
                                    + 3 * math.log(self.alg.policy.max_delta_rpy))
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
                        if _is_task_space_mode:
                            actions = _mask_frontres_task_actions(actions)
                            self.alg.transition.actions = actions.detach()
                            self.alg.transition.actions_log_prob = \
                                self.alg.policy.get_actions_log_prob(actions).detach()

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
                            if _is_task_space_mode:
                                _logp_zeros = _logp_zeros - (
                                    3 * math.log(self.alg.policy.max_delta_pos)
                                    + 3 * math.log(self.alg.policy.max_delta_rpy))
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

                        # ── Curriculum Oracle ──────────────────────────────────────────
                        # Mixes oracle (-OU ground truth) with FrontRES output to break the
                        # chicken-and-egg deadlock:
                        #   Untrained FrontRES → bad corrections → r_delta<0 → negative
                        #   advantages → PPO fights supervised → FrontRES never learns.
                        # With oracle mix, r_delta stays ≥ 0 throughout; PPO and supervised
                        # gradients are always aligned. Mix fades out as FrontRES improves.
                        #
                        #   mix = 1.0 → pure oracle   (cos_sim < oracle_mix_cos_low)
                        #   mix = 0.0 → pure FrontRES (cos_sim ≥ oracle_mix_cos_high)
                        #   linear interpolation in between
                        #
                        # This is standard curriculum / demonstration bootstrapping (c.f.
                        # DAgger, residual RL). The oracle is only the OU state, which is
                        # available during training (not at deployment).
                        if self.cfg.get("oracle_curriculum", False) and _task_corr is not None:
                            _env_oracle = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                            for _cmd_oracle in _env_oracle.command_manager._terms.values():
                                if hasattr(_cmd_oracle, 'supervised_target'):
                                    _sup = _cmd_oracle.supervised_target.to(self.device)
                                    _oracle_full = torch.zeros_like(_task_corr)
                                    _n = min(_sup.shape[-1], _oracle_full.shape[-1])
                                    _oracle_full[:, :_n] = _sup[:, :_n]

                                    # Per-step EMA of cos(FrontRES_output, oracle)
                                    # Tracks how close raw FrontRES is to the ideal correction.
                                    _fr_v  = _task_corr[:N_train, :_n]
                                    _or_v  = _oracle_full[:N_train, :_n]
                                    _cos_s = (_fr_v * _or_v).sum(-1) / (
                                        _fr_v.norm(dim=-1) * _or_v.norm(dim=-1) + 1e-8)
                                    _ema_alpha = 0.99
                                    _prev_ema  = getattr(self, '_oracle_cos_ema', 0.0)
                                    _new_ema   = _ema_alpha * _prev_ema + (1 - _ema_alpha) * float(_cos_s.mean())
                                    self._oracle_cos_ema = _new_ema

                                    # Mix coefficient
                                    _cos_lo = float(self.cfg.get("oracle_mix_cos_low",  0.3))
                                    _cos_hi = float(self.cfg.get("oracle_mix_cos_high", 0.85))
                                    if _new_ema < _cos_lo:
                                        _mix = 1.0
                                    elif _new_ema < _cos_hi:
                                        _mix = 1.0 - (_new_ema - _cos_lo) / (_cos_hi - _cos_lo)
                                    else:
                                        _mix = 0.0
                                    self._oracle_mix = _mix

                                    if _mix > 0.0:
                                        _task_corr = (1.0 - _mix) * _task_corr + _mix * _oracle_full
                                    break
                        # ─────────────────────────────────────────────────────────────

                        _task_corr = _mask_frontres_task_actions(_task_corr) if _task_corr is not None else None
                        if _task_corr is not None:
                            _env_raw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                            for _cmd_term in _env_raw.command_manager._terms.values():
                                if hasattr(_cmd_term, '_frontres_pos_correction'):
                                    # ── Contact-aware conservative projection ───────────────
                                    # Δxy is softened during ballistic phases to avoid
                                    # rewriting the flight trajectory. Δrpy is never gated:
                                    # orientation artifacts can occur in both contact and
                                    # airborne phases and do not inject vertical support.
                                    #
                                    # Δz is asymmetric:
                                    #   contact phase: positive Δz is blocked (no artificial lift)
                                    #   airborne phase: positive Δz is allowed only up to
                                    #                   penetration depth (remove ground
                                    #                   penetration, not recover flight height)
                                    # _task_corr: [Δpos(3), Δrpy(3), c_pos(1), c_rpy(1)]
                                    _pos_corr = _task_corr[:N_train, :3].clone()
                                    _rpy_corr = _task_corr[:N_train, 3:6].clone()
                                    _c_pos    = _task_corr[:N_train, 6:7].clone()
                                    _c_rpy    = _task_corr[:N_train, 7:8].clone()

                                    _z_upper = torch.zeros_like(_pos_corr[:, 2])
                                    if hasattr(_cmd_term, 'jump_degree'):
                                        _jd = _cmd_term.jump_degree[:N_train].to(_task_corr.device).clamp(0.0, 1.0)
                                        _contact_gate = (1.0 - _jd).unsqueeze(-1)
                                        _pos_corr[:, :2] = _pos_corr[:, :2] * _contact_gate
                                        if hasattr(_cmd_term, 'anchor_penetration_depth'):
                                            _penetration = _cmd_term.anchor_penetration_depth[:N_train].to(_task_corr.device)
                                            _z_upper = _jd * _penetration

                                    _z_lower = torch.full_like(_pos_corr[:, 2], -self.alg.policy.max_delta_pos)
                                    _pos_corr[:, 2] = torch.maximum(_pos_corr[:, 2], _z_lower)
                                    _pos_corr[:, 2] = torch.minimum(_pos_corr[:, 2], _z_upper)

                                    # ── Confidence: soft coupling ─────────────────────
                                    _pos_corr = _pos_corr * _c_pos
                                    _rpy_corr = _rpy_corr * _c_rpy

                                    _cmd_term._frontres_pos_correction[:N_train].copy_(_pos_corr)
                                    _quat_corr = quat_from_euler_xyz(
                                        _rpy_corr[:, 0], _rpy_corr[:, 1], _rpy_corr[:, 2])
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

                    # ── GMT baseline: OU zeroing now handled by set_baseline_envs() ─────
                    # Baseline envs [N_train:] are registered once at runner start via
                    # perturber.set_baseline_envs(). _ou_step() enforces zero state for
                    # them on every physics step, so no per-step reset_envs() is needed.

                    # ── FrontRES B1 delta-reward ────────────────────────────────────────
                    # GMT baseline envs [N_train:] ran with delta_q=0 → rewards ≈ GMT-only.
                    # r_delta isolates FrontRES contribution to anchor tracking only.
                    # GMT env rewards are zeroed → returns ≈ 0 → advantage ≈ 0 → no policy gradient.
                    #
                    # Fix 1: anchor-only r_delta (HRL intrinsic reward)
                    # Global reward (joint_torque, contact, body vel) conflates FrontRES with GMT:
                    #   - FrontRES corrects correctly but causes torque spike → penalises correct action
                    #   - FrontRES correction has no effect → should give 0, not negative
                    # Anchor error is the ONLY thing FrontRES directly controls. Using it as the
                    # intrinsic reward decouples FrontRES credit from GMT behaviour.
                    if _is_frontres:
                        _base_start = N_train
                        _base_end = N_train + N_base
                        _clean_start = _base_end
                        _clean_end = _clean_start + N_clean
                        r_raw_gmt = rewards[_base_start:_base_end].view(-1).clone()
                        r_clean_gmt = rewards[_clean_start:_clean_end].view(-1).clone()
                        r_total = rewards[:N_train].view(-1).clone()

                        # ── Anchor-only r_delta (GMT-free) ────────────────────────────
                        # Compares corrected anchor against ORIGINAL motion data,
                        # NOT against robot position.  GMT tracking noise ~28cm
                        # cannot drown a 2cm FrontRES correction anymore.
                        _env_for_rdelta = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        _mcmd_rdelta = _env_for_rdelta.command_manager._terms.get('motion')
                        _use_clean = (
                            _mcmd_rdelta is not None
                            and hasattr(_mcmd_rdelta, 'anchor_pos_w_original')
                            and hasattr(_mcmd_rdelta, 'anchor_quat_w_original')
                        )
                        if _use_clean:
                            # ── Per-axis r_step ───────────────────────────
                            # r_axis[i] = |DR_i| - |DR_i + correction_i|
                            # Keep this per-env; batch-averaging here destroys PPO
                            # credit assignment because every action receives the same reward.
                            _a_w   = _mcmd_rdelta.anchor_pos_w_original
                            _a_raw = _mcmd_rdelta.anchor_pos_w_raw
                            _a_fr  = _mcmd_rdelta.anchor_pos_w
                            _q_w   = _mcmd_rdelta.anchor_quat_w_original
                            _q_raw = _mcmd_rdelta.anchor_quat_w_raw
                            _q_fr  = _mcmd_rdelta.anchor_quat_w

                            def _r_axis(dr, corr):
                                return dr.abs() - (dr + corr).abs()

                            def _r_vec(dr_vec, corr_vec):
                                return dr_vec.norm(dim=-1) - (dr_vec + corr_vec).norm(dim=-1)

                            # Z
                            _dr_z_fr   = _a_raw[:N_train, 2] - _a_w[:N_train, 2]
                            _corr_z_fr = _a_fr[:N_train,  2] - _a_raw[:N_train, 2]
                            _r_z = _r_axis(_dr_z_fr, _corr_z_fr)
                            # XY
                            _dr_xy_fr   = _a_raw[:N_train, :2] - _a_w[:N_train, :2]
                            _corr_xy_fr = _a_fr[:N_train,  :2] - _a_raw[:N_train, :2]
                            _r_xy = _r_vec(_dr_xy_fr, _corr_xy_fr)
                            def _quat_to_rpy_xyzw(q: torch.Tensor):
                                # IsaacLab tensors here are xyzw. Keep yaw separate below so
                                # roll/pitch reward does not double-count heading errors.
                                x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                                roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                                pitch_arg = 2.0 * (w * y - z * x)
                                pitch = torch.asin(torch.clamp(pitch_arg, -1.0, 1.0))
                                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                                return roll, pitch, yaw

                            def _wrap_pi(a: torch.Tensor):
                                return torch.atan2(torch.sin(a), torch.cos(a))

                            # Roll/Pitch only.  The previous full-quaternion error also
                            # included yaw, which made the 0.4 RP term dominate r_delta.
                            _roll_raw, _pitch_raw, _yaw_raw = _quat_to_rpy_xyzw(_q_raw[:N_train])
                            _roll_fr,  _pitch_fr,  _yaw_fr  = _quat_to_rpy_xyzw(_q_fr[:N_train])
                            _roll_w,   _pitch_w,   _yaw_w   = _quat_to_rpy_xyzw(_q_w[:N_train])
                            _rp_raw = torch.stack(
                                [_wrap_pi(_roll_raw - _roll_w), _wrap_pi(_pitch_raw - _pitch_w)], dim=-1
                            )
                            _rp_fr = torch.stack(
                                [_wrap_pi(_roll_fr - _roll_w), _wrap_pi(_pitch_fr - _pitch_w)], dim=-1
                            )
                            _e_raw = _rp_raw.norm(dim=-1)
                            _e_fr  = _rp_fr.norm(dim=-1)
                            _r_rp = _e_raw - _e_fr
                            # Yaw
                            _yaw_err_raw = _wrap_pi(_yaw_raw - _yaw_w)
                            _yaw_corr = _wrap_pi(_yaw_fr - _yaw_raw)
                            _r_ya = _r_axis(_yaw_err_raw, _yaw_corr)

                            _r_step = 0.3*_r_z + 0.3*_r_xy + 0.15*_r_rp + 0.02*_r_ya
                            _dr_z_abs_log = _dr_z_fr.abs().mean()
                            _dr_xy_abs_log = _dr_xy_fr.norm(dim=-1).mean()
                            _dr_rp_abs_log = _e_raw.mean()
                            _dr_yaw_abs_log = _yaw_err_raw.abs().mean()
                            _corr_z_abs_log = _corr_z_fr.abs().mean()
                            _corr_xy_abs_log = _corr_xy_fr.norm(dim=-1).mean()
                            _corr_rp_abs_log = (_e_fr - _e_raw).abs().mean()
                            _corr_yaw_abs_log = _yaw_corr.abs().mean()

                            # ── r_rescue ─────────────────────────────────
                            _n_pair = min(N_train, N_base, N_clean)
                            _fell_base = dones[_base_start:_base_start+_n_pair].view(-1) > 0
                            _fell_fr   = dones[:_n_pair].view(-1) > 0
                            _r_rescue = torch.zeros(N_train, device=self.device)
                            _r_rescue_pair = torch.zeros(_n_pair, device=self.device)
                            # ±0.5 matches r_step magnitude (~0.02-0.05/step).
                            # ±10 was 200× larger, making Value learn fall-probability
                            # instead of correction quality, drowning the r_step signal.
                            _RESCUE_MAG = float(self.cfg.get("r_rescue_magnitude", 0.5))
                            _r_rescue_pair[_fell_base & ~_fell_fr] =  _RESCUE_MAG   # rescued
                            _r_rescue_pair[_fell_base &  _fell_fr] = -0.1 * _RESCUE_MAG   # both failed
                            _r_rescue_pair[~_fell_base & _fell_fr] = -_RESCUE_MAG   # FrontRES caused fall
                            _r_rescue[:_n_pair] = _r_rescue_pair

                            # ── Execution advantage (main HRL signal) ───────────────
                            # FrontRES is useful only if the corrected reference is
                            # easier for the frozen GMT to execute than the same
                            # perturbed reference without correction.  This paired
                            # advantage is the operational definition of
                            # "more executable"; clean-anchor geometry is only a
                            # small direction prior below.
                            _r_exec = torch.zeros(N_train, device=self.device)
                            _n_exec = min(N_train, N_base, N_clean)
                            _exec_temp = float(self.cfg.get("frontres_exec_reward_temp", 1.0))
                            _r_exec[:_n_exec] = torch.tanh(
                                (r_total[:_n_exec] - r_raw_gmt[:_n_exec]) / max(_exec_temp, 1e-6)
                            )

                            # ── Minimum-intervention cost ──────────────────────────
                            _intervention_cost = torch.zeros(N_train, device=self.device)
                            if _is_task_space_mode and actions.shape[-1] >= 6:
                                _delta = actions[:N_train, :6]
                                _max_delta = torch.tensor(
                                    [
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_pos,
                                        self.alg.policy.max_delta_rpy,
                                        self.alg.policy.max_delta_rpy,
                                        self.alg.policy.max_delta_rpy,
                                    ],
                                    device=self.device,
                                    dtype=_delta.dtype,
                                ).clamp(min=1e-6)
                                _weights = torch.tensor(
                                    self.cfg.get(
                                        "frontres_intervention_cost_weights",
                                        [0.02, 0.02, 0.05, 0.30, 0.30, 0.10],
                                    ),
                                    device=self.device,
                                    dtype=_delta.dtype,
                                )
                                _intervention_cost = (_weights * (_delta / _max_delta).pow(2)).sum(dim=-1)

                            _w_exec = float(self.cfg.get("frontres_exec_reward_weight", 1.0))
                            _w_geom = float(self.cfg.get("frontres_geometry_reward_weight", 0.05))
                            _w_rescue = float(self.cfg.get("frontres_rescue_reward_weight", 1.0))
                            # Executable-gap reward:
                            #   damage_gap  = R_clean - R_noisy
                            #   repair_gain = R_frontres - R_noisy
                            #   repair_ratio = repair_gain / damage_gap
                            #
                            # This makes the scale of r_delta invariant to the raw
                            # GMT reward magnitude.  Safe/no-op, repairable, and
                            # broken cases are separated with smooth sigmoid gates:
                            #   safe:    no meaningful damage -> learn minimum intervention
                            #   fragile: damage exists and is repairable -> learn repair
                            #   broken:  damage too large or currently unrepaired -> learn abstention cost
                            _gap_raw = r_clean_gmt[:_n_exec] - r_raw_gmt[:_n_exec]
                            _damage_gap = _gap_raw.clamp(min=0.0)
                            _repair_gain = r_total[:_n_exec] - r_raw_gmt[:_n_exec]
                            _gap_floor = float(self.cfg.get("frontres_gap_floor_per_step", 0.005))
                            _repair_ratio = (_repair_gain / _damage_gap.clamp(min=_gap_floor)).clamp(-1.0, 1.0)
                            _r_exec[:_n_exec] = _repair_ratio

                            _safe_gap = float(self.cfg.get("frontres_safe_gap_per_step", 0.003))
                            _broken_gap = float(self.cfg.get("frontres_broken_gap_per_step", 0.08))
                            _gate_temp = float(self.cfg.get("frontres_gap_gate_temp", 0.005))
                            _ratio_temp = float(self.cfg.get("frontres_repair_ratio_gate_temp", 0.20))
                            _broken_ratio_ref = float(self.cfg.get("frontres_broken_repair_ratio_ref", -0.25))
                            _damage_gate = torch.sigmoid((_damage_gap - _safe_gap) / max(_gate_temp, 1e-6))
                            _safe_gate = 1.0 - _damage_gate
                            _gap_broken_gate = torch.sigmoid((_damage_gap - _broken_gap) / max(_gate_temp, 1e-6))
                            _bad_repair_gate = torch.sigmoid(
                                (_broken_ratio_ref - _repair_ratio) / max(_ratio_temp, 1e-6)
                            )
                            _broken_gate = (_damage_gate * _gap_broken_gate * _bad_repair_gate).clamp(0.0, 1.0)
                            _fragile_gate = (_damage_gate * (1.0 - _broken_gate)).clamp(0.0, 1.0)

                            _safe_cost = float(self.cfg.get("frontres_safe_cost_weight", 1.0))
                            _fragile_cost = float(self.cfg.get("frontres_fragile_cost_weight", 0.10))
                            _broken_cost = float(self.cfg.get("frontres_broken_cost_weight", 1.0))
                            _cost_gate = _safe_cost * _safe_gate + _fragile_cost * _fragile_gate + _broken_cost * _broken_gate
                            _exec_gate = _fragile_gate

                            _actor_gate_floor = float(self.cfg.get("frontres_actor_gate_floor", 0.02))
                            _safe_actor_weight = float(self.cfg.get("frontres_safe_actor_gate_weight", 0.20))
                            _broken_actor_weight = float(self.cfg.get("frontres_broken_actor_gate_weight", 0.20))
                            _actor_gate = (
                                _actor_gate_floor
                                + (1.0 - _actor_gate_floor)
                                * (_fragile_gate + _safe_actor_weight * _safe_gate + _broken_actor_weight * _broken_gate)
                            ).clamp(0.0, 1.0)
                            _exec_weight = torch.zeros(N_train, device=self.device)
                            _cost_weight = torch.ones(N_train, device=self.device)
                            _exec_weight[:_n_exec] = _exec_gate
                            _cost_weight[:_n_exec] = _cost_gate
                            _frontres_actor_gate = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_actor_gate[:_n_exec, 0] = _actor_gate
                            self.alg.transition.frontres_actor_gate = _frontres_actor_gate
                            r_delta = (
                                _w_exec * _exec_weight * _repair_ratio
                                + _w_geom * _r_step
                                + _w_rescue * _r_rescue
                                - _cost_weight * _intervention_cost
                            )
                            _r_base_log   = r_raw_gmt.mean()
                            _r_rescue_log = _r_rescue.mean()
                        else:
                            r_raw_gmt = rewards[_base_start:_base_end].view(-1).clone()
                            r_total   = rewards[:N_train].view(-1)
                            if N_train == N_base:
                                r_delta = r_total - r_raw_gmt
                                _r_base_log = r_raw_gmt.mean()
                            else:
                                r_delta = r_total - r_raw_gmt.mean()
                                _r_base_log = r_raw_gmt.mean()
                            _r_rescue_log = 0.0
                            _r_z = _r_xy = _r_rp = _r_ya = None
                            _damage_gate = _broken_gate = _actor_gate = None
                            _frontres_actor_gate = torch.zeros(self.env.num_envs, 1, device=self.device)
                            _frontres_actor_gate[:N_train, 0] = 1.0
                            self.alg.transition.frontres_actor_gate = _frontres_actor_gate
                            _dr_z_abs_log = _dr_xy_abs_log = _dr_rp_abs_log = _dr_yaw_abs_log = None
                            _corr_z_abs_log = _corr_xy_abs_log = _corr_rp_abs_log = _corr_yaw_abs_log = None
                        # ─────────────────────────────────────────────────────────────

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
                        _frontres_baseline_sum += _r_base_log.item()
                        if _r_z is not None:
                            _frontres_r_z_sum          += _r_z.mean().item()
                            _frontres_r_xy_sum         += _r_xy.mean().item()
                            _frontres_r_rp_sum         += _r_rp.mean().item()
                            _frontres_r_yaw_sum        += _r_ya.mean().item()
                            _frontres_r_rescue_sum     += float(_r_rescue_log)
                            _frontres_r_exec_sum       += _r_exec.mean().item()
                            _frontres_r_geom_sum       += _r_step.mean().item()
                            _frontres_intervention_cost_sum += _intervention_cost.mean().item()
                            _frontres_reward_frontres_sum += r_total.mean().item()
                            _frontres_damage_gap_sum   += _damage_gap.mean().item()
                            _frontres_repair_gain_sum  += _repair_gain.mean().item()
                            _frontres_repair_ratio_sum += _repair_ratio.mean().item()
                            _frontres_safe_gate_sum    += _safe_gate.mean().item()
                            _frontres_fragile_gate_sum += _fragile_gate.mean().item()
                            _frontres_damage_gate_sum  += _damage_gate.mean().item()
                            _frontres_broken_gate_sum  += _broken_gate.mean().item()
                            _frontres_actor_gate_sum   += _actor_gate.mean().item()
                            _frontres_exec_gate_sum    += _exec_gate.mean().item()
                            _frontres_cost_gate_sum    += _cost_gate.mean().item()
                            _frontres_reward_diag_steps += 1
                            _frontres_dr_z_abs_sum     += _dr_z_abs_log.item()
                            _frontres_dr_xy_abs_sum    += _dr_xy_abs_log.item()
                            _frontres_dr_rp_abs_sum    += _dr_rp_abs_log.item()
                            _frontres_dr_yaw_abs_sum   += _dr_yaw_abs_log.item()
                            _frontres_corr_z_abs_sum   += _corr_z_abs_log.item()
                            _frontres_corr_xy_abs_sum  += _corr_xy_abs_log.item()
                            _frontres_corr_rp_abs_sum  += _corr_rp_abs_log.item()
                            _frontres_corr_yaw_abs_sum += _corr_yaw_abs_log.item()
                        # Task-space mode: split into Δpos and Δrpy; joint-space mode: log Δz
                        if _is_task_space_mode:
                            _tc = getattr(self.alg.policy, 'last_task_correction', None)
                            if _tc is not None:
                                _frontres_delta_pos_abs_sum += _tc[:N_train, :3].abs().mean().item()
                                _frontres_delta_rpy_abs_sum += _tc[:N_train, 3:6].abs().mean().item()
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

                        # Reset low-pass filter state for terminated FrontRES envs.
                        # Without this, the filter carries old correction into the new episode:
                        # new OU=0 → FrontRES predicts ≈0, but filter outputs 0.6×old_value
                        # for 4-5 steps → false anchor_pos/ori terminations immediately.
                        _prev_pos_c = getattr(self, '_prev_pos_correction', None)
                        if _prev_pos_c is not None:
                            _fr_done_mask = _done_mask[:N_train]
                            if _fr_done_mask.any():
                                _prev_pos_c[_fr_done_mask] = 0.0
                                self._prev_pos_correction = _prev_pos_c
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
                            cur_reward_sum_gmt[:N_base] += r_raw_gmt  # noisy GMT per-step reward
                            if N_clean > 0:
                                cur_reward_sum_gmt[N_base:N_base + N_clean] += r_clean_gmt
                        
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
                if self.training_type in ["rl", "mosaic", "frontres"]:
                    self.alg.compute_returns(privileged_obs)

            # update policy Rollout结束, 开始使用buffer计算Loss更新权重
            # Pass current iteration to algorithm for logging (needed by MOSAIC)
            self.alg.current_learning_iteration = it
            if _is_frontres and hasattr(self.alg, "ppo_actor_weight"):
                self.alg.ppo_actor_weight = _ppo_actor_weight_current
            # Pass oracle_mix so MOSAIC scales surrogate by (1 - oracle_mix):
            # PPO contribution ∝ FrontRES causal share of the correction applied.
            self.alg.oracle_mix = getattr(self, '_oracle_mix', 0.0)
            loss_dict = self.alg.update() # 调用mosaic.py中的update()函数进行权重更新

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # expose curriculum state and B1 delta-reward metrics to log() via locals()
            frontres_rdelta_mean   = (_frontres_rdelta_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_baseline_mean = (_frontres_baseline_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_z_mean      = (_frontres_r_z_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_xy_mean     = (_frontres_r_xy_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_rp_mean     = (_frontres_r_rp_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_yaw_mean    = (_frontres_r_yaw_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_rescue_mean = (_frontres_r_rescue_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_r_exec_mean = (_frontres_r_exec_sum / _frontres_reward_diag_steps
                                    if _is_frontres and _frontres_reward_diag_steps > 0 else None)
            frontres_r_geom_mean = (_frontres_r_geom_sum / _frontres_reward_diag_steps
                                    if _is_frontres and _frontres_reward_diag_steps > 0 else None)
            frontres_intervention_cost_mean = (
                _frontres_intervention_cost_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_reward_frontres_mean = (
                _frontres_reward_frontres_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_damage_gap_mean = (
                _frontres_damage_gap_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_gain_mean = (
                _frontres_repair_gain_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_repair_ratio_mean = (
                _frontres_repair_ratio_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_safe_gate_mean = (
                _frontres_safe_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_fragile_gate_mean = (
                _frontres_fragile_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_damage_gate_mean = (
                _frontres_damage_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_broken_gate_mean = (
                _frontres_broken_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_actor_gate_mean = (
                _frontres_actor_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_exec_gate_mean = (
                _frontres_exec_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_cost_gate_mean = (
                _frontres_cost_gate_sum / _frontres_reward_diag_steps
                if _is_frontres and _frontres_reward_diag_steps > 0 else None
            )
            frontres_dr_z_abs_mean = (_frontres_dr_z_abs_sum / _frontres_shaping_steps
                                      if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_xy_abs_mean = (_frontres_dr_xy_abs_sum / _frontres_shaping_steps
                                       if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_rp_abs_mean = (_frontres_dr_rp_abs_sum / _frontres_shaping_steps
                                       if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_dr_yaw_abs_mean = (_frontres_dr_yaw_abs_sum / _frontres_shaping_steps
                                        if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_z_abs_mean = (_frontres_corr_z_abs_sum / _frontres_shaping_steps
                                        if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_xy_abs_mean = (_frontres_corr_xy_abs_sum / _frontres_shaping_steps
                                         if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_rp_abs_mean = (_frontres_corr_rp_abs_sum / _frontres_shaping_steps
                                         if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_corr_yaw_abs_mean = (_frontres_corr_yaw_abs_sum / _frontres_shaping_steps
                                          if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_smooth_penalty_mean = (_frontres_smooth_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_reg_penalty_mean    = (_frontres_reg_penalty_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)
            frontres_survival_rate       = (1.0 - _frontres_term_count / _frontres_step_count
                                            if _is_frontres and _frontres_step_count > 0 else None)
            frontres_delta_pos_abs_mean  = (_frontres_delta_pos_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_delta_rpy_abs_mean  = (_frontres_delta_rpy_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_delta_z_abs_mean    = (_frontres_delta_z_abs_sum / _frontres_shaping_steps
                                            if _is_frontres and not _is_task_space_mode and _frontres_shaping_steps > 0 else None)
            frontres_jump_degree_mean    = (_frontres_jump_degree_sum / _frontres_shaping_steps
                                            if _is_frontres and _frontres_shaping_steps > 0 else None)

            # Store r_delta mean for next iteration's PI controller update.
            if frontres_rdelta_mean is not None:
                self._last_r_delta_mean = frontres_rdelta_mean

            # DR scale for logging: current value (set by PI controller at top of iteration)
            frontres_dr_scale = _dr_scale if _is_frontres else None

            # Removed: staircase advancement logic (replaced by PI controller above)
            _staircase_level_for_log = None
            _staircase_mult_for_log  = None

            # Phase flags for diagnostics (exposed to log() via locals())
            _supervised_warmup_active = False  # runs before main loop, always False here
            _critic_warmup_active     = _critic_warmup

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

        # -- Losses
        # Keys suppressed when their controlling lambda is 0 (avoids clutter for unused modes).
        _suppress_if_zero = {"bc_off_policy", "bc_teacher", "lambda_off_policy", "lambda_teacher"}
        # Keys reclassified out of Loss/ for clarity.
        _to_frontres  = {"supervised_cos_sim"}      # diagnostic, not a loss
        _to_curriculum = {"lambda_supervised", "ppo_actor_weight"}  # scheduler state, not a loss
        _frontres_diag_keys = {
            "delta_q_norm_mean",
            "delta_q_norm_std",
            "smooth_metric",
            "lambda_reg",
            "grad_cos_ppo_supervised",
            "grad_norm_ratio_ppo_to_supervised",
        }
        _stage1_dz_keys     = {"loss_dz", "dz_pred_abs", "dz_gt_abs", "dz_mae"}
        for key, value in locs["loss_dict"].items():
            if key in _suppress_if_zero and value == 0.0:
                continue
            if self.training_type == "supervise" and key in _stage1_dz_keys:
                self.writer.add_scalar(f"Stage1/DeltaZ/{key}", value, locs["it"])
            elif isinstance(self.alg.policy, FrontRESActorCritic) and key in _frontres_diag_keys:
                self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
            elif key in _to_frontres:
                self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
            elif key in _to_curriculum:
                self.writer.add_scalar(f"Curriculum/{key}", value, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy (not meaningful during supervised learning)
        if self.training_type != "supervise":
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- FrontRES B1 delta-reward + curriculum diagnostics
        if isinstance(self.alg.policy, FrontRESActorCritic):
            _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0

            # B1 delta-reward
            if locs.get("frontres_rdelta_mean") is not None:
                self.writer.add_scalar("FrontRES/r_delta_per_step",
                                       locs["frontres_rdelta_mean"], locs["it"])
            if locs.get("frontres_baseline_mean") is not None:
                self.writer.add_scalar("FrontRES/baseline_per_step",
                                       locs["frontres_baseline_mean"], locs["it"])
            for _name in (
                "r_exec", "r_geom", "r_rescue", "intervention_cost",
                "reward_frontres", "damage_gap", "repair_gain", "repair_ratio",
                "safe_gate", "fragile_gate", "damage_gate", "broken_gate",
                "actor_gate", "exec_gate", "cost_gate",
                "r_z", "r_xy", "r_rp", "r_yaw",
                "dr_z_abs", "dr_xy_abs", "dr_rp_abs", "dr_yaw_abs",
                "corr_z_abs", "corr_xy_abs", "corr_rp_abs", "corr_yaw_abs",
            ):
                _value = locs.get(f"frontres_{_name}_mean")
                if _value is not None:
                    self.writer.add_scalar(f"FrontRES/RewardComponents/{_name}",
                                           _value, locs["it"])

            # Correction magnitude (task-space: split Δpos/Δrpy; joint-space: Δz)
            if _ts_mode:
                if locs.get("frontres_delta_pos_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_pos_abs_mean",
                                           locs["frontres_delta_pos_abs_mean"], locs["it"])
                if locs.get("frontres_delta_rpy_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_rpy_abs_mean",
                                           locs["frontres_delta_rpy_abs_mean"], locs["it"])
            else:
                if locs.get("frontres_delta_z_abs_mean") is not None:
                    self.writer.add_scalar("FrontRES/delta_z_abs_mean",
                                           locs["frontres_delta_z_abs_mean"], locs["it"])

            # Optional shaping penalties (only log when non-zero)
            if locs.get("frontres_smooth_penalty_mean") not in (None, 0.0):
                self.writer.add_scalar("FrontRES/smooth_penalty_per_step",
                                       locs["frontres_smooth_penalty_mean"], locs["it"])
            if locs.get("frontres_reg_penalty_mean") not in (None, 0.0):
                self.writer.add_scalar("FrontRES/reg_penalty_per_step",
                                       locs["frontres_reg_penalty_mean"], locs["it"])

            # Jump-degree gate
            if locs.get("frontres_jump_degree_mean") is not None:
                self.writer.add_scalar("FrontRES/jump_degree_mean",
                                       locs["frontres_jump_degree_mean"], locs["it"])

            # Curriculum / DR schedule
            if locs.get("frontres_dr_scale") is not None:
                self.writer.add_scalar("Curriculum/dr_scale",
                                       locs["frontres_dr_scale"], locs["it"])
            if locs.get("frontres_survival_rate") is not None:
                self.writer.add_scalar("Curriculum/training_survival_rate",
                                       locs["frontres_survival_rate"], locs["it"])
            self.writer.add_scalar("Curriculum/r_delta_ema",
                                   locs.get("_r_delta_ema", 0.0), locs["it"])

            # Δq alpha ramp (joint-space mode only)
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
            # ── Phase indicator ──────────────────────────────────────────────
            _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0
            if _ts_mode:
                _is_warmup = locs.get("_supervised_warmup_active", False)
                _is_critic = locs.get("_critic_warmup_active", False)
                _is_actor_takeover = locs.get("_actor_takeover_active", False)
                _lam = getattr(self.alg, 'lambda_supervised', 0.0)
                if _is_warmup:
                    _phase = "SUPERVISED WARMUP"
                    _notes = "(GMT-only, FrontRES corrections disabled)"
                elif _is_critic:
                    _phase = "CRITIC WARMUP"
                    _paw = locs.get("loss_dict", {}).get("ppo_actor_weight", None)
                    if _paw is not None and _paw <= 0.0:
                        _notes = "(fixed low DR, PPO actor frozen; critic + supervised train)"
                    else:
                        _notes = "(fixed low DR; critic + supervised train)"
                elif _is_actor_takeover:
                    _phase = "ACTOR TAKEOVER"
                    _notes = "(fixed DR; PPO actor weight ramping)"
                elif _lam > 0.5:
                    _phase = "PPO + SUPERVISED ANCHOR"
                    _notes = ""
                elif _lam > 0.15:
                    _phase = "PPO + WEAK SUPERVISION"
                    _notes = ""
                else:
                    _phase = "PPO FINE-TUNING"
                    _notes = ""
            else:
                _phase = "PPO"
                _notes = ""
            _phase_str = f"  PHASE: {_phase}  "
            if _notes:
                _phase_str += f"\n  {_notes}  "

            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n"""
                f"""{_phase_str.center(width, ' ')}\n\n"""
                f"""{'─' * 30} PERFORMANCE {'─' * 30}\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

            # ── FrontRES: r_delta + cos_sim + curriculum (compact) ──────────
            if isinstance(self.alg.policy, FrontRESActorCritic):
                log_string += f"""\n{'─' * 30} FRONTRES {'─' * 31}\n"""
                log_string += f"""{'r_delta (FrontRES):':>{pad}} {statistics.mean(locs['rewbuffer']):.4f}\n"""
                if len(locs.get("rewbuffer_gmt", [])) > 0:
                    log_string += f"""{'reward_GMT (baseline):':>{pad}} {statistics.mean(locs['rewbuffer_gmt']):.4f}\n"""
                if len(locs.get("lenbuffer_gmt", [])) > 0:
                    log_string += f"""{'ep_len_GMT (baseline):':>{pad}} {statistics.mean(locs['lenbuffer_gmt']):.1f}\n"""
                # cos_sim
                _cs = locs.get("loss_dict", {}).get("supervised_cos_sim", None)
                if _cs is not None:
                    log_string += f"""{'supervised_cos_sim:':>{pad}} {_cs:.4f}\n"""
                if locs.get("frontres_delta_pos_abs_mean") is not None:
                    log_string += f"""{'|Δpos|:':>{pad}} {locs['frontres_delta_pos_abs_mean']:.4f} m\n"""
                if locs.get("frontres_delta_rpy_abs_mean") is not None:
                    log_string += f"""{'|Δrpy|:':>{pad}} {locs['frontres_delta_rpy_abs_mean']:.4f} rad\n"""
                if locs.get("frontres_r_z_mean") is not None:
                    log_string += f"""{'r_z/r_xy/r_rp/r_yaw:':>{pad}} """
                    log_string += (
                        f"{locs['frontres_r_z_mean']:+.4f} / "
                        f"{locs['frontres_r_xy_mean']:+.4f} / "
                        f"{locs['frontres_r_rp_mean']:+.4f} / "
                        f"{locs['frontres_r_yaw_mean']:+.4f}\n"
                    )
                if locs.get("frontres_r_exec_mean") is not None:
                    log_string += f"""{'repair/geom/rescue/action_cost:':>{pad}} """
                    log_string += (
                        f"{locs['frontres_r_exec_mean']:+.4f} / "
                        f"{locs['frontres_r_geom_mean']:+.4f} / "
                        f"{locs['frontres_r_rescue_mean']:+.4f} / "
                        f"{locs['frontres_intervention_cost_mean']:+.4f}\n"
                    )
                    if locs.get("frontres_reward_frontres_mean") is not None and locs.get("frontres_baseline_mean") is not None:
                        log_string += f"""{'raw reward FR/GMT:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_reward_frontres_mean']:+.4f} / "
                            f"{locs['frontres_baseline_mean']:+.4f}\n"
                        )
                    if locs.get("frontres_damage_gap_mean") is not None:
                        log_string += f"""{'gap/gain/ratio:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_damage_gap_mean']:+.4f} / "
                            f"{locs['frontres_repair_gain_mean']:+.4f} / "
                            f"{locs['frontres_repair_ratio_mean']:+.4f}\n"
                        )
                        log_string += f"""{'safe/fragile/broken:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_safe_gate_mean']:.3f} / "
                            f"{locs['frontres_fragile_gate_mean']:.3f} / "
                            f"{locs['frontres_broken_gate_mean']:.3f}\n"
                        )
                    if locs.get("frontres_damage_gate_mean") is not None:
                        log_string += f"""{'damage/broken/actor gate:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_damage_gate_mean']:.3f} / "
                            f"{locs['frontres_broken_gate_mean']:.3f} / "
                            f"{locs['frontres_actor_gate_mean']:.3f}\n"
                        )
                        log_string += f"""{'exec/cost gate:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_exec_gate_mean']:.3f} / "
                            f"{locs['frontres_cost_gate_mean']:.3f}\n"
                        )
                _gc = locs.get("loss_dict", {}).get("grad_cos_ppo_supervised", None)
                if _gc is not None:
                    _gr = locs.get("loss_dict", {}).get("grad_norm_ratio_ppo_to_supervised", 0.0)
                    log_string += f"""{'grad cos PPO/Sup:':>{pad}} {_gc:+.4f} (norm ratio={_gr:.3f})\n"""
                # Curriculum state
                if locs.get("frontres_dr_scale") is not None:
                    log_string += f"""{'DR scale:':>{pad}} {locs['frontres_dr_scale']:.4f}\n"""
                _rd_ema = locs.get("_r_delta_ema", 0.0)
                log_string += f"""{'r_delta EMA:':>{pad}} {_rd_ema:.4f}\n"""
                if locs.get("frontres_survival_rate") is not None:
                    log_string += f"""{'survival rate:':>{pad}} {locs['frontres_survival_rate']:.3f}\n"""
                _lam = locs.get("loss_dict", {}).get("lambda_supervised", None)
                if _lam is not None:
                    log_string += f"""{'λ_supervised:':>{pad}} {_lam:.3f}\n"""
                _paw = locs.get("loss_dict", {}).get("ppo_actor_weight", None)
                if _paw is not None:
                    log_string += f"""{'PPO actor weight:':>{pad}} {_paw:.3f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")

            if self.training_type == "supervise":
                log_string += f"""{'─' * 30} STAGE 1 {'─' * 33}\n"""
                if "behavior" in locs["loss_dict"]:
                    log_string += f"""{'behavior loss:':>{pad}} {locs['loss_dict']['behavior']:.4f}\n"""
                if len(locs["lenbuffer"]) > 0:
                    log_string += f"""{'ep length:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
            else:
                log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                for key, value in locs["loss_dict"].items():
                    log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # Episode_Reward / Metrics / Terminations → wandb only, not console
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
        if hasattr(self, '_dr_prev_error'):
            saved_dict["dr_prev_error"] = self._dr_prev_error
        if hasattr(self, '_frontres_warmup_complete'):
            saved_dict["frontres_warmup_complete"] = bool(self._frontres_warmup_complete)
        
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
        self._frontres_warmup_complete = bool(loaded_dict.get("frontres_warmup_complete", False))
        if self._frontres_warmup_complete:
            print("[Runner] Checkpoint marks FrontRES supervised warmup as complete.")

        # ── 断点续训模式控制 ────────────────────────────────────────────────────────
        # is_full_resume=True  (Stage2→Stage2 断点续训): 恢复优化器矩估计+学习率, 保留 std
        # is_full_resume=False (Stage1→Stage2 权重迁移): 仅权重, 重置优化器和 std.
        # Joint-warmup checkpoints are a special case: their critic has already
        # learned E(s)=R_clean-R_noisy and should be transferred into RL.
        # load_optimizer 参数仍可从外部显式覆盖（例如强制跳过优化器加载）。
        is_full_resume: bool = self.cfg.get('is_full_resume', True)
        if not is_full_resume:
            load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
            load_critic = self._frontres_warmup_complete
        print(f"[Runner] is_full_resume={is_full_resume} → "
              f"load_optimizer={load_optimizer}, load_critic={load_critic}, "
              f"reset_noise_std={not is_full_resume}")

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
                    # Task-space FrontRES: anchor-error dims [:num_extra] are not
                    # covered by the GMT normalizer.  Restore Stage-1 empirical stats
                    # for those dims when the checkpoint actually contains them.
                    # so Stage 2 sees the same normalized scale that Stage 1 trained on.
                    _s1_sd   = loaded_dict["obs_norm_state_dict"]
                    _s1_mean = _s1_sd.get("_mean", None)  # shape (1, 800)
                    _s1_std  = _s1_sd.get("_std",  None)  # shape (1, 800)
                    if _s1_mean is not None and _s1_std is not None:
                        gmt_dim = self._frontres_gmt_obs_dim
                        obs_dim = int(getattr(self.alg.policy, "num_actor_obs", gmt_dim))
                        num_extra = max(0, obs_dim - gmt_dim)
                        if num_extra > 0 and _s1_mean.shape[-1] >= obs_dim and _s1_std.shape[-1] >= obs_dim:
                            self._frontres_extra_mean = _s1_mean[:, :num_extra].to(self.device)
                            self._frontres_extra_std  = _s1_std[:,  :num_extra].to(self.device)
                            print(f"[Runner] Loaded Stage-1 anchor-error normalizer stats "
                                  f"(dims 0–{num_extra}) for FrontRES task-space.")
                        else:
                            self._frontres_extra_mean = None
                            self._frontres_extra_std = None
                            print("[Runner] Stage-1 checkpoint has no compatible anchor-error "
                                  "normalizer stats; FrontRES extra dims pass through unnormalized.")

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
            if is_full_resume:
                self.current_learning_iteration = loaded_dict["iter"]
            else:
                self.current_learning_iteration = 0
                print("[Runner] Stage1→Stage2 cold-start: current_learning_iteration reset to 0.")

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
        #   改用 cfg 中的 dr_scale_init（默认 1.0），确保 Stage2 从 Stage1 训练强度出发，
        #   避免 dr_scale=0 时 Stage1 修正策略作用于干净参考导致的即时崩溃。
        if is_full_resume:
            self._dr_scale      = loaded_dict.get("dr_scale",      0.0)
            self._dr_prev_error = loaded_dict.get("dr_prev_error", 0.0)
            print(f"[Runner] Adaptive DR scale restored from checkpoint: {self._dr_scale:.4f}")
        else:
            _dr_init = float(self.cfg.get("dr_scale_init", 1.0))
            self._dr_scale = _dr_init
            print(f"[Runner] Stage1→Stage2 cold-start: dr_scale initialised to "
                  f"dr_scale_init={_dr_init:.4f} (ignoring checkpoint value "
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

        IsaacLab places Optional obs terms (anchor_root_pos_error_w, anchor_root_rpy_error_w)
        BEFORE regular terms in the concatenated obs tensor, so the layout is:
          [0 : num_extra]           = anchor-error dims  (FrontRES-only, NOT in GMT training)
          [num_extra : num_extra+gmt_dim] = GMT-compatible dims (match GMT training obs exactly)

        where num_extra = obs_dim - gmt_dim  (= 30 = 6 dims/frame × 5 frames).

        We therefore normalize the LAST gmt_dim dims with the frozen GMT normalizer and
        optionally normalize the FIRST num_extra dims with Stage-1 empirical stats.
        Output shape is unchanged (800 dims); structure: [extra | gmt_part].
        """
        if self._frontres_gmt_obs_dim is not None and obs.shape[-1] > self._frontres_gmt_obs_dim:
            gmt_dim   = self._frontres_gmt_obs_dim
            num_extra = obs.shape[-1] - gmt_dim          # = 30 (anchor errors at front)
            extra     = obs[:, :num_extra]               # [0:30]   anchor errors
            gmt_part  = self.obs_normalizer(obs[:, num_extra:])  # [30:800] GMT-compatible → normalize
            _s1_mean = getattr(self, '_frontres_extra_mean', None)
            _s1_std  = getattr(self, '_frontres_extra_std',  None)
            if (_s1_mean is not None and _s1_std is not None
                    and _s1_mean.shape[-1] == num_extra
                    and _s1_std.shape[-1] == num_extra):
                extra = (extra - _s1_mean) / (_s1_std + 1e-8)
            return torch.cat([extra, gmt_part], dim=-1)  # [anchor_errors | normalized_gmt]
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
