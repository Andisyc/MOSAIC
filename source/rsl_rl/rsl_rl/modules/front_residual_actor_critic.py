# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Residual Learning Policy for Physical-Aware Motion Refiner

Implements ResMimic-style residual learning where:
- Front-End Residual network (trainable) provides Δq
- Final Ref Motion: q_final = q_origin + Δq
- GMT policy (frozen) provides base actions

This allows efficient task-specific refinement on top of a general motion tracking policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class ComposedActor(nn.Module):
    """
    Composed actor for ONNX export: combines frozen GMT + trainable FrontRES.

    FrontRES (pre-GMT residual) pipeline:
        obs → FrontRES → Δq
        q_ref_corrected = q_ref + Δq   (modify q_ref inside obs)
        obs_modified → GMT → actions
    """
    def __init__(self, gmt_policy: ActorCritic, residual_actor: nn.Module,
                 gmt_actor_input_dim: int, num_actor_obs: int,
                 q_ref_start_idx: int, num_actions: int):
        super().__init__()
        self.gmt_policy = gmt_policy
        self.residual_actor = residual_actor
        self.gmt_actor_input_dim = gmt_actor_input_dim
        self.num_actor_obs = num_actor_obs
        self.q_ref_start_idx = q_ref_start_idx
        self.num_actions = num_actions

    def forward(self, observations):
        """FrontRES pre-GMT pipeline: correct q_ref, then run GMT."""
        obs_dim = observations.shape[-1]

        if obs_dim == self.num_actor_obs:
            policy_obs = observations
        elif obs_dim == self.gmt_actor_input_dim:
            policy_obs = observations[:, :self.num_actor_obs]
        else:
            raise ValueError(
                f"Unexpected observation dimension: {obs_dim}. "
                f"Expected {self.num_actor_obs} or {self.gmt_actor_input_dim}")

        # 1. FrontRES computes Δq from policy observations
        delta_q = self.residual_actor(policy_obs)

        # 2. Apply Δq to q_ref inside the observation vector
        obs_modified = policy_obs.clone()
        q_ref_end_idx = self.q_ref_start_idx + self.num_actions
        obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] = (
            obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] + delta_q
        )

        # 3. Pad to GMT input dim if needed (e.g. ref_vel suffix)
        if obs_modified.shape[-1] < self.gmt_actor_input_dim:
            padding_size = self.gmt_actor_input_dim - obs_modified.shape[-1]
            padding = torch.zeros(
                *obs_modified.shape[:-1], padding_size,
                device=obs_modified.device,
                dtype=obs_modified.dtype)
            gmt_obs = torch.cat([obs_modified, padding], dim=-1)
        elif obs_dim == self.gmt_actor_input_dim:
            # Restore ref_vel suffix from original input
            ref_vel = observations[:, self.num_actor_obs:]
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        else:
            gmt_obs = obs_modified

        # 4. GMT forward (frozen)
        with torch.no_grad():
            actions = self.gmt_policy.act_inference(gmt_obs)

        return actions

    def __getitem__(self, idx):
        """Support subscript access for ONNX exporter (e.g., actor[0].in_features)"""
        return self.residual_actor[idx]


class FrontRESActorCritic(nn.Module):
    """
    Residual learning policy: frozen GMT + trainable residual network.

    Components:
    - residual_actor: Trainable residual network
    - critic: Trainable value function
    - gmt_policy: Frozen teacher policy (loaded from checkpoint)
    - gmt_normalizer: Frozen observation normalizer from GMT checkpoint

    Final Ref Motion: q_final = q_origin + Δq
    """

    is_recurrent = False
    is_encoding = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        # Residual network configuration
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,
        # GMT configuration
        q_ref_start_idx=0, # Added: Index where q_ref begins in the observation vector
        gmt_checkpoint_path=None,
        gmt_policy_cfg=None,  # Optional: specify GMT architecture (auto-inferred if None)
        # Ref vel estimator configuration
        num_ref_vel_estimator_obs=None,  # Dimension of ref_vel_estimator observations (e.g., 305)
        ref_vel_estimator_checkpoint_path=None,  # Path to estimator checkpoint
        ref_vel_estimator_type="mlp",  # Type of estimator: "mlp" or "transformer"
        # Critic configuration
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt: bool = False,
        # Standard ActorCritic parameters
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ResidualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if gmt_checkpoint_path is None:
            raise ValueError("gmt_checkpoint_path is required for ResidualActorCritic")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.q_ref_start_idx = q_ref_start_idx
        self.noise_std_type = noise_std_type

        activation_fn = resolve_nn_activation(activation)

        # ========== Load GMT Policy ==========

        print(f"[ResidualActorCritic] Loading GMT policy from: {gmt_checkpoint_path}")
        checkpoint = torch.load(gmt_checkpoint_path, map_location="cpu", weights_only=False) # 导入checkpoint

        # Infer GMT architecture from checkpoint
        state_dict = checkpoint["model_state_dict"] # 导入权重

        # Detect checkpoint format: standard or ref_vel skip connection
        has_skip_connection = "actor.actor_layer1.weight" in state_dict # 确定有跳连接的布尔变量

        if has_skip_connection:
            # Skip connection format: actor.actor_layer1, actor.actor_remaining.X
            print(f"[ResidualActorCritic] Detected ref_vel skip connection format in GMT checkpoint")

            # IMPORTANT: Layer1 input dimension tells us the ACTUAL policy_obs_dim used during training
            # This might differ from expected due to bugs or different configurations
            layer1_input_dim = state_dict["actor.actor_layer1.weight"].shape[1] # 设置第一层的输入维度 (观测向量的维度)
            gmt_critic_input_dim = state_dict["critic.0.weight"].shape[1] # 设置GMT的Critic第一层输入维度

            # Infer ref_vel_dim from the second layer input size difference 通过两层维度反推参考速度向量维度
            layer1_output = state_dict["actor.actor_layer1.weight"].shape[0] # 设置第一层的输出维度
            remaining_0_input = state_dict["actor.actor_remaining.0.weight"].shape[1] # 设置第二层输入维度

            # 参考速度维度是第二层输入维度减去第一层输出维度 (第二层输入时拼接了参考速度向量)
            # 在第一层结束才输入ref_vel是因为观测量太高维, 直接输入会导致信息淹没, 但ref_vel
            # 相比其他信息更重要 (代表了往哪走), 因此将第一层作为Encoder, 压缩信息后才进行拼接
            ref_vel_dim = remaining_0_input - layer1_output

            # Calculate gmt_actor_input_dim: layer1_input + ref_vel_dim
            # This is the total observation dimension expected by GMT policy
            gmt_actor_input_dim = layer1_input_dim + ref_vel_dim # 在IsaacLab中注册时需要总观测维度

            # Find the last actor layer in actor_remaining 使用remaining的命名方式是因为ref_vel在第二层才输入, 导致梯度截断, 因此特意申明
            # 寻找最后一层维度 (输出动作维度), 如果无法找到, 就使用输入层的维度
            actor_remaining_keys = [k for k in state_dict.keys() if k.startswith("actor.actor_remaining.") and ".weight" in k]
            if actor_remaining_keys:
                last_actor_key = max(actor_remaining_keys, key=lambda k: int(k.split(".")[2]))
                gmt_num_actions = state_dict[last_actor_key].shape[0]
            else:
                # Fallback: use actor_layer1 output as action dim (shouldn't happen)
                gmt_num_actions = state_dict["actor.actor_layer1.weight"].shape[0]
        else:
            # Standard format: actor.0, actor.2, ...
            gmt_actor_input_dim = state_dict["actor.0.weight"].shape[1]
            gmt_critic_input_dim = state_dict["critic.0.weight"].shape[1]

            # Find the last actor layer
            actor_keys = [k for k in state_dict.keys() if k.startswith("actor.") and ".weight" in k]
            last_actor_key = max(actor_keys, key=lambda k: int(k.split(".")[1]))
            gmt_num_actions = state_dict[last_actor_key].shape[0]

        if gmt_num_actions != num_actions:
            raise ValueError(
                f"GMT action dimension ({gmt_num_actions}) does not match "
                f"specified num_actions ({num_actions})")

        print(f"[ResidualActorCritic] GMT architecture: "
              f"actor_input={gmt_actor_input_dim}, "
              f"critic_input={gmt_critic_input_dim}, "
              f"actions={gmt_num_actions}")

        # Create GMT policy with correct dimensions
        if gmt_policy_cfg is None:
            # Auto-infer architecture from checkpoint
            gmt_policy_cfg = self._infer_gmt_architecture(state_dict, activation)

            # If skip connection format detected, add skip connection config
            if has_skip_connection:
                print("[ResidualActorCritic] GMT uses ref_vel skip connection, creating matching architecture")
                print(f"[ResidualActorCritic] Inferred ref_vel_dim={ref_vel_dim}")
                print(f"[ResidualActorCritic] Layer1 accepts {layer1_input_dim} dims (policy_obs)")
                print(f"[ResidualActorCritic] Setting gmt_actor_input_dim={gmt_actor_input_dim} (layer1_input + ref_vel_dim)")

                gmt_policy_cfg["ref_vel_skip_first_layer"] = True
                gmt_policy_cfg["ref_vel_dim"] = ref_vel_dim

        # GMT实例化: 依据架构创建实例
        self.gmt_policy = ActorCritic(
            num_actor_obs=gmt_actor_input_dim,
            num_critic_obs=gmt_critic_input_dim,
            num_actions=gmt_num_actions,
            **gmt_policy_cfg)

        # Load GMT weights directly (no conversion needed if architectures match)
        self.gmt_policy.load_state_dict(state_dict) # 导入GMT权重

        # Freeze GMT completely
        self.gmt_policy.eval() # GMT设为eval模式
        for param in self.gmt_policy.parameters(): # 梯度置为False
            param.requires_grad = False
        print("[ResidualActorCritic] GMT policy frozen (all parameters require_grad=False)")

        # Load GMT's observation normalizer (critical!)
        self.gmt_normalizer = None # 导入GMT观测量归一器
        if "obs_norm_state_dict" in checkpoint:
            # Infer normalizer dimension from checkpoint (usually policy_obs_dim, not gmt_actor_input_dim)
            # This is because normalizer operates on policy_obs before ref_vel is concatenated
            obs_norm_state = checkpoint["obs_norm_state_dict"]
            normalizer_dim = obs_norm_state["_mean"].shape[1]

            self.gmt_normalizer = EmpiricalNormalization(
                shape=[normalizer_dim], until=1.0e8)
            self.gmt_normalizer.load_state_dict(obs_norm_state)
            self.gmt_normalizer.eval()
            self.gmt_normalizer.until = 0  # Freeze statistics
            print(f"[ResidualActorCritic] GMT observation normalizer loaded (dim={normalizer_dim}) and frozen")
        else:
            print("[ResidualActorCritic] WARNING: No observation normalizer found in GMT checkpoint!")

        # ========== Load Ref Vel Estimator ==========

        self.ref_vel_estimator = None # 导入速度归一器
        self.num_ref_vel_estimator_obs = num_ref_vel_estimator_obs

        if ref_vel_estimator_checkpoint_path is not None:
            if num_ref_vel_estimator_obs is None:
                raise ValueError("num_ref_vel_estimator_obs must be provided when ref_vel_estimator_checkpoint_path is specified")

            print(f"[ResidualActorCritic] Loading ref_vel estimator from: {ref_vel_estimator_checkpoint_path}")
            print(f"[ResidualActorCritic] Estimator type: {ref_vel_estimator_type}")

            # Load estimator based on type
            if ref_vel_estimator_type == "mlp":
                from rsl_rl.modules import VelocityEstimator
                self.ref_vel_estimator = VelocityEstimator.load(
                    ref_vel_estimator_checkpoint_path,
                    device=str(next(self.gmt_policy.parameters()).device))
            elif ref_vel_estimator_type == "transformer":
                from rsl_rl.modules import VelocityEstimatorTransformer
                estimator_checkpoint = torch.load(
                    ref_vel_estimator_checkpoint_path,
                    map_location=str(next(self.gmt_policy.parameters()).device),
                    weights_only=False)
                self.ref_vel_estimator = VelocityEstimatorTransformer(
                    feature_dim=estimator_checkpoint.get('feature_dim', 61),
                    history_length=estimator_checkpoint.get('history_length', 5),
                    d_model=estimator_checkpoint.get('d_model', 128),
                    nhead=estimator_checkpoint.get('nhead', 4),
                    num_layers=estimator_checkpoint.get('num_layers', 2),)
                self.ref_vel_estimator.load_state_dict(estimator_checkpoint['model_state_dict'])
                self.ref_vel_estimator = self.ref_vel_estimator.to(next(self.gmt_policy.parameters()).device)
                print(f"[ResidualActorCritic] Transformer estimator loaded successfully")
            else:
                raise ValueError(f"Unknown ref_vel_estimator_type: {ref_vel_estimator_type}. Must be 'mlp' or 'transformer'")

            # Freeze estimator
            self.ref_vel_estimator.eval()
            for param in self.ref_vel_estimator.parameters():
                param.requires_grad = False
            print("[ResidualActorCritic] Ref vel estimator loaded and frozen")
        else:
            print("[ResidualActorCritic] WARNING: No ref_vel estimator provided, will use zero padding for GMT policy")
        
        # ========== Build Fron-End Residual Network ==========
        
        # 这就是我们在第一阶段使用监督学习训练的 FrontRES (student)
        # 它输出的是 Δq
        self.residual_actor = self._build_residual_actor(
            input_dim=num_actor_obs, 
            output_dim=num_actions,
            hidden_dims=residual_hidden_dims,
            activation=activation_fn,
            last_layer_gain=residual_last_layer_gain)
        print(f"[FrontEndResidualActorCritic] FrontRES network: {self.residual_actor}")

        # ========== Build Critic ==========

        critic_layers: list[nn.Module] = []
        prev_dim = num_critic_obs
        
        # 创建critic网络架构
        if critic_hidden_dims:
            # 创建首层: 线性层+激活层
            critic_layers.append(nn.Linear(prev_dim, critic_hidden_dims[0]))
            critic_layers.append(activation_fn)

            # 创建隐藏层
            for layer_index in range(len(critic_hidden_dims)):
                # 取出每层的输入维度
                in_dim = critic_hidden_dims[layer_index]

                # 达到最后一层时将输出维度设置为1, 因为只需要输出评分
                if layer_index == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(in_dim, 1))
                else: # 取出每层的输出维度并创建层级: 线性层+激活层
                    out_dim = critic_hidden_dims[layer_index + 1]
                    critic_layers.append(nn.Linear(in_dim, out_dim))
                    critic_layers.append(activation_fn)
        else:
            critic_layers.append(nn.Linear(prev_dim, 1))

        # List 2 Sequence实例化Critic网络
        self.critic = nn.Sequential(*critic_layers)
        print(f"[ResidualActorCritic] Critic MLP: {self.critic}")

        if init_critic_from_gmt: # 导入GMT的Critic权重作为RES的Critic
            self._load_critic_from_checkpoint(state_dict, num_critic_obs)

        # ========== Action Noise ==========

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # ========== Create Composed Actor for ONNX Export ==========

        # ONNX exporter expects policy.actor attribute
        # Create a wrapper module that composes GMT + residual
        self.actor = ComposedActor(
            self.gmt_policy,
            self.residual_actor,
            gmt_actor_input_dim,
            num_actor_obs,
            q_ref_start_idx,
            num_actions)
        print("[ResidualActorCritic] Created composed actor for ONNX export")

        # ========== Store GMT input dimension for padding ==========

        # If GMT expects more observations than provided, we'll need to pad
        self.gmt_actor_input_dim = gmt_actor_input_dim
        self.num_actor_obs = num_actor_obs

        if gmt_actor_input_dim != num_actor_obs:
            print(f"[ResidualActorCritic] WARNING: GMT expects {gmt_actor_input_dim} observations, "
                  f"but environment provides {num_actor_obs}. Will pad with zeros during inference.")

    def _infer_gmt_architecture(self, state_dict, activation):
        """Infer GMT policy architecture from checkpoint state_dict"""
        # Extract actor hidden dimensions
        actor_hidden_dims = []
        actor_keys = sorted([k for k in state_dict.keys() if k.startswith("actor.") and ".weight" in k])
        for i in range(len(actor_keys) - 1):  # Exclude last layer
            key = actor_keys[i]
            out_dim = state_dict[key].shape[0]
            actor_hidden_dims.append(out_dim)

        # Extract critic hidden dimensions
        critic_hidden_dims = []
        critic_keys = sorted([k for k in state_dict.keys() if k.startswith("critic.") and ".weight" in k])
        for i in range(len(critic_keys) - 1):  # Exclude last layer
            key = critic_keys[i]
            out_dim = state_dict[key].shape[0]
            critic_hidden_dims.append(out_dim)

        # Get noise std type and value
        if "std" in state_dict:
            noise_std_type = "scalar"
            init_noise_std = state_dict["std"][0].item()
        elif "log_std" in state_dict:
            noise_std_type = "log"
            init_noise_std = torch.exp(state_dict["log_std"][0]).item()
        else:
            noise_std_type = "scalar"
            init_noise_std = 1.0

        return {
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "activation": activation,
            "init_noise_std": init_noise_std,
            "noise_std_type": noise_std_type,
        }

    def _load_critic_from_checkpoint(self, checkpoint_state_dict, expected_input_dim):
        """Load critic weights from a checkpoint state_dict into the residual critic."""
        critic_state_dict = {
            k.replace("critic.", ""): v
            for k, v in checkpoint_state_dict.items()
            if k.startswith("critic.")
        }
        if not critic_state_dict:
            raise ValueError("No critic weights found in GMT checkpoint state_dict.")

        if "0.weight" not in critic_state_dict:
            raise ValueError("GMT critic state_dict missing first layer weights (critic.0.weight).")

        checkpoint_input_dim = critic_state_dict["0.weight"].shape[1]
        if checkpoint_input_dim != expected_input_dim:
            raise ValueError(
                "GMT critic input dim does not match residual critic input dim "
                f"({checkpoint_input_dim} != {expected_input_dim})."
            )

        try:
            self.critic.load_state_dict(critic_state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load GMT critic weights into residual critic. "
                "Check that critic_hidden_dims and input dims match the checkpoint."
            ) from exc

        print("[ResidualActorCritic] Critic weights loaded from GMT checkpoint")

    def _build_residual_actor(self, input_dim, output_dim, hidden_dims, activation, last_layer_gain):
        """
        Build residual network with small-gain Xavier initialization on last layer.

        The small gain (e.g., 0.01) ensures residual starts near zero:
        - Initial behavior: a_final ≈ a_gmt (GMT policy dominates)
        - Gradual learning: residual slowly learns corrections
        - Stable training: avoids large initial perturbations
        """
        layers = []
        prev_dim = input_dim

        # Hidden layers: standard Xavier init
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(activation)
            prev_dim = hidden_dim

        # Last layer: small gain Xavier (0.01) for near-zero initialization
        last_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(last_layer.weight, gain=last_layer_gain)
        nn.init.zeros_(last_layer.bias)
        layers.append(last_layer)

        # Verify initialization
        with torch.no_grad():
            weight_norm = torch.norm(last_layer.weight).item()
            print(f"[ResidualActorCritic] Residual last layer weight norm: {weight_norm:.6f} "
                  f"(gain={last_layer_gain})")

        return nn.Sequential(*layers)

    def _pad_observations_for_gmt(self, observations):
        """
        Pad observations if GMT policy expects more dimensions than provided.

        This handles the case where the GMT checkpoint was created with a different
        observation dimension than the current environment provides.
        """
        if observations.shape[-1] < self.gmt_actor_input_dim:
            # Pad with zeros to match GMT's expected input dimension
            padding_size = self.gmt_actor_input_dim - observations.shape[-1]
            padding = torch.zeros(
                *observations.shape[:-1], padding_size,
                device=observations.device,
                dtype=observations.dtype
            )
            observations = torch.cat([observations, padding], dim=-1)

        return observations

    def reset(self, dones=None):
        """Reset policy state (for recurrent policies)"""
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _parse_observations(self, observations):
        """
        Parse raw observation tensor (or dict) into (policy_obs, ref_vel).

        Returns:
            policy_obs (Tensor): observations for residual_actor  [N, num_actor_obs]
            ref_vel    (Tensor | None): reference velocity suffix  [N, ref_vel_dim] or None
            ref_vel_estimator_obs (Tensor | None): observations for vel estimator
        """
        if isinstance(observations, dict):
            policy_obs            = observations["policy"]
            ref_vel               = None
            ref_vel_estimator_obs = observations.get("ref_vel_estimator", None)
        elif isinstance(observations, torch.Tensor):
            obs_dim = observations.shape[-1]
            if obs_dim == self.num_actor_obs:
                policy_obs            = observations
                ref_vel               = None
                ref_vel_estimator_obs = None
            elif obs_dim == self.gmt_actor_input_dim:
                ref_vel_dim           = self.gmt_actor_input_dim - self.num_actor_obs
                policy_obs            = observations[:, :-ref_vel_dim]
                ref_vel               = observations[:, -ref_vel_dim:]
                ref_vel_estimator_obs = None
            else:
                raise ValueError(
                    f"Unexpected observation dimension: {obs_dim}. "
                    f"Expected {self.num_actor_obs} (policy_obs) or "
                    f"{self.gmt_actor_input_dim} (policy_obs+ref_vel)"
                )
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")
        return policy_obs, ref_vel, ref_vel_estimator_obs

    def _apply_delta_q_and_run_gmt(self, policy_obs, delta_q, ref_vel, ref_vel_estimator_obs):
        """
        Apply Δq to q_ref, build GMT observation, run frozen GMT.

        Always called inside torch.no_grad() – no gradient needed here because
        FrontRES distributes over Δq directly (see update_distribution).

        Returns:
            robot_actions (Tensor): motor commands output by GMT  [N, num_actions]
        """
        # Correct q_ref inside the observation vector
        obs_modified = policy_obs.clone()
        q_ref_end_idx = self.q_ref_start_idx + self.num_actions
        obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] = (
            obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] + delta_q
        )

        # Build GMT observation (add ref_vel suffix if available)
        if ref_vel is not None:
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        elif self.ref_vel_estimator is not None and ref_vel_estimator_obs is not None:
            ref_vel = self.ref_vel_estimator(ref_vel_estimator_obs)
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        else:
            gmt_obs = self._pad_observations_for_gmt(obs_modified)

        return self.gmt_policy.act_inference(gmt_obs)

    def _frontres_forward(self, observations):
        """
        Full FrontRES → GMT pipeline used only by act_inference (deployment / evaluation).

        For RL training, update_distribution + get_env_action is used instead, so that
        the policy distribution is defined over Δq-space and gradient never needs to
        flow through the frozen GMT network.

        Returns:
            robot_actions (Tensor): final motor commands from GMT after q_ref correction
            delta_q       (Tensor): Δq output by FrontRES (for diagnostic logging)
        """
        policy_obs, ref_vel, ref_vel_estimator_obs = self._parse_observations(observations)

        delta_q = self.residual_actor(policy_obs)

        with torch.no_grad():
            robot_actions = self._apply_delta_q_and_run_gmt(
                policy_obs, delta_q, ref_vel, ref_vel_estimator_obs)

        return robot_actions, delta_q

    def update_distribution(self, observations):
        """
        Define FrontRES action distribution over Δq-space for PPO training.

        Design rationale
        ----------------
        Treating Δq as the "action" of FrontRES (with frozen GMT + robot as the
        black-box environment) is the correct policy-gradient formulation:

            ∇J ∝ E[ A · ∇_θ log π_θ(Δq | obs) ]

        The gradient ∂ log π / ∂θ flows only through residual_actor(obs) → Δq_mean,
        with NO need to backpropagate through the frozen GMT network.

        The runner calls get_env_action() separately to map Δq_sample → robot_actions
        for env.step(). The rollout buffer stores Δq_sample for log_prob computation.
        """
        policy_obs, _, _ = self._parse_observations(observations)

        # Cache full observations so get_env_action can access ref_vel if present
        self._cached_observations = observations

        # FrontRES computes Δq mean (gradient flows through here during PPO update)
        delta_q_mean = self.residual_actor(policy_obs)

        # Store for logging / regularization access by MOSAIC algorithm
        self.last_residual_actions = delta_q_mean.detach()

        # Build distribution over Δq-space
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(delta_q_mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(delta_q_mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = Normal(delta_q_mean, std)

    def act(self, observations, **kwargs):
        """
        Sample Δq from FrontRES distribution (for rollout data collection).

        Returns Δq_sample – NOT robot motor actions.
        The runner converts Δq_sample → robot_actions via get_env_action().
        The rollout buffer stores Δq_sample so that PPO can compute log π(Δq|obs).
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_env_action(self, observations, delta_q_sample: torch.Tensor) -> torch.Tensor:
        """
        Convert FrontRES Δq sample to robot motor actions for env.step().

        Called by the runner immediately after policy.act():
            delta_q_sample = policy.act(obs)           # stored in rollout buffer
            robot_actions  = policy.get_env_action(obs, delta_q_sample)
            env.step(robot_actions)

        Uses cached observations from the preceding act() call so that ref_vel
        (possibly appended by the velocity estimator inside MOSAIC.act()) is
        correctly forwarded to GMT even when the runner only passes raw obs here.
        """
        # Prefer the cached obs from act() which may include ref_vel suffix
        obs = getattr(self, '_cached_observations', observations)
        policy_obs, ref_vel, ref_vel_estimator_obs = self._parse_observations(obs)

        with torch.no_grad():
            robot_actions = self._apply_delta_q_and_run_gmt(
                policy_obs, delta_q_sample, ref_vel, ref_vel_estimator_obs)

        return robot_actions

    def get_actions_log_prob(self, actions):
        """
        Log-probability of Δq samples under the current distribution.

        `actions` here are Δq_sample values stored during rollout (NOT robot actions).
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Deterministic robot actions for evaluation / deployment (uses mean Δq)."""
        actions, _ = self._frontres_forward(observations)
        return actions

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function"""
        value = self.critic(critic_observations)
        return value
