"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import torch
import argparse
import sys
import copy
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--skip_critic", action="store_true", default=False, help="Only load actor weights.")
parser.add_argument("--disable_motion_group_sampling", action="store_true", default=False, help="Disable motion group sampling ratios (use uniform sampling).")
parser.add_argument("--start_frame", type=int, default=10, help="Start frame index (0-based) for motion playback.")
parser.add_argument("--enable_motion_randomization", action="store_true", default=False, help="Keep motion randomization ranges (pose/velocity/joint) instead of zeroing them.")
parser.add_argument("--disable_obs_noise", action="store_true", default=True, help="Disable observation corruption/noise during playback.")
parser.add_argument("--disable_events", action="store_true", default=True, help="Disable event manager randomizations during playback.")

parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")

parser.add_argument("--num_envs", 
                    type=int, 
                    default=1, 
                    help="Number of environments to simulate."
                    )

parser.add_argument("--task",
                    type=str,
                    default="FrontRES-RLFinetune-Tracking-Flat-G1-v0",  # Stage 2 FrontRES task
                    help="Name of the task."
                    )

parser.add_argument("--motion", 
                    type=str, 
                    default="./q_npz/01_01_poses.npz", 
                    help="Path to the motion file."
                    )

from pathlib import Path

# Stage 2 FrontRES checkpoint (logs dir, populated after training completes).
# Falls back to GMT path so the script can also be used for GMT-only playback
# by passing --task Tracking-Flat-G1-Wo-State-Estimation-v0 on the CLI.
_s2_1 = Path("/home/chengyuxuan/MOSAIC/logs/rsl_rl/g1_flat_frontres_finetune")
_s2_2 = Path("/home/yuxuancheng/MOSAIC/logs/rsl_rl/g1_flat_frontres_finetune")
_gmt1 = Path("/home/chengyuxuan/MOSAIC/model/model_27000.pt")
_gmt2 = Path("/home/yuxuancheng/MOSAIC/model/model_27000.pt")

# Prefer the most recent Stage 2 checkpoint if the log dir exists
def _latest_ckpt(log_dir: Path):
    if not log_dir.exists():
        return None
    runs = sorted(log_dir.iterdir(), reverse=True)
    for run in runs:
        ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            return ckpts[-1]
    return None

_s2_ckpt = _latest_ckpt(_s2_1) or _latest_ckpt(_s2_2)
model_path = _s2_ckpt or (_gmt1 if _gmt1.exists() else (_gmt2 if _gmt2.exists() else None))

parser.add_argument("--resume_path",
                    type=str,
                    default=model_path,
                    help="Path to the checkpoint (.pt) to load."
                    )

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import onnxruntime as ort

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    env_cfg.commands.motion.motion_file = args_cli.motion

    if args_cli.wandb_path:
        # import wandb

        # run_path = args_cli.wandb_path

        # api = wandb.Api()
        # if "model" in args_cli.wandb_path:
        #     run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        # wandb_run = api.run(run_path)

        # # loop over files in the run
        # files = [file.name for file in wandb_run.files() if "model" in file.name]

        # # files are all model_xxx.pt find the largest filename
        # if "model" in args_cli.wandb_path:
        #     file = args_cli.wandb_path.split("/")[-1]
        # else:
        #     file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        # wandb_file = wandb_run.file(str(file))
        # wandb_file.download("./logs/rsl_rl/temp", replace=True)

        # print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        # resume_path = f"./logs/rsl_rl/temp/{file}"

        print(f"[INFO]: Loading model checkpoint from: {args_cli.resume_path}")
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {args_cli.resume_path}")

    # Load policy configuration from checkpoint's params/agent.yaml to ensure compatibility
    # This overrides the default configuration with the checkpoint's actual configuration
    checkpoint_dir = os.path.dirname(args_cli.resume_path)
    params_yaml_path = os.path.join(checkpoint_dir, "params", "agent.yaml")
    if os.path.exists(params_yaml_path):
        import yaml
        with open(params_yaml_path, 'r') as f:
            checkpoint_cfg = yaml.safe_load(f)

        # Override policy configuration from checkpoint
        if 'policy' in checkpoint_cfg:
            policy_cfg = checkpoint_cfg['policy']
            if 'ref_vel_skip_first_layer' in policy_cfg:
                agent_cfg.policy.ref_vel_skip_first_layer = policy_cfg['ref_vel_skip_first_layer']
                print(f"[Play] Loaded ref_vel_skip_first_layer={policy_cfg['ref_vel_skip_first_layer']} from checkpoint")
            if 'ref_vel_dim' in policy_cfg:
                agent_cfg.policy.ref_vel_dim = policy_cfg['ref_vel_dim']

    if args_cli.motion is not None:
        print(f"[INFO]: Using motion directory or file from CLI: {args_cli.motion}")
        env_cfg.commands.motion.motion = args_cli.motion

        # Optionally disable motion group sampling ratios for evaluation
        if args_cli.disable_motion_group_sampling and hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
            env_cfg.commands.motion.motion_group_sampling_ratios = None
            print("[INFO]: Disabled motion group sampling ratios for evaluation (uniform sampling).")

        # 强行从指定的帧开始执行
        if args_cli.start_frame is not None and hasattr(env_cfg.commands.motion, "start_frame"):
            env_cfg.commands.motion.start_from_beginning = True
            env_cfg.commands.motion.start_frame = args_cli.start_frame
            print(f"[INFO]: Forcing motion playback to start from frame {args_cli.start_frame}.")
    else:
        raise ValueError("Motion file or motion directory is required for evaluation.")

    # 将所有初始姿态与速度的随机化范围设为0, 保证初始动作贴合Ref Motion
    if not args_cli.enable_motion_randomization and hasattr(env_cfg, "commands"):
        motion_cfg = getattr(env_cfg.commands, "motion", None)
        if motion_cfg is not None:
            zero_ranges = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),}
            if hasattr(motion_cfg, "pose_range"):
                motion_cfg.pose_range = dict(zero_ranges)
            if hasattr(motion_cfg, "velocity_range"):
                motion_cfg.velocity_range = dict(zero_ranges)
            if hasattr(motion_cfg, "joint_position_range"):
                motion_cfg.joint_position_range = (0.0, 0.0)
            print("[INFO]: Zeroed motion randomization ranges for evaluation.")

    # 关闭观测噪音与随机事件
    if args_cli.disable_obs_noise and hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            if hasattr(env_cfg.observations, group_name):
                group_cfg = getattr(env_cfg.observations, group_name)
                if hasattr(group_cfg, "enable_corruption"):
                    group_cfg.enable_corruption = False
        print("[INFO]: Disabled observation corruption/noise for evaluation.")

    # 关闭事件管理器 (事件管理器用于控制域随机化和环境扰动)
    if args_cli.disable_events and hasattr(env_cfg, "events"):
        env_cfg.events = None
        print("[INFO]: Disabled event manager for evaluation.")

    # 关闭 q_ref 级别的 MotionPerturber（所有概率归零）
    # play 时使用预扰动的 .npz，不需要运行时再叠加随机扰动
    if hasattr(env_cfg, "motion_perturbations"):
        from whole_body_tracking.tasks.tracking.mdp.motion_perturbations import MotionPerturbationCfg
        env_cfg.motion_perturbations = MotionPerturbationCfg()
        print("[INFO]: Zeroed motion_perturbations for evaluation.")

    # Disable time-out termination during play to allow continuous replay
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        print("[INFO]: Disabling timeout termination for playback run.")
        env_cfg.terminations.time_out = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(args_cli.resume_path)

    # wrap for video recording
    # 将录像函数作为装饰器使用
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    # IsaacLab与rsl-rl是两个独立开发的库
    # 两者接口不同, 使用装饰器可以对齐接口
    env = RslRlVecEnvWrapper(env)

    # load previously trained model (通过Task参数控制导入的Policy)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner._move_normalizer_to_device(agent_cfg.device)
    ppo_runner.load(args_cli.resume_path, load_optimizer=False, load_critic=not args_cli.skip_critic)

    # obtain the trained policy for inference
    from rsl_rl.modules import FrontRESActorCritic

    # Check if velocity estimator is enabled
    use_velocity_estimator = (hasattr(ppo_runner.alg, 'ref_vel_estimator') and
                              ppo_runner.alg.ref_vel_estimator is not None and
                              hasattr(ppo_runner.alg, 'use_estimate_ref_vel') and
                              ppo_runner.alg.use_estimate_ref_vel)

    is_frontres = isinstance(ppo_runner.alg.policy, FrontRESActorCritic)
    if is_frontres:
        # Ensure alpha=1 for pure deployment (no curriculum scaling)
        ppo_runner.alg.policy.delta_q_alpha = 1.0
        print("[Play] FrontRES policy detected — delta_q_alpha set to 1.0 for inference.")

    if use_velocity_estimator:
        print("[Play] Using velocity estimator for inference")
        policy = None  # Will process observations manually in the loop
    else:
        # FrontRES: act_inference runs full ComposedActor pipeline (FrontRES → GMT → actions)
        # SuperviseLearning (Stage 1): has .student attribute
        if hasattr(ppo_runner.alg, "policy") and hasattr(ppo_runner.alg.policy, "student"):
            policy = ppo_runner.alg.policy.act_inference
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ============ 加载 GMT ONNX 模型 ============
    # gmt_model_path = ppo_runner.cfg.get("policy", {}).get("gmt_path")
    # gmt_session = None
    # if gmt_model_path and os.path.exists(gmt_model_path):
    #     try:
    #         print(f"[INFO] Loading GMT ONNX model from: {gmt_model_path}")
    #         gmt_session = ort.InferenceSession(gmt_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #         print(f"[INFO] GMT ONNX model loaded successfully. Provider: {gmt_session.get_providers()}")
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load GMT ONNX model: {e}")
    #         gmt_session = None
    # else:
    #     print(f"[WARNING] GMT ONNX model path not found or not specified in config. GMT will not be used.")

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(args_cli.resume_path), "exported")

    # FrontRES exports as a single composite ONNX (FrontRES + GMT packaged together)
    # so RobotBridge can use it as a drop-in replacement for the GMT ONNX:
    #   obs (770-dim, same format as MosaicEnv) → ONNX → motor actions
    # The ONNX file name is "policy.onnx" in both cases; update mosaic.yaml checkpoint path.
    onnx_filename = "policy.onnx"

    # Get velocity estimator info if available
    ref_vel_estimator = None
    ref_vel_estimator_obs_dim = None
    if hasattr(ppo_runner.alg, 'ref_vel_estimator') and ppo_runner.alg.ref_vel_estimator is not None:
        ref_vel_estimator = ppo_runner.alg.ref_vel_estimator
        if hasattr(ppo_runner.alg, 'ref_vel_estimator_obs_shape') and ppo_runner.alg.ref_vel_estimator_obs_shape is not None:
            ref_vel_estimator_obs_dim = ppo_runner.alg.ref_vel_estimator_obs_shape[0]

    export_motion_policy_as_onnx(
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename=onnx_filename,
        ref_vel_estimator=ref_vel_estimator,
        ref_vel_estimator_obs_dim=ref_vel_estimator_obs_dim,)

    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # ============ JIT导出代码 ============
    # try:
    #     class NormalizerWrapper(torch.nn.Module):
    #         def __init__(self, normalizer, model):
    #             super().__init__()
    #             self.normalizer = normalizer
    #             self.model = model

    #         def forward(self, obs):
    #             # 确保normalizer和模型在同一设备上
    #             obs = self.normalizer(obs)
    #             return self.model(obs)

    #     print(f"\n[INFO] Load in Normalizer")
        
    #     # 兼容 FrontRES 模型结构的深拷贝提取
    #     if hasattr(ppo_runner.alg, "policy") and hasattr(ppo_runner.alg.policy, "student"):
    #         actor_model_for_jit = copy.deepcopy(ppo_runner.alg.policy.student).eval().to("cpu")
    #     elif hasattr(ppo_runner.alg, "actor_critic"):
    #         actor_model_for_jit = copy.deepcopy(ppo_runner.alg.actor_critic.actor).eval().to("cpu")
    #     else:
    #         actor_model_for_jit = copy.deepcopy(ppo_runner.alg.policy.actor).eval().to("cpu")
            
    #     # 获取 Normalizer 的深拷贝
    #     if hasattr(ppo_runner, 'obs_normalizer') and ppo_runner.obs_normalizer is not None and not isinstance(ppo_runner.obs_normalizer, torch.nn.Identity):
    #         print(f"\n[INFO] Found Env Normalizer, Ready to Export JIT")
    #         normalizer_for_jit = copy.deepcopy(ppo_runner.obs_normalizer).eval().to("cpu")
    #         model_to_export = NormalizerWrapper(normalizer_for_jit, actor_model_for_jit)
    #     else:
    #         print(f"\n[WARNING] No active Normalizer found! Exporting raw actor.")
    #         model_to_export = actor_model_for_jit

    #     print(f"\n[INFO] Exporting TorchScript JIT model to: {export_model_dir}")
        
    #     # 创建一个虚拟输入
    #     dummy_input = torch.randn(1, env.observation_space.shape[0], device="cpu")

    #     # 将模型转换为 TorchScript (JIT)
    #     jit_model = torch.jit.trace(model_to_export, dummy_input)
        
    #     # 保存模型
    #     jit_path = os.path.join(export_model_dir, "policy_jit.pt")
    #     torch.jit.save(jit_model, jit_path)
    #     print(f"[INFO] Success! JIT saved as: {jit_path}\n")
    # except (Exception, ValueError) as e:
    #     print(f"[ERROR] TorchScript JIT export failed: {e}\n")
    # ====================================

    # reset environment
    env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if use_velocity_estimator:
                # Access observation manager directly to get all observation groups
                obs_manager = env.unwrapped.observation_manager

                # Compute observations for all groups - returns dict[group_name, tensor]
                obs_dict = obs_manager.compute()

                # Extract policy and ref_vel_estimator observations
                policy_obs = obs_dict["policy"].to(ppo_runner.device)
                ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].to(ppo_runner.device)

                # Normalize policy obs
                policy_obs_normalized = ppo_runner.obs_normalizer(policy_obs)

                # Estimate velocity
                estimated_ref_vel = ppo_runner.alg.ref_vel_estimator(ref_vel_estimator_obs) * 1.0
                print(f"[Play] Estimated ref vel: {estimated_ref_vel.cpu().numpy()}")

                # Augment observations
                obs_augmented = torch.cat([policy_obs_normalized, estimated_ref_vel], dim=-1)

                # Get actions
                actions = ppo_runner.alg.policy.act_inference(obs_augmented)
            else:
                # Standard inference without velocity estimator
                obs, _ = env.get_observations()
                actions = policy(obs)

                # ============ FrontRES + GMT 两阶段推理管线 ============
                # obs, _ = env.get_observations() # 获取原始观测

                # # 如果 GMT 模型已加载，则执行两阶段推理
                # if gmt_session is not None:
                #     # 1. FrontRES 推理，得到 Δq
                #     # policy 变量持有的是已加载的 FrontRES (SuperviseLearning.student)
                #     delta_q = policy(obs)

                #     # 2. 从观测中提取 q_ref
                #     # 根据 G1 机器人 URDF, 关节数为 29。'command' 包含 q_ref 和 qd_ref, 位于 obs 前 58 个维度
                #     num_joints = 29 # G1 机器人关节数
                #     q_ref = obs[:, :num_joints]

                #     # 3. 计算修正后的 q_corrected
                #     q_corrected = q_ref + delta_q

                #     # 4. 构造 GMT 的输入
                #     # 将 obs 中的 q_ref 部分替换为修正后的 q_corrected
                #     gmt_input_obs = obs.clone()
                #     gmt_input_obs[:, :num_joints] = q_corrected

                #     # 5. GMT ONNX 模型推理，得到最终 action
                #     gmt_input_name = gmt_session.get_inputs()[0].name
                #     gmt_output_name = gmt_session.get_outputs()[0].name
                #     actions_np = gmt_session.run([gmt_output_name], {gmt_input_name: gmt_input_obs.cpu().numpy()})[0]
                #     actions = torch.from_numpy(actions_np).to(ppo_runner.device)
                # else:
                #     # Fallback: 如果没有 GMT，则直接使用 FrontRES 的输出 (原始行为)
                #     actions = policy(obs)

            # env stepping
            _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
