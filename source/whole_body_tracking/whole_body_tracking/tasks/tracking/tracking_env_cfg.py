from __future__ import annotations

from dataclasses import MISSING
from typing import Union

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.mdp.motion_perturbations import MotionPerturbationCfg

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Stage 2 RL Finetune 专用推力范围：降低强度，适合运动跟踪微调阶段
FRONTRES_PUSH_VELOCITY_RANGE = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.1, 0.1),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.5, 0.5),
}

from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfRandomUniformTerrainCfg
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,),
    #     
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,),)

    # 创建地形: 50%的平地, 50%的轻微颠簸地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", 
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=0.5),
                "slightly_rough": HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.01, 0.03),
                    noise_step=0.01,),},),
        
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,),
        
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,),)
    
    robot: ArticulationCfg = MISSING # robots
    
    light = AssetBaseCfg( # lights
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),)
    
    sky_light = AssetBaseCfg( # 
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),)
    
    contact_forces = ContactSensorCfg( # 虚拟接触力传感器
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True)


##
# MDP settings: State/Observation, Action, Transition Probability, Reward
##


@configclass
class SingleMotionCommandsCfg: # 本Episode只执行一个动作序列
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),},
        
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),)

@configclass
class MultiMotionCommandsCfg: # 本Episode执行多个动作序列
    """Command specifications for the MDP."""

    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        resample_motions_every_s = 1e9,
        motion_sampling_warmup_s=1e9,
        motion_sampling_ramp_s=1e9,
        motion_sampling_schedule="cosine",
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),},
        
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),)

@configclass
class ActionsCfg: # 将Policy的输出结果翻译成目标关节角
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)

@configclass
class ObservationsCfg: # 学生模型观测量
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup): # 学生模型的Actor的观测量
        """Observations for policy group."""
        # 参考动作: 29 dim关节角 + 29 dim关节速度
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})

        # 局部坐标系 (base frame) 锚点姿态误差 6 dim
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25))

        # 局部坐标系 (base frame) 锚点位置误差 (特权信息)
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))

        # 机身线速度 (特权信息)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))

        # 机身角速度 3 dim
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        # 关节角位置误差 29 dim
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # 关节角速度误差 29 dim
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))

        # 前帧动作指令 29 dim
        actions = ObsTerm(func=mdp.last_action)

        # Task-space anchor error obs (None=disabled; enabled in Stage 2 __post_init__)
        anchor_root_pos_error_w: ObsTerm | None = None
        anchor_root_rpy_error_w: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5 # 储存过去4帧+当前1帧

    @configclass
    class PrivilegedCfg(ObsGroup): # 学生模型的Critic的特权信息
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class ObservationsExpertCfg: # 教师模型观测量
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg: # 域泛化
    """Configuration for events."""

    # startup
    physics_material = EventTerm( # 地面摩擦力
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,},)

    add_joint_default_pos = EventTerm( # 电机零位偏差
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",},)

    base_com = EventTerm( # 机器人质心
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},},)

    # interval
    push_robot = EventTerm( # 随机外力
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},)

    # -- 新增的物理属性域随机化 --
    # 说明: 以下随机化需要在 mdp 模块中实现对应的函数。

    randomize_gravity = EventTerm(
        func=mdp.randomize_gravity,
        mode="startup",
        params={
            "x_range": (-1.0, 1.0),
            "y_range": (-1.0, 1.0),
            "z_range": (-12.0, -7.0), # 在地球重力-9.8附近浮动
        },
    )

    randomize_actuator_properties = EventTerm(
        func=mdp.randomize_actuator_properties,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_range": (0.8, 1.2), # 将默认刚度乘以一个随机系数
            "damping_range": (0.8, 1.2),   # 将默认阻尼乘以一个随机系数
        },
    )

    add_payload = EventTerm(
        func=mdp.add_payload_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]), "mass_range": (0.0, 5.0)}, # 在躯干上增加0-5kg的随机载荷
    )

@configclass
class RewardsCfg: # 奖励项
    """Reward terms for the MDP."""

    # 动作平滑率的L2范数 (当前帧力矩与上帧力矩的差异)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)

    # 关节限位, 惩罚策略让关节超越极限
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},)
    
    # 关节速度与扭矩限制
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)

    # 全局锚点追踪: Robot在世界坐标系的位置
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},)
    
    # 全局锚点追踪: Robot在世界坐标系的朝向
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},)
    
    # 相对躯干追踪: 肢体中心点与Ref Motion对齐
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},)
    
    # 相对躯干追踪: 肢体朝向与Ref Motion对齐
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},)
    
    # 速度追踪: 躯干线速度与Ref Motion对齐
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},)
    
    # 速度追踪: 躯干角速度与Ref Motion对齐
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},)
    
    # 防摔倒: 除了手和脚, 其他区域触地则惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],),
            "threshold": 1.0,},)

@configclass
class RewardsExpertCfg: # 专家特调奖励项
    """
    Expert reward configuration - MOSAIC.
    """

    # 全局锚点追踪: Robot在世界坐标系的位置
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},)
    
    # 全局锚点追踪: Robot在世界坐标系的朝向
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},)
    
    # 相对躯干追踪: 肢体中心点与Ref Motion对齐
    motion_body_pos = RewTerm( # 跟踪奖励
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},)
    
    # 相对躯干追踪: 肢体朝向与Ref Motion对齐
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},)
    
    # 速度追踪: 躯干线速度与Ref Motion对齐
    motion_body_lin_vel = RewTerm( # 
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},)
    
    # 速度追踪: 躯干角速度与Ref Motion对齐
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 3.14},)
    
    # 速度追踪: 锚点(机器人盆骨)线速度与Ref Motion对齐
    motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_linear_velocity_error_exp,
        weight=1.0,  # 2*1.0
        params={"command_name": "motion", "std": 1.0},)

    # 上下半身分离追踪
    teleop_body_position_extend = RewTerm(
        func=mdp.teleop_body_position_extend,
        weight=1.0,
        params={
            "command_name": "motion",
            "upper_body_std": 0.5, 
            "lower_body_std": 0.5,  
            "upper_weight": 1.0,
            "lower_weight": 1.0,})
    
    teleop_vr_3point = RewTerm(
        func=mdp.teleop_vr_3point,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5})
    
    teleop_body_position_feet = RewTerm(
        func=mdp.teleop_body_position_feet,
        weight=1,  # 1.5*1
        params={"command_name": "motion", "std": 0.5})
    
    teleop_body_rotation_extend = RewTerm(
        func=mdp.teleop_body_rotation_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5})
    
    teleop_body_ang_velocity_extend = RewTerm(
        func=mdp.teleop_body_ang_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 3.14})
    
    teleop_body_velocity_extend = RewTerm(
        func=mdp.teleop_body_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0})

    # ===== Penalty terms (same as base) =====
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],),
            "threshold": 1.0,},)
    
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)  # 2*1e-1

    # 限制关节极限, 防止关节超限
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},)
    
    # 限制电机速度与扭矩, 防止烧毁电机
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # -2*2.5e-7
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # -2*1e-5
    
@configclass
class TerminationsCfg: # Episode结束判定
    """Termination terms for the MDP."""

    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 超时

    anchor_pos = DoneTerm( # 关节触地
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},)
    
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},)
    
    ee_body_pos = DoneTerm( # 根节点触地
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],},)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg): # 基础Env, 将场景、观测、动作空间、奖励项组装起来
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: SingleMotionCommandsCfg = SingleMotionCommandsCfg()
    
    # MDP settings
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    motion_perturbations: MotionPerturbationCfg = MotionPerturbationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 15 * 2**17

        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.origin_type = "env"
        self.viewer.asset_name = "robot"
    

@configclass
class GeneralTrackingEnvCfg(TrackingEnvCfg): # 泛化Env, 能同时加载多组动作
    """Configuration for the general tracking environment."""

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


@configclass
class ExpertGeneralTrackingEnvCfg(GeneralTrackingEnvCfg): # 教师模型训练Env, 将观测量替换为专家观测量, 奖励项替换成教师奖励项
    """
    Expert general tracking environment.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()
    observations: ObservationsExpertCfg = ObservationsExpertCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


@configclass
class DistillationTrackingEnvCfg(GeneralTrackingEnvCfg): # 师生蒸馏Env, 同时具有学生模型、教师模型、Critic
    """
    Student-teacher distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class DistillationObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup): # 学生模型观测量
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)
            # Root z-height error: z_sim - z_ref (m). Positive=robot above ref, Negative=float artifact.
            # Gives FrontRES a direct signal for Δz prediction (identity mapping: output ≈ input).
            anchor_height_error = ObsTerm(
                func=mdp.anchor_root_height_error, params={"command_name": "motion"},
                noise=Unoise(n_min=-0.01, n_max=0.01))

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup): # 教师模型观测量
            """Teacher observations - teacehr information."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)
            supervision_target = ObsTerm(func=mdp.get_supervision_target_delta_q, params={"command_name": "motion"})

            # def __post_init__(self):
            #     self.history_length = 5

        @configclass
        class RefVelEstimatorCfg(ObsGroup):
            """Observations for reference velocity estimator."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # [58] = [joint_pos(29), joint_vel(29)]
            ref_projected_gravity = ObsTerm(func=mdp.ref_projected_gravity, params={"command_name": "motion"})  # [3] using ANCHOR BODY quaternion

            def __post_init__(self):
                self.enable_corruption = False 
                self.concatenate_terms = True
                self.history_length = 5

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        critic: PrivilegedCfg = PrivilegedCfg()
        ref_vel_estimator: RefVelEstimatorCfg = RefVelEstimatorCfg() 

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


@configclass
class OneStageTrackingEnvCfg(GeneralTrackingEnvCfg): # 消融实验配置, 不使用教师模型, 直接训练学生模型
    """
    Teacher-student distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class OneStageObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # 参考指令集合
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}) # 机器人盆骨世界坐标系三维位置误差 (特权信息)
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}) # 机器人盆骨世界坐标系三维旋转误差
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            # def __post_init__(self):
            #     self.history_length = 5
        
        # observation groups
        policy: PolicyCfg = PolicyCfg()
        critic: PrivilegedCfg = PrivilegedCfg()

    observations: OneStageObservationsCfg = OneStageObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


@configclass
class MultiDistillationTrackingEnvCfg(GeneralTrackingEnvCfg): # 多专家蒸馏, 学生模型观测量, 教师模型观测量, 特权信息
    """
    Teacher-student distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class DistillationObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup): # 学生模型观测量: 无绝对坐标和绝对线速度, 只有本体感知 (姿态角, 角速度, 关节位置, 关节速度)
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # 参考指令集合
            motion_anchor_ori_b = ObsTerm( # 机器人盆骨世界坐标系三维旋转误差
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 机器人真实角速度
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # 关节角
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # 关节速度
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup): # 教师模型观测量: 与学生模型同样的观测量, 因为要计算KL散度, 如果观测量不一致会导致分布差异
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # 参考指令集合
            motion_anchor_ori_b = ObsTerm( # 机器人盆骨世界坐标系三维旋转误差
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 机器人真实角速度
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # 关节角
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # 关节速度
            actions = ObsTerm(func=mdp.last_action) # 上帧动作指令

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5
        
        @configclass
        class PrivilegedCfg(ObsGroup): # Critic的特权信息: 绝对位置与速度, 绝对肢体坐标
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # 参考指令集合
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}) # 机器人盆骨世界坐标系三维位置误差 (特权信息)
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}) # 机器人盆骨世界坐标系三维旋转误差
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}) # 机器人肢体的全局三维位置 (特权信息)
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}) # 机器人肢体的全局三维旋转 (特权信息)
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 世界坐标系的真实线速度 (特权信息)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"}) # Ref Motion的世界坐标系线速度 (特权信息)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # 机器人真实角速度
            joint_pos = ObsTerm(func=mdp.joint_pos_rel) # 关节角
            joint_vel = ObsTerm(func=mdp.joint_vel_rel) # 关节速度
            actions = ObsTerm(func=mdp.last_action) # 上帧动作指令

        @configclass
        class RefVelEstimatorCfg(ObsGroup):
            """Observations for reference velocity estimator.
            """
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # [58] = [joint_pos(29), joint_vel(29)]
            ref_projected_gravity = ObsTerm(func=mdp.ref_projected_gravity, params={"command_name": "motion"})  # [3] using ANCHOR BODY quaternion

            def __post_init__(self):
                self.enable_corruption = False 
                self.concatenate_terms = True
                self.history_length = 5

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        critic: PrivilegedCfg = PrivilegedCfg()
        ref_vel_estimator: RefVelEstimatorCfg = RefVelEstimatorCfg()

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()

# ======== FrontRES Two Stage Training EnvCfg ========

@configclass
class SupervisedTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Configuration for Stage 1 Supervised Learning environment.

    Trains FrontRES to output the anti-DR SE(3) correction using perturber
    deltas as ground-truth labels.  The supervision target is:

        target = -(perturbed_anchor - original_anchor) → "undo the DR"

    This replaces the old identity mapping (robot - anchor) which taught
    FrontRES to chase the DR-perturbed robot state.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    # -- Motion-level DR applied to the reference for Stage 1 training data --
    # Mild base values so FrontRES learns the anti-DR mapping without
    # destabilising GMT during supervised data collection.
    motion_perturbations: MotionPerturbationCfg = MotionPerturbationCfg(
        float_prob       = 0.3,
        float_ratio      = 0.05,   # max ≈ 5 cm
        sink_prob        = 0.3,
        sink_ratio       = 0.04,   # max ≈ 4 cm
        foot_slip_prob   = 0.2,
        foot_slip_ratio  = 0.008,
        root_tilt_prob   = 0.3,
        root_tilt_max_rad = 0.05,  # max ≈ 2.9°
        joint_noise_prob = 0.4,
        joint_noise_std  = 0.04,   # rad
        joint_noise_joint_indices = list(range(12)),
    )

    @configclass
    class SupervisedObservationsCfg:
        """Observation specifications for Stage 1 Supervised Learning."""

        @configclass
        class PolicyCfg(ObsGroup): # FrontRES 模型的输入特征
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)
            # Task-space position error: robot_root_pos_w - anchor_pos_w (world frame, 3 dims).
            # Gives FrontRES a direct signal for Δpos prediction (identity mapping: output ≈ input).
            anchor_root_pos_error_w = ObsTerm(
                func=mdp.anchor_root_pos_error_w, params={"command_name": "motion"},
                noise=Unoise(n_min=-0.01, n_max=0.01))
            # Task-space orientation error: RPY of quat_inv(q_anchor)*q_robot (3 dims).
            # Gives FrontRES a direct signal for Δrpy prediction (identity mapping: output ≈ input).
            anchor_root_rpy_error_w = ObsTerm(
                func=mdp.anchor_root_rpy_error_w, params={"command_name": "motion"},
                noise=Unoise(n_min=-0.01, n_max=0.01))

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TargetCfg(ObsGroup): # 监督信号目标: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw] = 6 dims
            supervision_target = ObsTerm(func=mdp.get_supervision_target_task_space, params={"command_name": "motion"})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        target: TargetCfg = TargetCfg()

    observations: SupervisedObservationsCfg = SupervisedObservationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()

        # Disable physics-level DR (material, mass, etc.) for Stage 1.
        # Motion-level DR is enabled via self.motion_perturbations above
        # to generate supervision targets.
        self.events = None
        self.curriculum = None


@configclass
class FrontRESFinetuneTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Environment configuration for Stage 2: RL Finetuning of FrontRES.

    Physics-level DR is handled by `events` (Isaac Lab EventManager).
    Motion-level DR (perturbations applied to q_ref, not the simulated robot)
    is handled by `motion_perturbations`, which is read by MultiMotionCommand
    to instantiate a MotionPerturber.
    """

    @configclass
    class RLFinetuneEventCfg:
        """
        Domain Randomization for FrontRES Stage 2 RL Finetune.

        设计原则：
          1. 只针对真实 sim2real gap，删除现实中不存在的干扰（重力随机化）
          2. 全部使用 mode="startup"（物理在仿真启动时固定，Critic 能为每个 env
             学出稳定的 V(s)，避免 reset 模式下 Critic 面对混合分布导致的高方差）
          3. 缩小各项范围，最坏组合概率约 0.03%（8192 env 中仅约 2~3 个 env），
             贡献近零梯度但不主动破坏收敛

        保留项（真实 sim2real gap 来源）：
          physics_material              — 地面材质不同
          add_joint_default_pos         — 编码器零点误差
          randomize_actuator_properties — PD 增益标定误差、关节摩擦
          base_com                      — URDF 质心与真机偏差
          add_payload                   — 真机负载
          push_robot                    — 真实扰动（降低频率和强度）

        删除项：
          randomize_gravity             — 重力在真实世界中不变，纯噪声
        """

        # startup（PhysX 材质重建代价高；收窄下限避免极端湿滑）
        physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.5),   # 原 (0.4, 2.0)，去掉极端湿滑
                "dynamic_friction_range": (0.5, 1.2),  # 原 (0.4, 1.5)
                "restitution_range": (0.0, 0.2),
                "num_buckets": 128,
            },
        )

        # startup（物理固定，Critic 能为每个 env 学出稳定的 V(s)）
        add_joint_default_pos = EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_distribution_params": (-0.02, 0.02),
                "operation": "add",
            },
        )

        base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

        randomize_actuator_properties = EventTerm(
            func=mdp.randomize_actuator_properties,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_range": (0.85, 1.15),  # 原 (0.7, 1.3)，±30% → ±15%
                "damping_range": (0.85, 1.15),
            },
        )

        add_payload = EventTerm(
            func=mdp.add_payload_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                "mass_range": (0.0, 2.0),  # 原 7.0 kg，G1 约 35kg，2kg ≈ +5.7%
            },
        )

        # interval（降低频率和强度，给机器人足够的恢复时间）
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),  # 原 (0.5, 2.5)s，大幅降低推力频率
            params={"velocity_range": FRONTRES_PUSH_VELOCITY_RANGE},
        )

    @configclass
    class RLFinetuneRewardsCfg:
        """Reward terms for FrontRES Stage 2 RL Finetune.

        Each reward term corresponds to one or more DR noise types:
          contact_feet            — 脚-地面穿模 / 漂浮 / 滑步 (接触状态)
          motion_feet_pos         — 脚-地面穿模 / 漂浮 / 滑步 (连续位置)
          motion_body_pos         — 关节偏置 / 关节解耦 / 身体自穿模
          motion_body_ori         — 关节偏置 / 关节解耦
          motion_body_lin_vel     — 关节偏置 / 关节解耦 (速度)
          motion_body_ang_vel     — 关节偏置 / 关节解耦 (角速度)
          motion_global_anchor_pos — 根节点错误 / 质心越界 / 漂浮
          motion_global_anchor_ori — 根节点错误
          undesired_contacts      — 身体自穿模 / 关节偏置自碰撞
          joint_limit             — 所有 DR 噪声类型的安全边界
          joint_torque            — 执行器过载保护

        Note: action_rate_l2 is removed — it penalises GMT output (robot_actions),
        not Δq directly.  Smoothness over Δq is handled by smooth_loss in ppo.py.
        """

        # --- 接触去噪: 脚-地面穿模 / 漂浮 / 滑步 ---
        # R = 0.5 * sum_f( 1[c_actual(f) == c_ref(f)] ), range [0, 1]
        contact_feet = RewTerm(
            func=mdp.contact_feet,
            weight=1.0,
            params={"command_name": "motion", "threshold": 0.05},
        )

        # R = exp(-E / sigma^2), E = mean_{j in J_feet}(||p_ref(j) - p_robot(j)||^2)
        # sigma = 0.05 m (stricter than full-body, reflects foot precision requirement)
        motion_feet_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.05,
                "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
        )

        # --- 关节去噪: 关节偏置 / 关节解耦 / 身体自穿模 ---
        # R = exp(-E / sigma^2), E = mean_{j in J_all}(||p_ref(j) - p_robot(j)||^2)
        motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        )

        # R = exp(-E / sigma^2), E = mean_{j in J_all}(quat_err(quat_ref(j), quat_robot(j))^2)
        motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        )

        # R = exp(-E / sigma^2), E = mean_{j in J_all}(||v_ref(j) - v_robot(j)||^2)
        motion_body_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=0.75,
            params={"command_name": "motion", "std": 1.0},
        )

        # R = exp(-E / sigma^2), E = mean_{j in J_all}(||w_ref(j) - w_robot(j)||^2)
        motion_body_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=0.75,
            params={"command_name": "motion", "std": 3.14},
        )

        # --- 根节点去噪: 根节点错误 / 质心越界 ---
        # R = exp(-E / sigma^2), E = ||p_anchor_ref - p_anchor_robot||^2
        motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        )

        # R = exp(-E / sigma^2), E = quat_err(quat_anchor_ref, quat_anchor_robot)^2
        motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        )

        # --- 物理安全约束 ---
        # 身体自穿模 / 关节偏置自碰撞: R = -1 if F(link) > 1.0 N
        undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )

        # 所有 DR 噪声类型的关节安全边界
        joint_limit = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-20.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        # 执行器过载保护: R = -sum_j(tau_j^2)
        joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)

    rewards: RLFinetuneRewardsCfg = RLFinetuneRewardsCfg()
    events: RLFinetuneEventCfg = RLFinetuneEventCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Motion-level domain randomization: perturb q_ref to simulate
        # physical artifacts (floating, penetration) in the reference motion.
        # Applied inside MultiMotionCommand via MotionPerturber — NOT via EventManager.
        #
        # Base values = "Level 1" mild perturbations that GMT can handle (episode_length ≈ 100+).
        # The adaptive DR PI controller multiplies these by dr_scale ∈ [0, dr_max_scale=4.0]:
        #   scale=1.0 → base values below (GMT comfortable)
        #   scale=4.0 → 4× base ≈ original design max (FrontRES fully required)
        self.motion_perturbations.float_prob       = 0.3
        self.motion_perturbations.float_ratio      = 0.05   # 4× → 0.20 m (original max)
        self.motion_perturbations.sink_prob        = 0.3
        self.motion_perturbations.sink_ratio       = 0.04   # 4× → 0.16 m
        self.motion_perturbations.foot_slip_prob   = 0.2
        self.motion_perturbations.foot_slip_ratio  = 0.008  # 4× → 0.032 m
        self.motion_perturbations.root_tilt_prob   = 0.3
        self.motion_perturbations.root_tilt_max_rad = 0.05  # 4× → 0.20 rad (~11.5°)
        self.motion_perturbations.joint_noise_prob = 0.4
        self.motion_perturbations.joint_noise_std  = 0.04   # 4× → 0.16 rad
        self.motion_perturbations.joint_noise_joint_indices = list(range(12))
