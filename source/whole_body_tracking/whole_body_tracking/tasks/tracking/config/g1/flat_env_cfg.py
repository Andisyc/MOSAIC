from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg, 
    GeneralTrackingEnvCfg, 
    ExpertGeneralTrackingEnvCfg, 
    DistillationTrackingEnvCfg, 
    MultiDistillationTrackingEnvCfg, 
    OneStageTrackingEnvCfg, 
    SupervisedTrackingEnvCfg,
    FrontRESFinetuneTrackingEnvCfg, # 引入第二阶段微调的基础环境配置
)


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatGeneralEnvCfg(GeneralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationGeneralEnvCfg(G1FlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqGeneralEnvCfg(G1FlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

@configclass
class G1FlatExpertGeneralEnvCfg(ExpertGeneralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

@configclass
class G1DistillationTrackingEnvCfg(DistillationTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.teacher.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FLDDistillationTrackingEnvCfg(DistillationTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.teacher.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1OneStageTrackingEnvCfg(OneStageTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # 刻意禁用两个特权观测量
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1MultiDistillationTrackingEnvCfg(MultiDistillationTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.teacher.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # Multi-teacher support: motion groups for different teachers
        self.commands.motion.motion_groups = {
            "motions": ["path/to/motions"],
            "teleop_motions": ["path/to/teleop_motions"],
        }
        # Motion group sampling ratios - controls proportion of environments using each group
        self.commands.motion.motion_group_sampling_ratios = {
            "motions": 0.5,
            "teleop_motions": 0.5,
        }

# ======== FrontRES Two Stage Trainig Configs ========

@configclass
class G1SupervisedTrackingEnvCfg(SupervisedTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatFrontRESFinetuneEnvCfg(FrontRESFinetuneTrackingEnvCfg):
    """
    Environment configuration for Stage 2: RL Finetuning of FrontRES.
    """

    def __post_init__(self):
        # Inherit from the FrontRES Finetuning base environment configuration
        super().__post_init__()

        # Set G1-specific configurations
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Match Stage 1 (SupervisedObservationsCfg.PolicyCfg) observation space → 800 dims.
        # Stage 1 was trained WITHOUT motion_anchor_pos_b (3 dims) and base_lin_vel (3 dims)
        # but WITH anchor_root_pos_error_w (3 dims) and anchor_root_rpy_error_w (3 dims).
        # Both stages therefore share a common 800-dim obs layout.
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        # Add task-space anchor error observations (3 + 3 dims, history × 5 = +30 dims total)
        self.observations.policy.anchor_root_pos_error_w = ObsTerm(
            func=mdp.anchor_root_pos_error_w, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.01, n_max=0.01))
        self.observations.policy.anchor_root_rpy_error_w = ObsTerm(
            func=mdp.anchor_root_rpy_error_w, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.01, n_max=0.01))

        # FrontRES corrects the global motion anchor (root-level perturbations), not
        # individual joint targets.  Wrist Z-tracking errors from fast dance arm gestures
        # are NOT fall indicators — they reflect PD-controller bandwidth limits and should
        # not terminate episodes.  Keep only ankle bodies for foot-contact quality check.
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis", 
            "left_hip_roll_link", 
            "left_knee_link", 
            "left_ankle_roll_link",
            "right_hip_roll_link", 
            "right_knee_link", 
            "right_ankle_roll_link", 
            "torso_link",
            "left_shoulder_roll_link", 
            "left_elbow_link", 
            "left_wrist_yaw_link",
            "right_shoulder_roll_link", 
            "right_elbow_link", 
            "right_wrist_yaw_link",
        ]
