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

        # GMT was already trained with Physics DR and is robust to PD gain
        # variations, COM offsets, and payload without FrontRES involvement.
        # FrontRES only corrects motion reference errors (ΔSE3); it cannot
        # compensate physical parameter variations via anchor corrections.
        # Physics DR here makes GMT fail as a B1 baseline (ep_len ≈ 12),
        # zeroing both r_delta and the supervised signal — a training deadlock.
        self.events = None

        # Obs layout (800 dims total):
        #   [0:770]  = GMT-compatible prefix:
        #              [cmd(58), ori(6), ang(3), jpos(29), jvel(29), act(29)] × 5
        #              = 154/frame × 5 = 770 dims — identical to the layout GMT was
        #              trained on (DistillationTrackingEnvCfg removes motion_anchor_pos_b
        #              and base_lin_vel, giving 154 dims/frame × 5 = 770).
        #   [770:800] = FrontRES-only anchor error signals (3+3 dims × 5 frames = 30):
        #              pass-through in the runner's partial normaliser.
        #
        # Removing motion_anchor_pos_b and base_lin_vel (from the 160-dim base) is
        # intentional: keeping them would make the prefix 160×5=800 dims, shifting
        # every subsequent index and corrupting GMT's fixed ONNX weight alignment.
        self.observations.policy.motion_anchor_pos_b = None   # keep 154-dim/frame GMT prefix
        self.observations.policy.base_lin_vel = None          # keep 154-dim/frame GMT prefix
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
