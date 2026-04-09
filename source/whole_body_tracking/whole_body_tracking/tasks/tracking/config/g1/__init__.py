import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg", # 网络结构 & 训练算法
    },
)


gym.register(
    id="General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatGeneralEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="General-Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationGeneralEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="General-Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqGeneralEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="Expert-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="Expert-General-Tracking-Flat-G1-MOSAIC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRunnerCfg", # 网络结构 & 训练算法
    },
)


gym.register(
    id="Distillation-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatDistillationRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="MOSAIC-Distill-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICHybridRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICPureDistillationRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0-fld",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FLDDistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICPureDistillationRunnerCfg", # 网络结构 & 训练算法
    },
)


gym.register(
    id="MOSAIC-RL-Continue-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRLContinueRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register(
    id="MOSAIC-Residual-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRLContinueResidualRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register( # RES训练
    id="MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1MultiDistillationTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICMultiTeacherResidualRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register( # GMT & Adapt训练
    id="General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1OneStageTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register( # FrontRES训练 (阶段1)
    id="FrontRES-Supervised-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1SupervisedTrackingEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化 (阶段一需要 target 监督信号)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatSupervisedRunnerCfg", # 网络结构 & 训练算法
    },
)

gym.register( # FrontRES训练 (阶段2)
    id="FrontRES-RLFinetune-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatFrontRESFinetuneEnvCfg, # 仿真环境 & 观测量 & 奖励项 & 域随机化 (阶段二RL需要 Critic)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatFrontRESFinetuneRunnerCfg", # 网络结构 & 训练算法 (supervise_learning.py & supervise.py)
    },
)