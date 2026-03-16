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
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg", # 奖励函数
    },
)


gym.register(
    id="General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatGeneralEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)

gym.register(
    id="General-Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationGeneralEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)

gym.register(
    id="General-Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqGeneralEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg", # 奖励函数
    },
)

gym.register(
    id="Expert-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)

gym.register(
    id="Expert-General-Tracking-Flat-G1-MOSAIC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRunnerCfg", # 奖励函数
    },
)


gym.register(
    id="Distillation-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatDistillationRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="MOSAIC-Distill-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICHybridRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICPureDistillationRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0-fld",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FLDDistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICPureDistillationRunnerCfg", # 奖励函数
    },
)


gym.register(
    id="MOSAIC-RL-Continue-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRLContinueRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="MOSAIC-Residual-General-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRLContinueResidualRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1MultiDistillationTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICMultiTeacherResidualRunnerCfg", # 奖励函数
    },
)

gym.register(
    id="General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1OneStageTrackingEnvCfg, # 环境导入
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", # 奖励函数
    },
)