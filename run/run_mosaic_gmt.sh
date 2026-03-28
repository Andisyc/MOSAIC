# Distribute Training
# HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
#     --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
#     --distributed \
#     --num_envs=24000 \
#     --motion /path/to/motion \
#     --headless \
#     --logger wandb \
#     --log_project_name GMT_MOSAIC_RL \
#     --run_name GMT_MOSAIC_GMT

# Single GPU Training
# HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
#     --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
#     --num_envs=12000 \
#     --motion /ssd1/chengyuxuan/AMASS_G1NPZ_Final \
#     --headless \
#     --logger wandb \
#     --log_project_name GMT_MOSAIC_RL \
#     --run_name GMT_MOSAIC_GMT

# stage pre: testing run
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
    --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
    --num_envs=2 \
    --motion /home/chengyuxuan/MOSAIC/q_npz \
    --logger wandb \
    --headless \
    --device cuda:0

# stage 0: MOSAIC GMT training
HYDRA_FULL_ERROR=1 nohup bash ~/IsaacLab_mosaic/isaaclab.sh -p ~/MOSAIC/scripts/rsl_rl/train.py \
    --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
    --num_envs=12000 \
    --motion /ssd1/chengyuxuan/AMASS_G1NPZ_Final \
    --logger wandb \
    --headless \
    --device cuda:0 >~/MOSAIC/train.txt 2>&1 &

# stage 1: FrontRES Supervised training
HYDRA_FULL_ERROR=1 bash ~/IsaacLab_mosaic/isaaclab.sh -p ~/MOSAIC/scripts/rsl_rl/train.py \
    --task=FrontRES-Supervised-Tracking-Flat-G1-v0 \
    --num_envs=12000 \
    --motion /ssd1/chengyuxuan/AMASS_G1NPZ_Final \
    --logger wandb \
    --headless \
    --device cuda:0

# stage 2: FrontRES RL training
HYDRA_FULL_ERROR=1 bash ~/IsaacLab_mosaic/isaaclab.sh -p ~/MOSAIC/scripts/rsl_rl/train.py \
    --task=FrontRES-RLFinetune-Tracking-Flat-G1-v0 \
    --num_envs=12000 \
    --motion /ssd1/chengyuxuan/AMASS_G1NPZ_Final \
    --logger wandb \
    --headless \
    --device cuda:0