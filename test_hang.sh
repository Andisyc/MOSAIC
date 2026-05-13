#!/bin/bash
export HYDRA_FULL_ERROR=1
bash ~/IsaacLab_mosaic/isaaclab.sh -p ~/MOSAIC/scripts/robustness_validation/run_minimal.py \
    --motion /home/chengyuxuan/MOSAIC/q_npz \
    --checkpoint ~/MOSAIC/model/model_27000.pt \
    --headless &
PID=$!
sleep 45
kill -ABRT $PID
