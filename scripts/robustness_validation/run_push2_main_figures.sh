#!/usr/bin/env bash
set -euo pipefail

# Run RobotBridge/MuJoCo push-2 validation for the main paper groups and
# generate grouped figures.  Intended working directory on the server:
#
#   /home/chengyuxuan/RobotBridge/deploy

PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOTBRIDGE_ROOT="${ROBOTBRIDGE_ROOT:-/home/chengyuxuan/RobotBridge}"
DEPLOY_DIR="${DEPLOY_DIR:-${ROBOTBRIDGE_ROOT}/deploy}"
MOTION_ROOT="${MOTION_ROOT:-${DEPLOY_DIR}/data/motion}"
CHECKPOINT="${CHECKPOINT:-${DEPLOY_DIR}/data/model/model_27000.onnx}"

PUSH_VELOCITY="${PUSH_VELOCITY:-2.0}"
FIXED_PUSH_OFFSET="${FIXED_PUSH_OFFSET:-40}"
PUSH_DIRECTION_ANGLE="${PUSH_DIRECTION_ANGLE:-0}"
NUM_TRIALS="${NUM_TRIALS:-10}"
PERTURBATION_MODE="${PERTURBATION_MODE:-rp}"

EPSILON_VALUES=(
  0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2
  0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4
)

MAIN_GROUPS=(Lateral Upper Walking)

cd "${DEPLOY_DIR}"

for GROUP in "${MAIN_GROUPS[@]}"; do
  "${PYTHON_BIN}" robustness_validation/run_validation_mujoco_batch.py \
    --robotbridge_root "${ROBOTBRIDGE_ROOT}" \
    --motion_root "${MOTION_ROOT}" \
    --groups "${GROUP}" \
    --file_glob "*.npz" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${DEPLOY_DIR}/verify/${GROUP}_push2" \
    --epsilon_values "${EPSILON_VALUES[@]}" \
    --push_velocities "${PUSH_VELOCITY}" \
    --fixed_push_offset "${FIXED_PUSH_OFFSET}" \
    --push_direction_angle "${PUSH_DIRECTION_ANGLE}" \
    --num_trials "${NUM_TRIALS}" \
    --perturbation_modes "${PERTURBATION_MODE}"
done

"${PYTHON_BIN}" robustness_validation/plot_results.py \
  --results_dirs \
  verify/Lateral_push2/motions/Lateral/*/baseline \
  verify/Upper_push2/motions/Upper/*/baseline \
  verify/Walking_push2/motions/Walking/*/baseline \
  --grouped \
  --perturbation_mode "${PERTURBATION_MODE}" \
  --output_dir verify/figures_push2

