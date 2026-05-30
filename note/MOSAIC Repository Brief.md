# MOSAIC Repository Brief

This file is a compact map for future coding sessions.  Read this before
modifying FrontRES/FEMR code so the assistant does not need to rediscover the
whole repository from scratch.

## One-Sentence Project Map

MOSAIC is an IsaacLab + RSL-RL motion-tracking codebase.  The active research
branch is FrontRES/FEMR: a trainable front-end residual refiner that edits the
root-level reference frame before a frozen GMT tracker consumes it.

The intended chain is:

```text
raw reference g_raw
  -> FrontRES / FEMR proposal Delta SE(3)
  -> action-cone and temporal-continuity filtering
  -> refined reference g_refin
  -> frozen GMT policy
  -> robot action
  -> IsaacLab / RobotBridge rollout diagnostics
```

## Core FrontRES Contract

FrontRES is not a replacement tracker.  It should correct reference-frame
artifacts while preserving GMT executability.

Current task-space output is:

```text
[dx, dy, dz, droll, dpitch, dyaw, tau]
```

where the first six entries form the HSL/FrontRES repair proposal and `tau` is
the HRL/PPO temporal mix variable in the current `hsl_hybrid` design.

The current design contract is in:

```text
note/FrontRES Design Contract.md
```

Use that document as the source of truth for fragile FrontRES changes.  The
most important invariant is:

```text
HSL owns the Delta SE(3) repair proposal.
HRL/PPO does not own repair direction.
HRL/PPO only owns the scalar temporal mix tau in hsl_hybrid.
```

The runtime write should follow:

\[
\Delta g^{write}_t =
(1-\tau_t)\Delta g^{HSL}_t + \tau_t \Delta g^{cont}_t .
\]

Do not silently reinterpret `tau` as amplitude, confidence, or direction.

## Main Files

### Training Entry

```text
scripts/rsl_rl/train.py
```

This launches Isaac Sim/IsaacLab, registers tasks through Hydra, builds the
environment, and uses:

```text
whole_body_tracking.utils.my_on_policy_runner.MotionOnPolicyRunner
```

Important CLI overrides:

```text
--task=FrontRES-Unified-Tracking-Flat-G1-v0
--num_envs=...
--motion ...
--headless
--device cuda:...
--supervised_warmup_iterations ...
--frontres_debug_training
--is_full_resume true|false
```

Common server command shape:

```text
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 nohup bash /hdd1/cyx/IsaacLab_mosaic/isaaclab.sh -p /hdd1/cyx/MOSAIC/scripts/rsl_rl/train.py \
    --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
    --num_envs=12000 \
    --motion /hdd1/cyx/AMASS_G1NPZ_Final \
    --logger tensorboard \
    --headless \
    --device cuda:0 \
    >/hdd1/cyx/MOSAIC/train.txt 2>&1 &
```

With `CUDA_VISIBLE_DEVICES=3`, Isaac/PyTorch see that physical GPU as
`cuda:0`, so `--device cuda:0` is intentional.

### FrontRES Runner Config

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py
```

Active class:

```text
G1FlatFrontRESUnifiedRunnerCfg
```

Important current defaults:

```text
max_iterations = 2000
frontres_training_objective = "hsl_hybrid"
frontres_specialist_mode = "rp"
frontres_active_task_dims = [0, 1, 2, 3, 4, 5, 6]
num_task_corrections = 6
task_conf_dim = 1
max_delta_pos = 0.3
max_delta_rpy = 0.4
supervised_warmup_iterations = 200
critic_warmup_iterations = 0
ppo_actor_warmup_iterations = 0
ppo_actor_ramp_iterations = 400
lambda_supervised_min = 0.20
```

Current curriculum scale notes:

```text
frontres_supervised_dr_scale_start = 1.25
frontres_supervised_dr_scale_end = 4.375
frontres_supervised_dr_ramp_iters = 1400
dr_min_scale = 1.25
dr_max_scale = 4.50
```

For `rp`, MOSAIC base perturbation is approximately `0.08 rad`, so
`dr_scale=4.375` corresponds to RobotBridge-like `epsilon=0.35`.

### Task Registration

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py
```

The active FrontRES task id is:

```text
FrontRES-Unified-Tracking-Flat-G1-v0
```

It maps to:

```text
env_cfg_entry_point: G1FlatFrontRESFinetuneEnvCfg
rsl_rl_cfg_entry_point: G1FlatFrontRESUnifiedRunnerCfg
```

### G1 FrontRES Env Config

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py
```

These define observations, reference-command layout, terminations, and the
FrontRES finetune environment.  The important idea is that FrontRES sees
anchor/reference error signals and GMT still acts on the refined reference.

### Motion Command / Perturbation Logic

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py
```

Important responsibilities:

- maintain raw / perturbed / FrontRES-corrected anchor references;
- synchronize FrontRES, noisy-GMT, and clean-GMT triplets;
- expose supervised target and anchor-error observations;
- reset FrontRES corrections and perturbation states on episode reset.

When debugging training labels, start with `observations.py` and `commands.py`.

### FrontRES Policy Module

```text
source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py
```

Important class:

```text
FrontRESActorCritic
```

It loads a frozen GMT checkpoint and owns:

- trainable `residual_actor`;
- trainable `critic`;
- frozen `gmt_policy`;
- GMT observation normalizer.

In task-space mode, the residual actor emits bounded:

```text
[Delta pos(3), Delta rpy(3), tau]
```

The module does not write task-space corrections into the environment; the
runner applies them to the motion command before GMT inference.

### FrontRES Algorithm

```text
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
```

Important class:

```text
FrontRESUnified
```

Objective modes:

```text
supervised_restore
basis_restore
hsl_hybrid
ppo_hrl and older RL-style modes
```

Current `hsl_hybrid` contract:

- supervised loss trains the Delta SE(3) proposal;
- PPO surrogate/value loss is active;
- PPO actor gradient is restricted to the `tau` output row;
- critic/value function still exists and is trained;
- `lambda_supervised` remains an anchor so PPO cannot freely rewrite direction.

Important methods and concepts:

```text
_compute_supervised_loss(...)
_compute_ppo_losses(...)
_ppo_tau_only_mode()
_keep_ppo_grad_on_tau_head_only(...)
```

If training suddenly degrades after PPO starts, inspect whether PPO is still
restricted to tau and whether diagnostics describe the value actually written
to GMT.

### Runner: FrontRES Rollout, Warmup, Write Path, Diagnostics

```text
source/rsl_rl/rsl_rl/runners/on_policy_runner.py
```

This is the densest file.  Do not edit without checking the design contract.

Key regions:

- mode detection and FrontRES setup near runner init;
- joint supervised warmup before PPO loop;
- triplet construction: FrontRES / noisy GMT / clean GMT;
- perturbation curriculum and DR scale;
- rollout-aware HSL label update;
- task-space action mask;
- runtime application of Delta SE(3);
- temporal continuity cache for `hsl_hybrid`;
- console diagnostics and save-time probe.

Key methods to search:

```text
_maybe_update_frontres_hsl_rollout_target
_frontres_project_task_target_to_action_cone
_frontres_apply_per_mode_supervised_mask
_frontres_exec_score
_frontres_exec_score_for_modes
_frontres_temporal_continuity_correction
_frontres_update_temporal_reference_cache
_frontres_invalidate_temporal_reference_cache
_apply_frontres_task_corrections
_maybe_print_frontres_restore_debug
_record_frontres_checkpoint_probe
```

Current intended runtime write in `hsl_hybrid`:

```text
proposal = HSL Delta SE(3)
continuity = previous refined reference advanced by raw frame-to-frame motion
tau = PPO scalar temporal mix
written = (1 - tau) * proposal + tau * continuity
```

The temporal cache must be invalidated on episode reset.

### Storage

```text
source/rsl_rl/rsl_rl/storage/rollout_storage.py
```

If adding a new training signal, check the storage tuple shape and minibatch
unpacking in `frontres_unified.py`.  Many FrontRES bugs are contract mismatches:
runner writes a field, algorithm expects a different field or shape.

## Validation / RobotBridge Scripts

Validation scripts live in:

```text
scripts/robustness_validation/
```

Important files:

```text
run_validation_mujoco.py
run_validation_mujoco_batch.py
results_io.py
plot_results.py
run_push2_main_figures.sh
metrics.py
push_controller.py
ou_injector.py
```

`run_validation_mujoco_batch.py` expects motion layout:

```text
motion_root/
  Walking/*.npz
  Turning/*.npz
  Upper/*.npz
  Lateral/*.npz
```

It saves each motion independently:

```text
output_dir/
  run_meta.json
  motions/<group>/<motion_stem>/
    meta.json
    results_raw.npz
    summary.csv
    status.json
    videos/*.mp4
```

Main push-2 validation script:

```text
scripts/robustness_validation/run_push2_main_figures.sh
```

It runs groups:

```text
Lateral Upper Walking
```

and then plots grouped `rp` results into:

```text
verify/figures_push2
```

Plotting script:

```text
scripts/robustness_validation/plot_results.py
```

Important plotting decisions already made:

- grouped paper figures should not include an Overall curve;
- figure titles are removed because LaTeX subfigure captions provide titles;
- Fig. 2 keeps the red zero-margin line but should not over-explain it in the
  legend;
- use `Reference frame noise epsilon` or `Reference frame noise ε`;
- avoid `Post-Push`; use `Push` if needed.

If RobotBridge code also needs updates, copy changed validation files from
this repo into:

```text
/Users/chengyuxuan/ArtiIntComVis/RobotBridge/deploy/robustness_validation/
```

Approved copy commands already exist for the main validation scripts.

## Paper / Notes

Main research notes:

```text
note/Front End Motion Refiner.md
note/FrontRES Design Contract.md
```

Paper LaTeX is not in this repository root in normal MOSAIC work; the FrontRES
paper folder has been:

```text
/Users/chengyuxuan/ArtiIntComVis/FrontRES
```

When writing paper text, use the `profleo` style if requested.  The paper
method narrative currently emphasizes:

- Reference Residual Refinement;
- Perturbation-Aligned Repair Space;
- Tracker-Aware Restoration Learning.

## Current Research Narrative

The paper should sell the architecture, not just one trained network.

The condensed story:

1. Video-extracted motion artifacts mainly appear as root-frame reference
   errors while many joint-angle estimates remain usable.
2. A frozen robust tracker has a finite robustness budget; corrupted reference
   frames consume that budget before external pushes or hardware noise appear.
3. FrontRES edits the reference in front of GMT, rather than changing GMT.
4. Perturbation family and repair space must be aligned.  High perturbation is
   not merely larger low perturbation; it can change contact, phase, and
   feasible repair directions.
5. HSL gives stable geometric/rollout-aware repair direction.
6. HRL/PPO should be restricted to physical filtering or temporal continuity
   choices, not free repair direction.
7. Demo-quality means approaching Clean behavior, not only avoiding falls.

## Training Status Concepts

Common console diagnostics:

```text
ep_len_FrontRES
ep_len_GMT (baseline)
DR scale
survival rate
supervised_cos_sim
restore ratio
mae/rmse all
mae/rmse rpy
valid target frac
L_pos/L_rot
L_mag/over/smooth
tau mix/active
write ratio/leakage
|Delta pos|
|Delta rpy|
restore rp/res/bias
grad cos PPO/Sup
lambda_supervised
PPO actor weight
learning rate
```

Interpretation reminders:

- If `ep_len_FrontRES` drops below GMT around the same DR scale repeatedly,
  suspect the repair proposal, temporal mix, or action-cone feasibility, not
  only the optimizer.
- A high supervised cosine can coexist with bad rollout performance.
- If `tau` starts acting like amplitude instead of temporal mix, the design is
  broken.
- If PPO can change Delta SE(3) direction, reward hacking is likely.
- If `restore ratio` is high but rollout worsens, the geometry target may be
  outside the dynamically feasible cone.

## Modification Checklist

Before editing FrontRES training logic:

1. Read this file.
2. Read `note/FrontRES Design Contract.md`.
3. Identify which owner is being changed:
   proposal, label, reward, tau, action cone, rollout cache, critic, storage, or
   diagnostic.
4. State the Design Delta in the response before nontrivial coding.

After editing:

1. Verify the path:

```text
config -> runner rollout construction -> storage fields -> algorithm update
-> runtime write -> diagnostics
```

2. Run `python -m py_compile` on touched Python files when practical.
3. Say whether the training command changes.
4. Say whether old branches/objectives were preserved.

## Common Failure Modes

- `frontres_training_objective` says one thing but runner/algorithm branches
  interpret it differently.
- Storage tuple shape changes in runner but minibatch unpacking is not updated.
- Debug config overrides formal config unexpectedly.
- `frontres_active_task_dims` samples perturbations that the action cone cannot
  repair.
- PPO actor gradient leaks into Delta SE(3) proposal direction.
- Runtime writes a different correction than diagnostics report.
- Temporal continuity cache is not reset when envs reset.
- Upward `dz` correction creates discontinuity and makes the robot float or
  fall.
- Mixed perturbations are introduced before single-family repairs are stable.
- RobotBridge and MOSAIC perturbation scales are compared without conversion.

