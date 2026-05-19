# MOSAIC / FrontRES Working Notes

This file is the local working contract for AI coding assistants in this
repository. Keep it concise and update it when the experiment design changes.

## Project Context

FrontRES is a lightweight residual corrector placed before the frozen GMT
tracker. It receives the tracking observation plus anchor-error history and
outputs task-space corrections:

```text
[dx, dy, dz, droll, dpitch, dyaw, conf_pos, conf_rpy]
```

The goal is not to replace GMT. The goal is to make corrupted reference frames
more executable by GMT, especially when visual/video extraction artifacts
consume robustness budget.

## Core Design Principles

- FrontRES should correct reference-frame artifacts, not learn a new tracker.
- Corrections must be executable by GMT. A correction that is geometrically
  closer but dynamically damaging is wrong.
- Use task-space `Delta SE(3)` rather than `Delta q` for the main FrontRES path.
- Root-level upward `dz` is dangerous because it can create dynamics
  discontinuities. Keep upward `dz` constrained unless a specific experiment
  intentionally relaxes it.
- Root sink/penetration artifacts are only partially repairable by FrontRES.
  Prefer feasible corrections such as roll/pitch or contact-consistent changes.
- Composite perturbations are a later curriculum stage. Warmup should first
  learn clear single-family correction signals.

## Training Pipeline

The intended training flow is:

1. Joint warmup
   - Actor learns the supervised anti-perturbation target.
   - Critic learns executable damage energy.
   - Warmup perturbations should be clear and balanced, usually one family at a
     time.

2. Actor takeover
   - PPO actor weight ramps up.
   - DR scale should remain controlled to avoid critic distribution shift.

3. PPO fine-tuning
   - Actor is fully active.
   - Perturbation curriculum can introduce mixed perturbation families.
   - Boundary DR should keep the batch near the repairable frontier, not deep in
     broken states.

## Perturbation Curriculum

Use two different schedules:

- Warmup: `balanced_single`
  - one perturbation family per rollout;
  - balanced across `planar`, `yaw`, `global_z`, `local_rp`;
  - purpose: clean supervised labels.

- RL: curriculum from single to mixed perturbations
  - early: single families;
  - middle: pairs;
  - late: occasional three/full combinations;
  - purpose: robustness to realistic composite artifacts.

The curriculum must respect `frontres_active_task_dims`. Do not sample
perturbation families that the active action cone cannot repair.

## Reward / Energy Notes

Avoid using the full environment reward directly for FrontRES. Teleoperation,
tracking, or unrelated task terms can introduce noise.

Prefer executable reward components:

- planar executability for `dx/dy/dyaw`;
- vertical/contact executability for `dz/droll/dpitch`;
- a weak task-consistency term only when needed to prevent trivial no-motion
  fixes.

Important diagnostics:

- `gap`: estimated executable damage before repair;
- `gain`: executable improvement from FrontRES;
- `ratio`: normalized repair gain;
- `positive_gain_frac`: fraction of samples with positive gain;
- `safe/fragile/broken`: distribution of sample difficulty;
- `damage/broken/actor_gate`: whether the actor is being updated on the right
  samples;
- `exec planar/vertical/task`: reward decomposition for mismatch debugging.

If gain becomes negative, first check whether the perturbation family, action
cone, and reward component are aligned.

## Validation Experiments

Validation is separate from FrontRES training. It demonstrates that reference
frame errors consume robustness budget.

Preferred story:

```text
reference-frame error epsilon increases
  -> post-push stability margin decreases
  -> push recovery rate drops
  -> FrontRES is motivated
```

Store each motion sequence independently so failures do not invalidate the
whole run. Plot scripts should read a results directory containing per-motion
subdirectories with both metadata and raw arrays.

For videos, RobotBridge/MuJoCo is preferred for presentation artifacts. For
training-side quantitative validation, IsaacLab remains acceptable if it matches
the training environment.

## Coding Rules For This Repo

- Do not revert user changes.
- Keep changes scoped to the current experiment.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Run at least `python -m py_compile` after Python code changes when practical.
- When touching FrontRES training logic, check:
  - resume/cold-start behavior;
  - debug mode overrides;
  - active action mask;
  - perturbation schedule;
  - reward diagnostics.

## Common Pitfalls

- Warmup diagnostics can be misleading if the current perturbation family has no
  signal for a dimension. Always inspect `modes=(...)` together with
  `valid_pos/valid_rpy`.
- A high supervised cosine does not guarantee PPO reward alignment.
- Composite perturbations can create reward conflict if one scalar executable
  reward is asked to represent multiple repair cones.
- Broken samples should not dominate actor updates.
- If `broken_frac` is too high, reduce DR scale or simplify the perturbation
  curriculum before changing the network.
