# FrontRES Design Contract

This document records the current experiment-level design contract for FrontRES.
It is more specific than the Dr.Cheng skill.  Read this before implementing
nontrivial changes to FrontRES training, rollout labels, PPO/HRL behavior, or
diagnostics.

## Version Goal

FrontRES is a front-end residual refiner before a frozen GMT tracker.  The
current version studies root-level reference artifacts, especially local
roll/pitch perturbations that become damaging at high strength.  The goal is
not only to keep the robot executable, but to make the repaired reference
approach the clean rollout closely enough for demo-quality behavior.

## Component Ownership

- **FEMR / FrontRES proposal** owns the task-space repair proposal
  \(\Delta g^{\mathrm{HSL}}_t \in SE(3)\), represented as
  \((\Delta x,\Delta y,\Delta z,\Delta r,\Delta p,\Delta y)\).
- **HSL** owns the main geometric restoration direction.  It uses Clean,
  Noisy, and Repaired rollout information to construct continuous sample
  weights, harmful-repair penalties, and rollout-aware supervised labels.
- **HRL / PPO** does not own the repair direction.  In the current hybrid
  design, PPO owns only the scalar position rejoin rate \(\rho^p_t\).
- **Position rejoin** chooses between the HSL position repair and a
  continuity-preserving position candidate:
  \[
  \Delta p^{\mathrm{write}}_t =
  (1-\rho^p_t)\Delta p^{\mathrm{cont}}_t
  + \rho^p_t\Delta p^{\mathrm{HSL}}_t.
  \]
- **Attitude repair** is written directly from HSL:
  \[
  \Delta rpy^{\mathrm{write}}_t=\Delta rpy^{\mathrm{HSL}}_t.
  \]
- **Continuity candidate** advances the previous refined reference position
  using the raw reference frame-to-frame motion.  It injects the prior that
  repaired root positions should remain dynamically continuous over time.
- **Action Cone** owns output feasibility constraints, including active
  dimensions, per-axis bounds, upward-\(z\) constraints, and jump/contact
  restrictions.
- **Diagnostics** must describe the value actually written into GMT.  They must
  not report old conceptual variables as if they were the deployed correction.

## Current Output Interface

The active `hsl_hybrid` branch uses a seven-dimensional task-space output:

\[
a^{\mathrm{FEMR}}_t =
(\Delta x,\Delta y,\Delta z,\Delta r,\Delta p,\Delta yaw,\rho^p_t).
\]

The first six dimensions are the HSL repair proposal.  The last dimension is
the PPO-owned position rejoin rate.  It is not a confidence score and it is not
an attitude gate.

The old `conf_pos` and `conf_rpy` interface is kept only for legacy objectives
and ablations.  It is not part of the current hybrid design because a pair of
confidence gates can shrink the clean-oriented repair but cannot express the
specific uncertainty we identified: how much root position should rejoin the
clean geometry versus preserve frame-to-frame dynamic continuity.

## Forbidden Freedoms

- Do not let PPO update the \(\Delta SE(3)\) proposal direction in
  `hsl_hybrid`.  PPO must be restricted to the scalar \(\rho^p_t\) output.
- Do not reinterpret \(\rho^p_t\) as a confidence score, amplitude gate, or
  attitude gate in the current `hsl_hybrid` design.
- Do not pre-shrink the HSL label and then shrink it again through
  \(\rho^p_t\).
- Do not let PPO mix or overwrite \(\Delta rpy^{\mathrm{HSL}}_t\) in the
  current position-rejoin design.
- Do not let Safe/Broken/Harmful samples dominate the proposal direction loss.
- Do not let temporal continuity cache survive an episode reset.
- Do not remove old HRL/HSL branches unless explicitly requested.  They are
  research assets for later papers.

## Sample Difficulty

Sample difficulty should remain continuous rather than hard categorical.
Use smooth gates for Safe, Repairable, Broken, and Harmful regions.  The
repairable weight should train the proposal on samples where an in-cone repair
has meaningful positive value.  Safe, Broken, and Harmful weights should either
encourage no-op behavior or suppress damaging proposals.

## Current Hybrid Training Contract

The current `hsl_hybrid` contract is:

- Supervised/HSL loss trains \(\Delta g^{\mathrm{HSL}}_t\).
- Harmful loss suppresses unsafe proposals.
- PPO uses rollout advantage but its actor gradient is restricted to
  \(\rho^p_t\).
- Runtime writes \((\Delta p^{\mathrm{write}}_t,\Delta rpy^{\mathrm{HSL}}_t)\),
  not \(\rho^p_t\Delta g^{\mathrm{HSL}}_t\).
- The first frame after reset has no valid temporal cache, so it falls back to
  HSL repair.

The core conceptual split is:

\[
\text{HSL}:\quad \text{where should the corrupted root frame move?}
\]

\[
\text{HRL/PPO}:\quad \text{how aggressively should root position rejoin that
target under dynamic continuity?}
\]

This split is deliberately asymmetric.  Roll/pitch repair is usually the
clean-oriented part that improves visual/demo quality.  Root position repair is
where strong geometric correction most often creates dynamic discontinuities,
so it receives the learned rejoin filter.

## Code Mapping

- Config:
  `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`
- Runner rollout, label construction, temporal cache, action-cone writing:
  `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- PPO/supervised loss and rho-only PPO restriction:
  `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`
- FrontRES actor output bounding:
  `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`

## Pre-Implementation Checklist

- State the Design Delta before coding.
- Identify the owner of every changed variable: proposal, label, reward, tau,
  action cone, rollout cache, or diagnostic.
- Check whether the change touches old objectives or only the active branch.
- Check whether the behavior is controlled by config or requires a command
  change.
- Check whether diagnostics still match the value actually written to GMT.

## Post-Implementation Audit

- Verify the path:
  config -> runner rollout construction -> storage fields -> algorithm update
  -> runtime write -> diagnostics.
- Run `python -m py_compile` on touched Python files when practical.
- Inspect whether old concepts remain in comments, log labels, or variable
  names in a way that can mislead future debugging.
- Explicitly report whether the training command changes.
