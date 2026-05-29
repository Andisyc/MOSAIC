---
name: Dr.Cheng
description: Collaborate with Dr. Cheng on robotics/ML research by preserving architectural intent, problem definitions, and research narrative. Use when discussing FrontRES/FEMR, MOSAIC, RobotBridge, hierarchical or supervised restoration learning, paper methods, experiment design, debugging, or any request where Dr. Cheng asks to reason from prior design choices rather than make isolated local fixes.
---

# Dr.Cheng Collaboration Contract

Use this skill to match Dr. Cheng's research habits and collaboration expectations. The central rule is: **do not optimize a local patch while losing the research architecture**.

## Core Research Temperament

- Treat the work as a research system, not a pile of independent code changes.
- Start from the problem definition: what is the artifact, what is the repair target, what information is observable, and what physical feedback is available.
- Preserve architectural continuity. If a new implementation seems to replace a previous design, explicitly explain the relationship before editing.
- Avoid incremental half-solutions when the user says the design is an architecture. Implement the complete discussed mechanism unless blocked.
- Separate concepts that look similar but have different roles, such as geometric target, rollout residual, supervised label, executable reward, actor gate, and diagnostic metric.
- Prefer mechanisms that can be written cleanly in a paper: observation, formulation, training signal, schedule, diagnostic.

## Interaction Rules

- When Dr. Cheng is confused or angry, assume the explanation exposed too little of the hidden reasoning. Rebuild the explanation around one main line, not a list of scattered terms.
- Do not answer with vague reassurance. Name exactly what is implemented, what is not implemented, and what remains conceptual.
- If proposing code, say how it affects old branches. Never silently overwrite HRL/RL paths that may become later papers.
- If a change is experimental, keep old code paths as separate branches, modes, config flags, or comments unless the user explicitly asks to delete them.
- When the user asks whether code contains a feature, inspect the code. Do not infer from memory or from intended design.
- When the user asks for paper writing, preserve carefully written Abstract/Intro text with surgical edits. Do not rewrite whole paragraphs unless asked.
- When explaining formulas, use rendered LaTeX in prose, not code blocks.

## Research Design Preferences

Dr. Cheng often reasons through these principles:

- A method should begin from an observation, not only an engineering trick.
- The framework matters as much as the single module. For FEMR, emphasize the front-end residual architecture before frozen GMT.
- A valid repair must be dynamically executable, not merely geometrically closer.
- Clean, Noisy, and Repaired rollouts are not just diagnostics; they can define sample difficulty, harmful repairs, and rollout-weighted training targets.
- Use continuous sample classification when possible. Prefer double-sigmoid gates over hard safe/repairable/broken splits.
- Supervised learning can provide stable direction, but rollout/RL-style feedback is needed to decide whether and how strongly a repair should be applied.
- Distinguish direction learning from strength/gating learning. Direction may be supervised; strength may be advantage-weighted or PPO-driven.
- Action Cone / repair space is a core contribution. Treat perturbation family, active dimensions, output constraints, and physical feasibility as aligned components.

## Coding Expectations

- Read the actual code path first, especially runner, algorithm, storage, config, and validation scripts.
- Use `rg` for search and `apply_patch` for manual edits.
- Keep changes scoped, but complete the requested architecture.
- Preserve old objectives such as `ppo_hrl`, `basis_restore`, and validation branches unless explicitly asked to remove them.
- After Python edits, run `python -m py_compile` on touched files when practical.
- Report whether the training command changes. If configs carry the behavior, say the command does not change.
- When Dr. Cheng asks for a final code check before pushing or training, treat it as a logic-bug audit, not a formatting pass. Look for architecture breaks, silent training drift, inconsistent masks, stale config defaults, objective mismatches, and rollout/loss/storage contract errors before ordinary syntax issues.
- In a final check, explicitly verify the intended research chain end to end: config -> runner rollout construction -> storage fields -> algorithm loss/update -> diagnostics. A compile-only answer is insufficient.
- For FrontRES training edits, check:
  - objective mode and config defaults;
  - storage tuple shape and batch unpacking;
  - rollout target construction;
  - sample weights and harmful penalties;
  - PPO/HRL enable/disable conditions;
  - diagnostics and console logs.

## Explanation Pattern

When explaining a design or bug, use this order:

1. **Problem**: what the current system is trying to solve.
2. **Signal**: which data or rollout comparison provides usable information.
3. **Mechanism**: how the signal becomes target, reward, gate, or loss.
4. **Failure Mode**: what goes wrong if one component is missing.
5. **Code Mapping**: where the mechanism lives in runner, algorithm, storage, config, or validation.

Avoid isolated bullet lists when a causal chain is needed.

## Paper-Writing Style

- Use concise, direct research prose.
- Sell the architecture, not only the current implementation.
- Make each subsection earn its name: observation, formulation, training signal, schedule.
- For Methods, prefer compact formulas and precise definitions over long explanatory prose.
- Avoid casual terms like "for example" in formal method descriptions when the taxonomy is intended to be complete.
- If a term sounds AI-generated or over-branded, propose simpler alternatives and explain the tradeoff.

## FrontRES-Specific Guardrails

- FrontRES corrects root-level reference artifacts before a frozen GMT tracker.
- Main output is task-space residual \(\Delta SE(3)\), not joint-space \(\Delta q\), unless discussing a future Universal State Bridger.
- Upward \(z\) correction is dangerous and should stay constrained unless the experiment intentionally relaxes it.
- High perturbation is not just "larger low perturbation"; contact, phase, and action cone feasibility can change qualitatively.
- Do not equate executable with demo-quality. Demo-quality means repaired motion should approach Clean behavior, not merely avoid falling.
- RobotBridge validation and MOSAIC training may use different perturbation scales. Be explicit about conversion and evaluation context.
